"""Scripted macro commander: live formation slots + phase machine.

Option C (docs/design-options-c-d.md §1): the macro layer -- which
formation to hold, where to push, when to switch -- is a deterministic
script; RL (variant C1) only supplies the micro. WarEnv injects
commander instances and builds the macro part of every observation and
the slot-shaping reward from their orders, so training and serving
share exactly one construction path (the "observations have a single
owner" lesson). The pure scripted variant (CommanderPolicy in
war_sim.scripted, league name "command") drives units straight at
their slots and doubles as the round-robin deadlock/oscillation probe.

One Commander instance serves ONE team. Its per-episode state (phase,
anchor, slot assignment) is re-initialized when the simulation step
counter moves backwards (env reset) and advanced at most once per
step, so calling order() repeatedly within one step is safe (cached).

Per-step order dict (Commander.order / ensure_order):

    {"phase":  "advance" | "engage" | "envelop" | "regroup",
     "anchor": (ax, ay),   # formation reference point (front-line center)
     "facing": float,      # formation axis (rad), toward the enemy
     "slots":  {unit_id: (sx, sy)}}   # world-space slot targets

Phase machine (thresholds hysteresis-paired; every judgment reads only
quantities the commander's own orders cannot flip -- contact distance,
enemy concentration, own losses):

    advance  default. The anchor pushes toward the enemy centroid, but
             only while the team keeps up (mean slot distance below
             ANCHOR_WAIT_K * spacing): a runaway anchor would dissolve
             the formation into a permanent chase.
    engage   contact: nearest own/enemy unit pair closer than
             ENGAGE_IN_K * weapon_range. The anchor freezes; micro
             takes over (slot reward = 0, macro_free = 1 in the obs).
             Exits only beyond ENGAGE_OUT_K * weapon_range.
    envelop  the enemy is concentrated (mean distance from the enemy
             units to their centroid -- RMS spread -- below
             ENVELOP_CONC_IN px; a stacked deathball sits near 10-20,
             any deployed line/wedge/crescent above 100) while our
             doctrine is line/crescent and contact is within
             ENVELOP_PROBE_K * weapon_range (a ring slot behind a
             far-away blob would be reached by running through it).
             Slots become a live firing ring around the enemy
             centroid: formation center at the front, wings sweeping
             around to the enemy rear (flank-bonus harvest).
    regroup  one-shot per episode when under REGROUP_FRAC of the team
             is left and we are out of contact: rebuild the slot grid
             at the team centroid (polar rematch) and re-form.

Slot assignment is stability-first (the doc's core requirement): the
polar-angle baseline matching runs exactly twice per episode -- at
spawn and at regroup entry -- and units then keep their slots forever;
dead units' slots simply stay vacant until a regroup rebuilds the
grid. Re-running a global assignment every step (or reshuffling on
every death) would send units crossing each other.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from .core import DEPLOY_X, FORMATION_NAMES

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycle
    from .env import WarEnv

PHASES = ("advance", "engage", "envelop", "regroup")

# Formations that make sense as a commander doctrine (scatter has no
# slot grid -- it is a spawn-only deployment).
DOCTRINE_NAMES = tuple(f for f in FORMATION_NAMES if f != "scatter")

# --- tunables (kept as constructor kwargs so a future meta-policy can
# --- replace the phase machine without touching the engine config) ---
ENGAGE_IN_K = 1.2     # enter engage: contact dist < 1.2 * weapon_range
ENGAGE_OUT_K = 1.5    # leave engage: contact dist > 1.5 * weapon_range
ENVELOP_CONC_IN = 40.0    # enter envelop: enemy RMS spread below this (px)
ENVELOP_CONC_OUT = 55.0   # leave envelop: enemy RMS spread above this (px)
ENVELOP_PROBE_K = 3.0     # only envelop within 3.0 * weapon_range contact
ENVELOP_RING_K = 0.9      # ring radius = 0.9 * weapon_range (firing range)
ENVELOP_SWEEP = math.radians(150.0)  # arc from the front to each wing tip
ENVELOP_ORBIT_W = 0.05    # ring angular speed (rad/step): a rotating ring
                          # keeps headings fresh, denying the flank bonus a
                          # stationary firing squad hands to a moving enemy
ENVELOP_MIN_FOES = 5       # fewer enemies: plain engage is enough
REGROUP_FRAC = 0.4         # alive fraction triggering a one-shot regroup
REGROUP_FORMED_K = 2.0     # regroup done: mean slot dist < 2.0 * spacing
ANCHOR_WAIT_K = 2.5        # anchor advances while mean slot dist < 2.5 * spacing
WORLD_MARGIN = 40.0        # anchor / slot / ring clamp margin (anti-wall)


def _centroid(units) -> Optional[Tuple[float, float]]:
    if not units:
        return None
    return (
        sum(u.x for u in units) / len(units),
        sum(u.y for u in units) / len(units),
    )


def _hull_area(pts: List[Tuple[float, float]]) -> float:
    """Convex-hull area (monotone chain + shoelace); 0 if degenerate."""
    if len(pts) < 3:
        return 0.0
    pts = sorted(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    s = 0.0
    for (x1, y1), (x2, y2) in zip(hull, hull[1:] + hull[:1]):
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


class Commander:
    """Per-team macro controller (option C). Fresh state per episode,
    everything else recomputed from the simulator state every step."""

    def __init__(
        self,
        formation: str = "line",
        *,
        engage_in_k: float = ENGAGE_IN_K,
        engage_out_k: float = ENGAGE_OUT_K,
        envelop_conc_in: float = ENVELOP_CONC_IN,
        envelop_conc_out: float = ENVELOP_CONC_OUT,
        envelop_probe_k: float = ENVELOP_PROBE_K,
        envelop_ring_k: float = ENVELOP_RING_K,
        envelop_sweep: float = ENVELOP_SWEEP,
        regroup_frac: float = REGROUP_FRAC,
        anchor_wait_k: float = ANCHOR_WAIT_K,
    ):
        if formation not in DOCTRINE_NAMES:
            raise ValueError(
                f"formation={formation!r} is not a doctrine;"
                f" choose one of {DOCTRINE_NAMES}"
            )
        if engage_out_k <= engage_in_k:
            raise ValueError("engage_out_k must exceed engage_in_k (hysteresis)")
        if envelop_conc_out <= envelop_conc_in:
            raise ValueError("envelop_conc_out must exceed envelop_conc_in")
        self.formation = formation
        self.engage_in_k = engage_in_k
        self.engage_out_k = engage_out_k
        self.envelop_conc_in = envelop_conc_in
        self.envelop_conc_out = envelop_conc_out
        self.envelop_probe_k = envelop_probe_k
        self.envelop_ring_k = envelop_ring_k
        self.envelop_sweep = envelop_sweep
        self.regroup_frac = regroup_frac
        self.anchor_wait_k = anchor_wait_k
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        """Forget the episode: initial phase, no slots, no anchor."""
        self.phase = "advance"
        self._anchor: Optional[Tuple[float, float]] = None
        self._facing = 0.0
        self._grid: List[Tuple[float, float]] = []   # local slot coords
        self._assign: Dict[str, int] = {}            # unit_id -> grid index
        self._order_t = -1                           # sim step of cached order
        self._order: Optional[Dict] = None
        self._regroup_done = False
        self._enemy_c: Optional[Tuple[float, float]] = None  # last seen
        self._sim_cfg = None
        self._ring_rot = 0.0  # envelop ring rotation accumulator

    @property
    def current(self) -> Optional[Dict]:
        """The cached order for the current step (None before the first)."""
        return self._order

    # ------------------------------------------------------------------
    def order(self, env: "WarEnv", team: int) -> Dict:
        """Macro order valid for the current simulator state."""
        return self.ensure_order(env, team, env.sim.step_count)

    def ensure_order(self, env: "WarEnv", team: int, t: int) -> Dict:
        """Order for sim step `t`; computed at most once per step.

        A step counter moving backwards (env reset) re-initializes the
        episode state, which is what keeps a shared policy-owned
        commander correct across episodes without explicit reset calls.
        """
        if self._order_t == t and self._order is not None:
            return self._order
        if t < self._order_t:
            self.reset()
        self._order_t = t
        self._order = self._compute(env, team)
        return self._order

    # ------------------------------------------------------------------
    def _compute(self, env: "WarEnv", team: int) -> Dict:
        sim = env.sim
        cfg = sim.cfg
        self._sim_cfg = cfg
        mine = sim.living(team)
        foes = sim.living(1 - team)

        if self._anchor is None:  # first order of the episode
            if team == 0:
                self._anchor = (DEPLOY_X, cfg.world_h / 2.0)
                self._facing = 0.0
            else:
                self._anchor = (cfg.world_w - DEPLOY_X, cfg.world_h / 2.0)
                self._facing = math.pi
            self._build_grid(sim, team, len(mine))

        enemy_c = _centroid(foes)
        if enemy_c is not None:
            self._enemy_c = enemy_c
        else:  # no enemies left: freeze the last meaningful geometry
            return self._slots_order(mine, self.phase)

        wr = cfg.weapon_range
        d_contact = self._contact_dist(
            mine, foes, early_below_sq=(wr * self.engage_in_k) ** 2
        )
        conc = self._concentration(foes)

        self._update_phase(sim, team, mine, d_contact, conc, cfg)
        if self.phase == "advance":
            self._advance_anchor(mine, enemy_c, cfg)
        return self._slots_order(mine, self.phase)

    # --- phase machine --------------------------------------------------
    def _update_phase(self, sim, team, mine, d_contact, conc, cfg):
        wr = cfg.weapon_range
        in_contact = d_contact < wr * self.engage_in_k
        out_contact = d_contact > wr * self.engage_out_k
        envelopable = self.formation in ("line", "crescent")

        if self.phase == "regroup":
            if in_contact or self._formed(mine):
                self.phase = "advance"  # enemy arrived, or team re-formed
            return

        low_strength = len(mine) < self.regroup_frac * cfg.n_units_per_side
        if not self._regroup_done and low_strength and out_contact:
            self.phase = "regroup"
            self._regroup_done = True  # one regroup per episode
            self._regroup(sim, team, mine)
            return

        if self.phase == "envelop":
            dispersed = conc is None or conc > self.envelop_conc_out
            too_far = d_contact >= wr * self.envelop_probe_k
            if dispersed or too_far:
                self.phase = "engage" if in_contact else "advance"
            return

        if (
            envelopable
            and conc is not None
            and conc < self.envelop_conc_in
            and d_contact < wr * self.envelop_probe_k
        ):
            self.phase = "envelop"
        elif in_contact:
            self.phase = "engage"
        elif out_contact:
            self.phase = "advance"
        # else: inside the engage hysteresis band -> keep the phase

    def _formed(self, mine) -> bool:
        """Regroup progress: mean distance to the current slots."""
        if not mine or not self._grid or not self._assign:
            return False
        slots = self._grid_slots(mine)
        ds = [
            math.hypot(u.x - slots[u.id][0], u.y - slots[u.id][1])
            for u in mine if u.id in slots
        ]
        if not ds:
            return True  # nobody left to form up
        return sum(ds) / len(ds) < REGROUP_FORMED_K * self._sim_cfg.formation_spacing

    # --- anchor ----------------------------------------------------------
    def _advance_anchor(self, mine, enemy_c, cfg):
        ax, ay = self._anchor
        dx, dy = enemy_c[0] - ax, enemy_c[1] - ay
        n = math.hypot(dx, dy)
        if n < 1e-9:
            return
        step = cfg.move_speed * cfg.dt
        cand = self._clamp((ax + dx / n * step, ay + dy / n * step), cfg)
        facing = math.atan2(enemy_c[1] - cand[1], enemy_c[0] - cand[0])
        # Gate on the team keeping up: while the formation is still
        # assembling, hold the anchor instead of dragging slots away.
        mean_d = self._mean_slot_dist(mine, cand, facing)
        if mean_d is not None and mean_d > self.anchor_wait_k * cfg.formation_spacing:
            return
        self._anchor, self._facing = cand, facing

    def _mean_slot_dist(self, mine, anchor, facing) -> Optional[float]:
        if not self._assign:
            return None
        cos_f, sin_f = math.cos(facing), math.sin(facing)
        total, count = 0.0, 0
        for u in mine:
            idx = self._assign.get(u.id)
            if idx is None:
                continue
            fx, fy = self._grid[idx]
            sx = anchor[0] + fx * cos_f - fy * sin_f
            sy = anchor[1] + fx * sin_f + fy * cos_f
            total += math.hypot(u.x - sx, u.y - sy)
            count += 1
        return total / count if count else None

    # --- slots -------------------------------------------------------------
    def _build_grid(self, sim, team, n: int):
        """(Re)build the local slot grid and (re)assign by polar match.

        Runs at spawn and at regroup entry -- the only two moments slot
        assignment ever changes."""
        if n <= 0:
            self._grid, self._assign = [], {}
            return
        self._grid = sim._formation_slots(self.formation, n)
        self._assign = self._polar_match(sim, team)

    def _polar_match(self, sim, team) -> Dict[str, int]:
        """Match units to slots by polar order around the anchor.

        Both sides are sorted by (angle around the anchor, radius) and
        zipped: no two matched pairs cross, and units already standing
        on their slots (spawned in the doctrine formation) match
        themselves."""
        ax, ay = self._anchor

        def sort_key(world_xy):
            dx, dy = world_xy[0] - ax, world_xy[1] - ay
            ang = math.atan2(dy, dx)
            if ang < 0.0:
                ang += 2.0 * math.pi
            return (ang, dx * dx + dy * dy)

        units = sorted(sim.living(team), key=lambda u: sort_key((u.x, u.y)))
        slots = sorted(
            range(len(self._grid)),
            key=lambda i: sort_key(self._slot_world(i)),
        )
        return {u.id: idx for u, idx in zip(units, slots)}

    def _slot_world(self, idx: int) -> Tuple[float, float]:
        """Grid slot -> world coordinates at the current anchor/facing.

        Local frame: +fx toward the enemy (0 = front line), +fy to the
        team's left; identical to BattleSimulator's deployment frame."""
        fx, fy = self._grid[idx]
        cos_f, sin_f = math.cos(self._facing), math.sin(self._facing)
        return (
            self._anchor[0] + fx * cos_f - fy * sin_f,
            self._anchor[1] + fx * sin_f + fy * cos_f,
        )

    def _regroup(self, sim, team, mine):
        """Rebuild the formation around the current team centroid."""
        c = _centroid(mine)
        if c is not None:
            self._anchor = self._clamp(c, sim.cfg)
        if self._enemy_c is not None:
            self._facing = math.atan2(
                self._enemy_c[1] - self._anchor[1],
                self._enemy_c[0] - self._anchor[0],
            )
        self._build_grid(sim, team, len(mine))

    def _slots_order(self, mine, phase) -> Dict:
        if phase == "envelop":
            slots = self._ring_slots(mine)
        else:
            slots = self._grid_slots(mine)
        return {
            "phase": phase,
            "anchor": self._anchor,
            "facing": self._facing,
            "slots": slots,
        }

    def _grid_slots(self, mine) -> Dict[str, Tuple[float, float]]:
        cfg = self._sim_cfg
        return {
            u.id: self._clamp(self._slot_world(self._assign[u.id]), cfg)
            for u in mine if u.id in self._assign
        }

    def _ring_slots(self, mine) -> Dict[str, Tuple[float, float]]:
        """Envelop: one live firing ring around the enemy centroid.

        Each slot's lateral grid coordinate fy decides its place on the
        ring: the formation center (fy ~ 0) holds the front arc, the
        outer wings sweep around toward the enemy rear. The ring is
        recomputed every step and follows the enemy centroid, and it
        slowly rotates (ENVELOP_ORBIT_W): moving units refresh their
        heading, so a counter-attacking blob cannot farm the flank
        bonus off a stationary firing squad (the 'static line gets
        backstabbed' lesson applies to rings just as much)."""
        cfg = self._sim_cfg
        ec = self._enemy_c
        self._ring_rot = (self._ring_rot + ENVELOP_ORBIT_W) % (2.0 * math.pi)
        theta_front = math.atan2(
            self._anchor[1] - ec[1], self._anchor[0] - ec[0]
        )
        max_fy = max((abs(fy) for _, fy in self._grid), default=1.0) or 1.0
        radius = self.envelop_ring_k * cfg.weapon_range
        slots = {}
        for u in mine:
            idx = self._assign.get(u.id)
            if idx is None:
                continue
            _, fy = self._grid[idx]
            side = 1.0 if fy >= 0.0 else -1.0
            t = min(1.0, abs(fy) / max_fy)
            theta = theta_front - side * self.envelop_sweep * t + self._ring_rot
            slots[u.id] = self._clamp(
                (ec[0] + radius * math.cos(theta),
                 ec[1] + radius * math.sin(theta)),
                cfg,
            )
        return slots

    # --- helpers -----------------------------------------------------------
    @staticmethod
    def _clamp(p, cfg) -> Tuple[float, float]:
        return (
            max(WORLD_MARGIN, min(cfg.world_w - WORLD_MARGIN, p[0])),
            max(WORLD_MARGIN, min(cfg.world_h - WORLD_MARGIN, p[1])),
        )

    @staticmethod
    def _contact_dist(mine, foes, early_below_sq: Optional[float] = None) -> float:
        """Nearest own/enemy unit pair distance. Early-exits once below
        `early_below_sq` (the engage threshold): the exact minimum only
        matters for the 'left contact' comparison, which such a distance
        can never satisfy."""
        best = math.inf
        for u in mine:
            for e in foes:
                dx, dy = u.x - e.x, u.y - e.y
                d2 = dx * dx + dy * dy
                if d2 < best:
                    best = d2
                    if early_below_sq is not None and best < early_below_sq:
                        return math.sqrt(best)
        return math.sqrt(best)

    @staticmethod
    def _concentration(foes) -> Optional[float]:
        """Enemy spread = RMS distance from the enemy units to their
        centroid (small = bunched). Chosen over convex-hull area per
        unit because collinear deployments have degenerate (near-zero)
        hull area and would read as perfectly concentrated. None when
        there are too few enemies for envelopment to matter."""
        if len(foes) < ENVELOP_MIN_FOES:
            return None
        cx = sum(e.x for e in foes) / len(foes)
        cy = sum(e.y for e in foes) / len(foes)
        ms = sum((e.x - cx) ** 2 + (e.y - cy) ** 2 for e in foes) / len(foes)
        return math.sqrt(ms)
