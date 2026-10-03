"""Standalone multi-agent environment around BattleSimulator.

No PettingZoo / gymnasium dependency: the only consumer is the standalone
PPO trainer (see war_sim.ppo and train.py), so the API stays minimal.

step() semantics (per env step):
    obs        next observation per unit. During an episode: only survivors.
               On the final step: every unit that acted this step gets a
               terminal observation (survivors need it for value
               bootstrapping at truncation).
    rewards    exactly for the units that acted this step (dense team signal).
    dead       set of acting units destroyed this step; their trajectory
               ends here (no bootstrap).
    truncated  True iff the episode ended by the step limit on this step.

The combat rules live in BattleSimulator; this file only adapts it for RL.

Macro layer (option C, variant C1 -- see war_sim.commander): WarEnv can
hold one Commander per team (constructor argument or per-episode
reset(commanders=...) override, same revert semantics as formations=).
Attached commanders advance once per simulation step and provide the
macro context; they never touch the simulator itself, so battles play
out identically with or without them.

Observations are structured sets (option D -- see
docs/design-options-c-d.md §2.2): each unit sees a SetObs record with

    self_feat  the full legacy flat vector (SELF_DIM = 55): self state,
               3-nearest-enemy/teammate summaries (a free local prior,
               kept as an incremental upgrade rather than a replacement),
               both centroids, local crowding and -- for commander teams
               -- the macro block (slot vector, phase one-hot, macro_free;
               all-zero phase one-hot marks "no macro")
    ally/enemy padded [max_units, REL_DIM] sets of relative features
               (dx, dy, d, hp, sin_h, cos_h) for ALL living others,
               sorted by distance (near first; distance sort gives the
               unordered set a stable canonical order)
    masks      [max_units] validity flags (0 = padding row)
    role       lifelong structural role id (left wing / center / right
               wing / vanguard / rear): from the team commander's slot
               grid when attached (the C+D combined form), else the spawn
               formation's grid, else a chirality-corrected initial-y
               thirds fallback for scatter deployments

ObsSpec/collate/zero_obs define the single shape contract and the single
batching code shared by training, evaluation and serving (the
"observations have one owner" lesson).

Rewards per acting unit, per step:
    * team kill/loss delta (+/- 2.0 per unit)
    * team damage delta (0.001 per hp) -- coordination signal
    * personal damage dealt (0.002 per hp) -- individual credit; flank
      bonuses from the engine flow through here, so maneuvering for
      flank shots pays the unit that did the maneuvering
    * potential-based shaping for closing on the nearest enemy (cold
      start; telescopes over an episode, bounded by ~0.6)
    * potential-based shaping for spacing: crowded units are rewarded
      for opening distance to their nearest teammate (bounded, cannot
      be farmed). Counters the natural collapse into a deathball.
    * potential-based slot shaping (commander teams only): rewarded for
      closing on the live slot target, zero in the engage phase (micro
      takes over) and on the death step; bounded by SLOT_K per episode.
      Note the design doc's phi = -max(0, 1 - d/ideal) has the sign
      flipped (it grows with distance -- that would reward fleeing the
      slot); the implemented potential is -min(1, d/ideal), which pays
      for approaching and saturates one spacing out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, Optional, Set, Tuple

import numpy as np

from .core import BattleConfig, BattleSimulator
from .commander import PHASES

# Self-feature block of the structured observation: the full legacy
# flat vector (K-NN summaries + macro block included).
SELF_DIM = 55
N_ACTIONS = 9  # stay + 8 movement directions (see BattleSimulator.ACTIONS)

# How many nearest enemies/teammates appear in the self-feature
# K-NN summaries (kept from the flat-obs era as a local prior).
K_NEIGHBORS = 3

# Macro observation block (option C): slot dx, dy, slot distance,
# phase one-hot (4), macro_free flag.
MACRO_DIM = 3 + len(PHASES) + 1

# Set-observation block (option D §2.2).
REL_DIM = 6  # per set element: (dx, dy, d, hp, sin_h, cos_h)
ROLE_NAMES = ("left_wing", "center", "right_wing", "vanguard", "rear")
N_ROLES = len(ROLE_NAMES)

# Potential-based reward shaping constants (all terms telescope over an
# episode, so none of them can be farmed by oscillating back and forth).
SHAPE_DIST_K = 0.5    # closing on the nearest enemy (cold-start signal)
SPACE_K = 0.25        # opening distance from a too-close teammate
SPACE_IDEAL = 30.0    # desired nearest-teammate distance (world units)
PERSONAL_DMG_K = 0.002  # personal damage-dealt credit (flanking pays)
SLOT_K = 0.2          # closing on the commander slot (macro shaping)

# Local-crowding observation: allies within this radius (world units).
DENSITY_RADIUS = 60.0


@dataclass(frozen=True)
class ObsSpec:
    """Shape contract of the structured observation (option D).

    max_units caps one side of a set. The design doc's
    U = 2 * n_units_per_side headroom is unnecessary: allies (<= n-1)
    and enemies (<= n) are each bounded by one side's headcount, and
    the doc's own memory estimate assumed U = n per set.
    """

    self_dim: int
    rel_dim: int
    max_units: int
    n_roles: int

    @property
    def flat_dim(self) -> int:
        """Size of the flattened set observation ("mlp" baseline arch)."""
        return (
            self.self_dim
            + 2 * self.max_units * self.rel_dim
            + 2 * self.max_units
        )

    def as_dict(self) -> dict:
        return {
            "self_dim": self.self_dim,
            "rel_dim": self.rel_dim,
            "max_units": self.max_units,
            "n_roles": self.n_roles,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ObsSpec":
        return cls(
            self_dim=int(d["self_dim"]),
            rel_dim=int(d["rel_dim"]),
            max_units=int(d["max_units"]),
            n_roles=int(d["n_roles"]),
        )


def obs_spec_for(cfg: BattleConfig) -> ObsSpec:
    """The observation spec a WarEnv with this config produces."""
    return ObsSpec(SELF_DIM, REL_DIM, cfg.n_units_per_side, N_ROLES)


@dataclass
class SetObs:
    """One unit's structured observation (option D §2.2)."""

    self_feat: np.ndarray  # [self_dim] float32
    ally: np.ndarray       # [U, rel_dim] distance-sorted, zero-padded
    enemy: np.ndarray      # [U, rel_dim]
    ally_mask: np.ndarray  # [U] float32, 1 = valid element
    enemy_mask: np.ndarray  # [U]
    role: int              # structural role id, lifelong per episode


def collate(obs) -> dict:
    """Batch SetObs records into numpy arrays.

    The single batching code for training, evaluation and serving --
    every consumer stacks observations through this function so the
    field order/dtypes can never drift between them.
    """
    items = list(obs)
    return {
        "self": np.stack([o.self_feat for o in items]),
        "ally": np.stack([o.ally for o in items]),
        "enemy": np.stack([o.enemy for o in items]),
        "ally_mask": np.stack([o.ally_mask for o in items]),
        "enemy_mask": np.stack([o.enemy_mask for o in items]),
        "role": np.asarray([o.role for o in items], dtype=np.int64),
    }


def zero_obs(spec: ObsSpec, role: int = ROLE_NAMES.index("center")) -> SetObs:
    """All-zero observation of the given spec (test placeholder)."""
    return SetObs(
        self_feat=np.zeros(spec.self_dim, np.float32),
        ally=np.zeros((spec.max_units, spec.rel_dim), np.float32),
        enemy=np.zeros((spec.max_units, spec.rel_dim), np.float32),
        ally_mask=np.zeros(spec.max_units, np.float32),
        enemy_mask=np.zeros(spec.max_units, np.float32),
        role=role,
    )


def _slot_role(fx: float, fy: float, grid) -> int:
    """Bucket a formation-slot coordinate into a structural role id.

    Lateral first: units in the outer WING_FRAC of the grid's lateral
    extent are wings (fy sign decides left/right -- the team's own left,
    matching the local deployment frame). The middle band then splits
    by depth into vanguard (front third, +fx faces the enemy), center
    and rear (back third). Degenerate extents (single rank / single
    file) collapse to the coarser distinction instead of crashing.
    """
    max_fy = max((abs(f) for _, f in grid), default=0.0)
    if max_fy > 1e-9 and abs(fy) > 0.35 * max_fy:
        return ROLE_NAMES.index("left_wing" if fy > 0.0 else "right_wing")
    fxs = [f for f, _ in grid]
    lo, hi = min(fxs), max(fxs)
    if hi - lo > 1e-9:
        t = (fx - lo) / (hi - lo)  # 0 = rearmost, 1 = frontmost
        if t > 2.0 / 3.0:
            return ROLE_NAMES.index("vanguard")
        if t < 1.0 / 3.0:
            return ROLE_NAMES.index("rear")
    return ROLE_NAMES.index("center")


class WarEnv:
    def __init__(
        self,
        config: BattleConfig | None = None,
        seed: int | None = None,
        commanders: tuple | dict | None = None,
    ):
        self.cfg = config or BattleConfig()
        self.seed = seed
        self.sim = BattleSimulator(self.cfg, seed=seed)
        self.episode_done = True
        # Base attachment (constructor) vs current episode attachment;
        # a plain reset() reverts to the base (same contract as the
        # formations= override).
        self._base_commanders = self._normalize_commanders(commanders)
        self._commanders: dict | None = None
        # Lifelong structural roles (option D), assigned per reset.
        self._roles: Dict[str, int] = {}

    @staticmethod
    def _normalize_commanders(commanders) -> dict | None:
        """(blue, red) tuple (entries may be None) or {team: Commander}
        -> {team: Commander} without None entries; None if empty."""
        if commanders is None:
            return None
        items = (
            dict(commanders).items() if isinstance(commanders, dict)
            else enumerate(commanders)
        )
        out = {int(team): c for team, c in items if c is not None}
        if len(set(map(id, out.values()))) != len(out):
            raise ValueError(
                "the same Commander instance cannot serve both teams:"
                " its per-episode state (phase/anchor/slots) is per-team"
            )
        return out or None

    @property
    def commanders(self) -> dict:
        """Commanders attached to the current episode (may be empty)."""
        return self._commanders or {}

    @property
    def obs_spec(self) -> ObsSpec:
        """The structured-observation shape contract of this env."""
        return obs_spec_for(self.cfg)

    def _macro(self, team: int) -> Optional[dict]:
        """Current-step macro order for the team, or None (no commander).

        ensure_order is cached per simulation step, so calling this from
        both the reward path and the observation path advances the
        commander exactly once per step."""
        cmd = (self._commanders or {}).get(team)
        if cmd is None:
            return None
        return cmd.ensure_order(self, team, self.sim.step_count)

    # ------------------------------------------------------------------
    @property
    def agents(self) -> list[str]:
        """Units that may act right now (empty once the episode is over)."""
        if self.episode_done:
            return []
        return [u.id for u in self.sim.units.values() if u.alive]

    def alive_counts(self) -> Tuple[int, int]:
        return len(self.sim.living(0)), len(self.sim.living(1))

    def state_dict(self):
        return self.sim.state_dict()

    # ------------------------------------------------------------------
    @staticmethod
    def _knn(unit, items, k: int):
        """The k nearest units to `unit` from `items` (self excluded),
        ascending by distance. O(len(items)) with a small insertion cost."""
        best = []  # [(d2, unit)] sorted ascending, len <= k
        for v in items:
            if v is unit:
                continue
            d2 = (v.x - unit.x) ** 2 + (v.y - unit.y) ** 2
            if len(best) < k:
                best.append((d2, v))
                best.sort(key=lambda t: t[0])
            elif d2 < best[-1][0]:
                best[-1] = (d2, v)
                best.sort(key=lambda t: t[0])
        return [v for _, v in best]

    def _macro_features(self, unit, order) -> list:
        """8 macro self-features (option C).

        Absent macro (no commander / dead unit's terminal obs): zero
        slot vector, max slot distance, all-zero phase one-hot and
        macro_free=0 -- distinguishable from every real phase."""
        if order is None:
            return [0.0, 0.0, 1.0, *([0.0] * len(PHASES)), 0.0]
        slot = order["slots"].get(unit.id)
        if slot is None:
            return [0.0, 0.0, 1.0, *([0.0] * len(PHASES)), 0.0]
        wx, wy = self.cfg.world_w, self.cfg.world_h
        d = math.hypot(slot[0] - unit.x, slot[1] - unit.y)
        onehot = [1.0 if order["phase"] == p else 0.0 for p in PHASES]
        return [
            (slot[0] - unit.x) / wx,
            (slot[1] - unit.y) / wy,
            min(1.0, d / (4.0 * self.cfg.formation_spacing)),
            *onehot,
            1.0 if order["phase"] == "engage" else 0.0,
        ]

    def _self_features(self, unit_id, living_by_team, centroids, macro=None
                       ) -> np.ndarray:
        u = self.sim.units[unit_id]
        wx, wy = self.cfg.world_w, self.cfg.world_h
        diag = (wx * wx + wy * wy) ** 0.5

        if u.team == 0:
            enemies, friends = living_by_team[1], living_by_team[0]
        else:
            enemies, friends = living_by_team[0], living_by_team[1]

        # --- k nearest enemies (with facing, so flanking is learnable) ---
        enemy_feats: list[float] = []
        for e in self._knn(u, enemies, K_NEIGHBORS):
            enemy_feats += [
                (e.x - u.x) / wx, (e.y - u.y) / wy,
                2 * math.hypot(e.x - u.x, e.y - u.y) / diag - 1,
                2 * e.hp / 100.0 - 1,
                math.sin(e.heading), math.cos(e.heading),
            ]
        while len(enemy_feats) < 6 * K_NEIGHBORS:  # pad missing slots
            enemy_feats += [0.0, 0.0, 1.0, -1.0, 0.0, 0.0]

        # --- k nearest teammates ---
        friend_feats: list[float] = []
        for f in self._knn(u, friends, K_NEIGHBORS):
            friend_feats += [
                (f.x - u.x) / wx, (f.y - u.y) / wy,
                2 * math.hypot(f.x - u.x, f.y - u.y) / diag - 1,
                2 * f.hp / 100.0 - 1,
            ]
        while len(friend_feats) < 4 * K_NEIGHBORS:
            friend_feats += [0.0, 0.0, 1.0, -1.0]

        # --- local crowding ---
        ally_density = sum(
            1 for f in friends
            if (f.x - u.x) ** 2 + (f.y - u.y) ** 2 < DENSITY_RADIUS ** 2
        )
        enemy_density = sum(
            1 for e in enemies
            if (e.x - u.x) ** 2 + (e.y - u.y) ** 2 < self.cfg.weapon_range ** 2
        )

        x = u.x / wx
        y = u.y / wy
        blue_alive, red_alive = len(living_by_team[0]), len(living_by_team[1])
        n = self.cfg.n_units_per_side
        time_norm = self.sim.step_count / self.cfg.max_steps
        own_c, enemy_c = centroids[u.team], centroids[1 - u.team]

        # 55 self features (see module docstring):
        #   self x,y,hp,heading | sin/cos heading | team one-hot (2)
        #   3x nearest enemy: dx,dy,d,hp,sin/cos heading
        #   3x nearest teammate: dx,dy,d,hp
        #   blue/red alive | time | own-centroid dx,dy | enemy-centroid dx,dy
        #   ally density | enemy density
        #   macro: slot dx,dy,d | phase one-hot (4) | macro_free
        return np.array(
            [
                2 * x - 1, 2 * y - 1, 2 * u.hp / 100.0 - 1, u.heading / math.pi,
                math.sin(u.heading), math.cos(u.heading),
                1.0 if u.team == 0 else 0.0,
                1.0 if u.team == 1 else 0.0,
                *enemy_feats,
                *friend_feats,
                2 * blue_alive / n - 1, 2 * red_alive / n - 1,
                2 * time_norm - 1,
                (own_c[0] - u.x) / wx, (own_c[1] - u.y) / wy,
                (enemy_c[0] - u.x) / wx, (enemy_c[1] - u.y) / wy,
                min(1.0, ally_density / 8.0),
                min(1.0, enemy_density / 8.0),
                *self._macro_features(u, macro),
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _team_arrays(members) -> dict:
        """Vectorized per-team state (set construction shares it)."""
        return {
            "x": np.asarray([m.x for m in members], dtype=np.float64),
            "y": np.asarray([m.y for m in members], dtype=np.float64),
            "hp": np.asarray([m.hp for m in members], dtype=np.float64),
            "sin": np.asarray([math.sin(m.heading) for m in members],
                              dtype=np.float64),
            "cos": np.asarray([math.cos(m.heading) for m in members],
                              dtype=np.float64),
            "idx": {m.id: i for i, m in enumerate(members)},
        }

    def _rel_set(self, unit, arr, exclude: bool):
        """Padded [U, REL_DIM] relative-feature set + validity mask.

        Elements are ALL living units of one side (self excluded for
        the ally set), sorted by distance ascending -- the canonical
        order that makes the permutation-invariant set trainable.
        Padding rows are exactly zero and masked 0.
        """
        U = self.cfg.n_units_per_side
        wx, wy = self.cfg.world_w, self.cfg.world_h
        diag = (wx * wx + wy * wy) ** 0.5

        keep = np.ones(len(arr["x"]), dtype=bool)
        if exclude and unit.id in arr["idx"]:
            keep[arr["idx"][unit.id]] = False
        dx = arr["x"][keep] - unit.x
        dy = arr["y"][keep] - unit.y
        d = np.hypot(dx, dy)

        feats = np.zeros((U, REL_DIM), dtype=np.float32)
        mask = np.zeros(U, dtype=np.float32)
        m = len(d)
        if m == 0:  # lone survivor: empty set, all-masked
            return feats, mask
        order = np.argsort(d, kind="stable")[:U]
        k = len(order)
        feats[:k, 0] = dx[order] / wx
        feats[:k, 1] = dy[order] / wy
        feats[:k, 2] = 2.0 * d[order] / diag - 1.0
        feats[:k, 3] = 2.0 * arr["hp"][keep][order] / 100.0 - 1.0
        feats[:k, 4] = arr["sin"][keep][order]
        feats[:k, 5] = arr["cos"][keep][order]
        mask[:k] = 1.0
        return feats, mask

    def _unit_obs(self, unit_id, living_by_team, centroids, arrays,
                  macro=None) -> SetObs:
        """One unit's full structured observation (option D §2.2)."""
        u = self.sim.units[unit_id]
        ally, ally_mask = self._rel_set(u, arrays[u.team], exclude=True)
        enemy, enemy_mask = self._rel_set(u, arrays[1 - u.team], exclude=False)
        return SetObs(
            self_feat=self._self_features(
                unit_id, living_by_team, centroids, macro
            ),
            ally=ally,
            enemy=enemy,
            ally_mask=ally_mask,
            enemy_mask=enemy_mask,
            role=self._roles.get(unit_id, ROLE_NAMES.index("center")),
        )

    def _observations(self, agent_ids, include_dead=False) -> Dict[str, SetObs]:
        # Share the living-team lists, centroids and vectorized state
        # across all agents' observations (they are the dominant cost
        # of this env otherwise).
        living_by_team = (
            [u for u in self.sim.units.values() if u.alive and u.team == 0],
            [u for u in self.sim.units.values() if u.alive and u.team == 1],
        )
        centroids = []
        for team in (0, 1):
            members = living_by_team[team]
            if members:
                centroids.append(
                    (sum(m.x for m in members) / len(members),
                     sum(m.y for m in members) / len(members))
                )
            else:
                centroids.append((0.0, 0.0))
        arrays = tuple(self._team_arrays(members) for members in living_by_team)
        if include_dead:
            return {
                a: self._unit_obs(a, living_by_team, centroids, arrays,
                                  self._macro(self.sim.units[a].team))
                for a in agent_ids
            }
        macros = {0: self._macro(0), 1: self._macro(1)}
        return {
            a: self._unit_obs(a, living_by_team, centroids, arrays,
                              macros[self.sim.units[a].team])
            for a in agent_ids
            if self.sim.units[a].alive
        }

    # ------------------------------------------------------------------
    def reset(
        self,
        seed: int | None = None,
        formations: Optional[Tuple[str, str]] = None,
        commanders: tuple | dict | None = None,
    ) -> Dict[str, SetObs]:
        """Start an episode.

        formations: optional (blue_formation, red_formation) override for
        this episode only (validated by BattleConfig); passing it always
        builds a fresh simulator, so pass a varying `seed` for spawn
        jitter variety across episodes.

        commanders: optional per-team Commander attachment for this
        episode only -- a (blue, red) tuple (entries may be None) or a
        {team: Commander} mapping. The same Commander instance may not
        serve both teams (per-episode state would tangle). Attached
        commanders drive only observations and the slot-shaping reward;
        they never alter the battle itself. Any reset() that does not
        pass commanders= reverts to the constructor attachment (same
        contract as the formations= override).

        A plain reset() (no arguments) keeps the legacy behavior of
        reusing the simulator and its rng stream -- unless the previous
        episode used a formation override, in which case the constructor
        config is restored.
        """
        if formations is not None:
            cfg = replace(
                self.cfg,
                blue_formation=formations[0],
                red_formation=formations[1],
            )
            self.sim = BattleSimulator(cfg, seed=self.seed if seed is None else seed)
        elif seed is not None:
            self.sim = BattleSimulator(self.cfg, seed=seed)
        elif self.sim.cfg == self.cfg:
            self.sim.reset()
        else:  # previous episode overrode the formations: restore cfg
            self.sim = BattleSimulator(self.cfg, seed=self.seed)

        if commanders is not None:
            self._commanders = self._normalize_commanders(commanders)
        else:  # revert to the constructor attachment
            self._commanders = (
                dict(self._base_commanders) if self._base_commanders else None
            )
        for cmd in (self._commanders or {}).values():
            cmd.reset()

        self._assign_roles()
        self.episode_done = False
        return self._observations([u.id for u in self.sim.units.values()])

    # ------------------------------------------------------------------
    def _assign_roles(self) -> None:
        """Lifelong structural role per unit, computed once at reset
        (option D §2.3; units never respawn, so roles never change).

        Source priority per team:
          1. the attached commander's slot grid (the C+D combined form:
             the commander grid is the authoritative structure, not the
             spawn formation -- training may spawn in one formation and
             command another);
          2. the spawn formation's slot grid (units are created in slot
             order, so id-sorted units zip with the grid);
          3. scatter deployments have no grid: coarse initial-y thirds
             fallback (chirality-corrected so both teams' "left wing"
             is their own left).
        """
        self._roles: Dict[str, int] = {}
        cfg = self.sim.cfg
        h = cfg.world_h
        for team in (0, 1):
            formation = cfg.blue_formation if team == 0 else cfg.red_formation
            members = sorted(self.sim.living(team), key=lambda u: u.id)

            slot_of: Dict[str, Tuple[float, float]] = {}
            grid: list = []
            cmd = (self._commanders or {}).get(team)
            if cmd is not None:
                cmd.ensure_order(self, team, self.sim.step_count)
                for u in members:
                    idx = cmd._assign.get(u.id)
                    if idx is not None:
                        slot_of[u.id] = cmd._grid[idx]
                grid = cmd._grid
            if not slot_of and formation != "scatter":
                # _spawn consumes slots[i] for the i-th created unit; ids
                # are zero-padded, so id order == creation order.
                grid = self.sim._formation_slots(formation, len(members))
                for u, s in zip(members, grid):
                    slot_of[u.id] = s

            if slot_of:
                for u in members:
                    fx, fy = slot_of.get(u.id, (0.0, 0.0))
                    self._roles[u.id] = _slot_role(fx, fy, grid)
            else:  # scatter fallback: chirality-corrected y thirds
                left, center, right = (
                    ROLE_NAMES.index("left_wing"),
                    ROLE_NAMES.index("center"),
                    ROLE_NAMES.index("right_wing"),
                )
                for u in members:
                    fy = (u.y - h / 2.0) if team == 0 else (h / 2.0 - u.y)
                    self._roles[u.id] = (
                        left if fy > h / 6.0
                        else right if fy < -h / 6.0
                        else center
                    )

    # ------------------------------------------------------------------
    def step(self, actions: Dict[str, int]):
        """Advance one battle step. Only units alive at step start may act."""
        assert not self.episode_done, "call reset() first"
        active = self.agents

        living_before = (
            [u for u in self.sim.units.values() if u.alive and u.team == 0],
            [u for u in self.sim.units.values() if u.alive and u.team == 1],
        )
        diag = (self.cfg.world_w ** 2 + self.cfg.world_h ** 2) ** 0.5

        def dist_to_nearest_enemy(unit, enemies):
            best = None
            for e in enemies:
                d2 = (e.x - unit.x) ** 2 + (e.y - unit.y) ** 2
                if best is None or d2 < best:
                    best = d2
            return 0.0 if best is None else best ** 0.5 / diag

        def dist_to_nearest_teammate(unit, friends):
            best = None
            for f in friends:
                if f is unit:
                    continue
                d2 = (f.x - unit.x) ** 2 + (f.y - unit.y) ** 2
                if best is None or d2 < best:
                    best = d2
            return None if best is None else best ** 0.5

        def spacing_potential(d: float | None) -> float:
            """0.0 at/above SPACE_IDEAL, down to -1.0 when stacked."""
            if d is None:
                return 0.0
            return -max(0.0, 1.0 - d / SPACE_IDEAL)

        d_before = {
            a: dist_to_nearest_enemy(
                self.sim.units[a], living_before[1 - self.sim.units[a].team]
            )
            for a in active
        }
        # Nearest-teammate distance (world units) for the spacing potential.
        nn_before = {
            a: dist_to_nearest_teammate(
                self.sim.units[a], living_before[self.sim.units[a].team]
            )
            for a in active
        }
        dmg_before = {
            a: self.sim.damage_by_unit.get(a, 0.0) for a in active
        }

        # Macro context (option C): slot targets as seen when the units
        # chose their actions. The slot potential is only collected
        # outside the engage phase (engage = micro takes over, reward
        # off) and only while the phase stays put across the step (phase
        # transitions move slots discontinuously -- grid <-> ring).
        macro_before = {team: self._macro(team) for team in (0, 1)}
        slot_ideal = self.cfg.formation_spacing

        def slot_potential(d: float) -> float:
            """0.0 at the slot, saturating at -1 one spacing out.

            (The design doc's phi = -max(0, 1 - d/ideal) has the sign
            flipped -- it grows with distance and would reward fleeing
            the slot.)"""
            return -min(1.0, d / slot_ideal)

        slot_d_before = {}
        for a in active:
            order = macro_before.get(self.sim.units[a].team)
            if order is None or order["phase"] == "engage":
                continue
            slot = order["slots"].get(a)
            if slot is not None:
                u = self.sim.units[a]
                slot_d_before[a] = math.hypot(u.x - slot[0], u.y - slot[1])

        before_alive = (len(living_before[0]), len(living_before[1]))
        before_damage = (self.sim.damage_by_team[0], self.sim.damage_by_team[1])

        _, done = self.sim.step(actions)

        after_alive = (len(self.sim.living(0)), len(self.sim.living(1)))
        truncated = self.sim.step_count >= self.cfg.max_steps

        dead: Set[str] = {a for a in active if not self.sim.units[a].alive}

        if dead:
            living_after = (
                [u for u in self.sim.units.values() if u.alive and u.team == 0],
                [u for u in self.sim.units.values() if u.alive and u.team == 1],
            )
        else:
            living_after = living_before

        macro_after = {team: self._macro(team) for team in (0, 1)}

        # Dense reward: kill/loss signal + per-step damage delta
        # (damage_by_team is cumulative over the episode -> use the delta)
        # + personal damage credit (flank bonuses flow through it)
        # + potential-based shaping on distance to the nearest enemy
        # + potential-based shaping on spacing from the nearest teammate.
        rewards: Dict[str, float] = {}
        for a in active:
            u = self.sim.units[a]
            team = u.team
            enemy = 1 - team
            r = (before_alive[enemy] - after_alive[enemy]) * 2.0
            r -= (before_alive[team] - after_alive[team]) * 2.0
            r += (self.sim.damage_by_team[team] - before_damage[team]) * 0.001
            r += (self.sim.damage_by_unit.get(a, 0.0) - dmg_before[a]) * PERSONAL_DMG_K
            if u.alive:
                d_after = dist_to_nearest_enemy(u, living_after[enemy])
                nn_after = dist_to_nearest_teammate(u, living_after[team])
                r += SHAPE_DIST_K * (d_before[a] - d_after)
                r += SPACE_K * (spacing_potential(nn_after)
                                - spacing_potential(nn_before[a]))
                # Slot potential (option C): alive units only, engage
                # phase excluded (see slot_d_before), phase transitions
                # excluded (slot positions jump grid <-> ring).
                if a in slot_d_before:
                    order_b = macro_before[team]
                    order_a = macro_after.get(team)
                    slot = order_a["slots"].get(a) if order_a else None
                    if (order_a is not None
                            and order_a["phase"] == order_b["phase"]
                            and slot is not None):
                        d_slot = math.hypot(u.x - slot[0], u.y - slot[1])
                        r += SLOT_K * (slot_potential(d_slot)
                                       - slot_potential(slot_d_before[a]))
            else:
                # Terminal step: no spacing change for a dead unit (the
                # closing-shaping term keeps its legacy d_after = 0 form).
                r += SHAPE_DIST_K * d_before[a]
            rewards[a] = float(r)

        if done:
            self.episode_done = True
            # Terminal observation for every unit that acted this final step.
            obs = self._observations(active, include_dead=True)
        else:
            obs = self._observations(active)

        return obs, rewards, dead, truncated
