from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import math
import random

# Deployment formations available for BattleConfig.blue/red_formation.
# "scatter" is the legacy uniform-random deployment; the rest place units
# on deterministic slot grids (see BattleSimulator._formation_slots).
FORMATION_NAMES = (
    "scatter",
    "line",
    "column",
    "wedge",
    "echelon_l",
    "echelon_r",
    "crescent",
)


@dataclass
class BattleConfig:
    n_units_per_side: int = 50
    world_w: float = 1000.0
    world_h: float = 600.0
    dt: float = 1.0
    max_steps: int = 1800
    move_speed: float = 7.0
    weapon_range: float = 85.0
    damage_per_shot: float = 4.0
    fire_interval: int = 4
    hit_probability: float = 0.72
    # --- tactical combat mechanics -------------------------------------
    # Anti-blob rules: massing everyone into one deathball must not be the
    # dominant strategy. Both rules are pure engine mechanics, so they apply
    # identically in training, serving and replays.
    # A single target can only be engaged by `focus_fire_cap` shooters per
    # step; further shooters retarget or waste their volley.
    focus_fire_cap: int = 3
    # Damage multiplier for attacks coming from outside the target's
    # protected frontal arc (flank/rear attacks).
    flank_bonus: float = 1.35
    # Half-angle (degrees) of the target's protected frontal arc.
    flank_arc_deg: float = 110.0
    # --- scripted initial deployment -----------------------------------
    blue_formation: str = "scatter"
    red_formation: str = "scatter"
    formation_spacing: float = 26.0
    formation_jitter: float = 6.0

    def __post_init__(self):
        for field, value in (
            ("blue_formation", self.blue_formation),
            ("red_formation", self.red_formation),
        ):
            if value not in FORMATION_NAMES:
                raise ValueError(
                    f"{field}={value!r} is not a formation;"
                    f" choose one of {FORMATION_NAMES}"
                )
        if self.focus_fire_cap < 1:
            raise ValueError("focus_fire_cap must be >= 1")
        if self.flank_bonus < 1.0:
            raise ValueError("flank_bonus must be >= 1.0")
        if self.formation_spacing <= 0:
            raise ValueError("formation_spacing must be > 0")


@dataclass
class Unit:
    id: str
    team: int  # 0=blue, 1=red
    x: float
    y: float
    hp: float = 100.0
    max_hp: float = 100.0
    heading: float = 0.0
    target_id: Optional[str] = None
    alive: bool = True
    cooldown: int = 0


class BattleSimulator:
    """Small deterministic-friendly battle engine.

    The simulation is intentionally simple:
      * 100 units total by default.
      * Agents choose a movement direction.
      * Combat is automatic: each living unit fires at the nearest enemy
        in range -- subject to the focus-fire cap (at most
        `focus_fire_cap` shooters may engage one target per step).
      * Attacks from outside a target's frontal arc deal `flank_bonus`
        times damage, so envelopment / flanking maneuvers pay off.
      * The engine owns damage, hit probability, deaths and metrics.
    """

    ACTIONS = {
        0: (0.0, 0.0),
        1: (0.0, -1.0),
        2: (1.0, -1.0),
        3: (1.0, 0.0),
        4: (1.0, 1.0),
        5: (0.0, 1.0),
        6: (-1.0, 1.0),
        7: (-1.0, 0.0),
        8: (-1.0, -1.0),
    }

    def __init__(self, config: BattleConfig | None = None, seed: int | None = None):
        self.cfg = config or BattleConfig()
        self.rng = random.Random(seed)
        self.seed = seed
        self.reset()

    def reset(self):
        self.time = 0.0
        self.step_count = 0
        self.units: Dict[str, Unit] = {}
        self.damage_by_team = {0: 0.0, 1: 0.0}
        self.shots_by_team = {0: 0, 1: 0}
        self.hits_by_team = {0: 0, 1: 0}
        self.flank_hits_by_team = {0: 0, 1: 0}
        # Per-attacker damage attribution (lets the RL env credit individual
        # flanking/aggressive units instead of only the team average).
        self.damage_by_unit: Dict[str, float] = {}
        self.events = []
        self._spawn()
        return self.state_dict()

    def _spawn(self):
        n = self.cfg.n_units_per_side
        for team, prefix, facing in ((0, "B", 0.0), (1, "R", math.pi)):
            formation = self.cfg.blue_formation if team == 0 else self.cfg.red_formation
            slots = (
                self._formation_slots(formation, n) if formation != "scatter" else None
            )
            for i in range(n):
                if slots is None:
                    # Legacy deployment: uniform scatter in the drop zone.
                    if team == 0:
                        x = self.rng.uniform(80, 300)
                        y = self.rng.uniform(80, self.cfg.world_h - 80)
                    else:
                        x = self.rng.uniform(
                            self.cfg.world_w - 300, self.cfg.world_w - 80
                        )
                        y = self.rng.uniform(80, self.cfg.world_h - 80)
                    heading = self.rng.uniform(-math.pi, math.pi)
                else:
                    # Scripted formation: local frame has +fx toward the
                    # enemy; the team's mirror keeps both sides' shapes
                    # chirality-identical from their own point of view.
                    fx, fy = slots[i]
                    if team == 0:
                        x = 260.0 + fx
                        y = self.cfg.world_h / 2 + fy
                    else:
                        x = self.cfg.world_w - 260.0 - fx
                        y = self.cfg.world_h / 2 - fy
                    x += self.rng.uniform(-1.0, 1.0) * self.cfg.formation_jitter
                    y += self.rng.uniform(-1.0, 1.0) * self.cfg.formation_jitter
                    x = max(10.0, min(self.cfg.world_w - 10.0, x))
                    y = max(10.0, min(self.cfg.world_h - 10.0, y))
                    heading = facing
                self.units[f"{prefix}{i:03d}"] = Unit(
                    id=f"{prefix}{i:03d}", team=team, x=x, y=y, heading=heading
                )

    def _formation_slots(self, formation: str, n: int) -> List[Tuple[float, float]]:
        """Slot offsets (fx, fy) in a team's local deployment frame.

        +fx points toward the enemy (0 = front line), +fy to the team's
        left. Slot grids are sized to the world so a full 50-unit team
        fits between the map edge and the midpoint; oversized teams wrap
        into additional ranks/blocks behind the front.
        """
        s = self.cfg.formation_spacing
        # Keep 80px lateral margins; depths stay inside the deploy zone.
        max_lat = max(2.0 * s, self.cfg.world_h - 160.0)
        per_rank = max(1, int(max_lat // s))

        slots: List[Tuple[float, float]] = []
        if formation == "line":
            # Wide ranks, shallow depth: the classic battle line.
            for i in range(n):
                r, c = divmod(i, per_rank)
                slots.append((-r * s * 0.8, (c - (per_rank - 1) / 2) * s))
        elif formation == "column":
            # Narrow front, deep file: a marching column.
            width = max(4, per_rank // 2)
            for i in range(n):
                r, c = divmod(i, width)
                slots.append((-r * s, (c - (width - 1) / 2) * s))
        elif formation == "wedge":
            # Solid triangle pointing at the enemy: rank r holds r+1
            # units (1 + 2 + ... + R >= n).
            r = 0
            while len(slots) < n:
                count = min(r + 1, n - len(slots))
                for j in range(count):
                    slots.append((-r * s * 0.9, (j - (count - 1) / 2) * s))
                r += 1
        elif formation in ("echelon_l", "echelon_r"):
            # Sheared ranks: each rank behind steps sideways, producing a
            # staircase edge on one flank (echelon_left / echelon_right).
            side = 1.0 if formation == "echelon_l" else -1.0
            per = max(2, int(per_rank * 0.6))
            for i in range(n):
                r, c = divmod(i, per)
                slots.append(
                    (
                        -(r * s * 0.8 + c * s * 0.3),
                        (c - (per - 1) / 2) * s - side * r * s,
                    )
                )
        elif formation == "crescent":
            # Concave bowl opening toward the enemy: the center is the
            # rearmost point, the horns reach forward to envelop anything
            # that steps into the bowl.
            span = max_lat
            depth = min(90.0, 0.45 * span)
            radius = (span / 2) ** 2 / (2 * depth) + depth / 2
            half = math.asin(min(1.0, (span / 2) / radius))
            for i in range(n):
                t = -half + (2 * half) * (i / max(1, n - 1))
                slots.append((radius * (1 - math.cos(t)), radius * math.sin(t)))
        else:  # pragma: no cover - validated in BattleConfig.__post_init__
            raise ValueError(f"unknown formation {formation!r}")
        return slots

    def living(self, team: int | None = None):
        return [
            u
            for u in self.units.values()
            if u.alive and (team is None or u.team == team)
        ]

    def _nearest_enemy(self, unit: Unit, candidates=None) -> Optional[Unit]:
        if candidates is None:
            candidates = self.living(1 - unit.team)
        if not candidates:
            return None
        return min(candidates, key=lambda e: (e.x - unit.x) ** 2 + (e.y - unit.y) ** 2)

    def _clamp_world(self, unit: Unit):
        unit.x = max(5.0, min(self.cfg.world_w - 5.0, unit.x))
        unit.y = max(5.0, min(self.cfg.world_h - 5.0, unit.y))

    def _move(self, unit: Unit, action: int):
        dx, dy = self.ACTIONS[int(action)]
        norm = math.hypot(dx, dy)
        if norm > 0:
            dx, dy = dx / norm, dy / norm
            unit.heading = math.atan2(dy, dx)
            unit.x += dx * self.cfg.move_speed * self.cfg.dt
            unit.y += dy * self.cfg.move_speed * self.cfg.dt
            self._clamp_world(unit)

    def _is_flank(self, attacker: Unit, target: Unit) -> bool:
        """True if the attack comes from outside the target's front arc.

        The target's protected arc is centered on its heading; a stationary
        target keeps its heading, so it must choose between holding fire
        discipline and turning to face threats (movement updates heading).
        """
        ang = math.atan2(attacker.y - target.y, attacker.x - target.x)
        rel = math.atan2(math.sin(ang - target.heading), math.cos(ang - target.heading))
        return abs(rel) > math.radians(self.cfg.flank_arc_deg)

    def _combat(self):
        # Snapshot both teams once; targeting must still respect deaths that
        # happen earlier in the same phase, hence the `alive` check below.
        blue = [u for u in self.units.values() if u.alive and u.team == 0]
        red = [u for u in self.units.values() if u.alive and u.team == 1]

        # focus-fire bookkeeping: target id -> shooters engaging it this step
        engagements: Dict[str, int] = {}

        for unit in blue + red:
            enemy_list = red if unit.team == 0 else blue
            target = None
            best_d2 = None
            # Nearest enemy that is not already saturated (focus-fire cap);
            # saturated shooters retarget instead of stacking more fire.
            for e in enemy_list:
                if not e.alive:
                    continue
                if engagements.get(e.id, 0) >= self.cfg.focus_fire_cap:
                    continue
                d2 = (e.x - unit.x) ** 2 + (e.y - unit.y) ** 2
                if best_d2 is None or d2 < best_d2:
                    target, best_d2 = e, d2
            unit.target_id = target.id if target else None
            if target is None:
                continue

            d = math.hypot(target.x - unit.x, target.y - unit.y)
            if d > self.cfg.weapon_range or unit.cooldown > 0:
                continue

            engagements[target.id] = engagements.get(target.id, 0) + 1
            self.shots_by_team[unit.team] += 1
            unit.cooldown = self.cfg.fire_interval

            if self.rng.random() <= self.cfg.hit_probability:
                self.hits_by_team[unit.team] += 1
                flank = self._is_flank(unit, target)
                damage = self.cfg.damage_per_shot * self.rng.uniform(0.75, 1.25)
                if flank:
                    damage *= self.cfg.flank_bonus
                    self.flank_hits_by_team[unit.team] += 1
                target.hp -= damage
                self.damage_by_team[unit.team] += damage
                self.damage_by_unit[unit.id] = (
                    self.damage_by_unit.get(unit.id, 0.0) + damage
                )

                event = {
                    "type": "hit",
                    "t": self.time,
                    "attacker": unit.id,
                    "target": target.id,
                    "damage": round(damage, 2),
                }
                if flank:
                    event["flank"] = True
                self.events.append(event)

                if target.hp <= 0 and target.alive:
                    target.hp = 0.0
                    target.alive = False
                    self.events.append(
                        {
                            "type": "destroyed",
                            "t": self.time,
                            "unit": target.id,
                            "by": unit.id,
                        }
                    )

    def step(self, actions: Dict[str, int] | None = None):
        actions = actions or {}

        # Default behavior is "move toward nearest enemy".
        for unit in self.living():
            if unit.cooldown > 0:
                unit.cooldown -= 1

            if unit.id in actions:
                action = int(actions[unit.id])
            else:
                target = self._nearest_enemy(unit)
                if target is None:
                    action = 0
                else:
                    angle = math.atan2(target.y - unit.y, target.x - unit.x)
                    # Quantize the angle into the 8 movement directions.
                    # dirs = [
                    #    (0, 0), (0, -math.pi/2), (math.pi/4),
                    #    (0, math.pi/2), (math.pi*3/4), (math.pi,),
                    #    (-math.pi*3/4), (-math.pi/2), (-math.pi/4)
                    # ]
                    # Easier direct quantization:
                    sector = int(round((angle + math.pi / 2) / (math.pi / 4))) % 8
                    action = [0, 1, 2, 3, 4, 5, 6, 7, 8][sector + 1]

            self._move(unit, action)

        self._combat()

        self.step_count += 1
        self.time += self.cfg.dt
        done = (
            len(self.living(0)) == 0
            or len(self.living(1)) == 0
            or self.step_count >= self.cfg.max_steps
        )

        # Keep only recent events for a live UI packet.
        self.events = self.events[-100:]
        return self.state_dict(), done

    def fire_efficiency(self, team: int) -> float:
        shots = self.shots_by_team[team]
        return self.hits_by_team[team] / shots if shots else 0.0

    def state_dict(self):
        return {
            "time": self.time,
            "step": self.step_count,
            "world": {
                "w": self.cfg.world_w,
                "h": self.cfg.world_h,
            },
            "units": [asdict(u) for u in self.units.values()],
            "stats": {
                "blue_alive": len(self.living(0)),
                "red_alive": len(self.living(1)),
                "blue_damage": self.damage_by_team[0],
                "red_damage": self.damage_by_team[1],
                "blue_fire_efficiency": self.fire_efficiency(0),
                "red_fire_efficiency": self.fire_efficiency(1),
                "blue_flank_hits": self.flank_hits_by_team[0],
                "red_flank_hits": self.flank_hits_by_team[1],
            },
            "events": list(self.events),
        }
