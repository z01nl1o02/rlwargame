from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional
import math
import random


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
      * Combat is automatic: each living unit fires at the nearest enemy in range.
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
        self.events = []
        self._spawn()
        return self.state_dict()

    def _spawn(self):
        n = self.cfg.n_units_per_side
        for i in range(n):
            # Blue starts on the left, red on the right.
            bx = self.rng.uniform(80, 300)
            by = self.rng.uniform(80, self.cfg.world_h - 80)
            rx = self.rng.uniform(self.cfg.world_w - 300, self.cfg.world_w - 80)
            ry = self.rng.uniform(80, self.cfg.world_h - 80)

            self.units[f"B{i:03d}"] = Unit(
                id=f"B{i:03d}", team=0, x=bx, y=by,
                heading=self.rng.uniform(-math.pi, math.pi)
            )
            self.units[f"R{i:03d}"] = Unit(
                id=f"R{i:03d}", team=1, x=rx, y=ry,
                heading=self.rng.uniform(-math.pi, math.pi)
            )

    def living(self, team: int | None = None):
        return [
            u for u in self.units.values()
            if u.alive and (team is None or u.team == team)
        ]

    def _nearest_enemy(self, unit: Unit, candidates=None) -> Optional[Unit]:
        if candidates is None:
            candidates = self.living(1 - unit.team)
        if not candidates:
            return None
        return min(candidates, key=lambda e: (e.x-unit.x)**2 + (e.y-unit.y)**2)

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

    def _combat(self):
        # Snapshot both teams once; targeting must still respect deaths that
        # happen earlier in the same phase, hence the `alive` check below.
        blue = [u for u in self.units.values() if u.alive and u.team == 0]
        red = [u for u in self.units.values() if u.alive and u.team == 1]

        for unit in blue + red:
            enemy_list = red if unit.team == 0 else blue
            target = None
            best_d2 = None
            for e in enemy_list:
                if not e.alive:
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

            self.shots_by_team[unit.team] += 1
            unit.cooldown = self.cfg.fire_interval

            if self.rng.random() <= self.cfg.hit_probability:
                self.hits_by_team[unit.team] += 1
                damage = self.cfg.damage_per_shot * self.rng.uniform(0.75, 1.25)
                target.hp -= damage
                self.damage_by_team[unit.team] += damage

                self.events.append({
                    "type": "hit",
                    "t": self.time,
                    "attacker": unit.id,
                    "target": target.id,
                    "damage": round(damage, 2),
                })

                if target.hp <= 0 and target.alive:
                    target.hp = 0.0
                    target.alive = False
                    self.events.append({
                        "type": "destroyed",
                        "t": self.time,
                        "unit": target.id,
                        "by": unit.id,
                    })

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
                    dirs = [
                        (0, 0), (0, -math.pi/2), (math.pi/4),
                        (0, math.pi/2), (math.pi*3/4), (math.pi,),
                        (-math.pi*3/4), (-math.pi/2), (-math.pi/4)
                    ]
                    # Easier direct quantization:
                    sector = int(round((angle + math.pi/2) / (math.pi/4))) % 8
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
            },
            "events": list(self.events),
        }
