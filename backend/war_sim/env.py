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
"""

from __future__ import annotations

import math
from typing import Dict, Set, Tuple

import numpy as np

from .core import BattleConfig, BattleSimulator

OBS_DIM = 18
N_ACTIONS = 9  # stay + 8 movement directions (see BattleSimulator.ACTIONS)

# Potential-based reward shaping (telescopes over an episode, bounded by
# k * d0 <= ~0.6, so it cannot be farmed): units get a small reward for
# closing on the nearest enemy. Without it, untrained policies random-walk
# and never come into weapon range within the step limit (cold start).
SHAPE_DIST_K = 0.5


class WarEnv:
    def __init__(self, config: BattleConfig | None = None, seed: int | None = None):
        self.cfg = config or BattleConfig()
        self.seed = seed
        self.sim = BattleSimulator(self.cfg, seed=seed)
        self.episode_done = True

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
    def _unit_obs(self, unit_id: str, living_by_team) -> np.ndarray:
        u = self.sim.units[unit_id]
        wx, wy = self.cfg.world_w, self.cfg.world_h
        diag = (wx * wx + wy * wy) ** 0.5

        def nearest(items, skip_self=False):
            best = None
            best_d2 = None
            for v in items:
                if skip_self and v is u:
                    continue
                d2 = (v.x - u.x) ** 2 + (v.y - u.y) ** 2
                if best_d2 is None or d2 < best_d2:
                    best, best_d2 = v, d2
            if best is None:
                return 0.0, 0.0, 1.0, 0.0
            # Positions are clamped to the world, so dx/dy and d are within
            # [-1, 1] by construction (no clipping needed).
            return (
                (best.x - u.x) / wx,
                (best.y - u.y) / wy,
                best_d2 ** 0.5 / diag,
                best.hp / 100.0,
            )

        if u.team == 0:
            enemies, friends = living_by_team[1], living_by_team[0]
        else:
            enemies, friends = living_by_team[0], living_by_team[1]
        edx, edy, ed, ehp = nearest(enemies)
        fdx, fdy, fd, fhp = nearest(friends, skip_self=True)

        x = u.x / wx
        y = u.y / wy
        blue_alive, red_alive = len(living_by_team[0]), len(living_by_team[1])
        n = self.cfg.n_units_per_side
        time_norm = self.sim.step_count / self.cfg.max_steps

        # 18 features:
        #   self x,y,hp,heading | nearest enemy dx,dy,d,hp |
        #   nearest friend dx,dy,d,hp | team one-hot (2) |
        #   blue_alive, red_alive | time_norm | (reserved 0.0)
        return np.array(
            [
                2 * x - 1, 2 * y - 1, 2 * u.hp / 100.0 - 1, u.heading / math.pi,
                edx, edy, 2 * ed - 1, 2 * ehp - 1,
                fdx, fdy, 2 * fd - 1, 2 * fhp - 1,
                1.0 if u.team == 0 else 0.0,
                1.0 if u.team == 1 else 0.0,
                2 * blue_alive / n - 1, 2 * red_alive / n - 1,
                2 * time_norm - 1,
                0.0,
            ],
            dtype=np.float32,
        )

    def _observations(self, agent_ids, include_dead=False) -> Dict[str, np.ndarray]:
        # Share the living-team lists across all agents' observations (they
        # are the dominant cost of this env otherwise).
        living_by_team = (
            [u for u in self.sim.units.values() if u.alive and u.team == 0],
            [u for u in self.sim.units.values() if u.alive and u.team == 1],
        )
        if include_dead:
            return {a: self._unit_obs(a, living_by_team) for a in agent_ids}
        return {
            a: self._unit_obs(a, living_by_team)
            for a in agent_ids
            if self.sim.units[a].alive
        }

    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> Dict[str, np.ndarray]:
        if seed is not None:
            self.sim = BattleSimulator(self.cfg, seed=seed)
        else:
            self.sim.reset()
        self.episode_done = False
        return self._observations([u.id for u in self.sim.units.values()])

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

        d_before = {
            a: dist_to_nearest_enemy(self.sim.units[a], living_before[1 - self.sim.units[a].team])
            for a in active
        }

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

        # Dense reward: kill/loss signal + per-step damage delta
        # (damage_by_team is cumulative over the episode -> use the delta)
        # + potential-based shaping on the distance to the nearest enemy.
        rewards: Dict[str, float] = {}
        for a in active:
            u = self.sim.units[a]
            team = u.team
            enemy = 1 - team
            r = (before_alive[enemy] - after_alive[enemy]) * 2.0
            r -= (before_alive[team] - after_alive[team]) * 2.0
            r += (self.sim.damage_by_team[team] - before_damage[team]) * 0.001
            if u.alive:
                d_after = dist_to_nearest_enemy(u, living_after[enemy])
            else:
                d_after = 0.0  # terminal potential is zero
            r += SHAPE_DIST_K * (d_before[a] - d_after)
            rewards[a] = float(r)

        if done:
            self.episode_done = True
            # Terminal observation for every unit that acted this final step.
            obs = self._observations(active, include_dead=True)
        else:
            obs = self._observations(active)

        return obs, rewards, dead, truncated
