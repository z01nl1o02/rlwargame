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

Observations (OBS_DIM = 47) are deliberately richer than "nearest enemy +
nearest friend": the policy can see the 3 nearest enemies (with facing)
and teammates, both teams' centroids, and local crowding -- without these,
"keep the team together in one ball" is the only representable strategy.

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
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, Optional, Set, Tuple

import numpy as np

from .core import BattleConfig, BattleSimulator

OBS_DIM = 47
N_ACTIONS = 9  # stay + 8 movement directions (see BattleSimulator.ACTIONS)

# How many nearest enemies/teammates appear in the observation.
K_NEIGHBORS = 3

# Potential-based reward shaping constants (all terms telescope over an
# episode, so none of them can be farmed by oscillating back and forth).
SHAPE_DIST_K = 0.5    # closing on the nearest enemy (cold-start signal)
SPACE_K = 0.25        # opening distance from a too-close teammate
SPACE_IDEAL = 30.0    # desired nearest-teammate distance (world units)
PERSONAL_DMG_K = 0.002  # personal damage-dealt credit (flanking pays)

# Local-crowding observation: allies within this radius (world units).
DENSITY_RADIUS = 60.0


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

    def _unit_obs(
        self, unit_id: str, living_by_team, centroids
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

        # 47 features (see module docstring):
        #   self x,y,hp,heading | sin/cos heading | team one-hot (2)
        #   3x nearest enemy: dx,dy,d,hp,sin/cos heading
        #   3x nearest teammate: dx,dy,d,hp
        #   blue/red alive | time | own-centroid dx,dy | enemy-centroid dx,dy
        #   ally density | enemy density
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
            ],
            dtype=np.float32,
        )

    def _observations(self, agent_ids, include_dead=False) -> Dict[str, np.ndarray]:
        # Share the living-team lists and centroids across all agents'
        # observations (they are the dominant cost of this env otherwise).
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
        if include_dead:
            return {a: self._unit_obs(a, living_by_team, centroids) for a in agent_ids}
        return {
            a: self._unit_obs(a, living_by_team, centroids)
            for a in agent_ids
            if self.sim.units[a].alive
        }

    # ------------------------------------------------------------------
    def reset(
        self,
        seed: int | None = None,
        formations: Optional[Tuple[str, str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Start an episode.

        formations: optional (blue_formation, red_formation) override for
        this episode only (validated by BattleConfig); passing it always
        builds a fresh simulator, so pass a varying `seed` for spawn
        jitter variety across episodes.

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
        self.episode_done = False
        return self._observations([u.id for u in self.sim.units.values()])

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
