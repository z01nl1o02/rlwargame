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

Observations (OBS_DIM = 55) are deliberately richer than "nearest enemy +
nearest friend": the policy can see the 3 nearest enemies (with facing)
and teammates, both teams' centroids, and local crowding -- without these,
"keep the team together in one ball" is the only representable strategy.
For teams with an attached commander, the last 8 features carry the macro
context: slot vector (dx, dy), slot distance, phase one-hot (4), and a
macro_free flag (engage phase: micro takes over). Teams without a
commander report zeros there (phase one-hot all-zero marks "no macro").

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
from dataclasses import replace
from typing import Dict, Optional, Set, Tuple

import numpy as np

from .core import BattleConfig, BattleSimulator
from .commander import PHASES

OBS_DIM = 55
N_ACTIONS = 9  # stay + 8 movement directions (see BattleSimulator.ACTIONS)

# How many nearest enemies/teammates appear in the observation.
K_NEIGHBORS = 3

# Macro observation block (option C): slot dx, dy, slot distance,
# phase one-hot (4), macro_free flag.
MACRO_DIM = 3 + len(PHASES) + 1

# Potential-based reward shaping constants (all terms telescope over an
# episode, so none of them can be farmed by oscillating back and forth).
SHAPE_DIST_K = 0.5    # closing on the nearest enemy (cold-start signal)
SPACE_K = 0.25        # opening distance from a too-close teammate
SPACE_IDEAL = 30.0    # desired nearest-teammate distance (world units)
PERSONAL_DMG_K = 0.002  # personal damage-dealt credit (flanking pays)
SLOT_K = 0.2          # closing on the commander slot (macro shaping)

# Local-crowding observation: allies within this radius (world units).
DENSITY_RADIUS = 60.0


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
        """8 macro observation features (option C).

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

    def _unit_obs(
        self, unit_id: str, living_by_team, centroids, macro=None
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

        # 55 features (see module docstring):
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
            return {
                a: self._unit_obs(a, living_by_team, centroids,
                                  self._macro(self.sim.units[a].team))
                for a in agent_ids
            }
        macros = {0: self._macro(0), 1: self._macro(1)}
        return {
            a: self._unit_obs(a, living_by_team, centroids,
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
    ) -> Dict[str, np.ndarray]:
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
