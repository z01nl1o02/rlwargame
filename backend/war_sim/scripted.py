"""Scripted benchmark opponents for league training and evaluation.

Four stateless baselines (they read only the current simulator state, so
they can be dropped into any episode, any team, any step):

    hold   -- everyone stands still (level-0 sanity baseline)
    blob   -- classic deathball: everyone charges the enemy centroid
              (the exact behavior we are training the RL policies out of;
              the anti-blob engine rules and a spread/flanking policy
              should beat it)
    line   -- advance together, stop at weapon range, keep spacing
              (a disciplined front line)
    flank  -- split into two wings along the map edges, sweep past the
              enemy and converge from behind (flank-bonus harvester)

These serve two purposes:
  * league curriculum in train.py: learning against fixed, diverse
    opponents prevents both self-play policies from collapsing into the
    same symmetric deathball equilibrium
  * benchmarks in evaluate.py: "beats blob 80%+" is a measurable bar,
    next to the formation-metrics report

Usage:
    from war_sim.scripted import SCRIPTED_POLICIES
    actions = SCRIPTED_POLICIES["blob"].act(env, team=1)
"""

from __future__ import annotations

import math
from typing import Dict

from .core import BattleSimulator
from .env import WarEnv


def dir_to_action(sim: BattleSimulator, dx: float, dy: float) -> int:
    """Best of the 9 discrete moves aligned with the vector (dx, dy).

    0 (hold) when the vector is (near-)zero or no direction aligns.
    """
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return 0
    best_a, best_score = 0, -2.0
    for a, (ax, ay) in sim.ACTIONS.items():
        m = math.hypot(ax, ay)
        if m == 0:
            continue
        score = (ax * dx + ay * dy) / (m * n)
        if score > best_score:
            best_a, best_score = a, score
    return best_a


def _centroid(units):
    if not units:
        return None
    return (
        sum(u.x for u in units) / len(units),
        sum(u.y for u in units) / len(units),
    )


def _nearest(unit, others):
    best, best_d2 = None, None
    for v in others:
        d2 = (v.x - unit.x) ** 2 + (v.y - unit.y) ** 2
        if best_d2 is None or d2 < best_d2:
            best, best_d2 = v, d2
    return best


class ScriptedPolicy:
    """Base class: override `unit_action` or `act`."""

    name = "base"

    def act(self, env: WarEnv, team: int) -> Dict[str, int]:
        sim = env.sim
        mine = sim.living(team)
        foes = sim.living(1 - team)
        if not mine or not foes:
            return {}
        return {
            u.id: self.unit_action(sim, u, mine, foes, team)
            for u in mine
        }

    def unit_action(self, sim, unit, mine, foes, team: int) -> int:
        raise NotImplementedError


class Hold(ScriptedPolicy):
    name = "hold"

    def unit_action(self, sim, unit, mine, foes, team):
        return 0


class BlobRush(ScriptedPolicy):
    """Deathball: everyone charges the enemy centroid. The strategy the
    anti-blob work is measured against."""
    name = "blob"

    def unit_action(self, sim, unit, mine, foes, team):
        cx, cy = _centroid(foes)
        return dir_to_action(sim, cx - unit.x, cy - unit.y)


class LineAdvance(ScriptedPolicy):
    """Advance as a loose front line: close to firing range, then hold,
    while keeping some distance from the nearest teammate."""
    name = "line"

    def unit_action(self, sim, unit, mine, foes, team):
        cx, cy = _centroid(foes)
        foe = _nearest(unit, foes)
        dx, dy = 0.0, 0.0
        if foe is None or math.hypot(foe.x - unit.x, foe.y - unit.y) > sim.cfg.weapon_range * 0.95:
            dx += cx - unit.x
            dy += cy - unit.y
        mate = _nearest(unit, [m for m in mine if m is not unit])
        if mate is not None:
            d = math.hypot(mate.x - unit.x, mate.y - unit.y)
            if d < 20.0:  # too crowded: separation dominates
                k = 2.0 * (1.0 - d / 20.0)
                dx += (unit.x - mate.x) * k
                dy += (unit.y - mate.y) * k
        return dir_to_action(sim, dx, dy)


class FlankWings(ScriptedPolicy):
    """Synchronized pincer: two wings sweep along the top/bottom map
    edges, and only converge on the enemy centroid once BOTH wings are
    level with it -- arriving one wing at a time just feeds the enemy
    piecemeal (verified: unsynchronized wings lose to blob 0/13).

    Phase ("swept") is judged from x position relative to the enemy
    centroid -- roughly level or beyond counts. A waypoint-distance test
    would flip back to "sweep" as soon as a wing leaves the waypoint
    zone to converge, locking it into a limit cycle that never engages
    (verified: eternal 146px standoff at a map wall)."""
    name = "flank"

    def act(self, env: WarEnv, team: int) -> Dict[str, int]:
        sim = env.sim
        mine = sim.living(team)
        foes = sim.living(1 - team)
        if not mine or not foes:
            return {}
        cx, cy = _centroid(foes)
        top = [m for m in mine if m.y < sim.cfg.world_h / 2]
        bottom = [m for m in mine if m.y >= sim.cfg.world_h / 2]
        wings = {}
        for edge_y, wing in ((50.0, top), (sim.cfg.world_h - 50.0, bottom)):
            swept = bool(wing) and (
                (sum(m.x for m in wing) / len(wing) - cx)
                * (1.0 if team == 0 else -1.0) > -50.0
            )
            wings[edge_y] = swept
        both_swept = all(wings.values())
        return {
            u.id: self.unit_action(sim, u, mine, foes, team, cx, cy,
                                   wings, both_swept)
            for u in mine
        }

    def unit_action(self, sim, unit, mine, foes, team,
                    cx, cy, wings, both_swept):
        cfg = sim.cfg
        dirx = 1.0 if team == 0 else -1.0  # which way "behind the enemy" is
        edge_y = 50.0 if unit.y < cfg.world_h / 2 else cfg.world_h - 50.0
        wx = max(40.0, min(cfg.world_w - 40.0, cx + 150.0 * dirx))
        my_swept = wings[edge_y]
        foe = _nearest(unit, foes)

        if foe is not None:
            d = math.hypot(foe.x - unit.x, foe.y - unit.y)
            if my_swept and both_swept and d <= cfg.weapon_range:
                return 0  # pincer is closed: stand and shoot (hold facing)
            if d < 25.0:
                # cornered while transiting: fight back
                return dir_to_action(sim, foe.x - unit.x, foe.y - unit.y)

        if not my_swept:
            tx, ty = wx, edge_y        # run to the sweep waypoint along my edge
        elif not both_swept:
            tx, ty = cx, edge_y        # in position: shadow the enemy, wait
        else:
            tx, ty = cx, cy            # both wings ready: converge (pincer)
        return dir_to_action(sim, tx - unit.x, ty - unit.y)


SCRIPTED_POLICIES = {
    p.name: p()
    for p in (Hold, BlobRush, LineAdvance, FlankWings)
}
