"""Evaluate trained policies: win rates + formation-quality metrics.

Win rate alone cannot tell "smart formations" from "a tighter deathball",
so every matchup also reports per-team geometry metrics averaged over the
battle (sampled every 10 steps after the opening, while >= 30% of the
team is alive):

    spacing   mean distance to the nearest teammate (a stacked deathball
              sits near 0; a line/wedge keeps its spacing)
    spread    std of distance from the team centroid
    hull/u    convex-hull area per living unit (how much room each unit
              actually occupies)
    allies60  mean number of teammates within 60px (local crowding)
    flank%    share of the team's hits that earned the flank bonus
              (maneuver quality: attacking from outside enemy facing)

For reference, the same metrics are printed for the scripted `blob`
policy -- the behavior this whole effort trains the policies out of.

Matchups (N episodes each, random initial formations per episode):
    blue_policy vs red_policy          (self-play mirror)
    each policy vs each scripted opponent, on its own side

Usage:
    cd backend
    python evaluate.py                          # checkpoints/war_100
    python evaluate.py --episodes 20
    python evaluate.py --formation wedge,line  # pin initial deployments
    python evaluate.py --record                # also save one replay
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass, field

import numpy as np

from war_sim.core import BattleConfig, FORMATION_NAMES
from war_sim.env import WarEnv
from war_sim.recorder import ReplayRecorder
from war_sim.runtime import PolicyRuntime, team_of
from war_sim.scripted import SCRIPTED_POLICIES

FORMATION_POOL = ("scatter", *FORMATION_NAMES[1:])


# ----------------------------------------------------------------------
# Formation metrics
# ----------------------------------------------------------------------
@dataclass
class MetricAccumulator:
    spacing: list = field(default_factory=list)   # nearest-teammate distance
    spread: list = field(default_factory=list)    # radial std from centroid
    hull_per_unit: list = field(default_factory=list)
    allies60: list = field(default_factory=list)

    def add(self, units) -> None:
        if len(units) < 2:
            return
        xs = np.array([u.x for u in units])
        ys = np.array([u.y for u in units])
        cx, cy = xs.mean(), ys.mean()
        for x, y in zip(xs, ys):
            d = np.hypot(xs - x, ys - y)
            d = d[d > 0]
            self.spacing.append(float(d.min()) if len(d) else 0.0)
            self.allies60.append(int((d < 60.0).sum()))
        self.spread.append(float(np.std(np.hypot(xs - cx, ys - cy))))
        area = _hull_area(xs, ys)
        if area is not None:
            self.hull_per_unit.append(area / len(units))

    def summary(self) -> dict:
        def m(vals):
            return float(np.mean(vals)) if vals else 0.0
        return {
            "spacing": m(self.spacing),
            "spread": m(self.spread),
            "hull/u": m(self.hull_per_unit),
            "allies60": m(self.allies60),
        }


def _hull_area(xs: np.ndarray, ys: np.ndarray) -> float | None:
    """Convex-hull area (monotone chain + shoelace); None if degenerate."""
    pts = sorted(zip(xs.tolist(), ys.tolist()))
    if len(pts) < 3:
        return None

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
        return None
    s = 0.0
    for (x1, y1), (x2, y2) in zip(hull, hull[1:] + hull[:1]):
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


# ----------------------------------------------------------------------
# Matchup runner
# ----------------------------------------------------------------------
def play_episode(
    env: WarEnv,
    controllers: dict,          # team -> ("ppo", PolicyRuntime-policies) | ("scripted", name)
    seed: int,
    formations: tuple[str, str],
    metrics: dict | None = None,  # team -> MetricAccumulator (sampled live)
    recorder: ReplayRecorder | None = None,
):
    obs = env.reset(seed=seed, formations=formations)
    if recorder is not None:
        recorder.start(env.sim, source="evaluate")

    while not env.episode_done:
        actions = {}
        for team in (0, 1):
            ids = [a for a in obs if team_of(a) == team]
            if not ids:
                continue
            kind, ctrl = controllers[team]
            if kind == "scripted":
                opp = SCRIPTED_POLICIES[ctrl].act(env, team)
                for a in ids:
                    actions[a] = int(opp.get(a, 0))
            else:
                batch = np.stack([obs[a] for a in ids])
                acts, _, _ = ctrl[team].model.act(batch, deterministic=True)
                for i, a in enumerate(ids):
                    actions[a] = int(acts[i])
        obs, _, _, _ = env.step(actions)
        if recorder is not None:
            recorder.record_step(env.state_dict())
        if metrics is not None and env.sim.step_count >= 50 \
                and env.sim.step_count % 10 == 0:
            for team in (0, 1):
                living = env.sim.living(team)
                if len(living) >= max(3, int(0.3 * env.cfg.n_units_per_side)):
                    metrics[team].add(living)

    if recorder is not None:
        recorder.finish("episode_end")
    blue, red = env.alive_counts()
    winner = 0 if (red == 0 and blue > 0) else (1 if (blue == 0 and red > 0) else -1)
    return winner, blue, red


def run_matchup(env, controllers, n_episodes, base_seed, formations=None):
    rng = random.Random(base_seed)
    w = l = d = 0
    alive = {t: [] for t in controllers}
    acc = {t: MetricAccumulator() for t in controllers}
    flank_share = {t: [] for t in controllers}
    for k in range(n_episodes):
        form = formations or (rng.choice(FORMATION_POOL), rng.choice(FORMATION_POOL))
        # alternate sides so a fixed "blue" slot cannot bias results
        if k % 2:
            form = (form[1], form[0])
        winner, blue, red = play_episode(
            env, controllers, seed=rng.randrange(2**31), formations=form,
            metrics=acc,
        )
        for t in controllers:
            alive[t].append(blue if t == 0 else red)
            hits = env.sim.hits_by_team[t]
            flank_share[t].append(
                env.sim.flank_hits_by_team[t] / hits if hits else 0.0
            )
        # result from team 0's perspective
        if winner == 0:
            w += 1
        elif winner == 1:
            l += 1
        else:
            d += 1
    return {
        "w": w, "l": l, "d": d,
        "alive": {t: float(np.mean(alive[t])) for t in controllers},
        "metrics": {t: acc[t].summary() for t in controllers},
        "flank%": {t: 100.0 * float(np.mean(flank_share[t])) for t in controllers},
    }


def fmt_row(name, res, team, n_units):
    m = res["metrics"][team]
    return (f"{name:<34s} {res['w']:2d}-{res['l']:2d}-{res['d']:2d} "
            f"{res['alive'][team]:6.1f} "
            f"{m['spacing']:7.1f} {m['spread']:7.1f} "
            f"{m['hull/u']:8.0f} {m['allies60']:7.1f} "
            f"{res['flank%'][team]:6.1f}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained war policies")
    p.add_argument("--checkpoint-dir",
                   default=os.environ.get("WAR_CHECKPOINT_DIR",
                                          "./checkpoints/war_100"))
    p.add_argument("--episodes", type=int, default=10,
                   help="episodes per matchup")
    p.add_argument("--n-units", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--formation", default=None,
                   help="pin deployments as 'blue,red' (e.g. 'wedge,line')")
    p.add_argument("--record", action="store_true",
                   help="save one replay of the first matchup for the viewer")
    return p.parse_args()


def main():
    args = parse_args()
    ai = PolicyRuntime(args.checkpoint_dir)
    status = ai.load()
    assert status["ready"], f"checkpoints not ready: {status['errors']}"
    print(f"checkpoints: {args.checkpoint_dir} "
          f"(blue+red loaded, obs_dim from files)")

    formations = None
    if args.formation:
        b, r = args.formation.split(",")
        formations = (b.strip(), r.strip())

    cfg = BattleConfig(n_units_per_side=args.n_units, max_steps=args.max_steps)
    env = WarEnv(cfg, seed=args.seed)

    header = (f"{'matchup (policy side first)':<34s} {'W-L-D':>8s} {'alive':>6s} "
              f"{'space':>7s} {'spread':>7s} {'hull/u':>8s} {'ally60':>7s} "
              f"{'flank%':>6s}")
    print(f"\n{args.episodes} episodes/matchup, n={args.n_units}, "
          f"max_steps={args.max_steps}\n{header}\n" + "-" * len(header))

    # 1) self-play mirror
    res = run_matchup(env, {0: ("ppo", ai.policies), 1: ("ppo", ai.policies)},
                      args.episodes, args.seed, formations)
    print(fmt_row("blue_policy vs red_policy", res, 0, args.n_units))
    print(fmt_row("", res, 1, args.n_units))

    # 2) each policy vs each scripted opponent (record the first one)
    first = True
    for team in (0, 1):
        for opp_idx, opp_name in enumerate(SCRIPTED_POLICIES):
            opp_team = 1 - team
            controllers = {team: ("ppo", ai.policies),
                           opp_team: ("scripted", opp_name)}
            if args.record and first:
                obs_env = WarEnv(cfg, seed=args.seed)
                rng = random.Random(args.seed)
                play_episode(
                    obs_env, controllers, seed=rng.randrange(2**31),
                    formations=formations or ("scatter", "scatter"),
                    recorder=ReplayRecorder(),
                )
                first = False
            res = run_matchup(env, controllers, args.episodes,
                              args.seed + 97 * (team + 1) + opp_idx,
                              formations)
            side = "blue" if team == 0 else "red"
            print(fmt_row(f"{side}_policy vs {opp_name}", res, team, args.n_units))

    # 3) reference: what a deathball looks like
    print("-" * len(header))
    res = run_matchup(env, {0: ("scripted", "blob"), 1: ("scripted", "line")},
                      args.episodes, args.seed + 999, formations)
    print(fmt_row("[reference] blob vs line", res, 0, args.n_units))
    res = run_matchup(env, {0: ("scripted", "line"), 1: ("scripted", "blob")},
                      args.episodes, args.seed + 1000, formations)
    print(fmt_row("[reference] line vs blob", res, 1, args.n_units))

    print(
        "\nreading the table:\n"
        "  spacing/allies60/hull-u: a stacked deathball scores ~0 spacing,\n"
        "  high allies60, low hull/u; formed teams keep spacing and room.\n"
        "  flank%: share of hits earned from outside the target's facing\n"
        "  arc (maneuver quality). Reference rows show the blob baseline."
    )


if __name__ == "__main__":
    main()
