"""One-off round-robin between all scripted policies (M1 acceptance).

Usage: uv run python scripted_roundrobin.py [episodes] [n_units] [max_steps]
"""
from __future__ import annotations

import random
import sys
import time

from war_sim.core import BattleConfig
from war_sim.env import WarEnv
from war_sim.scripted import SCRIPTED_POLICIES


def play(blue: str, red: str, seed: int, n: int, max_steps: int):
    env = WarEnv(BattleConfig(n_units_per_side=n, max_steps=max_steps),
                 seed=seed)
    env.reset(seed=seed)
    pol_b = SCRIPTED_POLICIES[blue]
    pol_r = SCRIPTED_POLICIES[red]
    while not env.episode_done:
        actions = {}
        actions.update(pol_b.act(env, 0))
        actions.update(pol_r.act(env, 1))
        env.step(actions)
    b, r = env.alive_counts()
    dmg = (env.sim.damage_by_team[0], env.sim.damage_by_team[1])
    steps = env.sim.step_count
    winner = 0 if (r == 0 and b > 0) else (1 if (b == 0 and r > 0) else -1)
    return winner, b, r, dmg, steps


def main() -> None:
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    names = list(SCRIPTED_POLICIES)
    table = {(a, b): [0, 0, 0] for a in names for b in names}
    standoffs = []
    t0 = time.time()
    rng = random.Random(20260929)
    for blue in names:
        for red in names:
            for k in range(episodes):
                w, b, r, dmg, steps = play(blue, red,
                                           rng.randrange(2**31), n, max_steps)
                table[(blue, red)][0 if w == 0 else 1 if w == 1 else 2] += 1
                if dmg[0] == 0.0 and dmg[1] == 0.0 and steps >= max_steps:
                    standoffs.append((blue, red))
    print(f"round-robin: {episodes} eps, n={n}, max_steps={max_steps}, "
          f"{time.time()-t0:.0f}s   (rows = blue, W-L-D vs red)")
    hdr = "blue\\red " + "".join(f"{name:>9s}" for name in names)
    print(hdr)
    for blue in names:
        row = f"{blue:<8s} "
        for red in names:
            w, l, d = table[(blue, red)]
            row += f"{w:>4d}-{l:<3d}" if d == 0 else f"{w:>3d}-{l}-{d:<3d}"
        print(row)
    if standoffs:
        print("ZERO-DAMAGE STANDOFFS:", standoffs)
    else:
        print("no zero-damage standoffs")


if __name__ == "__main__":
    main()
