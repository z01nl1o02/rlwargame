"""Standalone PPO training for the 100-unit war simulator.

No RLlib / PettingZoo — plain PyTorch on top of war_sim.env.

The training script creates two shared policies:

    Bxxx -> blue_policy
    Rxxx -> red_policy

Each policy controls 50 units; every unit is one trajectory.

Test the simulator first with:
    python run_sim.py

Then run:
    python train.py
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from war_sim.core import BattleConfig
from war_sim.env import N_ACTIONS, OBS_DIM, WarEnv
from war_sim.ppo import PPO, PPOParams, Trajectory

N_UNITS_PER_SIDE = 50
MAX_STEPS = 600
N_ENVS = 2            # independent battles collected per iteration
N_ITERS = int(os.environ.get("WAR_TRAIN_ITERS", "50"))  # quick runs: WAR_TRAIN_ITERS=3
SEED = 42
CHECKPOINT_DIR = os.environ.get("WAR_CHECKPOINT_DIR", "./checkpoints/war_100")


def team_of(agent_id: str) -> int:
    return 0 if agent_id.startswith("B") else 1


def run_episode(env: WarEnv, policies: dict, deterministic: bool = False):
    """Play one full battle with the current policies.

    Returns (per-agent trajectories, stats dict).
    """
    obs = env.reset()
    trajectories = {a: Trajectory() for a in obs}
    ended_by_truncation = False

    while not env.episode_done:
        actions = {}
        for team in (0, 1):
            ids = [a for a in obs if team_of(a) == team]
            if not ids:
                continue
            batch = np.stack([obs[a] for a in ids])
            acts, logps, values = policies[team].model.act(batch, deterministic)
            for i, a in enumerate(ids):
                actions[a] = int(acts[i])
                trajectories[a].add(
                    obs[a], int(acts[i]), float(logps[i]), float(values[i])
                )

        obs, rewards, dead, truncated = env.step(actions)
        ended_by_truncation = truncated

        # Every acting unit receives its (team-shaped) reward for this step.
        for a, r in rewards.items():
            trajectories[a].rewards.append(float(r))
        for a in dead:
            trajectories[a].terminal_by_death = True

        # Episode over: survivors bootstrap from the terminal observation
        # (units that died this step need no bootstrap).
        if env.episode_done:
            for a, o in obs.items():
                if a not in dead:
                    trajectories[a].bootstrap_value = policies[team_of(a)].model.value(o)

    blue, red = env.alive_counts()
    returns = {0: [], 1: []}
    for a, t in trajectories.items():
        returns[team_of(a)].append(sum(t.rewards))

    stats = {
        "blue_alive": blue,
        "red_alive": red,
        "winner": 0 if (red == 0 and blue > 0) else (1 if (blue == 0 and red > 0) else -1),
        "blue_return": float(np.mean(returns[0])) if returns[0] else 0.0,
        "red_return": float(np.mean(returns[1])) if returns[1] else 0.0,
        "truncated": ended_by_truncation,
    }
    return trajectories, stats


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cfg = BattleConfig(n_units_per_side=N_UNITS_PER_SIDE, max_steps=MAX_STEPS)
    envs = [WarEnv(cfg, seed=SEED + k) for k in range(N_ENVS)]
    policies = {}
    for team in (0, 1):
        torch.manual_seed(SEED + team)  # independent weight init per faction
        policies[team] = PPO(OBS_DIM, N_ACTIONS, PPOParams(), device="cpu")

    print(
        f"standalone PPO: {N_UNITS_PER_SIDE} units/side, max_steps={MAX_STEPS}, "
        f"envs={N_ENVS}, iters={N_ITERS}, seed={SEED}"
    )

    wins = {0: 0, 1: 0, -1: 0}
    for it in range(N_ITERS):
        t0 = time.perf_counter()

        trajs = {0: [], 1: []}
        ep_stats = []
        agent_steps = 0
        for env in envs:
            ep_trajs, s = run_episode(env, policies)
            for a, t in ep_trajs.items():
                trajs[team_of(a)].append(t)
                agent_steps += len(t)
            ep_stats.append(s)
            wins[s["winner"]] += 1

        upd = {team: policies[team].update(trajs[team]) for team in (0, 1)}
        wall = time.perf_counter() - t0

        blue_ret = float(np.mean([s["blue_return"] for s in ep_stats]))
        red_ret = float(np.mean([s["red_return"] for s in ep_stats]))
        kl = 0.5 * (upd[0]["approx_kl"] + upd[1]["approx_kl"])
        ent = 0.5 * (upd[0]["entropy"] + upd[1]["entropy"])
        pi_loss = 0.5 * (upd[0]["pi_loss"] + upd[1]["pi_loss"])
        v_loss = 0.5 * (upd[0]["v_loss"] + upd[1]["v_loss"])
        print(
            f"iter={it:03d} "
            f"blue_return={blue_ret:+8.2f} red_return={red_ret:+8.2f} "
            f"wins(B/R/draw)={wins[0]}/{wins[1]}/{wins[-1]} "
            f"steps={agent_steps} "
            f"pi_loss={pi_loss:+.3f} v_loss={v_loss:.3f} "
            f"ent={ent:.3f} kl={kl:.4f} "
            f"wall={wall:.1f}s",
            flush=True,
        )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    for team, name in ((0, "blue_policy"), (1, "red_policy")):
        path = os.path.join(CHECKPOINT_DIR, f"{name}.pt")
        policies[team].save(path)
        print(f"checkpoint saved to {path}")


if __name__ == "__main__":
    main()
