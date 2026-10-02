"""Standalone PPO training for the 100-unit war simulator.

No RLlib / PettingZoo — plain PyTorch on top of war_sim.env.

The training script creates two shared policies:

    Bxxx -> blue_policy
    Rxxx -> red_policy

Each policy controls 50 units; every unit is one trajectory.

Curriculum (anti-deathball):
    * every episode starts from a random scripted formation per team
      (scatter/line/column/wedge/echelon/crescent) so policies must cope
      with -- and learn to exploit -- structured deployments
    * league play: with probability --league-prob an episode pits one
      learning faction against a scripted opponent from
      war_sim.scripted (hold/blob/line/flank); the scripted faction's
      units act but contribute no trajectories. Fixed diverse opponents
      keep self-play from collapsing into one symmetric deathball
      equilibrium, and give a measurable win-rate bar per opponent.

Test the simulator first with:
    python run_sim.py

Then run:
    python train.py                       # defaults: 50 iters
    python train.py --iters 200 --envs 4  # longer run
"""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import torch

from war_sim.core import BattleConfig, FORMATION_NAMES
from war_sim.env import N_ACTIONS, OBS_DIM, WarEnv
from war_sim.ppo import PPO, PPOParams, Trajectory
from war_sim.scripted import SCRIPTED_POLICIES

N_UNITS_PER_SIDE = 50
MAX_STEPS = 600
N_ENVS = 2  # independent battles collected per iteration
SEED = 42
LEAGUE_PROB = 0.5  # chance an episode is league (vs scripted) play
CHECKPOINT_DIR = os.environ.get("WAR_CHECKPOINT_DIR", "./checkpoints/war_100")

# Deployment formations sampled per team per episode (curriculum variety).
FORMATION_POOL = ("scatter", *FORMATION_NAMES[1:])


def team_of(agent_id: str) -> int:
    return 0 if agent_id.startswith("B") else 1


def run_episode(
    env: WarEnv,
    policies: dict,
    seed: int | None = None,
    formations: tuple[str, str] | None = None,
    opponent_name: str | None = None,
    opponent_team: int | None = None,
    deterministic: bool = False,
):
    """Play one full battle with the current policies.

    opponent_name/opponent_team: if set, that faction is driven by the
    scripted policy (no trajectories collected for it) and only the
    other faction learns this episode.

    Returns (per-agent trajectories of the learning faction(s), stats).
    """
    opponent = SCRIPTED_POLICIES[opponent_name] if opponent_name else None
    learn_teams = (0, 1) if opponent is None else (1 - opponent_team,)

    obs = env.reset(seed=seed, formations=formations)
    trajectories = {a: Trajectory() for a in obs if team_of(a) in learn_teams}
    ended_by_truncation = False

    while not env.episode_done:
        actions = {}
        for team in (0, 1):
            ids = [a for a in obs if team_of(a) == team]
            if not ids:
                continue
            if opponent is not None and team == opponent_team:
                # Scripted faction: reads the pre-step simulator state.
                opp_actions = opponent.act(env, team)
                for a in ids:
                    actions[a] = int(opp_actions.get(a, 0))
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

        # Every acting learning unit receives its reward for this step.
        for a, r in rewards.items():
            if a in trajectories:
                trajectories[a].rewards.append(float(r))
        for a in dead:
            if a in trajectories:
                trajectories[a].terminal_by_death = True

        # Episode over: survivors bootstrap from the terminal observation
        # (units that died this step need no bootstrap).
        if env.episode_done:
            for a, o in obs.items():
                if a in trajectories and a not in dead:
                    trajectories[a].bootstrap_value = policies[team_of(a)].model.value(
                        o
                    )

    blue, red = env.alive_counts()
    winner = 0 if (red == 0 and blue > 0) else (1 if (blue == 0 and red > 0) else -1)
    returns = {t: [] for t in learn_teams}
    for a, t in trajectories.items():
        returns[team_of(a)].append(sum(t.rewards))

    stats = {
        "blue_alive": blue,
        "red_alive": red,
        "winner": winner,
        "blue_return": float(np.mean(returns[0])) if returns.get(0) else 0.0,
        "red_return": float(np.mean(returns[1])) if returns.get(1) else 0.0,
        "truncated": ended_by_truncation,
        "opponent": opponent_name,
        "opponent_team": opponent_team,
        "formations": formations,
    }
    return trajectories, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO training for the war simulator")
    p.add_argument(
        "--iters",
        type=int,
        default=int(os.environ.get("WAR_TRAIN_ITERS", "50")),
        help="training iterations (env: WAR_TRAIN_ITERS)",
    )
    p.add_argument(
        "--envs", type=int, default=N_ENVS, help="battles collected per iteration"
    )
    p.add_argument(
        "--n-units", type=int, default=N_UNITS_PER_SIDE, help="units per side"
    )
    p.add_argument(
        "--max-steps", type=int, default=MAX_STEPS, help="step limit per episode"
    )
    p.add_argument(
        "--league-prob",
        type=float,
        default=LEAGUE_PROB,
        help="probability an episode is played vs a scripted opponent",
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Episode-level randomness (seeds, formations, league draws).
    rng = random.Random(args.seed)

    cfg = BattleConfig(n_units_per_side=args.n_units, max_steps=args.max_steps)
    envs = [WarEnv(cfg, seed=args.seed + k) for k in range(args.envs)]
    policies = {}
    for team in (0, 1):
        torch.manual_seed(args.seed + team)  # independent weight init per faction
        policies[team] = PPO(OBS_DIM, N_ACTIONS, PPOParams(), device="cpu")

    print(
        f"standalone PPO: {args.n_units} units/side, max_steps={args.max_steps}, "
        f"envs={args.envs}, iters={args.iters}, seed={args.seed}, "
        f"league_prob={args.league_prob}"
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    def save_checkpoints():
        for team, name in ((0, "blue_policy"), (1, "red_policy")):
            policies[team].save(os.path.join(args.checkpoint_dir, f"{name}.pt"))

    wins = {0: 0, 1: 0, -1: 0}
    # League table from the learning faction's point of view: [W, L, D].
    league = {name: [0, 0, 0] for name in SCRIPTED_POLICIES}

    for it in range(args.iters):
        t0 = time.perf_counter()

        trajs = {0: [], 1: []}
        ep_stats = []
        agent_steps = 0
        matchups = []
        for env in envs:
            if rng.random() < args.league_prob:
                opponent_name = rng.choice(list(SCRIPTED_POLICIES))
                opponent_team = rng.randrange(2)
            else:
                opponent_name, opponent_team = None, None
            formations = (
                rng.choice(FORMATION_POOL),
                rng.choice(FORMATION_POOL),
            )
            ep_trajs, s = run_episode(
                env,
                policies,
                seed=rng.randrange(2**31),
                formations=formations,
                opponent_name=opponent_name,
                opponent_team=opponent_team,
            )
            for a, t in ep_trajs.items():
                trajs[team_of(a)].append(t)
                agent_steps += len(t)
            ep_stats.append(s)
            matchups.append(
                "self"
                if opponent_name is None
                else f"{opponent_name}@{'BR'[opponent_team]}"
            )
            wins[s["winner"]] += 1

            if opponent_name is not None:
                learner = 1 - opponent_team
                row = league[opponent_name]
                row[0 if s["winner"] == learner else 1 if s["winner"] >= 0 else 2] += 1
        # upd = {team: policies[team].update(trajs[team]) for team in (0, 1)}
        # FIXBUG: 如果某一次迭代里,3 局全部是 league 局、且脚本对手恰好都坐在同一队,那么另一队整轮 0 条轨迹
        #          → np.concatenate([]) → ValueError: need at least one array to concatenate。
        upd = {}
        for team in (0, 1):
            if trajs[team]:
                upd[team] = policies[team].update(trajs[team])
            else:
                upd[team] = {
                    "approx_kl": 0.0,
                    "entropy": 0.0,
                    "pi_loss": 0.0,
                    "v_loss": 0.0,
                }
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
            f"opp={matchups} "
            f"steps={agent_steps} "
            f"pi_loss={pi_loss:+.3f} v_loss={v_loss:.3f} "
            f"ent={ent:.3f} kl={kl:.4f} "
            f"wall={wall:.1f}s",
            flush=True,
        )
        if (it + 1) % 10 == 0:
            table = " ".join(f"{n}:{w}/{l}/{d}" for n, (w, l, d) in league.items())
            print(f"        league (W/L/D vs scripted): {table}", flush=True)
            save_checkpoints()

    save_checkpoints()
    for team, name in ((0, "blue_policy"), (1, "red_policy")):
        print(
            f"checkpoint saved to " f"{os.path.join(args.checkpoint_dir, f'{name}.pt')}"
        )


if __name__ == "__main__":
    main()
