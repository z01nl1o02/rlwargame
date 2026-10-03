"""debug reward settings
Usage:
    cd backend
    python evaluate.py                          # checkpoints/war_100
    python evaluate.py --episodes 20
    python evaluate.py --formation wedge,line  # pin initial deployments
    python evaluate.py --record                # also save one replay
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Any
import random
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from war_sim.core import BattleConfig, FORMATION_NAMES
from war_sim.commander import Commander
from war_sim.env import WarEnv
from war_sim.env import SHAPE_DIST_K, SPACE_K, TEAM_SIZE_K, PERSONAL_DMG_K, SPACE_IDEAL_RATIO
from war_sim.recorder import ReplayRecorder
from war_sim.runtime import PolicyRuntime, team_of
from war_sim.scripted import SCRIPTED_POLICIES

FORMATION_POOL = ("scatter", *FORMATION_NAMES[1:])
# Commander doctrines sampled per evaluated faction per episode
# (matches the training distribution; see train.COMMANDER_POOL).
COMMANDER_POOL = FORMATION_NAMES[1:]


#########################################################
#codes from war_sim.env
def dist_to_nearest_teammate(unit, friends):
    best = None
    for f in friends:
        if f is unit:
            continue
        d2 = (f.x - unit.x) ** 2 + (f.y - unit.y) ** 2
        if best is None or d2 < best:
            best = d2
    return None if best is None else best ** 0.5

def spacing_potential(d: float | None, ideal_d0: float, ideal_d1: float) -> float:
    """
    d < ideal_d0 时, d 越小,返回的值越小(负数),最小值 -1
    ideal_d0 <= d <= ideal_d1 时, 返回 0
    d > ideal_d1 时, d 越大,返回的值越小(负数), 最小值 -1 ( d >= 2 * ideal_d1 )
    """
    if d is None or ideal_d0 <= d <= ideal_d1:
        return 0.0
    if d < ideal_d0:
        return -max(0.0, 1.0 - d / ideal_d0) 
    
    return max(-1.0,  -(d - ideal_d1) / ideal_d1) 




def read_stats(env: WarEnv) -> dict[str, dict]:

    
    macro = {
        team: env._macro(team) 
        for team in (0,1)
        }
   
    damage_by_team = {
        team: env.sim.damage_by_team[team]
        for team in (0,1)
    } 
    
    damage_by_unit = dict((a,env.sim.damage_by_unit.get(a, 0.0)) for a in env.sim.units.keys())
    hp_unit = dict((a,env.sim.units[a].hp) for a in env.sim.units.keys())
 
    alived_teams = {
        0: [u for u in env.sim.units.values() if u.team==0],
        1: [u for u in env.sim.units.values() if u.team==1]
    }
    nn_to_teammate = {
        a : -1 if not u.alive else dist_to_nearest_teammate(u, alived_teams[u.team]) 
        for a,u in env.sim.units.items()
    }
   
    flag_aliving ={
        a: u.alive
        for a,u in env.sim.units.items()
    } 
    
    return {
        "macro": macro,
        "damage_by_team": damage_by_team,
        "damage_by_unit": damage_by_unit,
        "hp_unit": hp_unit,
        "nn_to_teammate": nn_to_teammate,
        "flag_aliving": flag_aliving
    } 

     
def calc_rewards(env:WarEnv, before:dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    damage_by_team = {0:0, 1:0}
    for team in (0,1):
        damage_by_team[team] = (after['damage_by_team'][team] - before['damage_by_team'][team]) * 0.001
    hp_reward_of_unit = {}
    for a, hp_before in before['hp_unit'].items():
        hp_after = after['hp_unit'].get(a,0.0)
        hp_reward_of_unit[a] = 0.01 * (hp_after - hp_before)
        
    nn_to_teammate = {}
    ideal_d0 = env.cfg.formation_spacing  - env.cfg.formation_jitter * SPACE_IDEAL_RATIO
    ideal_d1 = env.cfg.formation_spacing + env.cfg.formation_jitter * SPACE_IDEAL_RATIO
    for a, nn in before['nn_to_teammate'].items():
        if not after['flag_aliving'][a]:
            nn_to_teammate[a] = -1
            continue 
        nn_before = before['nn_to_teammate'][a]
        nn_after = after['nn_to_teammate'][a]
        nn_to_teammate[a] = SPACE_K * (spacing_potential(nn_after,ideal_d0, ideal_d1) - spacing_potential(nn_before, ideal_d0, ideal_d1)) 
    return {
        "macro": after['macro'],
        "r_damage_by_team": damage_by_team,
        "r_hp_of_unit": hp_reward_of_unit,
        "r_nn_to_teammate": nn_to_teammate,
        "nn_to_teammate": after['nn_to_teammate'],
        'aliving': after['flag_aliving']
    }

def save_rewards_csv(path:str, rewards_all:list[dict[str,Any]]):
    data = {}
    data['phase_blue'] = [r['macro'][0]['phase'] for r in rewards_all]
    data['phase_red'] = [r['macro'][1]['phase'] if r['macro'][1] else "" for r in rewards_all]
    
    data["r_damage_by_team_0"] = [r['r_damage_by_team'][0] for r in rewards_all]
    data["r_damage_by_team_1"] = [r['r_damage_by_team'][1] for r in rewards_all]
   
    uints_id_all = rewards_all[0]['r_hp_of_unit'].keys()
    for a in uints_id_all: 
        data[f'aliving_{a}'] = [r['aliving'][a] for r in rewards_all]
        data[f'nn_to_teammate_{a}'] = [r['nn_to_teammate'][a] for r in rewards_all]
        data[f"r_nn_to_teammate_{a}"] = [r['r_nn_to_teammate'][a] for r in rewards_all]
        data[f'r_hp_of_{a}'] = [r['r_hp_of_unit'][a] for r in rewards_all]
        
    pd.DataFrame(data).to_csv(path,index=True)
    return 

# ----------------------------------------------------------------------
# player
# ----------------------------------------------------------------------
def play_episode(
    env: WarEnv,
    controllers: dict,   # team -> ("ppo"|"ppo-nomacro"|"scripted", ctrl)
    seed: int,
    formations: tuple[str, str],
    recorder: ReplayRecorder | None = None,
):
    """Play one episode. Controller kinds:

      "ppo"         trained policy + Commander macro context (the
                    observations match training: a random doctrine is
                    sampled per episode from COMMANDER_POOL)
      "ppo-nomacro" same policy, no commander attached -- ablation for
                    how much the policy relies on the macro block
      "scripted"    a war_sim.scripted policy (reads the sim state)
    """
    rng = random.Random(seed)
    commanders = {}
    for team, (kind, _ctrl) in controllers.items():
        if kind == "ppo":
            commanders[team] = Commander(rng.choice(COMMANDER_POOL))
    obs = env.reset(seed=seed, formations=formations,
                    commanders=(commanders.get(0), commanders.get(1)))
    if recorder is not None:
        recorder.start(env.sim, source="evaluate")
    rewards_all = []
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
        stats_before = read_stats(env)
        obs, _, _, _ = env.step(actions)
        stats_after = read_stats(env)
        rewards = calc_rewards(env,stats_before, stats_after) 
        rewards_all.append(rewards)
        if recorder is not None:
            recorder.record_step(env.state_dict())
    if recorder is not None:
        recorder.finish("episode_end")
        
    save_rewards_csv("./tests_reward.csv",rewards_all)
    blue, red = env.alive_counts()
    winner = 0 if (red == 0 and blue > 0) else (1 if (blue == 0 and red > 0) else -1)
    return winner, blue, red


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
              f"{'flank%':>6s} {'slot_d':>6s}")
    print(f"\n{args.episodes} episodes/matchup, n={args.n_units}, "
          f"max_steps={args.max_steps}\n{header}\n" + "-" * len(header))

    controllers = {
                    0: ("ppo", ai.policies),
                   1: ("scripted", "line")}
    rng = random.Random(args.seed)
    play_episode(
        env, controllers, seed=rng.randrange(2**31),
        formations=formations or ("wedge", "wedge"),
        recorder=ReplayRecorder(),
    )

if __name__ == "__main__":
    main()
