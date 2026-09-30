"""Run one scripted battle and record a replay.

Initial deployments are configurable from the command line, e.g.:

    python run_sim.py                          # legacy uniform scatter
    python run_sim.py --blue wedge --red line
    python run_sim.py --list-formations

Units use the engine's default "move toward nearest enemy" behavior, so
this doubles as a quick visual check of the tactical combat mechanics
(focus-fire cap, flank bonus) in the replay viewer.
"""

import argparse
import json

from war_sim.core import BattleConfig, BattleSimulator, FORMATION_NAMES
from war_sim.recorder import ReplayRecorder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one scripted battle")
    p.add_argument("--blue", default="scatter", choices=FORMATION_NAMES,
                   help="blue initial deployment")
    p.add_argument("--red", default="scatter", choices=FORMATION_NAMES,
                   help="red initial deployment")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--list-formations", action="store_true",
                   help="list the available formations and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_formations:
        print("formations:", " ".join(FORMATION_NAMES))
        return

    cfg = BattleConfig(blue_formation=args.blue, red_formation=args.red)
    sim = BattleSimulator(cfg, seed=args.seed)
    recorder = ReplayRecorder()
    recorder.start(sim, source="run_sim")
    print(f"deployments: blue={args.blue} red={args.red} seed={args.seed}")

    done = False
    for _ in range(args.steps):
        state, done = sim.step()
        recorder.record_step(state)
        if sim.step_count % 30 == 0:
            print(
                f"t={state['time']:5.0f}s "
                f"blue={state['stats']['blue_alive']:2d} "
                f"red={state['stats']['red_alive']:2d} "
                f"blue_eff={state['stats']['blue_fire_efficiency']:.2f} "
                f"red_eff={state['stats']['red_fire_efficiency']:.2f} "
                f"flank(B/R)={state['stats']['blue_flank_hits']}/"
                f"{state['stats']['red_flank_hits']}"
            )
        if done:
            break

    replay_path = recorder.finish(reason="episode_end" if done else "script_exit")
    print(f"saved replay: {replay_path}")

    with open("battle_final.json", "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print("saved battle_final.json")


if __name__ == "__main__":
    main()
