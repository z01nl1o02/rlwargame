import json
import time

from war_sim.core import BattleSimulator, BattleConfig
from war_sim.recorder import ReplayRecorder

if __name__ == "__main__":
    sim = BattleSimulator(BattleConfig(), seed=7)
    recorder = ReplayRecorder()
    recorder.start(sim, source="run_sim")

    done = False
    for _ in range(600):
        state, done = sim.step()
        recorder.record_step(state)
        if sim.step_count % 30 == 0:
            print(
                f"t={state['time']:5.0f}s "
                f"blue={state['stats']['blue_alive']:2d} "
                f"red={state['stats']['red_alive']:2d} "
                f"blue_eff={state['stats']['blue_fire_efficiency']:.2f} "
                f"red_eff={state['stats']['red_fire_efficiency']:.2f}"
            )
        if done:
            break

    replay_path = recorder.finish(reason="episode_end" if done else "script_exit")
    print(f"saved replay: {replay_path}")

    with open("battle_final.json", "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    print("saved battle_final.json")
