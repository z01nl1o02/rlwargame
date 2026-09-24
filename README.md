# 100-Unit War Simulator

A minimal architecture that keeps one battle engine behind three interfaces:

    BattleSimulator
          |
          +---- FastAPI/WebSocket ----> PixiJS UI
          |
          +---- WarEnv (RL adapter)
          |        |
          |        +---- standalone PPO training (train.py)
          |        +---- PolicyRuntime serving (war_sim/runtime.py)


## 1. Backend

Python 3.11+ is recommended.
    uv sync

Run a pure simulation:

    uv run python run_sim.py

Run the web server:

    uv run uvicorn server:app --reload --port 8000

## 2. Frontend

In another terminal:

    cd frontend
    npm install
    npm run dev

Open the Vite URL, normally:

    http://localhost:5173

You should see 50 blue + 50 red units, movement trails, live force counts,
fire efficiency, and simulation time.

## 3. Training (standalone PPO)

After the simulator and environment work:

    cd backend
    uv run python train.py

The training script creates two shared policies:

    Bxxx -> blue_policy
    Rxxx -> red_policy

Each policy controls 50 units; every unit is one trajectory.
Checkpoints are written to `backend/checkpoints/war_100/`.

Quick runs for smoke-testing the pipeline (full training is ~50 iters):

    WAR_TRAIN_ITERS=3 uv run python train.py

`WAR_CHECKPOINT_DIR` overrides the checkpoint directory for both
training and serving.

## 4. Serving trained policies in the backend

The web server loads the checkpoints at startup and drives every unit
with the trained policies. Observations are built by the same `WarEnv`
the policies were trained on, so serving matches training exactly.

    uv run uvicorn server:app --reload --port 8000

If a checkpoint is missing or corrupt, that faction silently falls back
to the engine's default "move toward nearest enemy" behavior -- the
server always runs.

Endpoints:

    GET  /ai/status                 loaded policies + enabled/active flags
    POST /control/ai/on|off         toggle policy-driven vs default behavior
    POST /control/ai/reload         hot-reload checkpoints after retraining

Headless check of the serving path (no browser needed):

    uv run python ai_check.py

## 5. Replay recording

Every simulation is recorded to one JSON file per battle, so the full
course of the fight can be visualized later by external tools:

- `run_sim.py` records its run.
- The web server records every episode; a manual `/control/reset`
  finalizes the current file and starts a new recording.

Files go to `backend/replays/` (gitignored) by default:

    WAR_REPLAY_DIR=/path/to/dir    # override the output directory
    WAR_REPLAY_KEEP=20             # keep only the newest n files (default: all)

File format (format_version 1): `meta` holds recording time, source,
seed, the full `BattleConfig`, the winner and final stats; `frames` holds
one snapshot per step. `frames[0]` is the initial deployment after reset;
every following frame carries `step`, `time`, `units`, `stats` and the
`events` of that single step, in exactly the shape of the live
`state_dict()` packets -- a viewer can reuse the frontend rendering code
directly.

## 6. Architecture

The important boundary is:

    action
      |
      v
    BattleSimulator.step()
      |
      +--> movement
      +--> target selection
      +--> fire
      +--> hit probability
      +--> damage
      +--> destruction
      |
      v
    BattleState

The UI never changes the combat rules.

## 7. Next extensions

The first version deliberately keeps combat simple. The next layer should add:

- formations / groups
- sensors and fog of war
- terrain
- morale
- ammunition and logistics
- command delay
- objectives
- event/replay files
- squad-level hierarchical agents
- GNN/Transformer observation encoder
- self-play
- world-model battle prediction
