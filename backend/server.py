from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from war_sim.core import BattleConfig
from war_sim.env import WarEnv
from war_sim.recorder import ReplayRecorder
from war_sim.runtime import PolicyRuntime


# WarEnv wraps BattleSimulator: identical combat rules, plus the exact
# observation construction the policies were trained on. state_dict() is
# delegated to the simulator, so the frontend contract is unchanged.
env = WarEnv(BattleConfig(), seed=42)

ai = PolicyRuntime()
ai_info: dict = {"ready": False, "loaded": [], "errors": {}, "checkpoint_dir": ""}

# One replay file per episode; a manual reset finalizes the current file.
recorder = ReplayRecorder()

running = True
speed = 10.0  # simulation steps per real second
use_ai = True  # drive units with trained policies; falls back to engine default


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ai_info, obs
    ai_info = ai.load()  # degrade gracefully if checkpoints are missing
    print(f"[ai] policies: {ai_info}", flush=True)
    obs = env.reset()
    recorder.start(env.sim, source="server")
    task = asyncio.create_task(sim_loop())
    yield
    task.cancel()


app = FastAPI(title="War Simulator 100", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

obs: dict = {}


async def sim_loop():
    global running, obs
    while True:
        if running:
            actions = ai.act(obs) if (use_ai and ai.ready) else {}
            obs, _, _, _ = env.step(actions)
            recorder.record_step(env.state_dict())
            if env.episode_done:
                recorder.finish("episode_end")
                obs = env.reset()
                recorder.start(env.sim, source="server")
        await asyncio.sleep(1 / max(1.0, speed))


@app.get("/state")
def state():
    return env.state_dict()


@app.get("/ai/status")
def ai_status():
    return {**ai_info, "enabled": use_ai, "active": use_ai and ai.ready}


@app.post("/control/{command}")
@app.post("/control/{command}/{value}")
def control(command: str, value: str | None = None):
    global running, speed, use_ai, obs, ai_info

    if command == "pause":
        running = False
    elif command == "play":
        running = True
    elif command == "speed" and value is not None:
        speed = max(1.0, float(value))
    elif command == "reset":
        recorder.finish("manual_reset")
        obs = env.reset()
        recorder.start(env.sim, source="server")
    elif command == "ai" and value in ("on", "off"):
        use_ai = value == "on"
    elif command == "ai" and value == "reload":
        ai_info = ai.load()
        use_ai = True

    return {
        "running": running,
        "speed": speed,
        "ai": {"enabled": use_ai, "ready": ai.ready, "active": use_ai and ai.ready},
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_text(json.dumps(env.state_dict()))
            await asyncio.sleep(1 / 15)
    except Exception:
        pass
