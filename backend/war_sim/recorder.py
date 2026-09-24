"""Replay recording: persist the full course of a battle to one JSON file.

The recorder is a passive observer: it only reads state_dict() snapshots of
a BattleSimulator (or WarEnv, whose sim is a BattleSimulator). Combat rules
(core.py) and the RL adapter (env.py) stay untouched, and training is
unaffected -- recording is opt-in per entry point.

File format (format_version 1):

    {
      "meta": {
        "format_version", "recorded_at", "source", "seed",
        "config": {...BattleConfig fields...},
        "steps", "duration", "ended_by",
        "result": {winner, alive counts, damage, fire efficiency}
      },
      "frames": [
        {"step", "time", "units": [...], "stats": {...}, "events": [...]},
        ...
      ]
    }

frames[0] is the initial deployment after reset; every following frame is
one engine step. units/stats have exactly the shape of the live
state_dict() packets the web frontend consumes, so an external viewer can
render any frame with the same code. "events" holds only the events that
happened in that single step: the engine keeps just the most recent events
in each state packet, so the recorder diffs consecutive packets.

One file per episode, written when the episode ends (or when the recording
is replaced, e.g. by a manual reset). If the process dies mid-episode, that
recording is lost. Files go to backend/replays/ by default (override with
WAR_REPLAY_DIR); WAR_REPLAY_KEEP=<n> keeps only the newest n files.

The lock matters for the server: sim_loop runs on the event loop while
sync control endpoints run in FastAPI's thread pool, so start/record/finish
can interleave.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .core import BattleSimulator

FORMAT_VERSION = 1


def default_replay_dir() -> Path:
    env_dir = os.environ.get("WAR_REPLAY_DIR")
    if env_dir:
        return Path(env_dir)
    # Resolve against this file so the location is stable regardless of CWD
    # (same convention as runtime.default_checkpoint_dir).
    return Path(__file__).resolve().parent.parent / "replays"


class ReplayRecorder:
    """Accumulates per-step snapshots of one episode and writes them to a
    single JSON file on finish()."""

    def __init__(self, replay_dir: str | os.PathLike | None = None):
        self.replay_dir = Path(replay_dir) if replay_dir else default_replay_dir()
        self._lock = threading.Lock()
        self._meta: dict | None = None
        self._frames: list[dict] = []
        self._prev_events: list[dict] = []

    @property
    def active(self) -> bool:
        return self._meta is not None

    def start(self, sim: BattleSimulator, source: str) -> None:
        """Start recording a freshly reset simulator. Frame 0 captures the
        initial deployment."""
        with self._lock:
            self._meta = {
                "format_version": FORMAT_VERSION,
                "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source": source,
                "seed": sim.seed,
                "config": asdict(sim.cfg),
            }
            self._frames = []
            self._prev_events = []
            self._append_locked(sim.state_dict())

    def record_step(self, state: dict) -> None:
        """Append one snapshot: state_dict() taken right after sim.step()
        returned. No-op while no recording is active."""
        with self._lock:
            if self._meta is None:
                return
            self._append_locked(state)

    def finish(self, reason: str = "episode_end") -> Path | None:
        """Write the recording to disk and stop recording. Returns the file
        path, or None if nothing was being recorded."""
        with self._lock:
            if self._meta is None:
                return None

            last = self._frames[-1]
            stats = last["stats"]
            blue, red = stats["blue_alive"], stats["red_alive"]
            winner = "blue" if blue and not red else "red" if red and not blue else "draw"
            meta = {
                **self._meta,
                "steps": last["step"],
                "duration": last["time"],
                "ended_by": reason,
                "result": {
                    "winner": winner,
                    "blue_alive": blue,
                    "red_alive": red,
                    "blue_damage": stats["blue_damage"],
                    "red_damage": stats["red_damage"],
                    "blue_fire_efficiency": stats["blue_fire_efficiency"],
                    "red_fire_efficiency": stats["red_fire_efficiency"],
                },
            }

            self.replay_dir.mkdir(parents=True, exist_ok=True)
            base = datetime.now().strftime("battle_%Y%m%d_%H%M%S")
            path = self.replay_dir / f"{base}.json"
            n = 2
            while path.exists():
                path = self.replay_dir / f"{base}_{n}.json"
                n += 1

            with open(path, "w", encoding="utf-8") as f:
                json.dump({"meta": meta, "frames": self._frames}, f, ensure_ascii=False)

            self._meta = None
            self._frames = []
            self._prev_events = []

        self._enforce_retention()
        return path

    # ------------------------------------------------------------------
    def _append_locked(self, state: dict) -> None:
        events = state.get("events", [])
        self._frames.append({
            "step": state["step"],
            "time": state["time"],
            "units": state["units"],
            "stats": state["stats"],
            "events": self._new_events(events),
        })
        self._prev_events = list(events)

    def _new_events(self, events: list[dict]) -> list[dict]:
        """Events that happened in the latest step.

        The engine trims its cumulative event list to the most recent ones
        per state packet, so the new events are the part of the current
        packet extending beyond the overlap with the previous packet's
        retained window (a suffix/prefix match)."""
        prev = self._prev_events
        n, m = len(prev), len(events)
        for k in range(min(n, m), -1, -1):
            if prev[n - k:] == events[:k]:
                return list(events[k:])
        return []  # unreachable: k == 0 always matches

    def _enforce_retention(self) -> None:
        keep = os.environ.get("WAR_REPLAY_KEEP", "")
        try:
            keep_n = int(keep) if keep else 0
        except ValueError:
            return
        if keep_n <= 0:
            return
        # Timestamped names sort chronologically (collision suffixes sort
        # after their base name), so the newest files are the last ones.
        files = sorted(self.replay_dir.glob("battle_*.json"))
        for old in files[:-keep_n]:
            old.unlink(missing_ok=True)
