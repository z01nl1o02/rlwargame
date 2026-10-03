"""Serving layer: load trained PPO checkpoints and drive the simulator.

The web server keeps a WarEnv (exactly the same observation construction
as training -- see train.py/run_episode) and asks PolicyRuntime for actions
every step. Degradation contract: if a faction's checkpoint is missing or
corrupt, act() simply emits no actions for that faction and BattleSimulator
falls back to its built-in "move toward nearest enemy" behavior, so the
server always runs -- with or without trained models.

Usage:
    ai = PolicyRuntime()          # default: backend/checkpoints/war_100
    status = ai.load()            # or reload() to pick up fresh checkpoints
    actions = ai.act(obs)         # {unit_id: action} for observed agents
"""

from __future__ import annotations

import os
from pathlib import Path

from .core import BattleConfig
from .env import ObsSpec, collate, obs_spec_for
from .ppo import PPO

# (team, checkpoint stem) -- must match train.py's checkpoint names.
TEAM_POLICIES: tuple[tuple[int, str], ...] = ((0, "blue_policy"), (1, "red_policy"))


def default_checkpoint_dir() -> Path:
    env_dir = os.environ.get("WAR_CHECKPOINT_DIR")
    if env_dir:
        return Path(env_dir)
    # train.py saves to ./checkpoints/war_100 relative to backend/; resolve
    # against this file so the server finds the checkpoints regardless of CWD.
    return Path(__file__).resolve().parent.parent / "checkpoints" / "war_100"


def team_of(agent_id: str) -> int:
    return 0 if agent_id.startswith("B") else 1


class PolicyRuntime:
    """Loads blue/red PPO checkpoints and batches inference per faction."""

    def __init__(
        self,
        checkpoint_dir: str | os.PathLike | None = None,
        device: str = "cpu",
        expected_spec: ObsSpec | None = None,
    ):
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir else default_checkpoint_dir()
        )
        self.device = device
        # What the serving env produces (default BattleConfig shape);
        # checkpoints trained for another spec are refused explicitly.
        self.expected_spec = expected_spec or obs_spec_for(BattleConfig())
        self.policies: dict[int, PPO] = {}

    @property
    def ready(self) -> bool:
        return len(self.policies) == len(TEAM_POLICIES)

    def load(self) -> dict:
        """(Re)load checkpoints from disk. Never raises; returns a status.

        Checkpoints that do not match the current observation contract
        are refused explicitly (old-format files must fail loudly per
        checkpoint, not mis-read silently): legacy flat-vector files
        (predating option D) and files trained for a different ObsSpec
        both degrade to the engine default behavior with a clear error.
        """
        self.policies.clear()
        errors: dict[str, str] = {}
        for team, name in TEAM_POLICIES:
            path = self.checkpoint_dir / f"{name}.pt"
            if not path.exists():
                errors[name] = f"missing: {path}"
                continue
            try:
                ppo = PPO.load(str(path), device=self.device)
            except Exception as exc:  # legacy / corrupted / incompatible
                errors[name] = repr(exc)
                continue
            if ppo.spec != self.expected_spec:
                errors[name] = (
                    f"obs spec {ppo.spec} != serving {self.expected_spec}: "
                    "checkpoint trained for a different observation format"
                )
                continue
            self.policies[team] = ppo
        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "loaded": [name for t, name in TEAM_POLICIES if t in self.policies],
            "errors": errors,
            "ready": self.ready,
        }

    def act(
        self, obs: dict, deterministic: bool = True
    ) -> dict[str, int]:
        """Actions for the given observations (one batched forward per
        faction). Agents whose faction has no loaded policy get no entry,
        which the engine treats as its default behavior."""
        actions: dict[str, int] = {}
        for team, ppo in self.policies.items():
            ids = [a for a in obs if team_of(a) == team]
            if not ids:
                continue
            batch = collate([obs[a] for a in ids])
            acts, _, _ = ppo.model.act(batch, deterministic=deterministic)
            for i, a in enumerate(ids):
                actions[a] = int(acts[i])
        return actions
