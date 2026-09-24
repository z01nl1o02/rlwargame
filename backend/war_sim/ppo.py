"""Minimal standalone PPO for the 100-unit war simulator.

No RLlib / PettingZoo: plain PyTorch. Two independent Actor-Critic
policies (blue / red); within a faction every unit shares one policy
(parameter sharing) and each unit is one trajectory.

Components:
    ActorCritic  small MLP with policy and value heads
    Trajectory   per-unit step storage (obs/action/logp/value/reward)
    compute_gae  GAE(lambda) advantages for one trajectory
    PPO          clipped-surrogate update with entropy bonus
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.pi_head = nn.Linear(hidden, n_actions)
        self.v_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        """obs [N, obs_dim] -> (logits [N, A], value [N])."""
        h = self.trunk(obs)
        return self.pi_head(h), self.v_head(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False):
        """obs [N, obs_dim] -> (actions [N], logps [N], values [N]) numpy."""
        logits, values = self.forward(torch.as_tensor(obs, dtype=torch.float32))
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            actions = dist.probs.argmax(dim=-1)
        else:
            actions = dist.sample()
        return (
            actions.numpy(),
            dist.log_prob(actions).numpy().astype(np.float64),
            values.numpy().astype(np.float64),
        )

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> float:
        """V(s) for a single observation (terminal bootstrap)."""
        _, v = self.forward(torch.as_tensor(obs[None], dtype=torch.float32))
        return float(v.item())


@dataclass
class Trajectory:
    """One unit's episode. Steps are contiguous: the unit acts every env
    step from reset until it dies (terminal_by_death) or the episode ends
    (bootstrap from the terminal observation)."""

    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    logps: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    terminal_by_death: bool = False
    bootstrap_value: float = 0.0

    def add(self, obs: np.ndarray, action: int, logp: float, value: float):
        self.obs.append(obs)
        self.actions.append(action)
        self.logps.append(logp)
        self.values.append(value)

    def __len__(self):
        return len(self.rewards)


def compute_gae(traj: Trajectory, gamma: float = 0.99, lam: float = 0.95):
    """GAE(lambda) advantages and value targets for one trajectory."""
    T = len(traj)
    adv = np.zeros(T, dtype=np.float64)
    last = 0.0
    for t in reversed(range(T)):
        # value[t+1] is V(s_{t+1}) because the next obs is exactly the obs
        # the unit acts from at t+1; at the end bootstrap from the terminal
        # observation (0.0 for death -> target collapses to the reward).
        next_v = traj.values[t + 1] if t + 1 < T else traj.bootstrap_value
        delta = traj.rewards[t] + gamma * next_v - traj.values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    returns = adv + np.asarray(traj.values, dtype=np.float64)
    return adv, returns


@dataclass
class PPOParams:
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    epochs: int = 4
    minibatch_size: int = 2048
    max_grad_norm: float = 0.5
    hidden: int = 128


class PPO:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        params: Optional[PPOParams] = None,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.p = params or PPOParams()
        self.device = torch.device(device)
        self.model = ActorCritic(obs_dim, n_actions, self.p.hidden).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.p.lr)

    # ------------------------------------------------------------------
    def update(self, trajectories: List[Trajectory]) -> dict:
        trajectories = [t for t in trajectories if len(t) > 0]
        obs = np.concatenate([t.obs for t in trajectories], axis=0)
        actions = np.concatenate([t.actions for t in trajectories]).astype(np.int64)
        old_logp = np.concatenate([t.logps for t in trajectories])
        advs, rets = zip(*(compute_gae(t, self.p.gamma, self.p.lam) for t in trajectories))
        adv = np.concatenate(advs)
        ret = np.concatenate(rets)

        # Advantage normalization over the whole batch.
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device)
        old_logp_t = torch.as_tensor(old_logp, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=self.device)

        N = obs_t.shape[0]
        stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0,
                 "clip_frac": 0.0, "n_updates": 0, "n_samples": N}
        mb = min(self.p.minibatch_size, N)

        for _ in range(self.p.epochs):
            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, mb):
                idx = perm[start:start + mb]
                logits, value = self.model(obs_t[idx])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(actions_t[idx])

                ratio = (logp - old_logp_t[idx]).exp()
                adv_b = adv_t[idx]
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.p.clip, 1.0 + self.p.clip) * adv_b
                pi_loss = -torch.min(surr1, surr2).mean()

                v_loss = 0.5 * (value - ret_t[idx]).pow(2).mean()
                entropy = dist.entropy().mean()

                loss = pi_loss + self.p.vf_coef * v_loss - self.p.ent_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.p.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    stats["pi_loss"] += float(pi_loss)
                    stats["v_loss"] += float(v_loss)
                    stats["entropy"] += float(entropy)
                    stats["approx_kl"] += float((old_logp_t[idx] - logp).mean())
                    stats["clip_frac"] += float(
                        ((ratio - 1.0).abs() > self.p.clip).float().mean()
                    )
                    stats["n_updates"] += 1

        if stats["n_updates"]:
            for k in ("pi_loss", "v_loss", "entropy", "approx_kl", "clip_frac"):
                stats[k] /= stats["n_updates"]
        return stats

    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save(
            {
                "model": self.model.state_dict(),
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
                "params": self.p.__dict__,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "PPO":
        data = torch.load(path, map_location=device)
        params = PPOParams(**data["params"])
        agent = cls(data["obs_dim"], data["n_actions"], params, device)
        agent.model.load_state_dict(data["model"])
        return agent
