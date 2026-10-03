"""Minimal standalone PPO for the 100-unit war simulator.

No RLlib / PettingZoo: plain PyTorch. Two independent Actor-Critic
policies (blue / red); within a faction every unit shares one policy
(parameter sharing) and each unit is one trajectory.

Option D (docs/design-options-c-d.md §2.3): policies consume the
structured set observation (war_sim.env.SetObs). Three selectable
architectures (PPO(..., arch=...), stored in every checkpoint):

    "mlp"       flat baseline: the structured record is flattened and
                fed through the legacy MLP -- it sees exactly the same
                information as the set encoders, so D-vs-MLP controlled
                experiments differ in architecture, not in information
    "deepsets"  D1: shared per-element encoder + masked mean pooling;
                trunk input = self features | pooled ally | pooled
                enemy | role embedding
    "setformer" D2: small transformer over [self, ally*, enemy*] tokens
                with token-type embeddings; ally/enemy cross-attention
                happens inside one joint token sequence ("see what the
                teammates are looking at"). Role embedding rides on the
                self token; padded set tokens are masked attention keys
                and only the self token feeds the heads.

Components:
    build_actor_critic  architecture factory (arch name -> module)
    Trajectory          per-unit step storage (obs/action/logp/value/reward;
                        obs entries are SetObs records)
    compute_gae         GAE(lambda) advantages for one trajectory
    PPO                 clipped-surrogate update with entropy bonus

Checkpoint compatibility: every checkpoint stores its arch + ObsSpec.
Files without an "arch" key are the legacy flat-vector format and are
REFUSED at load time with an explicit error (mis-reading them silently
would be worse than failing loudly -- see docs/lessons-learned.md §7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .env import ObsSpec, SetObs, collate

ARCHS = ("mlp", "deepsets", "setformer")

# Long-tensor fields of the batch dict (everything else is float32).
_INT_FIELDS = ("role",)


def to_tensors(batch: dict, device: str | torch.device = "cpu") -> dict:
    """numpy collate batch -> tensor batch on the given device."""
    out = {}
    for k, v in batch.items():
        dtype = torch.int64 if k in _INT_FIELDS else torch.float32
        out[k] = torch.as_tensor(v, dtype=dtype, device=device)
    return out


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over valid set elements [N,U,D] x [N,U] -> [N,D].

    All-masked rows (a lone survivor sees no allies) pool to zeros.
    """
    m = mask.unsqueeze(-1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


class BaseActorCritic(nn.Module):
    """Shared act/value interface over structured batches."""

    def forward(self, batch: dict):
        """tensor batch dict -> (logits [N, A], value [N])."""
        raise NotImplementedError

    @torch.no_grad()
    def act(self, batch: dict, deterministic: bool = False):
        """batch (numpy collate output) -> (actions, logps, values)."""
        logits, values = self.forward(to_tensors(batch))
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
    def value(self, obs: SetObs) -> float:
        """V(s) for a single observation (terminal bootstrap)."""
        _, v = self.forward(to_tensors(collate([obs])))
        return float(v.item())


class MLPActorCritic(BaseActorCritic):
    """Flat baseline (arch "mlp"): legacy trunk over the flattened record.

    The masks are part of the flattened vector (padded rows are zero,
    but the explicit valid-count information costs little and keeps the
    baseline from having to infer set sizes from zeros)."""

    def __init__(self, spec: ObsSpec, n_actions: int, hidden: int = 128):
        super().__init__()
        self.spec = spec
        self.trunk = nn.Sequential(
            nn.Linear(spec.flat_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.pi_head = nn.Linear(hidden, n_actions)
        self.v_head = nn.Linear(hidden, 1)

    def forward(self, batch: dict):
        flat = torch.cat(
            [
                batch["self"],
                batch["ally"].flatten(1),
                batch["enemy"].flatten(1),
                batch["ally_mask"],
                batch["enemy_mask"],
            ],
            dim=1,
        )
        h = self.trunk(flat)
        return self.pi_head(h), self.v_head(h).squeeze(-1)


class DeepSetsActorCritic(BaseActorCritic):
    """D1 set encoder (arch "deepsets"): per-element MLP + masked mean.

    One shared encoder embeds both sets (ally and enemy elements are
    the same kind of quantity); mean pooling is permutation-invariant
    and has none of the optimization quirks of attention, which is why
    D1 goes first through the pipeline."""

    def __init__(self, spec: ObsSpec, n_actions: int, hidden: int = 128,
                 set_hidden: int = 64, role_dim: int = 16):
        super().__init__()
        self.spec = spec
        self.set_enc = nn.Sequential(
            nn.Linear(spec.rel_dim, set_hidden),
            nn.ReLU(),
            nn.Linear(set_hidden, set_hidden),
        )
        self.role_emb = nn.Embedding(spec.n_roles, role_dim)
        self.trunk = nn.Sequential(
            nn.Linear(spec.self_dim + 2 * set_hidden + role_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.pi_head = nn.Linear(hidden, n_actions)
        self.v_head = nn.Linear(hidden, 1)

    def forward(self, batch: dict):
        pooled_ally = _masked_mean(self.set_enc(batch["ally"]), batch["ally_mask"])
        pooled_enemy = _masked_mean(self.set_enc(batch["enemy"]),
                                    batch["enemy_mask"])
        h = self.trunk(torch.cat(
            [batch["self"], pooled_ally, pooled_enemy,
             self.role_emb(batch["role"])], dim=1))
        return self.pi_head(h), self.v_head(h).squeeze(-1)


class SetformerActorCritic(BaseActorCritic):
    """D2 set encoder (arch "setformer"): joint-token transformer.

    tokens = [self | ally* | enemy*], each set element projected to
    d_model and tagged with a token-type embedding; the observing
    unit's role embedding rides on the self token. A couple of
    self-attention layers give every ally token direct access to every
    enemy token (the cross-attention the design doc calls the point of
    D2: a unit can see which enemies its teammates are facing). Padded
    elements are masked as attention KEYS; their query outputs are
    garbage but never read -- only the self token feeds the trunk, so
    the model stays permutation-invariant over valid elements.

    max_set truncates each set to its nearest elements (the sets are
    distance-sorted, so slicing = nearest-k): the doc's explicit
    fallback when full-set iteration time is unacceptable. Measured
    on this 4-core CPU: 8.8s per 2048-sample update with all 101
    tokens vs 1.6s with 33 -- full sets would cost ~25 min/iter,
    ~25x the doc's whole D2 budget. DeepSets keeps full sets (its
    cost is linear and stays at ~1.75x MLP).
    """

    def __init__(self, spec: ObsSpec, n_actions: int, hidden: int = 128,
                 d_model: int = 64, nhead: int = 4, layers: int = 2,
                 ffn: int = 128, role_dim: int = 16, max_set: int = 16):
        super().__init__()
        self.spec = spec
        self.max_set = min(max_set, spec.max_units)
        self.self_proj = nn.Linear(spec.self_dim, d_model)
        self.set_proj = nn.Linear(spec.rel_dim, d_model)
        self.type_emb = nn.Embedding(3, d_model)  # self / ally / enemy
        self.role_emb = nn.Embedding(spec.n_roles, role_dim)
        self.role_proj = nn.Linear(role_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.trunk = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.pi_head = nn.Linear(hidden, n_actions)
        self.v_head = nn.Linear(hidden, 1)

    def forward(self, batch: dict):
        n = batch["self"].shape[0]
        device = batch["self"].device
        k = self.max_set
        self_tok = (
            self.self_proj(batch["self"])
            + self.type_emb.weight[0]
            + self.role_proj(self.role_emb(batch["role"]))
        )
        ally_tok = (self.set_proj(batch["ally"][:, :k])
                    + self.type_emb.weight[1])
        enemy_tok = (self.set_proj(batch["enemy"][:, :k])
                     + self.type_emb.weight[2])
        tokens = torch.cat([self_tok[:, None], ally_tok, enemy_tok], dim=1)
        # True = ignore as an attention key; the self token is always
        # valid, so no row is ever fully masked (that would NaN).
        pad = torch.cat(
            [
                torch.zeros(n, 1, dtype=torch.bool, device=device),
                batch["ally_mask"][:, :k] == 0.0,
                batch["enemy_mask"][:, :k] == 0.0,
            ],
            dim=1,
        )
        h = self.encoder(tokens, src_key_padding_mask=pad)
        out = self.trunk(h[:, 0])
        return self.pi_head(out), self.v_head(out).squeeze(-1)


def build_actor_critic(arch: str, spec: ObsSpec, n_actions: int,
                       hidden: int = 128) -> BaseActorCritic:
    if arch == "mlp":
        return MLPActorCritic(spec, n_actions, hidden)
    if arch == "deepsets":
        return DeepSetsActorCritic(spec, n_actions, hidden)
    if arch == "setformer":
        return SetformerActorCritic(spec, n_actions, hidden)
    raise ValueError(f"unknown arch {arch!r}; choose one of {ARCHS}")


@dataclass
class Trajectory:
    """One unit's episode. Steps are contiguous: the unit acts every env
    step from reset until it dies (terminal_by_death) or the episode ends
    (bootstrap from the terminal observation)."""

    obs: List[SetObs] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    logps: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    terminal_by_death: bool = False
    bootstrap_value: float = 0.0

    def add(self, obs: SetObs, action: int, logp: float, value: float):
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
        spec: ObsSpec,
        n_actions: int,
        params: Optional[PPOParams] = None,
        device: str = "cpu",
        arch: str = "deepsets",
    ):
        if arch not in ARCHS:
            raise ValueError(f"unknown arch {arch!r}; choose one of {ARCHS}")
        self.spec = spec
        self.n_actions = n_actions
        self.arch = arch
        self.p = params or PPOParams()
        self.device = torch.device(device)
        self.model = build_actor_critic(
            arch, spec, n_actions, self.p.hidden
        ).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.p.lr)

    # ------------------------------------------------------------------
    def update(self, trajectories: List[Trajectory]) -> dict:
        trajectories = [t for t in trajectories if len(t) > 0]
        obs = [o for t in trajectories for o in t.obs]
        actions = np.concatenate(
            [t.actions for t in trajectories]
        ).astype(np.int64)
        old_logp = np.concatenate([t.logps for t in trajectories])
        advs, rets = zip(
            *(compute_gae(t, self.p.gamma, self.p.lam) for t in trajectories)
        )
        adv = np.concatenate(advs)
        ret = np.concatenate(rets)

        # Advantage normalization over the whole batch.
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # One collate for the whole batch (single batching code); the
        # structured fields are stacked once and sliced per minibatch.
        obs_t = to_tensors(collate(obs), device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device)
        old_logp_t = torch.as_tensor(old_logp, dtype=torch.float32,
                                     device=self.device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=self.device)

        N = actions_t.shape[0]
        stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0,
                 "clip_frac": 0.0, "n_updates": 0, "n_samples": N}
        mb = min(self.p.minibatch_size, N)

        for _ in range(self.p.epochs):
            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, mb):
                idx = perm[start:start + mb]
                batch = {k: v[idx] for k, v in obs_t.items()}
                logits, value = self.model(batch)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(actions_t[idx])

                ratio = (logp - old_logp_t[idx]).exp()
                adv_b = adv_t[idx]
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.p.clip,
                                    1.0 + self.p.clip) * adv_b
                pi_loss = -torch.min(surr1, surr2).mean()

                v_loss = 0.5 * (value - ret_t[idx]).pow(2).mean()
                entropy = dist.entropy().mean()

                loss = pi_loss + self.p.vf_coef * v_loss - self.p.ent_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.p.max_grad_norm)
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
                "arch": self.arch,
                "spec": self.spec.as_dict(),
                "n_actions": self.n_actions,
                "params": self.p.__dict__,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "PPO":
        data = torch.load(path, map_location=device)
        if "arch" not in data or "spec" not in data:
            raise ValueError(
                f"{path}: legacy flat-observation checkpoint (predates "
                "option D's structured sets; no arch/spec fields). Refused "
                "instead of mis-read -- retrain with the current train.py"
            )
        spec = ObsSpec.from_dict(data["spec"])
        params = PPOParams(**data["params"])
        agent = cls(spec, data["n_actions"], params, device,
                    arch=data["arch"])
        agent.model.load_state_dict(data["model"])
        return agent
