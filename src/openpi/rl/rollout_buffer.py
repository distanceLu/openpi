from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import torch

from openpi.rl.returns import compute_gae

ObservationCollator = Callable[[list[Any]], Any]


def _scalar_tensor(value: torch.Tensor | float | bool | int, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu().reshape(-1)
    if tensor.numel() != 1:
        raise ValueError(f"{name} must contain one scalar per SMDP transition, got shape {tuple(tensor.shape)}")
    return tensor


@dataclass
class Pi05RolloutBatch:
    observation: Any
    chains: torch.Tensor  # [B, K + 1, H, action_dim]
    old_logprobs: torch.Tensor  # [B, K, H, action_dim]
    denoise_indices: torch.Tensor  # [B, K]
    denoise_timesteps: torch.Tensor  # [B, K]
    old_velocities: torch.Tensor | None  # [B, K, H, action_dim]
    advantages: torch.Tensor  # [B]
    returns: torch.Tensor  # [B]
    loss_mask: torch.Tensor | None = None
    outer_loss_mask: torch.Tensor | None = None


@dataclass
class Pi05RolloutBuffer:
    """CPU SMDP rollout storage; each row is one executed action chunk."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    observations: list[Any] = field(default_factory=list)
    chains: list[torch.Tensor] = field(default_factory=list)
    old_logprobs: list[torch.Tensor] = field(default_factory=list)
    denoise_indices: list[torch.Tensor] = field(default_factory=list)
    denoise_timesteps: list[torch.Tensor] = field(default_factory=list)
    old_velocities: list[torch.Tensor] = field(default_factory=list)
    rewards: list[torch.Tensor] = field(default_factory=list)
    terminated: list[torch.Tensor] = field(default_factory=list)
    truncated: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)
    next_values: list[torch.Tensor] = field(default_factory=list)
    durations: list[torch.Tensor] = field(default_factory=list)

    def add(
        self,
        observation: Any,
        chains: torch.Tensor,
        old_logprobs: torch.Tensor,
        denoise_indices: torch.Tensor,
        denoise_timesteps: torch.Tensor,
        reward: torch.Tensor | float,
        terminated: torch.Tensor | bool,
        truncated: torch.Tensor | bool,
        value: torch.Tensor | float,
        next_value: torch.Tensor | float,
        duration: torch.Tensor | int,
        old_velocities: torch.Tensor | None = None,
    ) -> None:
        self.observations.append(observation)
        self.chains.append(chains.detach().cpu())
        self.old_logprobs.append(old_logprobs.detach().cpu())
        self.denoise_indices.append(denoise_indices.detach().cpu())
        self.denoise_timesteps.append(denoise_timesteps.detach().cpu())
        if old_velocities is not None:
            if len(self.old_velocities) != len(self.observations) - 1:
                raise ValueError("old_velocities must be supplied for every transition or none")
            self.old_velocities.append(old_velocities.detach().cpu())
        elif self.old_velocities:
            raise ValueError("old_velocities must be supplied for every transition or none")
        self.rewards.append(_scalar_tensor(reward, "reward"))
        self.terminated.append(_scalar_tensor(terminated, "terminated"))
        self.truncated.append(_scalar_tensor(truncated, "truncated"))
        self.values.append(_scalar_tensor(value, "value"))
        self.next_values.append(_scalar_tensor(next_value, "next_value"))
        self.durations.append(_scalar_tensor(duration, "duration"))

    def clear(self) -> None:
        for items in (
            self.observations, self.chains, self.old_logprobs, self.denoise_indices, self.denoise_timesteps,
            self.old_velocities, self.rewards, self.terminated, self.truncated, self.values, self.next_values, self.durations,
        ):
            items.clear()

    def __len__(self) -> int:
        return len(self.rewards)

    def compute_advantages(self) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = torch.stack(self.rewards)
        values = torch.stack(self.values)
        next_values = torch.stack(self.next_values)
        terminated = torch.stack(self.terminated)
        episode_ends = torch.maximum(terminated, torch.stack(self.truncated))
        durations = torch.stack(self.durations)
        return compute_gae(rewards, values, next_values, terminated, episode_ends, durations, self.gamma, self.gae_lambda)

    def iter_minibatches(
        self, collate_observations: ObservationCollator, minibatch_size: int, device: torch.device, shuffle: bool = True,
        indices: torch.Tensor | None = None, precomputed: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Iterator[Pi05RolloutBatch]:
        if len(self) == 0:
            raise ValueError("rollout buffer is empty")
        if precomputed is None:
            advantages, returns = self.compute_advantages()
        else:
            advantages, returns = precomputed
        flat_count = len(self.observations)
        if indices is None:
            indices = torch.randperm(flat_count) if shuffle else torch.arange(flat_count)
        chains, old_logprobs = torch.cat(self.chains), torch.cat(self.old_logprobs)
        denoise_indices, denoise_timesteps = torch.cat(self.denoise_indices), torch.cat(self.denoise_timesteps)
        old_velocities = torch.cat(self.old_velocities) if self.old_velocities else None
        if chains.shape[0] != flat_count:
            raise ValueError(f"Expected {flat_count} chain rows, got {chains.shape[0]}")
        for start in range(0, flat_count, minibatch_size):
            idx = indices[start : start + minibatch_size]
            yield Pi05RolloutBatch(
                observation=collate_observations([self.observations[i] for i in idx.tolist()]),
                chains=chains[idx].to(device), old_logprobs=old_logprobs[idx].to(device),
                denoise_indices=denoise_indices[idx].to(device), denoise_timesteps=denoise_timesteps[idx].to(device),
                old_velocities=old_velocities[idx].to(device) if old_velocities is not None else None,
                advantages=advantages.reshape(-1)[idx].to(device), returns=returns.reshape(-1)[idx].to(device),
                outer_loss_mask=torch.ones(idx.numel(), dtype=torch.bool, device=device),
            )
