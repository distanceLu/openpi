from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Pi05DenoiseTransition:
    """One inner-MDP transition: x_k -> x_{k+1} inside one environment step."""

    step_index: torch.Tensor
    timestep: torch.Tensor
    x_cur: torch.Tensor
    x_next: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    logprob: torch.Tensor
    velocity: torch.Tensor


@dataclass
class Pi05RolloutOutput:
    """Data saved during rollout for PPO-style pi0.5 RL updates.

    Shapes use B for environment batch size, K for denoise steps, H for action
    horizon, and D for action dimension.
    """

    actions: torch.Tensor  # [B, H, D]
    chains: torch.Tensor  # [B, K + 1, H, D]
    denoise_logprobs: torch.Tensor  # [B, K, H, D]
    denoise_means: torch.Tensor  # [B, K, H, D]
    denoise_stds: torch.Tensor  # [B, K, H, D]
    denoise_timesteps: torch.Tensor  # [B, K]
    denoise_indices: torch.Tensor  # [B, K]
    velocities: torch.Tensor  # [B, K, H, D]
    values: torch.Tensor | None = None  # [B, K] or [B]


@dataclass
class Pi05LogProbOutput:
    """Current-policy logprobs recomputed from a saved denoise chain."""

    logprobs: torch.Tensor  # [B, K, H, D]
    means: torch.Tensor  # [B, K, H, D]
    stds: torch.Tensor  # [B, K, H, D]
    velocities: torch.Tensor  # [B, K, H, D]
    values: torch.Tensor | None = None
