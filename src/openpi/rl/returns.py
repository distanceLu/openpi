from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    episode_ends: torch.Tensor,
    durations: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE for variable-duration outer-SMDP transitions.

    Rewards are already discounted within each action chunk. True termination
    disables bootstrap; termination or truncation stops the trace at a reset.
    """

    tensors = (rewards, values, next_values, terminated, episode_ends, durations)
    if any(tensor.shape != rewards.shape for tensor in tensors[1:]):
        shapes = [tuple(tensor.shape) for tensor in tensors]
        raise ValueError(f"All GAE inputs must have identical shapes, got {shapes}")
    if rewards.numel() == 0:
        raise ValueError("GAE inputs must not be empty")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")
    if torch.any(durations < 1):
        raise ValueError("Every SMDP transition duration must be positive")
    if torch.any((terminated != 0) & (terminated != 1)) or torch.any((episode_ends != 0) & (episode_ends != 1)):
        raise ValueError("terminated and episode_ends must be binary masks")
    if torch.any(terminated > episode_ends):
        raise ValueError("Every terminated transition must also be an episode end")

    dtype = values.dtype if values.is_floating_point() else torch.float32
    device = values.device
    rewards = rewards.to(device=device, dtype=dtype)
    values = values.to(device=device, dtype=dtype)
    next_values = next_values.to(device=device, dtype=dtype)
    terminated = terminated.to(device=device, dtype=dtype)
    episode_ends = episode_ends.to(device=device, dtype=dtype)
    durations = durations.to(device=device, dtype=dtype)

    gamma_t = torch.as_tensor(gamma, device=device, dtype=dtype).pow(durations)
    trace_t = torch.as_tensor(gamma * gae_lambda, device=device, dtype=dtype).pow(durations)
    deltas = rewards + gamma_t * (1.0 - terminated) * next_values - values
    advantages = torch.zeros_like(deltas)
    running_advantage = torch.zeros_like(deltas[-1])
    for index in range(deltas.shape[0] - 1, -1, -1):
        running_advantage = deltas[index] + trace_t[index] * (1.0 - episode_ends[index]) * running_advantage
        advantages[index] = running_advantage
    return advantages, advantages + values
