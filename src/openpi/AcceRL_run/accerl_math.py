"""Small framework-level math helpers shared by AcceRL runners."""

from __future__ import annotations

import numpy as np


def compute_smdp_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    durations: np.ndarray,
    *,
    bootstrap_value: float,
    is_terminal: bool,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE for variable-duration outer-MDP transitions."""

    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    durations = np.asarray(durations, dtype=np.int64)
    if rewards.ndim != 1 or values.shape != rewards.shape or durations.shape != rewards.shape:
        raise ValueError("rewards, values, and durations must be matching one-dimensional arrays")
    if np.any(durations <= 0):
        raise ValueError("all SMDP transition durations must be positive")

    advantages = np.zeros_like(rewards)
    gae = 0.0
    for index in reversed(range(len(rewards))):
        if index == len(rewards) - 1:
            next_value = 0.0 if is_terminal else float(bootstrap_value)
        else:
            next_value = float(values[index + 1])
        duration = int(durations[index])
        discount = gamma**duration
        trace_discount = (gamma * gae_lambda) ** duration
        delta = float(rewards[index]) + discount * next_value - float(values[index])
        gae = delta + trace_discount * gae
        advantages[index] = gae
    returns = advantages + values
    return advantages, returns.astype(np.float32)


def warmup_cosine_lr_scale(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup followed by cosine decay, evaluated before ``step`` updates."""

    if step < 0 or warmup_steps < 0 or total_steps <= 0:
        raise ValueError("step/warmup_steps must be non-negative and total_steps must be positive")
    if warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be smaller than total_steps")
    if step < warmup_steps:
        return float(step) / max(warmup_steps, 1)
    progress = min(max((step - warmup_steps) / (total_steps - warmup_steps), 0.0), 1.0)
    return float(0.5 * (1.0 + np.cos(np.pi * progress)))
