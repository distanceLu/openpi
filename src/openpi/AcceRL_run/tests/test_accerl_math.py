from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ACCE_RL_RUN_DIR = Path(__file__).resolve().parents[1]
if str(ACCE_RL_RUN_DIR) not in sys.path:
    sys.path.insert(0, str(ACCE_RL_RUN_DIR))

from accerl_math import compute_smdp_gae  # noqa: E402
from accerl_math import warmup_cosine_lr_scale  # noqa: E402
from ds_libero_ppo_pi05 import ReplayBufferActor  # noqa: E402
from ds_libero_ppo_pi05 import Trajectory  # noqa: E402

from openpi.rl.pi05_losses import compute_diagonal_gaussian_kl_loss  # noqa: E402


def _trajectory(version: int, timestamp: int, steps: int = 2) -> Trajectory:
    scalar = np.zeros((steps, 1), dtype=np.float32)
    return Trajectory(
        observations=[None] * steps,
        chains=scalar.copy(),
        old_logprobs=scalar.copy(),
        old_means=scalar.copy(),
        old_stds=np.ones_like(scalar),
        denoise_timesteps=scalar.copy(),
        denoise_indices=np.zeros((steps, 1), dtype=np.int64),
        old_velocities=scalar.copy(),
        rewards=np.zeros(steps, dtype=np.float32),
        old_values=np.zeros(steps, dtype=np.float32),
        durations=np.ones(steps, dtype=np.int64),
        bootstrap_value=0.0,
        bootstrap_observation=None,
        is_terminal=True,
        policy_versions=np.full(steps, version, dtype=np.int64),
        insert_times_ms=np.full(steps, timestamp, dtype=np.int64),
    )


def test_smdp_gae_uses_duration_and_bootstrap() -> None:
    advantages, returns = compute_smdp_gae(
        rewards=np.array([1.0, 2.0], dtype=np.float32),
        values=np.array([0.5, 0.25], dtype=np.float32),
        durations=np.array([2, 1], dtype=np.int64),
        bootstrap_value=0.75,
        is_terminal=False,
        gamma=0.9,
        gae_lambda=0.8,
    )

    last_delta = 2.0 + 0.9 * 0.75 - 0.25
    first_delta = 1.0 + 0.9**2 * 0.25 - 0.5
    expected_first = first_delta + (0.9 * 0.8) ** 2 * last_delta
    np.testing.assert_allclose(advantages, [expected_first, last_delta], rtol=1e-6)
    np.testing.assert_allclose(returns, advantages + np.array([0.5, 0.25]), rtol=1e-6)


def test_terminal_gae_ignores_bootstrap() -> None:
    advantages, _ = compute_smdp_gae(
        rewards=np.array([1.0], dtype=np.float32),
        values=np.array([0.4], dtype=np.float32),
        durations=np.array([1], dtype=np.int64),
        bootstrap_value=100.0,
        is_terminal=True,
        gamma=0.99,
        gae_lambda=0.95,
    )
    np.testing.assert_allclose(advantages, [0.6])


def test_warmup_cosine_schedule_boundaries() -> None:
    assert warmup_cosine_lr_scale(0, 10, 100) == 0.0
    assert warmup_cosine_lr_scale(5, 10, 100) == 0.5
    assert warmup_cosine_lr_scale(10, 10, 100) == 1.0
    assert warmup_cosine_lr_scale(100, 10, 100) == pytest.approx(0.0)


def test_diagonal_gaussian_kl_is_exact_and_differentiable() -> None:
    old_means = torch.tensor([0.0, 1.0])
    old_stds = torch.tensor([1.0, 2.0])
    new_means = torch.tensor([1.0, 1.0], requires_grad=True)
    new_stds = torch.tensor([1.0, 1.0], requires_grad=True)

    loss = compute_diagonal_gaussian_kl_loss(old_means, old_stds, new_means, new_stds)
    expected = torch.tensor([0.5, -torch.log(torch.tensor(2.0)) + 1.5]).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert new_means.grad is not None
    assert new_stds.grad is not None


def test_diagonal_gaussian_kl_zero_for_identical_policies() -> None:
    means = torch.randn(2, 3, 4)
    stds = torch.rand(2, 3, 4) + 0.1
    loss = compute_diagonal_gaussian_kl_loss(means, stds, means, stds)
    torch.testing.assert_close(loss, torch.zeros(()))


def test_replay_discards_stale_and_consumes_selected_trajectory() -> None:
    replay_class = ReplayBufferActor.__ray_metadata__.modified_class
    replay = replay_class(10)
    replay.add_trajectory(_trajectory(version=0, timestamp=1))
    replay.add_trajectory(_trajectory(version=9, timestamp=2))

    selected = replay.sample_trajectories(
        minimum_steps=2,
        current_policy_version=10,
        max_policy_lag=8,
        max_sample_reuse=1,
    )

    assert [int(item.policy_versions[0]) for item in selected] == [9]
    assert replay.size() == 0
    assert replay.get_stats()["replay/stale_discarded"] == 1.0


def test_replay_honors_bounded_reuse() -> None:
    replay_class = ReplayBufferActor.__ray_metadata__.modified_class
    replay = replay_class(10)
    replay.add_trajectory(_trajectory(version=3, timestamp=1))

    first = replay.sample_trajectories(2, 3, 1, 2)
    second = replay.sample_trajectories(2, 3, 1, 2)

    assert len(first) == len(second) == 1
    assert replay.size() == 0
