from __future__ import annotations

import time
from typing import Any

import torch

from openpi.rl.http_protocol import RolloutUpdateResponse
from openpi.rl.rollout_buffer import Pi05RolloutBuffer


def update_from_payload(
    state: Any, payload: dict[str, Any], minibatch_size: int, rank: int = 0, world_size: int = 1
) -> dict[str, Any]:
    req = payload
    count = len(req["observations"])
    full_buffer = Pi05RolloutBuffer(gamma=state.gamma)
    for i in range(count):
        from openpi.rl.openpi_policy_server import _jsonable_observation_to_torch
        from openpi.models import model as _model
        observation = _model.Observation.from_dict(
            _jsonable_observation_to_torch(req["observations"][i], state.nested_mdp.device)
        )
        full_buffer.add(
            observation=observation,
            chains=torch.as_tensor(req["chains"][i]),
            old_logprobs=torch.as_tensor(req["old_logprobs"][i]),
            denoise_indices=torch.as_tensor(req["denoise_indices"][i]),
            denoise_timesteps=torch.as_tensor(req["denoise_timesteps"][i]),
            reward=req["rewards"][i], terminated=req["terminated"][i], truncated=req["truncated"][i],
            value=torch.as_tensor(req["values"][i]), next_value=torch.as_tensor(req["next_values"][i]),
            duration=req["durations"][i],
            old_velocities=torch.as_tensor(req["old_velocities"][i]) if req.get("old_velocities") is not None else None,
        )
    advantages, returns = full_buffer.compute_advantages()
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    selected = torch.arange(rank, count, world_size)
    buffer = Pi05RolloutBuffer(gamma=state.gamma)
    for index in selected.tolist():
        kwargs = dict(
            observation=full_buffer.observations[index], chains=full_buffer.chains[index],
            old_logprobs=full_buffer.old_logprobs[index], denoise_indices=full_buffer.denoise_indices[index],
            denoise_timesteps=full_buffer.denoise_timesteps[index], reward=full_buffer.rewards[index],
            terminated=full_buffer.terminated[index], truncated=full_buffer.truncated[index],
            value=full_buffer.values[index], next_value=full_buffer.next_values[index],
            duration=full_buffer.durations[index],
            old_velocities=full_buffer.old_velocities[index] if full_buffer.old_velocities else None,
        )
        buffer.add(**kwargs)
    local_advantages = advantages[selected]
    local_returns = returns[selected]
    metrics_list = []
    for batch in buffer.iter_minibatches(
        collate_observations=state.adapter.collate_observations,
        minibatch_size=max(1, min(minibatch_size, len(buffer))), device=state.nested_mdp.device,
        shuffle=False, precomputed=(local_advantages, local_returns),
    ):
        metrics_list.append(state.trainer.update(batch.__dict__))
    metrics = {key: torch.stack([item[key] for item in metrics_list]).mean() for key in metrics_list[0]}
    if torch.distributed.is_initialized():
        for value in metrics.values():
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
        metrics = {key: (value / world_size).item() for key, value in metrics.items()}
    else:
        metrics = {key: value.item() for key, value in metrics.items()}
    rewards = torch.as_tensor(req["rewards"], dtype=torch.float32)
    metrics.update({
        "rollout/reward_sum": rewards.sum().item(), "rollout/reward_mean": rewards.mean().item(),
        "rollout/reward_std": rewards.std(unbiased=False).item(), "rollout/episode_length": count,
        "optimizer/learning_rate": state.optimizer.param_groups[0]["lr"] if state.optimizer is not None else 0.0,
    })
    return metrics


def worker_loop(state: Any, minibatch_size: int) -> None:
    if not torch.distributed.is_initialized():
        raise RuntimeError("Distributed worker requires an initialized process group")
    while True:
        box: list[Any] = [None]
        torch.distributed.broadcast_object_list(box, src=0)
        payload = box[0]
        if payload is None:
            break
        update_from_payload(
            state, payload, minibatch_size,
            rank=torch.distributed.get_rank(), world_size=torch.distributed.get_world_size(),
        )
        torch.distributed.barrier()
