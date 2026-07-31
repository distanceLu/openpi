from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from torch.utils.tensorboard import SummaryWriter

from openpi.models import model as _model
from openpi.rl.http_protocol import RolloutUpdateRequest, RolloutUpdateResponse, SampleResponse
from openpi.rl.libero_adapter import Pi05LiberoRLAdapter
from openpi.rl.pi05_action_head import Pi05GaussianActionHead
from openpi.rl.pi05_nested_mdp import Pi05NestedMDP, Pi05NestedMDPConfig
from openpi.rl.pi05_trainer import Pi05PPOTrainConfig, Pi05PPOTrainer
from openpi.rl.run_pi05_libero_rl import configure_trainable_parameters
from openpi.rl.rollout_buffer import Pi05RolloutBuffer
from openpi.rl.monitor import TrainingMonitor


class SampleRequestModel(BaseModel):
    observation: Any
    mode: str = "train"
    return_values: bool = True


class RolloutUpdateRequestModel(BaseModel):
    observations: list[Any]
    chains: list[Any]
    old_logprobs: list[Any]
    denoise_indices: list[Any]
    denoise_timesteps: list[Any]
    rewards: list[float]
    terminated: list[bool]
    truncated: list[bool]
    values: list[Any]
    next_values: list[Any]
    durations: list[int]
    old_velocities: list[Any] | None = None


def _jsonable_observation_to_torch(observation: Any, device: torch.device) -> dict[str, Any]:
    if not isinstance(observation, dict):
        raise TypeError(f"Expected an observation object, got {type(observation)!r}")

    def convert(value: Any, dtype: torch.dtype | None = None) -> Any:
        if isinstance(value, dict):
            return {key: convert(item, dtype) for key, item in value.items()}
        if value is None:
            return None
        return torch.as_tensor(value, dtype=dtype, device=device)

    result = {
        "image": convert(observation.get("image"), torch.float32),
        "image_mask": convert(observation.get("image_mask"), torch.bool),
        "state": convert(observation.get("state"), torch.float32),
        "tokenized_prompt": convert(observation.get("tokenized_prompt"), torch.long),
        "tokenized_prompt_mask": convert(observation.get("tokenized_prompt_mask"), torch.bool),
        "token_ar_mask": convert(observation.get("token_ar_mask"), torch.long),
        "token_loss_mask": convert(observation.get("token_loss_mask"), torch.bool),
    }
    for key in ("image", "image_mask", "state"):
        if result[key] is None:
            raise ValueError(f"Serialized observation is missing required field {key!r}")
    if not isinstance(result["image"], dict) or set(result["image"]) != {
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    }:
        raise ValueError(f"Unexpected serialized image keys: {list(result['image']) if isinstance(result['image'], dict) else type(result['image'])}")
    if not isinstance(result["image_mask"], dict) or set(result["image_mask"]) != set(result["image"]):
        raise ValueError("image_mask keys must exactly match image keys")
    batch_size = result["state"].shape[0]
    if result["state"].shape != (batch_size, 32):
        raise ValueError(f"Expected normalized state [B, 32], got {tuple(result['state'].shape)}")
    for key, image in result["image"].items():
        if image.ndim != 4 or image.shape[0] != batch_size or image.shape[1] != 3:
            raise ValueError(f"Expected serialized image {key!r} as [B, 3, H, W], got {tuple(image.shape)}")
        if not torch.isfinite(image).all() or image.min() < -1.001 or image.max() > 1.001:
            raise ValueError(f"Serialized image {key!r} must be finite and scaled to [-1, 1]")
    if not torch.isfinite(result["state"]).all():
        raise ValueError("Serialized state contains NaN/Inf")
    if result["tokenized_prompt"] is None or result["tokenized_prompt_mask"] is None:
        raise ValueError("Serialized pi0.5 observation must include tokenized prompt and mask")
    if result["tokenized_prompt"].shape != result["tokenized_prompt_mask"].shape:
        raise ValueError("tokenized_prompt and tokenized_prompt_mask shapes must match")
    return result


def _validate_sample_output(rollout: Any, batch_size: int, action_horizon: int, action_dim: int) -> None:
    expected = (batch_size, action_horizon, action_dim)
    if tuple(rollout.actions.shape) != expected:
        raise ValueError(f"Expected pi0.5 actions {expected}, got {tuple(rollout.actions.shape)}")
    if not torch.isfinite(rollout.actions).all():
        raise ValueError("pi0.5 sampled actions contain NaN/Inf")
    if rollout.values is not None and tuple(rollout.values.reshape(-1).shape) != (batch_size,):
        raise ValueError(f"Expected one value per observation, got {tuple(rollout.values.shape)}")


@dataclass
class Pi05OpenPIServerState:
    adapter: Pi05LiberoRLAdapter
    nested_mdp: Pi05NestedMDP
    reference_nested_mdp: Pi05NestedMDP | None = None
    trainer: Pi05PPOTrainer | None = None
    optimizer: torch.optim.Optimizer | None = None
    engine: Any | None = None
    update_count: int = 0
    checkpoint_dir: Path | None = None
    checkpoint_interval: int = 0
    tensorboard_writer: SummaryWriter | None = None
    gamma: float = 0.99

    def save_checkpoint(self) -> Path | None:
        if self.checkpoint_dir is None or self.checkpoint_interval <= 0:
            return None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_dir / f"pi05_rl_update_{self.update_count:08d}.pt"
        temp_path = checkpoint_path.with_suffix(".tmp")
        torch.save(
            {
                "update_count": self.update_count,
                "model_state_dict": self.nested_mdp.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
            },
            temp_path,
        )
        temp_path.replace(checkpoint_path)
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(
            {
                "update_count": self.update_count,
                "model_state_dict": self.nested_mdp.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
            },
            latest_path,
        )
        return checkpoint_path


def create_app(
    checkpoint_path: str = "/mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT/model.safetensors",
    lora_adapter_path: str | None = None,
    reference_dir: str = "/mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT",
    assets_dir: str = "/mnt/data/lcx1/yiqinworkspace/openpi/assets",
    device: str = "cuda",
    num_denoise_steps: int = 10,
    sample_method: str = "flow_noise",
    full_model_training: bool = False,
    default_prompt: str = "",
    lr: float = 1e-5,
    target_kl: float = 0.03,
    checkpoint_dir: str | Path | None = None,
    checkpoint_interval: int = 0,
    tensorboard_log_dir: str | Path | None = None,
    gamma: float = 0.99,
    deepspeed_config: str | Path | None = None,
    ppo_minibatch_size: int = 1,
    monitor_log_dir: str | Path | None = None,
    distributed_coordinator: Any | None = None,
) -> FastAPI:
    adapter = Pi05LiberoRLAdapter(
        checkpoint_path=checkpoint_path,
        lora_adapter_path=lora_adapter_path,
        reference_dir=reference_dir,
        assets_dir=assets_dir,
        device=device,
        default_prompt=default_prompt,
    )
    model = adapter.load_pi05_model_from_checkpoint()
    model.rl_action_head = Pi05GaussianActionHead(
        input_dim=model.action_out_proj.in_features,
        action_dim=model.config.action_dim,
    ).to(device=next(model.parameters()).device, dtype=torch.float32)
    configure_trainable_parameters(model, full_model_training)
    nested_config = Pi05NestedMDPConfig(num_denoise_steps=num_denoise_steps, sample_method=sample_method)
    nested_mdp = Pi05NestedMDP(model, nested_config)

    reference_model = adapter.load_pi05_model_from_checkpoint()
    reference_model.rl_action_head = Pi05GaussianActionHead(
        input_dim=reference_model.action_out_proj.in_features,
        action_dim=reference_model.config.action_dim,
    ).to(device=next(reference_model.parameters()).device, dtype=torch.float32)
    reference_model.rl_action_head.load_state_dict(model.rl_action_head.state_dict())
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad_(False)
    reference_nested_config = Pi05NestedMDPConfig(
        num_denoise_steps=num_denoise_steps,
        sample_method=sample_method,
        require_trainable_heads=False,
    )
    reference_nested_mdp = Pi05NestedMDP(reference_model, reference_nested_config)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=lr)
    engine = None
    if deepspeed_config is not None:
        import deepspeed

        if not torch.distributed.is_initialized():
            deepspeed.init_distributed(dist_backend="nccl")
        engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=trainable_parameters,
            optimizer=optimizer,
            config=str(deepspeed_config),
        )
        nested_mdp.model = engine.module
    trainer = Pi05PPOTrainer(
        nested_mdp=nested_mdp,
        optimizer=optimizer,
        config=Pi05PPOTrainConfig(
            ppo_loss_mode="path",
            path_logprob_reduce="sum",
            reference_kl_coef=0.01,
            fm_anchor_coef=0.01,
            target_kl=target_kl,
            normalize_advantages=False,
        ),
        reference_mdp=reference_nested_mdp,
        engine=engine,
    )

    monitor = TrainingMonitor(monitor_log_dir, enabled=not deepspeed_config or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)
    state = Pi05OpenPIServerState(
        adapter=adapter,
        nested_mdp=nested_mdp,
        reference_nested_mdp=reference_nested_mdp,
        trainer=trainer,
        optimizer=optimizer,
        engine=engine,
        checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir is not None else None,
        checkpoint_interval=checkpoint_interval,
        tensorboard_writer=SummaryWriter(log_dir=str(tensorboard_log_dir)) if tensorboard_log_dir is not None else None,
        gamma=gamma,
    )
    app_monitor = monitor

    app = FastAPI(title="Pi0.5 OpenPI RL Server")
    app.state.pi05 = state

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/sample")
    def sample(req: SampleRequestModel) -> dict[str, Any]:
        observation = _model.Observation.from_dict(_jsonable_observation_to_torch(req.observation, state.nested_mdp.device))
        rollout = state.nested_mdp.sample_inner_mdp(observation, mode=req.mode, return_values=req.return_values)
        _validate_sample_output(
            rollout,
            batch_size=observation.state.shape[0],
            action_horizon=state.nested_mdp.model.config.action_horizon,
            action_dim=state.nested_mdp.model.config.action_dim,
        )
        return SampleResponse(
            action=rollout.actions,
            chains=rollout.chains,
            denoise_logprobs=rollout.denoise_logprobs,
            denoise_means=rollout.denoise_means,
            denoise_stds=rollout.denoise_stds,
            denoise_timesteps=rollout.denoise_timesteps,
            denoise_indices=rollout.denoise_indices,
            velocities=rollout.velocities,
            values=rollout.values,
        ).to_dict()

    @app.post("/update")
    def update(req: RolloutUpdateRequestModel) -> dict[str, Any]:
        coordinator = distributed_coordinator or getattr(app.state, "pi05_coordinator", None)
        if coordinator is not None:
            return coordinator.submit(req)

        if state.trainer is None:
            return {"error": "trainer not configured"}
        if not 0.0 < state.gamma <= 1.0:
            raise ValueError(f"Server gamma must be in (0, 1], got {state.gamma}")
        field_lengths = {
            name: len(getattr(req, name))
            for name in ("chains", "old_logprobs", "denoise_indices", "denoise_timesteps", "rewards", "terminated", "truncated", "values", "next_values", "durations")
        }
        if not req.observations or any(length != len(req.observations) for length in field_lengths.values()):
            raise ValueError(f"SMDP update fields must be non-empty and aligned with observations: {field_lengths}")
        if req.old_velocities is not None and len(req.old_velocities) != len(req.observations):
            raise ValueError("old_velocities must align with observations when provided")
        buffer = Pi05RolloutBuffer(gamma=state.gamma)
        for i in range(len(req.observations)):
            observation = _model.Observation.from_dict(
                _jsonable_observation_to_torch(req.observations[i], state.nested_mdp.device)
            )
            buffer.add(
                observation=observation,
                chains=torch.as_tensor(req.chains[i]),
                old_logprobs=torch.as_tensor(req.old_logprobs[i]),
                denoise_indices=torch.as_tensor(req.denoise_indices[i]),
                denoise_timesteps=torch.as_tensor(req.denoise_timesteps[i]),
                reward=req.rewards[i],
                terminated=req.terminated[i],
                truncated=req.truncated[i],
                value=torch.as_tensor(req.values[i]),
                next_value=torch.as_tensor(req.next_values[i]),
                duration=req.durations[i],
                old_velocities=torch.as_tensor(req.old_velocities[i]) if req.old_velocities is not None else None,
            )
        advantages, _ = buffer.compute_advantages()
        advantage_mean = advantages.mean().item()
        advantage_std = advantages.std(unbiased=False).item()
        batch_metrics = []
        total_minibatches = (len(req.observations) + ppo_minibatch_size - 1) // ppo_minibatch_size
        for minibatch_index, batch in enumerate(
            buffer.iter_minibatches(
                collate_observations=state.adapter.collate_observations,
                minibatch_size=max(1, min(ppo_minibatch_size, len(req.observations))),
                device=state.nested_mdp.device,
            ),
            start=1,
        ):
            minibatch_metrics = state.trainer.update(batch.__dict__)
            batch_metrics.append(minibatch_metrics)
            print(
                {
                    "event": "ppo_minibatch",
                    "minibatch": minibatch_index,
                    "minibatches": total_minibatches,
                    "progress_percent": round(100.0 * minibatch_index / total_minibatches, 2),
                    "returns_mean": batch.returns.float().mean().item(),
                    "advantages_mean": batch.advantages.float().mean().item(),
                    **{name: value.detach().float().item() for name, value in minibatch_metrics.items()},
                },
                flush=True,
            )
        metrics = {k: torch.stack([m[k] for m in batch_metrics]).mean().item() for k in batch_metrics[0]}
        rewards = torch.as_tensor(req.rewards, dtype=torch.float32)
        values = torch.as_tensor(req.values, dtype=torch.float32)
        next_values = torch.as_tensor(req.next_values, dtype=torch.float32)
        durations = torch.as_tensor(req.durations, dtype=torch.float32)
        metrics.update(
            {
                "rollout/advantage_mean_raw": advantage_mean,
                "rollout/advantage_std_raw": advantage_std,
                "rollout/reward_sum": rewards.sum().item(),
                "rollout/reward_mean": rewards.mean().item(),
                "rollout/reward_min": rewards.min().item(),
                "rollout/reward_max": rewards.max().item(),
                "rollout/value_mean": values.mean().item(),
                "rollout/value_std": values.std(unbiased=False).item(),
                "rollout/next_value_mean": next_values.mean().item(),
                "rollout/duration_mean": durations.mean().item(),
                "rollout/terminated_fraction": sum(req.terminated) / len(req.terminated),
                "rollout/truncated_fraction": sum(req.truncated) / len(req.truncated),
                "optimizer/learning_rate": state.optimizer.param_groups[0]["lr"] if state.optimizer is not None else 0.0,
            }
        )
        state.update_count += 1
        metrics["pi05_rl/update_count"] = state.update_count
        if app_monitor is not None:
            app_monitor.log(state.update_count, metrics)
        if state.tensorboard_writer is not None:
            for name, value in metrics.items():
                state.tensorboard_writer.add_scalar(name, value, state.update_count)
            state.tensorboard_writer.flush()
        checkpoint_path = None
        if state.checkpoint_interval > 0 and state.update_count % state.checkpoint_interval == 0:
            checkpoint_path = state.save_checkpoint()
        if checkpoint_path is not None:
            metrics["pi05_rl/checkpoint_path"] = str(checkpoint_path)
        print({"event": "ppo_update", **metrics}, flush=True)
        return RolloutUpdateResponse(metrics=metrics).to_dict()

    return app
