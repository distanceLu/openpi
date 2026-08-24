"""Trajectory-ordered PI0.5 LoRA imitation learning from three-camera HDF5 episodes."""

from __future__ import annotations

import argparse
import dataclasses
from datetime import UTC
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import random
import time

import h5py
import numpy as np
import safetensors.torch
import torch
from torch.utils.data import default_collate
from torch.utils.tensorboard import SummaryWriter

from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.sft_shiji.dataset import batch_to_model
from openpi.sft_shiji.hdf5_dataset import HDF5EpisodeDataset
from openpi.sft_shiji.hdf5_dataset import compute_hdf5_norm_stats
from openpi.sft_shiji.hdf5_dataset import list_trajectory_files
from openpi.sft_shiji.hdf5_dataset import trajectory_length
from openpi.sft_shiji.lora import EXPERIMENTS
from openpi.sft_shiji.lora import configure_finetuning
from openpi.sft_shiji.lora import load_adapter
from openpi.sft_shiji.lora import save_adapter
from openpi.sft_shiji.lora import trainable_parameters
from openpi.shared import normalize as normalize_api

ACTION_NAMES = ("x", "y", "z", "rx", "ry", "rz")


@dataclasses.dataclass
class TrainConfig:
    hdf5_root: Path = Path("/mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/sft_shiji/hdf5_data/2026-08-05-command-delta")
    checkpoint: Path = Path("/mnt/data/lcx1/yiqinworkspace/openpi/asset_pi05_base")
    output_dir: Path = Path("/mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/sft_shiji/output_total")
    run_name: str | None = None
    prompt: str = "follow the demonstrated robot tool trajectory"
    steps: int = 5_000
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    warmup_steps: int = 200
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    save_interval: int = 500
    # 0 validates after each complete pass over all training trajectories.
    validation_interval: int = 0
    validation_batches: int = 0
    validation_inference_batches: int = 0
    validation_diffusion_steps: int = 10
    validation_noise_seed: int = 20260811
    validation_recording: str = "14-40-19"
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    early_stopping_overfit_ratio: float = 1.25
    tensorboard_dir: Path | None = None
    resume_from: Path | None = None
    seed: int = 42
    action_horizon: int = 10
    finetune_mode: str = "a"
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = False
    max_token_len: int = 200
    use_robot_state: bool = False
    mask_non_z_actions: bool = False
    motion_threshold: float = 1e-4
    balance_motion_samples: bool = True
    max_balance_repeat: int = 8
    action_abs_quantile: float = 0.99


def _hdf5_use_robot_state(path: Path) -> bool:
    with h5py.File(path, "r") as file:
        value = file["metadata"].attrs.get("use_robot_state", False)
        return bool(value)


def _motion_counts(paths: list[Path], threshold: float) -> tuple[int, int]:
    moving = 0
    total = 0
    for path in paths:
        with h5py.File(path, "r") as file:
            actions = np.asarray(file["actions/trajectory"][:, :, 2])
        moving += int(np.sum(np.any(np.abs(actions) >= threshold, axis=1)))
        total += actions.shape[0]
    return moving, total - moving


def _action_mask(mask_non_z_actions: bool, device: torch.device) -> torch.Tensor | None:
    if not mask_non_z_actions:
        return None
    mask = torch.zeros(32, dtype=torch.float32, device=device)
    mask[2] = 1.0
    return mask


def _masked_training_loss(
    model: PI0Pytorch,
    observation,
    actions: torch.Tensor,
    action_mask: torch.Tensor | None,
) -> torch.Tensor:
    losses = model(observation, actions.float(), action_mask=action_mask)
    if action_mask is None:
        return losses.mean()
    return losses.sum() / (losses.shape[0] * losses.shape[1] * action_mask.sum())


def _model_file(checkpoint: Path) -> Path:
    path = checkpoint if checkpoint.suffix == ".safetensors" else checkpoint / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"PI0.5 PyTorch checkpoint not found: {path}")
    return path


def _resolve_run_output_dir(config: TrainConfig) -> Path:
    if config.output_dir.name != "output_total":
        return config.output_dir
    run_name = config.run_name or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return config.output_dir / run_name


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move(child, device) for key, child in value.items()}
    return value


def _learning_rate(step: int, config: TrainConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(1, config.warmup_steps)
    progress = min(1.0, (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * cosine


def _optimizer_step(
    model_parameters: list[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    accumulated: int,
    config: TrainConfig,
    step: int,
) -> tuple[float, float]:
    if accumulated <= 0:
        raise ValueError("Cannot perform an optimizer step without accumulated micro batches")
    if accumulated < config.gradient_accumulation_steps:
        correction = config.gradient_accumulation_steps / accumulated
        for parameter in model_parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(correction)
    learning_rate = _learning_rate(step, config)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    grad_norm = torch.nn.utils.clip_grad_norm_(model_parameters, config.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return learning_rate, float(grad_norm)


@torch.no_grad()
def evaluate_trajectories(
    model: PI0Pytorch,
    dataset: HDF5EpisodeDataset,
    paths: list[Path],
    device: torch.device,
    max_steps: int,
    action_mask: torch.Tensor | None,
) -> float:
    model.eval()
    losses = []
    for path in paths:
        for _, _, sample in dataset.iter_trajectory(path):
            batch = _move(default_collate([sample]), device)
            observation, actions = batch_to_model(batch)
            losses.append(float(_masked_training_loss(model, observation, actions, action_mask).item()))
            if max_steps > 0 and len(losses) >= max_steps:
                model.train()
                return sum(losses) / len(losses)
    model.train()
    return sum(losses) / max(1, len(losses))


def _physical_error(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = predicted - target
    epsilon = np.finfo(np.float64).eps
    return {
        "translation_rmse": float(np.sqrt(np.mean(np.square(error[..., :3])))),
        "rotation_rmse_rad": float(np.sqrt(np.mean(np.square(error[..., 3:6])))),
        "translation_wape_percent": float(
            100.0 * np.sum(np.abs(error[..., :3]))
            / max(float(np.sum(np.abs(target[..., :3]))), epsilon)
        ),
        "rotation_wape_percent": float(
            100.0 * np.sum(np.abs(error[..., 3:6]))
            / max(float(np.sum(np.abs(target[..., 3:6]))), epsilon)
        ),
    }


def _z_physical_error(
    predicted: np.ndarray,
    target: np.ndarray,
    motion_threshold: float,
) -> dict[str, float]:
    predicted_z = np.asarray(predicted, dtype=np.float64)[..., 2]
    target_z = np.asarray(target, dtype=np.float64)[..., 2]
    error = predicted_z - target_z
    absolute_error = np.abs(error)
    absolute_target = np.abs(target_z)
    epsilon = np.finfo(np.float64).eps
    rmse = float(np.sqrt(np.mean(np.square(error))))
    zero_rmse = float(np.sqrt(np.mean(np.square(target_z))))
    moving = absolute_target >= motion_threshold
    stationary = ~moving
    result = {
        "z_mae": float(np.mean(absolute_error)),
        "z_rmse": rmse,
        "z_wape_percent": float(
            100.0 * np.sum(absolute_error) / max(float(np.sum(absolute_target)), epsilon)
        ),
        "z_advantage_vs_zero_percent": 100.0 * (1.0 - rmse / max(zero_rmse, epsilon)),
        "moving_fraction": float(np.mean(moving)),
    }
    if np.any(moving):
        moving_error = error[moving]
        result["moving_z_rmse"] = float(np.sqrt(np.mean(np.square(moving_error))))
        result["moving_z_wape_percent"] = float(
            100.0 * np.sum(np.abs(moving_error))
            / max(float(np.sum(absolute_target[moving])), epsilon)
        )
    if np.any(stationary):
        result["stationary_z_mae"] = float(np.mean(absolute_error[stationary]))
        result["stationary_false_motion_rate"] = float(
            np.mean(np.abs(predicted_z[stationary]) >= motion_threshold)
        )
    return result


@torch.no_grad()
def evaluate_inference_metrics(
    model: PI0Pytorch,
    dataset: HDF5EpisodeDataset,
    paths: list[Path],
    device: torch.device,
    max_steps: int,
    diffusion_steps: int,
    noise_seed: int,
    action_mask: torch.Tensor | None,
    motion_threshold: float,
) -> dict[str, float]:
    model.eval()
    predictions = []
    targets = []
    generator = torch.Generator(device=device).manual_seed(noise_seed)
    for path in paths:
        for _, _, sample in dataset.iter_trajectory(path):
            batch = _move(default_collate([sample]), device)
            observation, normalized_actions = batch_to_model(batch)
            noise = torch.randn(normalized_actions.shape, generator=generator, device=device, dtype=normalized_actions.dtype)
            predicted_normalized = model.sample_actions(
                device, observation, noise=noise, num_steps=diffusion_steps, action_mask=action_mask
            )[..., :6]
            predicted_physical = dataset.transform.unnormalizer(
                {"actions": np.clip(predicted_normalized.float().cpu().numpy(), -1.0, 1.0)}
            )["actions"]
            target_physical = dataset.transform.unnormalizer(
                {"actions": normalized_actions[..., :6].float().cpu().numpy()}
            )["actions"]
            if action_mask is not None:
                physical_mask = action_mask[:6].detach().cpu().numpy().astype(bool)
                predicted_physical[..., ~physical_mask] = 0.0
                target_physical[..., ~physical_mask] = 0.0
            predictions.append(predicted_physical)
            targets.append(target_physical)
            if max_steps > 0 and len(predictions) >= max_steps:
                break
        if max_steps > 0 and len(predictions) >= max_steps:
            break
    if not predictions:
        model.train()
        return {}
    predicted = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    model_metrics = _physical_error(predicted, target)
    zero_metrics = _physical_error(np.zeros_like(target), target)
    result = {
        "physical/translation_rmse": model_metrics["translation_rmse"],
        "physical/rotation_rmse_rad": model_metrics["rotation_rmse_rad"],
        "physical/translation_wape_percent": model_metrics["translation_wape_percent"],
        "physical/rotation_wape_percent": model_metrics["rotation_wape_percent"],
        "physical/zero_translation_rmse": zero_metrics["translation_rmse"],
        "physical/zero_rotation_rmse_rad": zero_metrics["rotation_rmse_rad"],
        "physical/translation_improvement_vs_zero_percent": 100.0 * (1.0 - model_metrics["translation_rmse"] / max(zero_metrics["translation_rmse"], 1e-8)),
        "physical/rotation_improvement_vs_zero_percent": 100.0 * (1.0 - model_metrics["rotation_rmse_rad"] / max(zero_metrics["rotation_rmse_rad"], 1e-8)),
    }
    for axis, axis_name in enumerate(ACTION_NAMES):
        result[f"physical/axis_{axis_name}/wape_percent"] = float(
            100.0 * np.sum(np.abs(predicted[..., axis] - target[..., axis]))
            / max(float(np.sum(np.abs(target[..., axis]))), np.finfo(np.float64).eps)
        )
        result[f"physical/axis_{axis_name}/rmse"] = float(
            np.sqrt(np.mean(np.square(predicted[..., axis] - target[..., axis])))
        )
    for horizon in range(target.shape[1]):
        horizon_metrics = _physical_error(predicted[:, horizon], target[:, horizon])
        result[f"physical/horizon_{horizon + 1:02d}/translation_rmse"] = horizon_metrics["translation_rmse"]
        result[f"physical/horizon_{horizon + 1:02d}/rotation_rmse_rad"] = horizon_metrics["rotation_rmse_rad"]
    for name, value in _z_physical_error(predicted, target, motion_threshold).items():
        result[f"physical/{name}"] = value
    model.train()
    return result


def _save_checkpoint(
    model: PI0Pytorch,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    step: int,
    epoch: int,
    episode_id: str,
    train_loss: float,
    validation_loss: float | None,
    best_validation_loss: float,
    stop_reason: str | None,
) -> None:
    checkpoint_dir = output_dir / f"step-{step:08d}"
    save_adapter(model, checkpoint_dir / "adapter_model.safetensors")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    (checkpoint_dir / "trainer_state.json").write_text(json.dumps({
        "step": step,
        "epoch": epoch,
        "completed_episode": episode_id,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "best_validation_loss": best_validation_loss,
        "stop_reason": stop_reason,
        "resume_policy": "resume from the next complete trajectory boundary",
    }, indent=2))
    (output_dir / "latest").write_text(checkpoint_dir.name)


def train(config: TrainConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("PI0.5 SFT requires CUDA")
    if config.steps <= 0 or config.micro_batch_size <= 0 or config.gradient_accumulation_steps <= 0:
        raise ValueError("steps, micro_batch_size, and gradient_accumulation_steps must be positive")
    if config.validation_interval < 0 or config.early_stopping_patience <= 0:
        raise ValueError("validation_interval must be non-negative and patience must be positive")
    if config.validation_batches < 0 or config.validation_inference_batches < 0:
        raise ValueError("validation batch limits must be non-negative")
    if config.validation_diffusion_steps <= 0:
        raise ValueError("validation_diffusion_steps must be positive")
    if config.early_stopping_min_delta < 0 or config.early_stopping_overfit_ratio <= 1.0:
        raise ValueError("early stopping min_delta must be non-negative and overfit_ratio must exceed 1")
    if not np.isfinite(config.motion_threshold) or config.motion_threshold <= 0:
        raise ValueError("motion_threshold must be finite and positive")
    if config.max_balance_repeat <= 0 or not 0.5 < config.action_abs_quantile <= 1.0:
        raise ValueError("max_balance_repeat must be positive and action_abs_quantile must be in (0.5, 1.0]")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = _resolve_run_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    _seed_everything(config.seed)
    all_trajectory_files = list_trajectory_files(config.hdf5_root)
    excluded_inference_files = [path for path in all_trajectory_files if "infer" in path.stem.lower()]
    trajectory_files = [path for path in all_trajectory_files if "infer" not in path.stem.lower()]
    validation_recordings = {
        name.strip() for name in config.validation_recording.split(",") if name.strip()
    }
    validation_files = [path for path in trajectory_files if path.stem in validation_recordings]
    train_files = [path for path in trajectory_files if path.stem not in validation_recordings]
    found_validation_recordings = {path.stem for path in validation_files}
    if not train_files or found_validation_recordings != validation_recordings:
        raise ValueError(
            "Need training HDF5 files and every comma-separated validation trajectory; "
            f"requested={sorted(validation_recordings)}, found={sorted(found_validation_recordings)}"
        )

    action_shapes = {}
    data_use_robot_state = {}
    for path in trajectory_files:
        with h5py.File(path, "r") as file:
            action_shapes[path.stem] = tuple(file["actions/trajectory"].shape[1:])
        data_use_robot_state[path.stem] = _hdf5_use_robot_state(path)
    invalid_shapes = {
        episode: shape for episode, shape in action_shapes.items()
        if shape != (config.action_horizon, 6)
    }
    if invalid_shapes:
        raise ValueError(
            f"HDF5 action shapes must be ({config.action_horizon}, 6); invalid trajectories: {invalid_shapes}"
        )
    state_mismatches = {
        episode: enabled
        for episode, enabled in data_use_robot_state.items()
        if enabled != config.use_robot_state
    }
    if state_mismatches:
        logging.warning(
            "Training use_robot_state=%s while HDF5 metadata is %s. HDF5 use_robot_state=False means "
            "the stored state is a constant zero vector; enabling it still adds constant discrete state tokens.",
            config.use_robot_state,
            state_mismatches,
        )

    lengths = {path.stem: trajectory_length(path) for path in trajectory_files}
    train_timesteps = sum(lengths[path.stem] for path in train_files)
    samples_per_update = config.micro_batch_size * config.gradient_accumulation_steps
    norm_stats = compute_hdf5_norm_stats(
        train_files,
        action_abs_quantile=config.action_abs_quantile,
    )
    moving_steps, stationary_steps = _motion_counts(train_files, config.motion_threshold)
    moving_repeat = 1
    stationary_repeat = 1
    if config.balance_motion_samples and moving_steps > 0 and stationary_steps > 0:
        if moving_steps < stationary_steps:
            moving_repeat = min(
                config.max_balance_repeat,
                max(1, round(stationary_steps / moving_steps)),
            )
        elif stationary_steps < moving_steps:
            stationary_repeat = min(
                config.max_balance_repeat,
                max(1, round(moving_steps / stationary_steps)),
            )
    balanced_train_timesteps = (
        stationary_steps * stationary_repeat + moving_steps * moving_repeat
    )
    optimizer_steps_per_epoch = math.ceil(balanced_train_timesteps / samples_per_update)
    validation_interval = config.validation_interval or optimizer_steps_per_epoch
    normalize_api.save(output_dir / "assets" / "real_robot", norm_stats)
    (output_dir / "train_config.json").write_text(json.dumps({
        field.name: str(value) if isinstance(value, Path) else value
        for field in dataclasses.fields(config)
        if (value := getattr(config, field.name)) is not None
    }, indent=2, ensure_ascii=False))
    (output_dir / "trajectory_manifest.json").write_text(json.dumps({
        "training_trajectories": [path.stem for path in train_files],
        "validation_trajectories": [path.stem for path in validation_files],
        "excluded_inference_trajectories": [path.stem for path in excluded_inference_files],
        "timesteps_per_trajectory": lengths,
        "shuffle_trajectories": True,
        "trajectory_shuffle_seed": config.seed,
        "trajectory_shuffle_policy": "deterministic permutation per epoch using seed + epoch",
        "shuffle_timesteps": False,
        "gradient_accumulation_crosses_trajectory_boundary": False,
        "motion_sampling": {
            "enabled": config.balance_motion_samples,
            "threshold": config.motion_threshold,
            "moving_steps": moving_steps,
            "stationary_steps": stationary_steps,
            "moving_sample_repeat": moving_repeat,
            "stationary_sample_repeat": stationary_repeat,
        },
        "action_normalization": {
            "type": "symmetric_absolute_quantile",
            "absolute_quantile": config.action_abs_quantile,
            "clip_range": [-1.0, 1.0],
            "scale": np.asarray(norm_stats["actions"].q99).tolist(),
        },
    }, indent=2, ensure_ascii=False))
    logging.info(
        "Sequential schedule: %d training trajectories, %d timesteps, %d optimizer steps/epoch; validation=%s",
        len(train_files), train_timesteps, optimizer_steps_per_epoch,
        [path.stem for path in validation_files],
    )

    device = torch.device("cuda:0")
    action_mask = _action_mask(config.mask_non_z_actions, device)
    logging.info(
        "Training action axes=%s; disabled axes are forced to zero",
        [name for index, name in enumerate(ACTION_NAMES) if action_mask is None or action_mask[index] > 0],
    )
    model_config = Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
        discrete_state_input=config.use_robot_state,
        dtype="bfloat16",
        pytorch_compile_mode=None,
    )
    logging.info("Constructing PI0.5 and loading %s", _model_file(config.checkpoint))
    model = PI0Pytorch(model_config)
    safetensors.torch.load_model(model, _model_file(config.checkpoint), device="cpu", strict=False)
    if config.finetune_mode not in EXPERIMENTS:
        raise ValueError(f"Unknown finetune mode {config.finetune_mode!r}; choose from {sorted(EXPERIMENTS)}")
    finetune_spec, replaced = configure_finetuning(
        model,
        config.finetune_mode,
        dropout=config.lora_dropout,
    )
    model.to(device)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    trainable = trainable_parameters(model)
    logging.info(
        "Fine-tune mode=%s spec=%s LoRA_modules=%d trainable=%s / total=%s",
        config.finetune_mode,
        dataclasses.asdict(finetune_spec),
        len(replaced),
        f"{sum(p.numel() for p in trainable):,}",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    resume_step = 0
    resume_epoch = 0
    resume_episode: str | None = None
    resumed_validation_loss: float | None = None
    resumed_best_validation_loss = float("inf")
    if config.resume_from is not None:
        resume_dir = config.resume_from
        adapter_path = resume_dir / "adapter_model.safetensors"
        state_path = resume_dir / "trainer_state.json"
        if not adapter_path.is_file() or not state_path.is_file():
            raise FileNotFoundError(f"Incomplete resume checkpoint: {resume_dir}")
        load_adapter(model, adapter_path)
        state = json.loads(state_path.read_text())
        resume_step = int(state["step"])
        resume_epoch = int(state["epoch"])
        resume_episode = str(state["completed_episode"])
        resumed_validation_loss = state.get("validation_loss")
        resumed_best_validation_loss = float(state.get("best_validation_loss", float("inf")))
        optimizer_path = resume_dir / "optimizer.pt"
        if optimizer_path.is_file():
            optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu", weights_only=True))
        else:
            logging.warning("No optimizer.pt in %s; adapter and schedule step resume, optimizer moments restart", resume_dir)
        logging.info(
            "Resuming from %s: step=%d epoch=%d completed_episode=%s",
            resume_dir, resume_step, resume_epoch, resume_episode,
        )
    optimizer.zero_grad(set_to_none=True)
    train_dataset = HDF5EpisodeDataset(
        config.prompt, norm_stats, config.max_token_len, use_robot_state=config.use_robot_state
    )
    validation_dataset = HDF5EpisodeDataset(
        config.prompt, norm_stats, config.max_token_len, use_robot_state=config.use_robot_state
    )
    tensorboard_dir = config.tensorboard_dir or output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    writer.add_text("config/train", (output_dir / "train_config.json").read_text(), 0)
    logging.info("TensorBoard logs: %s", tensorboard_dir)

    step = resume_step
    epoch = resume_epoch
    best_validation_loss = resumed_best_validation_loss
    latest_validation_loss = resumed_validation_loss
    no_improvement_evaluations = 0
    best_motion_z_rmse = float("inf")
    next_validation_step = (step // validation_interval + 1) * validation_interval
    resume_pending = resume_episode is not None
    stop_reason: str | None = None
    started = time.perf_counter()
    last_saved_step = -1

    def epoch_trajectories(epoch_number: int) -> list[Path]:
        ordered = list(train_files)
        random.Random(config.seed + epoch_number).shuffle(ordered)
        return ordered

    try:
        while step < config.steps and stop_reason is None:
            if not resume_pending:
                epoch += 1
            current_train_files = epoch_trajectories(epoch)
            logging.info("epoch=%d shuffled trajectory order=%s", epoch, [path.stem for path in current_train_files])
            for episode_index, path in enumerate(current_train_files):
                if resume_pending:
                    if path.stem != resume_episode:
                        continue
                    resume_pending = False
                    continue
                episode_loss = 0.0
                episode_timesteps = 0
                accumulated = 0
                accumulated_loss = 0.0
                last_update_loss = 0.0
                logging.info(
                    "epoch=%d trajectory=%d/%d episode=%s timesteps=%d",
                    epoch, episode_index + 1, len(current_train_files), path.stem, lengths[path.stem],
                )
                pending_samples = []
                pending_timesteps = []
                for _, timestep, sample, is_moving in train_dataset.iter_trajectory_with_motion(
                    path, config.motion_threshold
                ):
                    repeats = moving_repeat if is_moving else stationary_repeat
                    pending_samples.extend([sample] * repeats)
                    pending_timesteps.extend([timestep] * repeats)
                    while len(pending_samples) >= config.micro_batch_size:
                        current_samples = pending_samples[: config.micro_batch_size]
                        current_timesteps = pending_timesteps[: config.micro_batch_size]
                        del pending_samples[: config.micro_batch_size]
                        del pending_timesteps[: config.micro_batch_size]
                        batch = _move(default_collate(current_samples), device)
                        observation, actions = batch_to_model(batch)
                        loss = _masked_training_loss(model, observation, actions, action_mask)
                        (loss / config.gradient_accumulation_steps).backward()
                        value = float(loss.detach().item())
                        batch_timesteps = len(current_samples)
                        episode_loss += value * batch_timesteps
                        accumulated_loss += value
                        episode_timesteps += batch_timesteps
                        accumulated += 1
                        writer.add_scalar("trajectory/timestep", current_timesteps[-1], step)
                        if accumulated == config.gradient_accumulation_steps:
                            learning_rate, grad_norm = _optimizer_step(trainable, optimizer, accumulated, config, step)
                            step += 1
                            last_update_loss = accumulated_loss / accumulated
                            accumulated = 0
                            accumulated_loss = 0.0
                            writer.add_scalar("loss/train", last_update_loss, step)
                            writer.add_scalar("optimization/learning_rate", learning_rate, step)
                            writer.add_scalar("optimization/gradient_norm", grad_norm, step)
                if pending_samples:
                    batch = _move(default_collate(pending_samples), device)
                    observation, actions = batch_to_model(batch)
                    loss = _masked_training_loss(model, observation, actions, action_mask)
                    (loss / config.gradient_accumulation_steps).backward()
                    value = float(loss.detach().item())
                    episode_loss += value * len(pending_samples)
                    accumulated_loss += value
                    episode_timesteps += len(pending_samples)
                    accumulated += 1
                if accumulated:
                    learning_rate, grad_norm = _optimizer_step(trainable, optimizer, accumulated, config, step)
                    step += 1
                    last_update_loss = accumulated_loss / accumulated
                    writer.add_scalar("loss/train", last_update_loss, step)
                    writer.add_scalar("optimization/learning_rate", learning_rate, step)
                    writer.add_scalar("optimization/gradient_norm", grad_norm, step)

                mean_episode_loss = episode_loss / max(1, episode_timesteps)
                writer.add_scalar("trajectory/train_mean_loss", mean_episode_loss, step)
                writer.add_scalar("trajectory/length", episode_timesteps, step)
                writer.add_scalar("system/peak_vram_gib", torch.cuda.max_memory_allocated(device) / 2**30, step)
                writer.flush()
                logging.info(
                    "completed episode=%s in order: timesteps=%d step=%d/%d mean_loss=%.6f elapsed=%.0fs",
                    path.stem, episode_timesteps, step, config.steps, mean_episode_loss, time.perf_counter() - started,
                )

                validation_due = step >= next_validation_step or step >= config.steps
                if validation_due:
                    latest_validation_loss = evaluate_trajectories(
                        model, validation_dataset, validation_files, device,
                        config.validation_batches, action_mask,
                    )
                    improved = latest_validation_loss < best_validation_loss - config.early_stopping_min_delta
                    if improved:
                        best_validation_loss = latest_validation_loss
                        no_improvement_evaluations = 0
                        save_adapter(model, output_dir / "best" / "adapter_model.safetensors")
                    else:
                        no_improvement_evaluations += 1
                    ratio = latest_validation_loss / max(mean_episode_loss, 1e-12)
                    inference_metrics = evaluate_inference_metrics(
                        model,
                        validation_dataset,
                        validation_files,
                        device,
                        config.validation_inference_batches,
                        config.validation_diffusion_steps,
                        config.validation_noise_seed,
                        action_mask,
                        config.motion_threshold,
                    )
                    motion_z_rmse = inference_metrics.get("physical/moving_z_rmse")
                    if motion_z_rmse is not None and motion_z_rmse < best_motion_z_rmse:
                        best_motion_z_rmse = motion_z_rmse
                        save_adapter(model, output_dir / "best_physical" / "adapter_model.safetensors")
                    writer.add_scalar("loss/validation", latest_validation_loss, step)
                    writer.add_scalar("loss/validation_train_ratio", ratio, step)
                    for metric_name, metric_value in inference_metrics.items():
                        writer.add_scalar(metric_name, metric_value, step)
                    logging.info(
                        "boundary validation: step=%d validation_loss=%.6f best=%.6f no_improvement=%d/%d",
                        step, latest_validation_loss, best_validation_loss,
                        no_improvement_evaluations, config.early_stopping_patience,
                    )
                    while next_validation_step <= step:
                        next_validation_step += validation_interval
                    if no_improvement_evaluations >= config.early_stopping_patience:
                        stop_reason = "validation loss stopped improving at complete trajectory boundaries"
                    elif ratio >= config.early_stopping_overfit_ratio and no_improvement_evaluations >= 2:
                        stop_reason = f"overfitting at trajectory boundary: validation/train ratio={ratio:.3f}"

                save_due = step - last_saved_step >= config.save_interval or validation_due or stop_reason is not None
                if save_due:
                    _save_checkpoint(
                        model, optimizer, output_dir, step, epoch, path.stem, last_update_loss,
                        latest_validation_loss, best_validation_loss, stop_reason,
                    )
                    last_saved_step = step
                if step >= config.steps or stop_reason is not None:
                    break
    finally:
        if stop_reason is not None:
            logging.warning("Early stopping at complete trajectory boundary: %s", stop_reason)
        writer.flush()
        writer.close()


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="PI0.5 ordered three-camera HDF5 trajectory fine-tuning")
    optional_types = {"tensorboard_dir": Path, "resume_from": Path, "run_name": str}
    for field in dataclasses.fields(TrainConfig):
        default = field.default
        flag = "--" + field.name.replace("_", "-")
        if field.name == "finetune_mode":
            parser.add_argument(flag, choices=sorted(EXPERIMENTS), default=default)
        elif isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(flag, type=optional_types.get(field.name, type(default)), default=default)
    return TrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(parse_args())
