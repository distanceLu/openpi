"""Single-RTX-3090 PI0.5 LoRA imitation-learning entry point."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
from pathlib import Path
import random
import time

import numpy as np
import safetensors.torch
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.sft_shiji.dataset import AlignmentConfig
from openpi.sft_shiji.dataset import RealRobotDataset
from openpi.sft_shiji.dataset import batch_to_model
from openpi.sft_shiji.dataset import build_aligned_samples
from openpi.sft_shiji.dataset import compute_norm_stats
from openpi.sft_shiji.dataset import save_alignment_report
from openpi.sft_shiji.dataset import split_by_recording
from openpi.sft_shiji.lora import inject_lora
from openpi.sft_shiji.lora import save_adapter
from openpi.sft_shiji.lora import trainable_parameters
from openpi.shared import normalize as normalize_api


@dataclasses.dataclass
class TrainConfig:
    data_root: Path = Path("/home/shugen/yanjie/ros2_ws/data_collect/2026-08-03")
    checkpoint: Path = Path("/data/yiqinworkspace/openpi/pi05_base_checkpoint/Checkpoint_pi05")
    output_dir: Path = Path("/data/yiqinworkspace/openpi/src/openpi/sft_shiji/output")
    prompt: str = "follow the demonstrated robot tool trajectory"
    steps: int = 5_000
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_workers: int = 2
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    warmup_steps: int = 200
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    save_interval: int = 500
    # 0 evaluates once after every complete pass over all training recording folders.
    validation_interval: int = 0
    validation_batches: int = 0
    validation_recording: str = "17-01-09"
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    early_stopping_overfit_ratio: float = 1.25
    tensorboard_dir: Path | None = None
    seed: int = 42
    action_horizon: int = 10
    control_hz: float = 10.0
    sample_hz: float = 10.0
    timestamp_ticks_per_second: int = 1_000_000
    camera_tolerance_ms: float = 100.0
    state_tolerance_ms: float = 30.0
    action_mode: str = "relative_pose"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    adapt_backbone: bool = True
    adapt_expert: bool = True
    adapt_projector: bool = True
    adapt_vision: bool = False
    gradient_checkpointing: bool = True
    max_token_len: int = 200


def _model_file(checkpoint: Path) -> Path:
    path = checkpoint if checkpoint.suffix == ".safetensors" else checkpoint / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"PI0.5 PyTorch checkpoint not found: {path}")
    return path


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


def _make_loader(dataset, config: TrainConfig, *, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed + int(not shuffle))
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        drop_last=shuffle,
        generator=generator,
    )


@torch.no_grad()
def evaluate(model: PI0Pytorch, loader: DataLoader, device: torch.device, max_batches: int) -> float:
    """Evaluate the fixed recording; max_batches=0 means its complete aligned sample set."""
    model.eval()
    losses = []
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        device_batch = _move(batch, device)
        observation, actions = batch_to_model(device_batch)
        losses.append(float(model(observation, actions.float()).mean().item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def train(config: TrainConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("PI0.5 SFT requires CUDA; this profile targets a 24 GiB RTX 3090")
    if config.batch_size != 1:
        logging.warning("Three 224x224 views make batch_size > 1 likely to OOM on a 24 GiB RTX 3090")
    if config.action_mode not in {"absolute_pose", "relative_pose"}:
        raise ValueError(f"Unknown action_mode: {config.action_mode}")
    if config.steps <= 0 or config.gradient_accumulation_steps <= 0:
        raise ValueError("steps and gradient_accumulation_steps must be positive")
    if config.validation_interval < 0 or config.early_stopping_patience <= 0:
        raise ValueError("validation_interval must be non-negative and early_stopping_patience must be positive")
    if config.early_stopping_min_delta < 0 or config.early_stopping_overfit_ratio <= 1.0:
        raise ValueError("early stopping min_delta must be non-negative and overfit_ratio must exceed 1")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    _seed_everything(config.seed)

    alignment = AlignmentConfig(
        action_horizon=config.action_horizon,
        control_hz=config.control_hz,
        sample_hz=config.sample_hz,
        timestamp_ticks_per_second=config.timestamp_ticks_per_second,
        camera_tolerance_ms=config.camera_tolerance_ms,
        state_tolerance_ms=config.state_tolerance_ms,
        action_mode=config.action_mode,
    )
    logging.info("Aligning demonstrations under %s", config.data_root)
    samples = build_aligned_samples(config.data_root, alignment)
    train_samples, validation_samples = split_by_recording(samples, config.validation_recording)
    norm_stats = compute_norm_stats(train_samples)
    normalize_api.save(config.output_dir / "assets" / "real_robot", norm_stats)
    save_alignment_report(
        config.output_dir / "alignment_report.json", samples, train_samples, validation_samples, alignment
    )
    (config.output_dir / "train_config.json").write_text(
        json.dumps(
            {
                field.name: str(value) if isinstance(value, Path) else value
                for field in dataclasses.fields(config)
                if (value := getattr(config, field.name)) is not None
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    logging.info(
        "Aligned %d steps from %d complete episode folders: train=%d steps from %d folders; "
        "validation=%d steps from fixed episode %s",
        len(samples), len({sample.episode_id for sample in samples}), len(train_samples),
        len({sample.episode_id for sample in train_samples}), len(validation_samples), config.validation_recording,
    )

    train_dataset = RealRobotDataset(
        train_samples, config.prompt, norm_stats, max_token_len=config.max_token_len
    )
    validation_dataset = RealRobotDataset(
        validation_samples, config.prompt, norm_stats, max_token_len=config.max_token_len
    )
    train_loader = _make_loader(train_dataset, config, shuffle=True)
    validation_loader = _make_loader(validation_dataset, config, shuffle=False)
    micro_batches_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = math.ceil(micro_batches_per_epoch / config.gradient_accumulation_steps)
    validation_interval = config.validation_interval or optimizer_steps_per_epoch
    planned_epochs = config.steps / optimizer_steps_per_epoch
    logging.info(
        "Schedule: %d aligned training steps from complete episode folders, %d micro-batches/epoch, %d optimizer steps/epoch, "
        "validation every %d optimizer steps (one complete epoch), %.2f planned epochs",
        len(train_dataset), micro_batches_per_epoch, optimizer_steps_per_epoch, validation_interval, planned_epochs,
    )

    device = torch.device("cuda:0")
    model_config = Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
        discrete_state_input=True,
        dtype="bfloat16",
        pytorch_compile_mode=None,
    )
    logging.info("Constructing PI0.5 and loading %s", _model_file(config.checkpoint))
    model = PI0Pytorch(model_config)
    safetensors.torch.load_model(model, _model_file(config.checkpoint), device="cpu", strict=False)
    replaced = inject_lora(
        model,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
        adapt_backbone=config.adapt_backbone,
        adapt_expert=config.adapt_expert,
        adapt_vision=config.adapt_vision,
        adapt_projector=config.adapt_projector,
    )
    model.to(device)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()

    trainable = trainable_parameters(model)
    trainable_count = sum(parameter.numel() for parameter in trainable)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    logging.info(
        "Injected %d LoRA modules; trainable=%s / total=%s (%.3f%%)",
        len(replaced), f"{trainable_count:,}", f"{total_count:,}", 100 * trainable_count / total_count,
    )
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    optimizer.zero_grad(set_to_none=True)
    tensorboard_dir = config.tensorboard_dir or config.output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    writer.add_text("config/train", json.dumps({
        field.name: str(value) if isinstance(value, Path) else value
        for field in dataclasses.fields(config)
        if (value := getattr(config, field.name)) is not None
    }, indent=2, ensure_ascii=False), 0)
    logging.info("TensorBoard logs: %s", tensorboard_dir)

    step = 0
    micro_steps_in_update = 0
    running_loss = 0.0
    best_validation_loss = float("inf")
    no_improvement_evaluations = 0
    latest_validation_loss: float | None = None
    stop_reason: str | None = None
    started = time.perf_counter()
    try:
        while step < config.steps and stop_reason is None:
            for batch in train_loader:
                device_batch = _move(batch, device)
                observation, actions = batch_to_model(device_batch)
                loss = model(observation, actions.float()).mean()
                (loss / config.gradient_accumulation_steps).backward()
                running_loss += float(loss.detach().item())
                micro_steps_in_update += 1
                if micro_steps_in_update < config.gradient_accumulation_steps:
                    continue

                learning_rate = _learning_rate(step, config)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                mean_loss = running_loss / micro_steps_in_update
                running_loss = 0.0
                micro_steps_in_update = 0
                elapsed = time.perf_counter() - started
                memory = torch.cuda.max_memory_allocated(device) / 2**30
                writer.add_scalar("loss/train", mean_loss, step)
                writer.add_scalar("optimization/learning_rate", learning_rate, step)
                writer.add_scalar("optimization/gradient_norm", float(grad_norm), step)
                writer.add_scalar("system/peak_vram_gib", memory, step)
                logging.info(
                    "step=%d/%d train_loss=%.6f lr=%.2e grad=%.3f elapsed=%.0fs peak_vram=%.2fGiB",
                    step, config.steps, mean_loss, learning_rate, float(grad_norm), elapsed, memory,
                )

                if step % validation_interval == 0 or step == config.steps:
                    validation_loss = evaluate(model, validation_loader, device, config.validation_batches)
                    latest_validation_loss = validation_loss
                    loss_gap = validation_loss - mean_loss
                    loss_ratio = validation_loss / max(mean_loss, 1e-12)
                    improved = validation_loss < best_validation_loss - config.early_stopping_min_delta
                    if improved:
                        best_validation_loss = validation_loss
                        no_improvement_evaluations = 0
                        save_adapter(model, config.output_dir / "best" / "adapter_model.safetensors")
                        (config.output_dir / "best" / "trainer_state.json").write_text(json.dumps({
                            "step": step,
                            "train_loss": mean_loss,
                            "validation_loss": validation_loss,
                        }, indent=2))
                    else:
                        no_improvement_evaluations += 1

                    writer.add_scalar("loss/validation", validation_loss, step)
                    writer.add_scalar("loss/validation_train_gap", loss_gap, step)
                    writer.add_scalar("loss/validation_train_ratio", loss_ratio, step)
                    writer.add_scalar("early_stopping/no_improvement_evaluations", no_improvement_evaluations, step)
                    writer.flush()
                    logging.info(
                        "step=%d validation_loss=%.6f train_loss=%.6f gap=%+.6f ratio=%.3f "
                        "best=%.6f no_improvement=%d/%d",
                        step, validation_loss, mean_loss, loss_gap, loss_ratio, best_validation_loss,
                        no_improvement_evaluations, config.early_stopping_patience,
                    )
                    if no_improvement_evaluations >= config.early_stopping_patience:
                        stop_reason = (
                            f"validation loss did not improve by {config.early_stopping_min_delta:g} for "
                            f"{config.early_stopping_patience} evaluations"
                        )
                    elif loss_ratio >= config.early_stopping_overfit_ratio and no_improvement_evaluations >= 2:
                        stop_reason = (
                            f"overfitting: validation/train loss ratio {loss_ratio:.3f} reached "
                            f"{config.early_stopping_overfit_ratio:.3f}"
                        )

                if (
                    step % config.save_interval == 0
                    or step % validation_interval == 0
                    or step == config.steps
                    or stop_reason is not None
                ):
                    checkpoint_dir = config.output_dir / f"step-{step:08d}"
                    save_adapter(model, checkpoint_dir / "adapter_model.safetensors")
                    (checkpoint_dir / "trainer_state.json").write_text(json.dumps({
                        "step": step,
                        "train_loss": mean_loss,
                        "validation_loss": latest_validation_loss,
                        "best_validation_loss": best_validation_loss,
                        "stop_reason": stop_reason,
                    }, indent=2))
                    (config.output_dir / "latest").write_text(checkpoint_dir.name)
                if step >= config.steps or stop_reason is not None:
                    break
    finally:
        if stop_reason is not None:
            logging.warning("Early stopping at step %d: %s", step, stop_reason)
            (config.output_dir / "early_stop.json").write_text(json.dumps({
                "step": step,
                "reason": stop_reason,
                "best_validation_loss": best_validation_loss,
            }, indent=2))
        writer.flush()
        writer.close()


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="PI0.5 three-camera real-robot LoRA SFT")
    optional_types = {"tensorboard_dir": Path}
    for field in dataclasses.fields(TrainConfig):
        default = field.default
        flag = "--" + field.name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(flag, type=optional_types.get(field.name, type(default)), default=default)
    return TrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(parse_args())
