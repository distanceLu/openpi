"""LoRA supervised fine-tuning for PI0.5 on a local LIBERO LeRobot dataset.

The PI0.5 training objective is conditional flow matching.  For each action chunk
``a`` the model samples noise ``eps`` and time ``t``, constructs
``x_t = t * eps + (1 - t) * a``, predicts the velocity, and minimizes MSE against
``eps - a``.  Only LoRA matrices are optimized; base checkpoint weights remain
frozen.
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import timedelta
import json
import logging
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import deepspeed
import numpy as np
import safetensors.torch
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.sft.libero_hdf5_dataset import LiberoHDF5Dataset
from openpi.training import config as training_config
from openpi.training import data_loader

_DEFAULT_ROOT = Path("/mnt/data/lcx1/yiqinworkspace/openpi")


@dataclasses.dataclass
class SFTConfig:
    initial_checkpoint: Path = _DEFAULT_ROOT / "asset_pi05_base/pytorch"
    dataset_dir: Path = _DEFAULT_ROOT / "src/openpi/sft/dataset_sft"
    output_dir: Path = _DEFAULT_ROOT / "src/openpi/sft/SFT-PI05-LIBERO-VAL"
    norm_stats_dir: Path = _DEFAULT_ROOT / "asset_pi05/pytorch/assets"
    norm_stats_asset_id: str = "physical-intelligence/libero"
    steps: int = 30_000
    batch_size: int = 16
    num_workers: int = 4
    validation_fraction: float = 0.1
    validation_batches: int = 100
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.005
    early_stopping_ema_alpha: float = 0.3
    hard_stop_ratio: float = 1.5
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    save_interval: int = 500
    log_interval: int = 1
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    seed: int = 42
    resume: bool = False
    gradient_checkpointing: bool = True
    default_prompt: str | None = None
    tensorboard: bool = True
    tensorboard_dir: Path | None = None
    tensorboard_run_name: str = "formal"
    log_dir: Path | None = None
    moving_average_window: int = 20
    monitor_interval: int = 1
    gpu_util_interval: int = 10
    deepspeed_config: Path | None = None
    local_rank: int = -1


@dataclasses.dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


class LoRALinear(nn.Module):
    """Frozen linear layer with a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.rank = rank
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    @property
    def weight(self) -> nn.Parameter:
        return self.base.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        residual = torch.nn.functional.linear(self.dropout(inputs), self.lora_a)
        residual = torch.nn.functional.linear(residual, self.lora_b)
        return base_output + residual * self.scaling


def inject_lora(model: nn.Module, config: SFTConfig) -> list[str]:
    """Inject adapters into language/action-expert attention and MLP projections."""
    targets = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    for parameter in model.parameters():
        parameter.requires_grad = False

    replaced: list[str] = []
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(module, child_name, LoRALinear(child, config.lora_rank, config.lora_alpha, config.lora_dropout))
                replaced.append(full_name)
    if not replaced:
        raise RuntimeError("No target Linear layers found; model projection names have changed")
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    unexpected = [name for name in trainable_names if ".lora_a" not in name and ".lora_b" not in name]
    if unexpected:
        raise RuntimeError(f"Non-LoRA parameters remained trainable after injection: {unexpected[:10]}")
    expected_trainable = 2 * len(replaced)
    if len(trainable_names) != expected_trainable:
        raise RuntimeError(
            f"LoRA injection produced {len(trainable_names)} trainable tensors; expected {expected_trainable} "
            f"for {len(replaced)} wrapped Linear layers"
        )
    return replaced


def _find_model_file(checkpoint: Path) -> Path:
    direct = checkpoint / "model.safetensors"
    candidates = [direct, *checkpoint.glob("**/model.safetensors")]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No model.safetensors under {checkpoint}. The supplied Hugging Face directory appears incomplete; "
        "download its LFS weight file before training."
    )


def setup_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    enabled = world_size > 1
    if enabled:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL DDP requires CUDA")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        # NCCL 2.26 enables its RAS listener by default. On shared nodes, concurrent
        # jobs can contend for its fixed local address even though torchrun selected
        # a unique rendezvous port. RAS is diagnostic-only and is not needed by DDP.
        os.environ.setdefault("NCCL_RAS_ENABLE", "0")
        if int(os.environ.get("RANK", "0")) == 0:
            logging.warning(
                "Initializing NCCL: rank=%s local_rank=%d world_size=%d master=%s:%s device=%s",
                os.environ.get("RANK"), local_rank, world_size,
                os.environ.get("MASTER_ADDR"), os.environ.get("MASTER_PORT"), device,
            )
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=3),
            device_id=device,
        )
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return DistributedContext(enabled, rank, local_rank, world_size, device)


def build_loader(config: SFTConfig, context: DistributedContext):
    if not config.dataset_dir.is_dir():
        raise FileNotFoundError(f"Local HDF5 dataset directory does not exist: {config.dataset_dir}")
    model_config = Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False, dtype="bfloat16")
    data_factory = training_config.LeRobotLiberoDataConfig(
        repo_id="libero_hdf5",
        assets=training_config.AssetsConfig(
            assets_dir=str(config.norm_stats_dir),
            asset_id=config.norm_stats_asset_id,
        ),
        base_config=training_config.DataConfig(prompt_from_task=False),
        extra_delta_transform=False,
    )
    data_config = data_factory.create(config.initial_checkpoint, model_config)
    if data_config.norm_stats is None:
        expected = config.norm_stats_dir / config.norm_stats_asset_id / "norm_stats.json"
        raise FileNotFoundError(f"PI0.5 LIBERO normalization statistics not found at {expected}")

    train_dataset = LiberoHDF5Dataset(
        config.dataset_dir,
        action_horizon=model_config.action_horizon,
        default_prompt=config.default_prompt,
        split="train",
        validation_fraction=config.validation_fraction,
        split_seed=config.seed,
    )
    val_dataset = LiberoHDF5Dataset(
        config.dataset_dir,
        action_horizon=model_config.action_horizon,
        default_prompt=config.default_prompt,
        split="validation",
        validation_fraction=config.validation_fraction,
        split_seed=config.seed,
    )
    logging.info(
        "Detected task IDs %s; per-task train episodes=%s; validation episodes=%s; frames train=%d validation=%d",
        list(train_dataset.task_ids),
        train_dataset.task_episode_counts,
        val_dataset.task_episode_counts,
        len(train_dataset),
        len(val_dataset),
    )
    if train_dataset.task_ids != val_dataset.task_ids:
        raise ValueError(
            f"Task mismatch between splits: train={train_dataset.task_ids}, validation={val_dataset.task_ids}"
        )
    if len(train_dataset) < config.batch_size or len(val_dataset) < config.batch_size:
        raise ValueError(
            f"Demonstration-level split is too small for batch_size={config.batch_size}: "
            f"train_frames={len(train_dataset)}, validation_frames={len(val_dataset)}"
        )
    train_dataset = data_loader.transform_dataset(train_dataset, data_config)
    val_dataset = data_loader.transform_dataset(val_dataset, data_config)
    sampler = (
        torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=config.seed,
            drop_last=True,
        )
        if context.enabled
        else None
    )
    torch_loader = data_loader.TorchDataLoader(
        train_dataset,
        local_batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
        seed=config.seed + context.rank,
        framework="pytorch",
    )
    val_torch_loader = data_loader.TorchDataLoader(
        val_dataset,
        local_batch_size=config.batch_size,
        shuffle=False,
        num_batches=config.validation_batches,
        num_workers=config.num_workers,
        seed=config.seed + context.rank + 1,
        framework="pytorch",
    )
    return (
        data_loader.DataLoaderImpl(data_config, torch_loader),
        data_loader.DataLoaderImpl(data_config, val_torch_loader),
        model_config,
    )


@torch.no_grad()
def evaluate_loss(model, loader, device: torch.device, context: DistributedContext, max_batches: int) -> float:
    was_training = model.training
    model.eval()
    total = torch.zeros(2, dtype=torch.float64, device=device)
    for batch_index, (observation, actions) in enumerate(loader):
        if batch_index >= max_batches:
            break
        observation = move_to_device(observation, device)
        actions = actions.to(device=device, dtype=torch.float32)
        loss = model(observation, actions).mean()
        total[0] += loss.detach().float().item()
        total[1] += 1
    if context.enabled:
        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
    if was_training:
        model.train()
    return (total[0] / total[1].clamp_min(1)).item()


def _format_duration(seconds: float) -> str:
    """Format a duration compactly for terminal progress output."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def log_metrics(step: int, metrics: dict[str, Any], pbar: tqdm | None) -> str:
    """Print exactly one permanent, ordered metric line for a completed optimizer step."""
    total_steps = int(metrics["total_steps"])
    elapsed = float(metrics["elapsed_seconds"])
    completed_this_run = max(1, step - int(metrics.get("started_step", 0)))
    steps_per_second = completed_this_run / max(elapsed, 1e-9)
    eta = (total_steps - step) / max(steps_per_second, 1e-9)
    progress = min(1.0, step / total_steps)
    bar_width = 20
    filled = min(bar_width, int(progress * bar_width))
    bar = "=" * filled + ">" + "." * max(0, bar_width - filled - 1) if filled < bar_width else "=" * bar_width
    metrics["steps_per_second"] = steps_per_second
    metrics["eta_seconds"] = eta
    row = (
        f"[{bar}] {progress:6.2%} | "
        f"step {step:>{len(str(total_steps))}}/{total_steps} | "
        f"epoch {float(metrics['epoch']):6.2f}/{float(metrics['total_epochs']):.2f} | "
        f"loss {float(metrics['loss']):.6f} | "
        f"lr {float(metrics['learning_rate']):.2e} | "
        f"grad {float(metrics['grad_norm']):.3f} | "
        f"{float(metrics['samples_per_second']):.2f} samples/s | "
        f"{steps_per_second:.2f} steps/s | elapsed {_format_duration(elapsed)} | ETA {_format_duration(eta)}"
    )
    if pbar is not None:
        pbar.update(max(0, step - pbar.n))
    print(row, flush=True)
    return row


class TrainingMonitor:
    """Rank-zero progress, JSONL/file logging, TensorBoard, and low-overhead GPU monitoring."""

    def __init__(self, config: SFTConfig, context: DistributedContext, start_step: int):
        self.config = config
        self.context = context
        self.started = time.perf_counter()
        self.losses: list[float] = []
        self.epoch_loss_sum = 0.0
        self.epoch_updates = 0
        self.last_epoch = 0
        self.log_dir = config.log_dir or config.output_dir / "logs"
        self.writer = None
        self.log_file = None
        self.metrics_file = None
        self.progress = None
        self.best_val_loss = float("inf")
        self.no_improve_count = 0
        self.ema_val_loss = None
        self.best_step = 0
        if not context.is_main:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = (self.log_dir / "train.log").open("a", buffering=1)
        self.metrics_file = (self.log_dir / "metrics.json").open("a", buffering=1)
        tensorboard_root = config.tensorboard_dir or self.log_dir
        self.writer = SummaryWriter(
            log_dir=str(tensorboard_root / config.tensorboard_run_name),
            purge_step=start_step if config.resume else None,
        ) if config.tensorboard else None
        # Disable tqdm's carriage-return display. Each optimizer step is printed
        # as one permanent line by log_metrics, which remains readable in redirected
        # logs, tmux, and multi-process launchers without overwriting prior steps.
        self.progress = tqdm(
            total=config.steps,
            initial=start_step,
            desc="PI0.5 SFT",
            unit="step",
            disable=True,
        )
        self.started_step = start_step

    @staticmethod
    def _gpu_metrics(device: torch.device, include_utilization: bool) -> dict[str, float]:
        if device.type != "cuda":
            return {"gpu_memory_allocated_gib": 0.0, "gpu_memory_reserved_gib": 0.0, "gpu_utilization_pct": 0.0}
        metrics = {
            "gpu_memory_allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
            "gpu_memory_reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
            "gpu_memory_peak_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        }
        if include_utilization:
            try:
                metrics["gpu_utilization_pct"] = float(torch.cuda.utilization(device))
            except (AttributeError, RuntimeError):
                metrics["gpu_utilization_pct"] = -1.0
        return metrics

    def update(self, metrics: dict[str, Any]) -> None:
        if not self.context.is_main:
            return
        step = int(metrics["step"])
        loss = float(metrics["loss"])
        epoch_number = int(metrics.get("epoch_number", metrics["epoch"]))
        if epoch_number != self.last_epoch:
            self.epoch_loss_sum = 0.0
            self.epoch_updates = 0
            self.last_epoch = epoch_number
        self.losses.append(loss)
        self.losses = self.losses[-self.config.moving_average_window :]
        self.epoch_loss_sum += loss
        self.epoch_updates += 1
        metrics["loss_moving_average"] = sum(self.losses) / len(self.losses)
        metrics["epoch_average_loss"] = self.epoch_loss_sum / self.epoch_updates
        metrics["elapsed_seconds"] = time.perf_counter() - self.started
        metrics["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        metrics.update(self._gpu_metrics(self.context.device, step % self.config.gpu_util_interval == 0))
        metrics["total_steps"] = self.config.steps
        metrics["started_step"] = self.started_step
        compact = log_metrics(step, metrics, self.progress)
        if self.log_file is not None and (step % self.config.log_interval == 0 or step == self.config.steps):
            self.log_file.write(compact + "\n")
        if self.metrics_file is not None and (step % self.config.monitor_interval == 0 or step == self.config.steps):
            self.metrics_file.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        if self.writer is not None:
            scalar_keys = (
                "loss", "batch_loss", "loss_moving_average", "epoch_average_loss", "action_mse_loss",
                "action_l1_loss", "learning_rate", "grad_norm", "samples_per_second", "data_time_ms",
                "forward_time_ms", "backward_time_ms", "optimizer_time_ms", "gpu_memory_allocated_gib",
                "gpu_memory_reserved_gib", "gpu_memory_peak_gib", "language_token_count", "image_token_count",
                "action_token_count", "episode_length_mean",
            )
            for key in scalar_keys:
                if key in metrics:
                    self.writer.add_scalar(key.replace("_", "/", 1), metrics[key], step)
            if step % self.config.log_interval == 0:
                self.writer.flush()

    def log_validation(self, step: int, train_loss: float, val_loss: float) -> bool:
        alpha = self.config.early_stopping_ema_alpha
        self.ema_val_loss = (
            val_loss if self.ema_val_loss is None
            else alpha * val_loss + (1.0 - alpha) * self.ema_val_loss
        )
        improved = self.ema_val_loss < self.best_val_loss - self.config.early_stopping_min_delta
        if improved:
            self.best_val_loss = self.ema_val_loss
            self.best_step = step
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1
        patience_warning = self.no_improve_count >= self.config.early_stopping_patience
        gap = val_loss - train_loss
        ratio = val_loss / max(train_loss, 1e-12)
        hard_stop = ratio > self.config.hard_stop_ratio
        if not self.context.is_main:
            return hard_stop
        metrics = {
            "step": step,
            "train_loss_at_checkpoint": train_loss,
            "validation_loss": val_loss,
            "validation_loss_ema": self.ema_val_loss,
            "best_validation_loss": self.best_val_loss,
            "best_step": self.best_step,
            "no_improve_count": self.no_improve_count,
            "overfit_gap": gap,
            "validation_train_ratio": ratio,
            "patience_warning": patience_warning,
            "hard_stop": hard_stop,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if self.metrics_file is not None:
            self.metrics_file.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        message = (
            f"step {step} | checkpoint evaluation | train_loss {train_loss:.6f} | "
            f"val_loss {val_loss:.6f} | ema_val_loss {self.ema_val_loss:.6f} | "
            f"best_val_loss {self.best_val_loss:.6f} at step {self.best_step} | "
            f"no_improve {self.no_improve_count}/{self.config.early_stopping_patience} | "
            f"gap {gap:+.6f} | val/train {ratio:.3f} | patience_warning {patience_warning} | "
            f"hard_stop {hard_stop}"
        )
        if self.log_file is not None:
            self.log_file.write(message + "\n")
        print(f"[checkpoint {step}] {message}", flush=True)
        if patience_warning:
            logging.warning(
                "Validation loss has not materially improved for %d evaluations; continuing training.",
                self.no_improve_count,
            )
        if self.writer is not None:
            self.writer.add_scalars("loss_comparison", {"train": train_loss, "validation": val_loss}, step)
            self.writer.add_scalar("validation/ema_loss", self.ema_val_loss, step)
            self.writer.add_scalar("validation/best_loss", self.best_val_loss, step)
            self.writer.add_scalar("validation/no_improve_count", self.no_improve_count, step)
            self.writer.add_scalar("overfitting/gap", gap, step)
            self.writer.add_scalar("overfitting/validation_train_ratio", ratio, step)
            self.writer.add_scalar("overfitting/warning", float(patience_warning), step)
            self.writer.add_scalar("overfitting/hard_stop", float(hard_stop), step)
            self.writer.flush()
        return hard_stop

    def close(self) -> None:
        if self.progress is not None:
            self.progress.close()
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        if self.log_file is not None:
            self.log_file.close()
        if self.metrics_file is not None:
            self.metrics_file.close()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in unwrap_model(model).state_dict().items() if ".lora_" in name}


def _json_config(config: SFTConfig) -> dict:
    return {
        field.name: str(value) if isinstance(value, Path) else value
        for field in dataclasses.fields(config)
        if (value := getattr(config, field.name)) is not None
    }


def save_checkpoint(engine, config: SFTConfig, context: DistributedContext, step: int) -> None:
    tag = f"step-{step:08d}"
    client_state = {
        "step": step,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "effective_batch_size": config.batch_size * config.gradient_accumulation_steps * context.world_size,
        "world_size": context.world_size,
    }
    # Every rank must participate because ZeRO-2 optimizer state is partitioned.
    engine.save_checkpoint(str(config.output_dir / "deepspeed"), tag=tag, client_state=client_state)
    if context.is_main:
        destination = config.output_dir / tag
        destination.mkdir(parents=True, exist_ok=True)
        adapter = adapter_state_dict(engine.module)
        if not adapter:
            raise RuntimeError("Refusing to save an empty LoRA adapter")
        safetensors.torch.save_file(adapter, destination / "adapter_model.safetensors")
        (destination / "adapter_config.json").write_text(json.dumps(_json_config(config), indent=2))
        (destination / "trainer_state.json").write_text(json.dumps(client_state, indent=2))
        (config.output_dir / "latest").write_text(tag)


def load_resume(engine, config: SFTConfig) -> int:
    if not config.resume:
        return 0
    latest_file = config.output_dir / "latest"
    if not latest_file.exists():
        raise FileNotFoundError(f"--resume was specified but no latest checkpoint exists in {config.output_dir}")
    tag = latest_file.read_text().strip()
    load_path, client_state = engine.load_checkpoint(
        str(config.output_dir / "deepspeed"),
        tag=tag,
        load_optimizer_states=True,
        load_lr_scheduler_states=False,
    )
    if load_path is None:
        raise FileNotFoundError(f"DeepSpeed checkpoint {tag} is incomplete under {config.output_dir / 'deepspeed'}")
    step = int(client_state["step"])
    old_batch = client_state.get("batch_size")
    old_accumulation = client_state.get("gradient_accumulation_steps")
    if old_batch != config.batch_size or old_accumulation != config.gradient_accumulation_steps:
        logging.warning(
            "Resuming step %d with changed batching: batch_size %s -> %d, accumulation %s -> %d",
            step, old_batch, config.batch_size, old_accumulation, config.gradient_accumulation_steps,
        )
    return step


def learning_rate(step: int, config: SFTConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(1, config.warmup_steps)
    progress = min(1.0, (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * cosine


def move_to_device(tree, device: torch.device):
    if isinstance(tree, torch.Tensor):
        return tree.to(device)
    if dataclasses.is_dataclass(tree) and not isinstance(tree, type):
        return dataclasses.replace(
            tree,
            **{field.name: move_to_device(getattr(tree, field.name), device) for field in dataclasses.fields(tree)},
        )
    if isinstance(tree, dict):
        return type(tree)((key, move_to_device(value, device)) for key, value in tree.items())
    if isinstance(tree, tuple):
        return type(tree)(*(move_to_device(value, device) for value in tree)) if hasattr(tree, "_fields") else tuple(
            move_to_device(value, device) for value in tree
        )
    if isinstance(tree, list):
        return [move_to_device(value, device) for value in tree]
    return tree


def train(config: SFTConfig) -> None:
    context = setup_distributed()
    try:
        _train(config, context)
    finally:
        if context.enabled and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _train(config: SFTConfig, context: DistributedContext) -> None:
    if config.steps <= 0:
        raise ValueError(f"steps must be positive, got {config.steps}")
    if config.batch_size <= 0 or config.gradient_accumulation_steps <= 0:
        raise ValueError("batch_size and gradient_accumulation_steps must be positive")
    if config.save_interval <= 0 or config.log_interval <= 0:
        raise ValueError("save_interval and log_interval must be positive")
    if config.early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive")
    if config.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be non-negative")
    if not 0.0 < config.early_stopping_ema_alpha <= 1.0:
        raise ValueError("early_stopping_ema_alpha must be in (0, 1]")
    if config.hard_stop_ratio <= 0:
        raise ValueError("hard_stop_ratio must be positive")
    if not config.resume and config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to start a fresh run in non-empty output directory {config.output_dir}. "
            "Choose an empty --output-dir or pass --resume to continue its latest checkpoint."
        )
    logging.basicConfig(
        level=logging.INFO if context.is_main else logging.ERROR,
        format=f"%(asctime)s %(levelname)s [rank={context.rank}] %(message)s",
        force=True,
    )
    random.seed(config.seed + context.rank)
    np.random.seed(config.seed + context.rank)
    torch.manual_seed(config.seed + context.rank)
    device = context.device
    if device.type != "cuda":
        logging.warning("CUDA is unavailable; PI0.5 training on CPU will be extremely slow")

    model_file = _find_model_file(config.initial_checkpoint)
    if model_file.stat().st_size == 0:
        raise ValueError(f"Initial checkpoint is empty: {model_file}. Re-download the safetensors file before training.")
    logging.info("Building HDF5 data pipeline")
    loader, val_loader, model_config = build_loader(config, context)
    logging.info("Creating PI0.5 model on %s", device)
    model = PI0Pytorch(model_config).to(device)
    logging.info("Loading base checkpoint from %s", model_file)
    safetensors.torch.load_model(model, model_file, device=str(device), strict=False)
    logging.info("Base checkpoint loaded; injecting LoRA")
    replaced = inject_lora(model, config)
    # LoRA parameters are newly created after the base model was moved, so explicitly
    # place the complete adapted model on this rank's GPU before constructing DDP.
    model = model.to(device)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA injection left no trainable parameters")
    for module_name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        for parameter_name, parameter in (("lora_a", module.lora_a), ("lora_b", module.lora_b)):
            if parameter.dtype != module.base.weight.dtype:
                raise RuntimeError(
                    f"{module_name}.{parameter_name} dtype {parameter.dtype} does not match "
                    f"base weight dtype {module.base.weight.dtype}"
                )
            if parameter.device != module.base.weight.device:
                raise RuntimeError(
                    f"{module_name}.{parameter_name} device {parameter.device} does not match "
                    f"base weight device {module.base.weight.device}"
                )
    if any(parameter.device != device for parameter in trainable):
        devices = sorted({str(parameter.device) for parameter in trainable})
        raise RuntimeError(f"LoRA parameter device mismatch: expected {device}, found {devices}")
    if config.deepspeed_config is None:
        raise ValueError("--deepspeed-config is required for ZeRO-2 SFT")
    if not config.deepspeed_config.is_file():
        raise FileNotFoundError(f"DeepSpeed config does not exist: {config.deepspeed_config}")
    ds_config = json.loads(config.deepspeed_config.read_text())
    if ds_config.get("zero_optimization", {}).get("stage") != 2:
        raise ValueError(f"SFT requires DeepSpeed ZeRO stage 2, got config {config.deepspeed_config}")
    ds_config["train_micro_batch_size_per_gpu"] = config.batch_size
    ds_config["gradient_accumulation_steps"] = config.gradient_accumulation_steps
    ds_config["train_batch_size"] = config.batch_size * config.gradient_accumulation_steps * context.world_size
    ds_config["gradient_clipping"] = config.max_grad_norm
    if context.is_main:
        config.output_dir.mkdir(parents=True, exist_ok=True)
    if context.enabled:
        torch.distributed.barrier()
    logging.info("Initializing DeepSpeed ZeRO-2 engine from %s", config.deepspeed_config)
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=trainable,
        optimizer=optimizer,
        config=ds_config,
        dist_init_required=False,
    )
    step = load_resume(model, config)
    effective_batch_size = config.batch_size * config.gradient_accumulation_steps * context.world_size
    logging.info(
        "Injected LoRA into %d layers; trainable parameters: %s; world_size=%d; per_gpu_batch=%d; "
        "gradient_accumulation=%d; global_effective_batch=%d",
        len(replaced), f"{sum(p.numel() for p in trainable):,}", context.world_size, config.batch_size,
        config.gradient_accumulation_steps, effective_batch_size,
    )
    logging.info("Training loop ready; requesting first HDF5 batch")
    monitor = TrainingMonitor(config, context, step)
    if monitor.writer is not None:
        monitor.writer.add_text("config", json.dumps(_json_config(config), indent=2), step)
        logging.info("Training logs and TensorBoard data: %s", monitor.log_dir)
    model.train()
    micro_steps = 0
    running_loss = 0.0
    running_l1 = 0.0
    running_data = running_forward = running_backward = running_optimizer = 0.0
    update_started = time.perf_counter()
    batches_per_epoch = max(1, len(loader._data_loader.torch_loader))  # noqa: SLF001
    # One optimizer step consumes gradient_accumulation_steps loader batches.
    # Seed this counter from the checkpoint so epoch progress remains monotonic
    # after --resume (the resumed loader itself starts at a fresh iterator).
    batch_index = step * config.gradient_accumulation_steps
    total_epochs = config.steps * config.gradient_accumulation_steps / batches_per_epoch
    should_stop = False
    try:
        while step < config.steps and not should_stop:
            for observation, actions in loader:
                data_done = time.perf_counter()
                running_data += data_done - update_started if micro_steps == 0 else data_done - batch_started
                batch_index += 1
                observation = move_to_device(observation, device)
                actions = actions.to(device=device, dtype=torch.float32)
                forward_started = time.perf_counter()
                losses = model(observation, actions)
                raw_loss = losses.mean()
                running_forward += time.perf_counter() - forward_started
                backward_started = time.perf_counter()
                model.backward(raw_loss)
                running_backward += time.perf_counter() - backward_started
                running_loss += raw_loss.detach().float().item()
                running_l1 += losses.detach().float().sqrt().mean().item()
                micro_steps += 1
                boundary = model.is_gradient_accumulation_boundary()
                lr = learning_rate(step, config)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer_started = time.perf_counter()
                model.step()
                running_optimizer += time.perf_counter() - optimizer_started
                batch_started = time.perf_counter()
                if not boundary:
                    continue
                step += 1
                update_elapsed = time.perf_counter() - update_started
                update_loss = running_loss / micro_steps
                update_l1 = running_l1 / micro_steps
                grad_norm = model.get_global_grad_norm()
                if grad_norm is None:
                    raise RuntimeError("DeepSpeed did not publish a global gradient norm after optimizer step")
                language_tokens = observation.tokenized_prompt_mask.sum().item() / max(1, config.batch_size)
                image_tokens = sum(mask.sum().item() for mask in observation.image_masks.values()) / max(1, config.batch_size)
                local_metrics = torch.tensor(
                    [update_loss, update_l1, float(grad_norm), running_data, running_forward, running_backward,
                     running_optimizer, language_tokens, image_tokens],
                    dtype=torch.float64,
                    device=device,
                )
                if context.enabled:
                    torch.distributed.all_reduce(local_metrics, op=torch.distributed.ReduceOp.SUM)
                    local_metrics /= context.world_size
                (global_loss, global_l1, global_grad_norm, data_time, forward_time, backward_time,
                 optimizer_time, language_tokens, image_tokens) = local_metrics.tolist()
                batch_in_epoch = (batch_index - 1) % batches_per_epoch + 1
                epoch = (batch_index - 1) // batches_per_epoch + 1
                epoch_progress = epoch - 1 + batch_in_epoch / batches_per_epoch
                monitor.update({
                    "step": step,
                    "global_step": step,
                    "optimizer_step": step,
                    "epoch": epoch_progress,
                    "epoch_number": epoch,
                    "total_epochs": total_epochs,
                    "batch_in_epoch": batch_in_epoch,
                    "batches_per_epoch": batches_per_epoch,
                    "loss": global_loss,
                    "batch_loss": update_loss,
                    "train_loss": global_loss,
                    "action_prediction_loss": global_loss,
                    "action_mse_loss": global_loss,
                    "action_l1_loss": global_l1,
                    "learning_rate": lr,
                    "grad_norm": global_grad_norm,
                    "gradient_clipped": global_grad_norm > config.max_grad_norm,
                    "gradient_clip_threshold": config.max_grad_norm,
                    "batch_size_per_gpu": config.batch_size,
                    "global_effective_batch_size": effective_batch_size,
                    "gradient_accumulation_steps": config.gradient_accumulation_steps,
                    "action_chunk_length": actions.shape[1],
                    "action_token_count": actions.shape[1],
                    "language_token_count": language_tokens,
                    "image_token_count": image_tokens,
                    "episode_length_mean": float("nan"),
                    "data_time_ms": data_time * 1000,
                    "forward_time_ms": forward_time * 1000,
                    "backward_time_ms": backward_time * 1000,
                    "optimizer_time_ms": optimizer_time * 1000,
                    "update_time_seconds": update_elapsed,
                    "samples_per_second": effective_batch_size / max(update_elapsed, 1e-9),
                })
                running_loss = running_l1 = 0.0
                running_data = running_forward = running_backward = running_optimizer = 0.0
                micro_steps = 0
                update_started = time.perf_counter()
                if step % config.save_interval == 0 or step == config.steps:
                    val_loss = evaluate_loss(model, val_loader, device, context, config.validation_batches)
                    hard_stop = monitor.log_validation(step, global_loss, val_loss)
                    if hard_stop:
                        should_stop = True
                    save_checkpoint(model, config, context, step)
                    if context.is_main and monitor.best_step == step:
                        (config.output_dir / "best").write_text(f"step-{step:08d}")
                    if hard_stop and context.is_main:
                        logging.warning(
                            "Hard stopping at step %d: validation/train loss ratio %.3f exceeded %.3f. "
                            "Confirm checkpoint quality with downstream LIBERO evaluation.",
                            step,
                            val_loss / max(global_loss, 1e-12),
                            config.hard_stop_ratio,
                        )
                if step >= config.steps or should_stop:
                    break
    finally:
        monitor.close()


def parse_args() -> SFTConfig:
    parser = argparse.ArgumentParser(description="PI0.5 LoRA SFT on local LIBERO data")
    optional_types = {
        "default_prompt": str,
        "tensorboard_dir": Path,
        "log_dir": Path,
        "deepspeed_config": Path,
    }
    for field in dataclasses.fields(SFTConfig):
        default = field.default
        flag = "--" + field.name.replace("_", "-")
        flags = (flag, "--local_rank") if field.name == "local_rank" else (flag,)
        if isinstance(default, bool):
            parser.add_argument(*flags, action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(
                *flags,
                dest=field.name,
                type=optional_types.get(field.name, type(default)),
                default=default,
            )
    return SFTConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(parse_args())
