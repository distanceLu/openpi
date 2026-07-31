from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from openpi.rl.pi05_trainer import Pi05PPOTrainer
from openpi.rl.rollout_buffer import ObservationCollator, Pi05RolloutBuffer
from openpi.rl.rollout_collector import Pi05RolloutCollector

MetricLogger = Callable[[dict[str, float | torch.Tensor], int], None]


@dataclass
class Pi05RLLoopConfig:
    total_iterations: int = 1000
    rollout_steps: int = 128
    ppo_epochs: int = 4
    minibatch_size: int = 16
    checkpoint_dir: str | None = None
    checkpoint_every: int = 50
    clear_buffer_after_update: bool = True


class Pi05RLTrainingLoop:
    """Full single-GPU/on-policy PPO loop: rollout -> GAE -> loss -> update."""

    def __init__(
        self,
        collector: Pi05RolloutCollector,
        trainer: Pi05PPOTrainer,
        collate_observations: ObservationCollator,
        config: Pi05RLLoopConfig | None = None,
        logger: MetricLogger | None = None,
    ):
        self.collector = collector
        self.trainer = trainer
        self.collate_observations = collate_observations
        self.config = config or Pi05RLLoopConfig()
        self.logger = logger
        self.buffer = Pi05RolloutBuffer(gamma=self.collector.config.gamma)

    def run(self) -> None:
        device = self.trainer.nested_mdp.device
        for iteration in range(1, self.config.total_iterations + 1):
            self.collector.config.rollout_steps = self.config.rollout_steps
            rollout_metrics = self.collector.collect(self.buffer)
            update_metrics = self._update_from_buffer(device)
            metrics = {**rollout_metrics, **update_metrics}

            if self.logger is not None:
                self.logger(metrics, iteration)
            else:
                printable = {k: self._metric_to_float(v) for k, v in metrics.items()}
                print(f"[pi05-rl] iter={iteration} metrics={printable}")

            if self.config.checkpoint_dir and iteration % self.config.checkpoint_every == 0:
                self.save_checkpoint(iteration)
            if self.config.clear_buffer_after_update:
                self.buffer.clear()

    def _update_from_buffer(self, device: torch.device) -> dict[str, torch.Tensor]:
        metric_accumulator: dict[str, list[torch.Tensor]] = {}
        for _ in range(self.config.ppo_epochs):
            for batch in self.buffer.iter_minibatches(
                collate_observations=self.collate_observations,
                minibatch_size=self.config.minibatch_size,
                device=device,
            ):
                metrics = self.trainer.update(batch.__dict__)
                for key, value in metrics.items():
                    metric_accumulator.setdefault(key, []).append(value.detach().float().cpu())
                if self.trainer.config.target_kl is not None:
                    outer_kl = metrics.get("pi05_rl/outer_approx_kl")
                    if outer_kl is not None and float(outer_kl) > self.trainer.config.target_kl:
                        return {key: torch.stack(values).mean() for key, values in metric_accumulator.items()}

        return {key: torch.stack(values).mean() for key, values in metric_accumulator.items()}

    def save_checkpoint(self, iteration: int) -> Path:
        if self.config.checkpoint_dir is None:
            raise ValueError("checkpoint_dir is not configured")
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"pi05_rl_iter_{iteration:06d}.pt"
        torch.save(
            {
                "iteration": iteration,
                "model": self.trainer.model.state_dict(),
                "optimizer": self.trainer.optimizer.state_dict(),
            },
            path,
        )
        return path

    @staticmethod
    def _metric_to_float(value: float | torch.Tensor) -> float:
        if torch.is_tensor(value):
            return float(value.detach().cpu())
        return float(value)
