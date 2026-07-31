from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from openpi.rl.pi05_denoising import normal_entropy
from openpi.rl.pi05_losses import (
    PPOLossMode,
    compute_fm_anchor_loss,
    compute_pi05_ppo_loss,
    compute_reference_kl_loss,
    compute_value_loss,
)
from openpi.rl.pi05_nested_mdp import Pi05NestedMDP


@dataclass
class Pi05PPOTrainConfig:
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0
    normalize_advantages: bool = True
    ppo_loss_mode: PPOLossMode = "path"
    inner_loss_coef: float = 0.1
    log_ratio_clip: float | None = 20.0
    path_logprob_reduce: str = "sum"
    fm_anchor_coef: float = 0.0
    reference_kl_coef: float = 0.0
    target_kl: float | None = None


class Pi05PPOTrainer:
    """Minimal PPO update helper for pi0.5 nested-MDP rollouts."""

    def __init__(
        self,
        nested_mdp: Pi05NestedMDP,
        optimizer: torch.optim.Optimizer,
        config: Pi05PPOTrainConfig | None = None,
        engine: Any | None = None,
        reference_mdp: Pi05NestedMDP | None = None,
    ):
        self.nested_mdp = nested_mdp
        self.model = nested_mdp.model
        self.optimizer = optimizer
        self.engine = engine
        self.config = config or Pi05PPOTrainConfig()
        self.reference_mdp = reference_mdp
        if self.reference_mdp is not None:
            self.reference_mdp.model.eval()
            for param in self.reference_mdp.model.parameters():
                param.requires_grad_(False)

    def update(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Run one PPO update.

        Required batch keys:
            observation: model observation object for the flattened rollout batch
            chains: [B, K + 1, H, D]
            old_logprobs: [B, K, H, D]
            advantages: [B] or broadcastable to old_logprobs

        Optional batch keys:
            returns: critic targets
            old_values/current values: for value loss if model exposes value_head
            denoise_indices: [B, K]
            loss_mask: bool mask broadcastable to logprobs
        """

        # Keep the policy in the same training mode used during rollout. PPO
        # compares logprobs from the behavior policy and the current policy;
        # switching train/eval behavior between those two evaluations would
        # make their ratio depend on mode rather than only on parameters.
        self.model.train()
        if self.config.fm_anchor_coef > 0 and batch.get("old_velocities") is None and batch.get("target_velocities") is None:
            raise ValueError("fm_anchor_coef > 0 requires old_velocities or target_velocities in every PPO batch")
        if self.config.reference_kl_coef > 0 and self.reference_mdp is None and batch.get("reference_logprobs") is None:
            raise ValueError("reference_kl_coef > 0 requires reference_mdp or reference_logprobs")
        recomputed = self.nested_mdp.recompute_logprobs(
            observation=batch["observation"],
            chains=batch["chains"],
            denoise_indices=batch.get("denoise_indices"),
            denoise_timesteps=batch.get("denoise_timesteps"),
            return_values="returns" in batch,
        )
        entropy = normal_entropy(recomputed.stds)
        policy_loss, policy_metrics = compute_pi05_ppo_loss(
            new_logprobs=recomputed.logprobs,
            old_logprobs=batch["old_logprobs"],
            advantages=batch["advantages"],
            clip_eps=self.config.clip_eps,
            loss_mask=batch.get("loss_mask"),
            normalize_advantages=self.config.normalize_advantages,
            entropy=entropy,
            mode=self.config.ppo_loss_mode,
            inner_loss_coef=self.config.inner_loss_coef,
            log_ratio_clip=self.config.log_ratio_clip,
            path_logprob_reduce=self.config.path_logprob_reduce,
        )

        if "returns" not in batch:
            raise ValueError("PPO update requires returns computed from GAE.")
        if recomputed.values is None:
            raise ValueError("PPO update requires model.value_head to recompute V(s).")
        value_loss = compute_value_loss(recomputed.values, batch["returns"], batch.get("outer_loss_mask"))

        fm_anchor_loss = self._compute_fm_anchor_loss(batch, recomputed)
        reference_kl_loss = self._compute_reference_kl_loss(batch, recomputed)
        total_loss = (
            policy_loss
            + self.config.value_coef * value_loss
            - self.config.entropy_coef * policy_metrics.entropy
            + self.config.fm_anchor_coef * fm_anchor_loss
            + self.config.reference_kl_coef * reference_kl_loss
        )

        if self.engine is not None:
            self.engine.zero_grad()
            self.engine.backward(total_loss)
            self.engine.step()
            engine_grad_norm = self.engine.get_global_grad_norm()
            if engine_grad_norm is None:
                raise RuntimeError(
                    "DeepSpeed did not publish a global gradient norm after engine.step(); "
                    "ensure gradient_clipping is enabled and the step is a gradient accumulation boundary"
                )
            grad_norm = torch.as_tensor(engine_grad_norm, device=total_loss.device, dtype=torch.float32)
        else:
            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            trainable_parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
            if self.config.max_grad_norm is not None and self.config.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, self.config.max_grad_norm)
            else:
                grad_norm = torch.linalg.vector_norm(
                    torch.stack([parameter.grad.detach().norm() for parameter in trainable_parameters if parameter.grad is not None])
                )
            self.optimizer.step()

        with torch.no_grad():
            advantages = batch["advantages"].float()
            returns = batch["returns"].float()
            predicted_values = recomputed.values.float()
            return_variance = torch.var(returns, unbiased=False)
            explained_variance = 1.0 - torch.var(returns - predicted_values, unbiased=False) / return_variance.clamp_min(1e-8)

        metrics = policy_metrics.as_dict()
        metrics.update(
            {
                "pi05_rl/value_loss": value_loss.detach(),
                "pi05_rl/fm_anchor_loss": fm_anchor_loss.detach(),
                "pi05_rl/reference_kl_loss": reference_kl_loss.detach(),
                "pi05_rl/total_loss": total_loss.detach(),
                "pi05_rl/grad_norm": grad_norm.detach(),
                "pi05_rl/advantage_mean": advantages.mean().detach(),
                "pi05_rl/advantage_std": advantages.std(unbiased=False).detach(),
                "pi05_rl/return_mean": returns.mean().detach(),
                "pi05_rl/return_std": returns.std(unbiased=False).detach(),
                "pi05_rl/value_mean": predicted_values.mean().detach(),
                "pi05_rl/explained_variance": explained_variance.detach(),
            }
        )
        return metrics

    def _compute_fm_anchor_loss(self, batch: dict[str, Any], recomputed: Any) -> torch.Tensor:
        if self.config.fm_anchor_coef <= 0:
            return torch.zeros((), device=self.nested_mdp.device)
        if "target_velocities" in batch:
            target_velocities = batch["target_velocities"]
        elif "old_velocities" in batch:
            target_velocities = batch["old_velocities"]
        else:
            # Conservative default: anchor current velocities to the rollout-time
            # velocities if they are provided by future buffers; otherwise no-op.
            return torch.zeros((), device=self.nested_mdp.device)
        return compute_fm_anchor_loss(recomputed.velocities, target_velocities, batch.get("loss_mask"))

    def _compute_reference_kl_loss(self, batch: dict[str, Any], recomputed: Any) -> torch.Tensor:
        if self.config.reference_kl_coef <= 0:
            return torch.zeros((), device=self.nested_mdp.device)
        if self.reference_mdp is None and "reference_logprobs" not in batch:
            return torch.zeros((), device=self.nested_mdp.device)
        if "reference_logprobs" in batch:
            reference_logprobs = batch["reference_logprobs"]
        else:
            with torch.no_grad():
                reference = self.reference_mdp.recompute_logprobs(
                    observation=batch["observation"],
                    chains=batch["chains"],
                    denoise_indices=batch.get("denoise_indices"),
                    denoise_timesteps=batch.get("denoise_timesteps"),
                    return_values=False,
                )
                reference_logprobs = reference.logprobs
        return compute_reference_kl_loss(recomputed.logprobs, reference_logprobs, batch.get("loss_mask"))
