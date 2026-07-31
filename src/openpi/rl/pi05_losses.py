from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


PPOLossMode = Literal["element", "path", "path_and_element"]
PathLogprobReduceMode = Literal["mean", "sum"]


@dataclass
class Pi05PPOMetrics:
    policy_loss: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor
    ratio_mean: torch.Tensor
    ratio_min: torch.Tensor
    ratio_max: torch.Tensor
    log_ratio_mean: torch.Tensor
    entropy: torch.Tensor
    valid_count: torch.Tensor
    outer_policy_loss: torch.Tensor | None = None
    inner_policy_loss: torch.Tensor | None = None
    outer_approx_kl: torch.Tensor | None = None
    inner_approx_kl: torch.Tensor | None = None
    outer_clip_fraction: torch.Tensor | None = None
    inner_clip_fraction: torch.Tensor | None = None

    def as_dict(self, prefix: str = "pi05_rl") -> dict[str, torch.Tensor]:
        metrics = {
            f"{prefix}/policy_loss": self.policy_loss.detach(),
            f"{prefix}/approx_kl": self.approx_kl.detach(),
            f"{prefix}/clip_fraction": self.clip_fraction.detach(),
            f"{prefix}/ratio_mean": self.ratio_mean.detach(),
            f"{prefix}/ratio_min": self.ratio_min.detach(),
            f"{prefix}/ratio_max": self.ratio_max.detach(),
            f"{prefix}/log_ratio_mean": self.log_ratio_mean.detach(),
            f"{prefix}/entropy": self.entropy.detach(),
            f"{prefix}/valid_count": self.valid_count.detach(),
        }
        optional = {
            "outer_policy_loss": self.outer_policy_loss,
            "inner_policy_loss": self.inner_policy_loss,
            "outer_approx_kl": self.outer_approx_kl,
            "inner_approx_kl": self.inner_approx_kl,
            "outer_clip_fraction": self.outer_clip_fraction,
            "inner_clip_fraction": self.inner_clip_fraction,
        }
        for name, value in optional.items():
            if value is not None:
                metrics[f"{prefix}/{name}"] = value.detach()
        return metrics


def expand_outer_advantage(advantages: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Broadcast outer-MDP advantages onto inner denoise transitions."""

    advantages = advantages.to(device=target.device, dtype=target.dtype)
    while advantages.dim() < target.dim():
        advantages = advantages.unsqueeze(-1)
    return advantages


def compute_pi05_ppo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    loss_mask: torch.Tensor | None = None,
    normalize_advantages: bool = True,
    entropy: torch.Tensor | None = None,
    eps: float = 1e-8,
    mode: PPOLossMode = "element",
    inner_loss_coef: float = 1.0,
    log_ratio_clip: float | None = 20.0,
    path_logprob_reduce: PathLogprobReduceMode = "mean",
) -> tuple[torch.Tensor, Pi05PPOMetrics]:
    """Nested-MDP PPO actor loss for pi0.5 denoise chains.

    Expected shapes:
        new_logprobs / old_logprobs: [B, K, H, D]
        advantages: [B], [B, 1], or broadcastable outer-MDP shape
        loss_mask: optional broadcastable bool mask

    Modes:
        element: PPO is applied elementwise to every denoise transition/action dim.
        path: PPO is applied once per environment transition using the full
            denoise-chain logprob sum. This is the outer path-level PPO.
        path_and_element: path-level outer PPO plus elementwise inner PPO.
    """

    if mode not in ("element", "path", "path_and_element"):
        raise ValueError(f"Unknown pi0.5 PPO loss mode: {mode}")

    new_logprobs = new_logprobs.float()
    old_logprobs = old_logprobs.to(device=new_logprobs.device, dtype=torch.float32)
    raw_advantages = advantages.to(device=new_logprobs.device, dtype=torch.float32)

    if normalize_advantages:
        raw_advantages = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std(unbiased=False) + eps)

    element_mask = _expand_mask(loss_mask, new_logprobs)
    element_loss, element_metrics = _compute_element_ppo_loss(
        new_logprobs=new_logprobs,
        old_logprobs=old_logprobs,
        advantages=raw_advantages,
        mask=element_mask,
        clip_eps=clip_eps,
        entropy=entropy,
        log_ratio_clip=log_ratio_clip,
    )
    path_loss, path_metrics = _compute_path_ppo_loss(
        new_logprobs=new_logprobs,
        old_logprobs=old_logprobs,
        advantages=raw_advantages,
        mask=element_mask,
        clip_eps=clip_eps,
        entropy=entropy,
        log_ratio_clip=log_ratio_clip,
        eps=eps,
        path_logprob_reduce=path_logprob_reduce,
    )

    if mode == "element":
        policy_loss = element_loss
        selected = element_metrics
        outer_policy_loss = None
        inner_policy_loss = element_loss
    elif mode == "path":
        policy_loss = path_loss
        selected = path_metrics
        outer_policy_loss = path_loss
        inner_policy_loss = None
    else:
        policy_loss = path_loss + inner_loss_coef * element_loss
        selected = path_metrics
        outer_policy_loss = path_loss
        inner_policy_loss = element_loss

    metrics = Pi05PPOMetrics(
        policy_loss=policy_loss,
        approx_kl=selected["approx_kl"],
        clip_fraction=selected["clip_fraction"],
        ratio_mean=selected["ratio_mean"],
        ratio_min=selected["ratio_min"],
        ratio_max=selected["ratio_max"],
        log_ratio_mean=selected["log_ratio_mean"],
        entropy=selected["entropy"],
        valid_count=selected["valid_count"],
        outer_policy_loss=outer_policy_loss,
        inner_policy_loss=inner_policy_loss,
        outer_approx_kl=path_metrics["approx_kl"],
        inner_approx_kl=element_metrics["approx_kl"],
        outer_clip_fraction=path_metrics["clip_fraction"],
        inner_clip_fraction=element_metrics["clip_fraction"],
    )
    return policy_loss, metrics


def _expand_mask(loss_mask: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor:
    if loss_mask is None:
        return torch.ones_like(target, dtype=torch.bool)
    mask = loss_mask.to(device=target.device, dtype=torch.bool)
    while mask.dim() < target.dim():
        mask = mask.unsqueeze(-1)
    return mask.expand_as(target)


def _safe_ratio(log_ratio: torch.Tensor, log_ratio_clip: float | None) -> torch.Tensor:
    if log_ratio_clip is not None:
        log_ratio = torch.clamp(log_ratio, -log_ratio_clip, log_ratio_clip)
    return torch.exp(log_ratio)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, value, torch.zeros_like(value)).sum() / mask.count_nonzero().clamp_min(1).to(value.dtype)


def _ppo_kl_estimate(log_ratio: torch.Tensor) -> torch.Tensor:
    # Schulman-style non-negative sample estimate for KL(old || new):
    # E_old[(r - 1) - log r]. Clamp the same log-ratio used for ``ratio`` so
    # rare numerical outliers cannot make the diagnostic negative or overflow.
    clipped_log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
    ratio = torch.exp(clipped_log_ratio)
    return (ratio - 1.0) - clipped_log_ratio


def _compute_element_ppo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float,
    entropy: torch.Tensor | None,
    log_ratio_clip: float | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    expanded_advantages = expand_outer_advantage(advantages, new_logprobs)
    log_ratio = new_logprobs - old_logprobs
    ratio = _safe_ratio(log_ratio, log_ratio_clip)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    element_loss = -torch.minimum(ratio * expanded_advantages, clipped_ratio * expanded_advantages)
    policy_loss = _masked_mean(element_loss, mask)

    if entropy is None:
        entropy_value = torch.zeros((), device=new_logprobs.device)
    else:
        entropy_value = _masked_mean(entropy.to(device=new_logprobs.device, dtype=torch.float32), mask)

    with torch.no_grad():
        metrics = _ratio_metrics(log_ratio, ratio, mask, clip_eps)
        metrics["entropy"] = entropy_value.detach()
        metrics["valid_count"] = mask.count_nonzero().clamp_min(1).to(new_logprobs.dtype)
    return policy_loss, metrics


def _compute_path_ppo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float,
    entropy: torch.Tensor | None,
    log_ratio_clip: float | None,
    eps: float,
    path_logprob_reduce: PathLogprobReduceMode,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    path_mask = mask.reshape(mask.shape[0], -1)
    valid_per_path = path_mask.sum(dim=1).clamp_min(1).to(new_logprobs.dtype)
    flat_new = torch.where(mask, new_logprobs, torch.zeros_like(new_logprobs)).reshape(new_logprobs.shape[0], -1)
    flat_old = torch.where(mask, old_logprobs, torch.zeros_like(old_logprobs)).reshape(old_logprobs.shape[0], -1)
    if path_logprob_reduce == "sum":
        new_path_logprob = flat_new.sum(dim=1)
        old_path_logprob = flat_old.sum(dim=1)
    else:
        # Mean keeps path-level ratios numerically comparable to element-level
        # ratios for high-dimensional action chunks.
        new_path_logprob = flat_new.sum(dim=1) / valid_per_path
        old_path_logprob = flat_old.sum(dim=1) / valid_per_path
    log_ratio = new_path_logprob - old_path_logprob
    ratio = _safe_ratio(log_ratio, log_ratio_clip)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

    adv = advantages.reshape(advantages.shape[0], -1).mean(dim=1) if advantages.dim() > 1 else advantages.reshape(-1)
    valid_path_mask = path_mask.any(dim=1)
    path_loss = -torch.minimum(ratio * adv, clipped_ratio * adv)
    policy_loss = _masked_mean(path_loss, valid_path_mask)

    if entropy is None:
        entropy_value = torch.zeros((), device=new_logprobs.device)
    else:
        flat_entropy = torch.where(
            mask,
            entropy.to(device=new_logprobs.device, dtype=torch.float32),
            torch.zeros_like(new_logprobs),
        ).reshape(new_logprobs.shape[0], -1)
        entropy_value = (
            torch.where(valid_path_mask, flat_entropy.sum(dim=1) / valid_per_path, torch.zeros_like(valid_per_path))
            .sum()
            / valid_path_mask.count_nonzero().clamp_min(1).to(new_logprobs.dtype)
        )

    with torch.no_grad():
        metrics = _ratio_metrics(log_ratio, ratio, valid_path_mask, clip_eps)
        metrics["entropy"] = entropy_value.detach()
        metrics["valid_count"] = valid_path_mask.count_nonzero().clamp_min(1).to(new_logprobs.dtype)
    return policy_loss, metrics


def _ratio_metrics(
    log_ratio: torch.Tensor,
    ratio: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float,
) -> dict[str, torch.Tensor]:
    approx_kl_values = _ppo_kl_estimate(log_ratio)
    masked_ratio = torch.where(mask, ratio, torch.zeros_like(ratio))
    valid_count = mask.count_nonzero().clamp_min(1).to(ratio.dtype)
    if mask.any():
        ratio_min = ratio[mask].min()
        ratio_max = ratio[mask].max()
    else:
        ratio_min = torch.zeros((), device=ratio.device)
        ratio_max = torch.zeros((), device=ratio.device)
    return {
        "approx_kl": _masked_mean(approx_kl_values, mask),
        "clip_fraction": _masked_mean((torch.abs(ratio - 1.0) > clip_eps).float(), mask),
        "ratio_mean": masked_ratio.sum() / valid_count,
        "ratio_min": ratio_min,
        "ratio_max": ratio_max,
        "log_ratio_mean": _masked_mean(log_ratio, mask),
    }


def compute_value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Plain MSE critic loss with optional outer-MDP mask."""

    values = values.float()
    returns = returns.to(device=values.device, dtype=torch.float32)
    loss = 0.5 * (values - returns).pow(2)
    if loss_mask is None:
        return loss.mean()
    mask = loss_mask.to(device=values.device, dtype=torch.bool)
    return torch.where(mask, loss, torch.zeros_like(loss)).sum() / mask.count_nonzero().clamp_min(1)


def compute_fm_anchor_loss(
    current_velocities: torch.Tensor,
    target_velocities: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Flow-matching anchor on denoise velocities saved from rollout/reference."""

    loss = (current_velocities.float() - target_velocities.to(current_velocities.device, dtype=torch.float32)).pow(2)
    if loss_mask is None:
        return loss.mean()
    mask = _expand_mask(loss_mask, loss)
    return _masked_mean(loss, mask)


def compute_reference_kl_loss(
    new_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sampled KL-style regularizer against a frozen reference pi0.5 policy.

    With rollout samples, E[log pi_theta - log pi_ref] is the standard sampled
    policy KL penalty used to keep RL updates close to the reference policy.
    """

    logprob_diff = new_logprobs.float() - reference_logprobs.to(new_logprobs.device, dtype=torch.float32)
    if loss_mask is None:
        return logprob_diff.mean()
    mask = _expand_mask(loss_mask, logprob_diff)
    return _masked_mean(logprob_diff, mask)
