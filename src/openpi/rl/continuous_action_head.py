"""Continuous-action Gaussian policy head for standard PPO.

This is a conventional PPO policy head.  pi0.5's native policy is a
multi-step flow/denoising policy; use this module when the action is modeled
as one continuous vector at each environment step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal


@dataclass(frozen=True)
class ContinuousActionHeadConfig:
    input_dim: int
    action_dim: int
    hidden_dims: tuple[int, ...] = (256, 256)
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    squash_actions: bool = True
    action_low: float | torch.Tensor | None = None
    action_high: float | torch.Tensor | None = None


class ContinuousActionHead(nn.Module):
    """Diagonal-Gaussian PPO actor with optional tanh action bounds.

    ``log_prob`` always returns one scalar per batch item: the sum over all
    action dimensions.  Store that scalar during rollout as ``old_logprob``
    and recompute it from the same sampled action during PPO updates.
    """

    def __init__(self, config: ContinuousActionHeadConfig):
        super().__init__()
        self.config = config

        if (config.action_low is None) != (config.action_high is None):
            raise ValueError("action_low and action_high must either both be set or both be None")
        if not config.squash_actions and config.action_low is not None:
            raise ValueError("action bounds require squash_actions=True")
        if config.log_std_min >= config.log_std_max:
            raise ValueError("log_std_min must be smaller than log_std_max")

        layers: list[nn.Module] = []
        last_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.extend((nn.Linear(last_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()))
            last_dim = hidden_dim
        self.shared_net = nn.Sequential(*layers) if layers else nn.Identity()
        self.mean_layer = nn.Linear(last_dim, config.action_dim)
        self.log_std_layer = nn.Linear(last_dim, config.action_dim)
        self._init_output_layers()

        if config.action_low is not None:
            action_low = torch.as_tensor(config.action_low, dtype=torch.float32)
            action_high = torch.as_tensor(config.action_high, dtype=torch.float32)
            try:
                broadcast_shape = torch.broadcast_shapes(action_low.shape, action_high.shape, (config.action_dim,))
            except RuntimeError as error:
                raise ValueError("action bounds must be scalar or broadcastable to [action_dim]") from error
            if broadcast_shape != (config.action_dim,):
                raise ValueError("action bounds must be scalar or broadcastable to [action_dim]")
            action_low = action_low.expand(config.action_dim).clone()
            action_high = action_high.expand(config.action_dim).clone()
            if torch.any(action_high <= action_low):
                raise ValueError("Every action_high value must be greater than action_low")
            self.register_buffer("action_low", action_low)
            self.register_buffer("action_high", action_high)
        else:
            self.action_low = None
            self.action_high = None

    def _init_output_layers(self) -> None:
        nn.init.xavier_uniform_(self.mean_layer.weight, gain=0.01)
        nn.init.zeros_(self.mean_layer.bias)
        nn.init.xavier_uniform_(self.log_std_layer.weight, gain=0.01)
        nn.init.zeros_(self.log_std_layer.bias)

    def _distribution_parameters(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared_net(features)
        mean = self.mean_layer(hidden)
        log_std = self.log_std_layer(hidden).clamp(self.config.log_std_min, self.config.log_std_max)
        return mean, log_std.exp()

    def distribution(self, features: torch.Tensor) -> Normal:
        """Return the elementwise diagonal Normal before tanh squashing."""
        mean, std = self._distribution_parameters(features)
        return Normal(mean, std)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return state-dependent ``(mean, std)`` tensors, each ``[B, action_dim]``."""
        return self._distribution_parameters(features)

    def sample(self, features: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Return environment action and its exact log probability, both [B]."""
        dist = self.distribution(features)
        latent_action = dist.mean if deterministic else dist.rsample()
        action = self._squash(latent_action) if self.config.squash_actions else latent_action
        log_prob = self._log_prob_from_latent(dist, latent_action)
        return action, log_prob

    def log_prob(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log pi(action | features), expecting actions from ``sample``.

        For tanh-squashed policies this applies the change-of-variables
        correction.  It is essential: using Normal.log_prob(action) directly
        after tanh produces an incorrect PPO importance ratio.
        """
        dist = self.distribution(features)
        if not self.config.squash_actions:
            return dist.log_prob(action).sum(dim=-1)

        latent_action = self._unsquash(action)
        return self._log_prob_from_latent(dist, latent_action)

    def evaluate_actions(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recompute current-policy log probability and approximate entropy."""
        return self.log_prob(features, actions), self.entropy(features)

    def entropy(self, features: torch.Tensor) -> torch.Tensor:
        """Base Gaussian entropy [B], used as a squashed-policy approximation."""
        return self.distribution(features).entropy().sum(dim=-1)

    def _bounds_like(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.action_low.to(tensor), self.action_high.to(tensor)

    def _squash(self, latent_action: torch.Tensor) -> torch.Tensor:
        action = torch.tanh(latent_action)
        if self.action_low is None:
            return action
        action_low, action_high = self._bounds_like(action)
        return action * ((action_high - action_low) / 2) + ((action_high + action_low) / 2)

    def _unsquash(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_low is not None:
            action_low, action_high = self._bounds_like(action)
            action = (action - (action_high + action_low) / 2) / ((action_high - action_low) / 2)
        eps = torch.finfo(action.dtype).eps
        return torch.atanh(action.clamp(-1.0 + eps, 1.0 - eps))

    def _log_prob_from_latent(self, dist: Normal, latent_action: torch.Tensor) -> torch.Tensor:
        log_prob = dist.log_prob(latent_action)
        if self.config.squash_actions:
            # log |d tanh(u)/du| = log(1 - tanh(u)^2).
            correction = 2.0 * (
                torch.log(torch.as_tensor(2.0, device=latent_action.device, dtype=latent_action.dtype))
                - latent_action
                - torch.nn.functional.softplus(-2.0 * latent_action)
            )
            log_prob = log_prob - correction
            if self.action_low is not None:
                action_low, action_high = self._bounds_like(latent_action)
                action_scale = (action_high - action_low) / 2
                log_prob = log_prob - action_scale.abs().log()
        return log_prob.sum(dim=-1)


def clipped_ppo_policy_loss(
    new_logprob: torch.Tensor,
    old_logprob: torch.Tensor,
    advantage: torch.Tensor,
    clip_eps: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return PPO clipped surrogate loss and useful detached diagnostics."""
    if new_logprob.shape != old_logprob.shape:
        raise ValueError(f"new/old logprob shapes must match, got {new_logprob.shape} and {old_logprob.shape}")
    advantage = advantage.expand_as(new_logprob)
    log_ratio = new_logprob - old_logprob
    ratio = log_ratio.exp()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantage
    loss = -torch.minimum(unclipped, clipped).mean()
    return loss, {
        "ratio": ratio.mean().detach(),
        "clip_fraction": ((ratio - 1.0).abs() > clip_eps).float().mean().detach(),
        "approx_kl": ((ratio - 1.0) - log_ratio).mean().detach(),
    }
