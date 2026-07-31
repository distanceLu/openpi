from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


@dataclass
class Pi05ValueHeadConfig:
    """Value head config for the outer environment MDP critic."""

    input_dim: int
    hidden_dims: tuple[int, ...] = (1024, 512, 256)
    activation: Literal["relu", "gelu", "silu", "tanh"] = "relu"
    dropout: float = 0.0
    pool_mode: Literal["masked_mean", "last_valid", "first"] = "masked_mean"


class Pi05ValueHead(nn.Module):
    """MLP critic head: pooled prefix hidden states -> scalar V(s)."""

    def __init__(self, config: Pi05ValueHeadConfig):
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        last_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(_make_activation(config.activation))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, prefix_hidden_states: torch.Tensor, prefix_pad_mask: torch.Tensor) -> torch.Tensor:
        features = pool_prefix_hidden_states(
            prefix_hidden_states=prefix_hidden_states,
            prefix_pad_mask=prefix_pad_mask,
            mode=self.config.pool_mode,
        )
        parameter = next(self.net.parameters())
        return self.net(features.to(device=parameter.device, dtype=parameter.dtype)).squeeze(-1)


def attach_pi05_value_head(
    model: nn.Module,
    input_dim: int | None = None,
    hidden_dims: tuple[int, ...] = (1024, 512, 256),
    pool_mode: Literal["masked_mean", "last_valid", "first"] = "masked_mean",
) -> Pi05ValueHead:
    """Attach a prefix-state critic to a pi0.5 model if it does not already have one."""

    if hasattr(model, "value_head"):
        value_head = getattr(model, "value_head")
        if not isinstance(value_head, Pi05ValueHead):
            return value_head
        return value_head

    if input_dim is None:
        input_dim = _infer_prefix_hidden_dim(model)
    value_head = Pi05ValueHead(
        Pi05ValueHeadConfig(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            pool_mode=pool_mode,
        )
    )
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.value_head = value_head.to(device=device, dtype=dtype if dtype.is_floating_point else torch.float32)
    return model.value_head


def pool_prefix_hidden_states(
    prefix_hidden_states: torch.Tensor,
    prefix_pad_mask: torch.Tensor,
    mode: Literal["masked_mean", "last_valid", "first"] = "masked_mean",
) -> torch.Tensor:
    """Pool image-language prefix hidden states into one state feature per sample."""

    hidden = prefix_hidden_states.to(torch.float32)
    mask = prefix_pad_mask.to(device=hidden.device, dtype=torch.bool)
    if mode == "masked_mean":
        weights = mask.to(hidden.dtype)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (hidden * weights.unsqueeze(-1)).sum(dim=1) / denom
    if mode == "last_valid":
        lengths = mask.long().sum(dim=1).clamp_min(1) - 1
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
    if mode == "first":
        return hidden[:, 0]
    raise ValueError(f"Unknown prefix pooling mode: {mode}")


def _make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def _infer_prefix_hidden_dim(model: nn.Module) -> int:
    try:
        return int(model.paligemma_with_expert.paligemma.language_model.config.hidden_size)
    except AttributeError:
        pass
    try:
        return int(model.paligemma_with_expert.paligemma.language_model.config.width)
    except AttributeError:
        pass
    try:
        return int(model.paligemma_with_expert.paligemma.config.width)
    except AttributeError:
        pass
    raise ValueError("Cannot infer prefix hidden dim; pass input_dim explicitly to attach_pi05_value_head().")
