from __future__ import annotations

import torch
from torch import nn


class Pi05GaussianActionHead(nn.Module):
    """Predict flow-noise standard deviation from pi0.5 suffix tokens.

    As in RLinf's ``flow_noise`` mode, pi0.5 flow dynamics determine the
    transition mean and this head learns its state-dependent diagonal std.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        min_std: float = 0.08,
        max_std: float = 0.16,
    ):
        super().__init__()
        if input_dim <= 0 or action_dim <= 0:
            raise ValueError("input_dim and action_dim must be positive")
        if not 0 < min_std < max_std:
            raise ValueError("std bounds must satisfy 0 < min_std < max_std")

        layers: list[nn.Module] = []
        feature_dim = input_dim
        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("hidden dimensions must be positive")
            layers.extend((nn.Linear(feature_dim, hidden_dim), nn.Tanh()))
            feature_dim = hidden_dim

        self.shared_net = nn.Sequential(*layers) if layers else nn.Identity()
        self.log_std_layer = nn.Linear(feature_dim, action_dim)
        self.register_buffer("logvar_min", torch.log(torch.tensor(min_std**2, dtype=torch.float32)))
        self.register_buffer("logvar_max", torch.log(torch.tensor(max_std**2, dtype=torch.float32)))
        self._init_output_layer()

    def _init_output_layer(self) -> None:
        nn.init.xavier_uniform_(self.log_std_layer.weight, gain=0.01)
        nn.init.zeros_(self.log_std_layer.bias)

    def forward(self, suffix_features: torch.Tensor) -> torch.Tensor:
        parameter = self.log_std_layer.weight
        suffix_features = suffix_features.to(device=parameter.device, dtype=parameter.dtype)
        hidden = self.shared_net(suffix_features)
        raw_logvar = torch.tanh(self.log_std_layer(hidden))
        logvar_min = self.logvar_min.to(device=raw_logvar.device, dtype=raw_logvar.dtype)
        logvar_max = self.logvar_max.to(device=raw_logvar.device, dtype=raw_logvar.dtype)
        logvar = logvar_min + (logvar_max - logvar_min) * (raw_logvar + 1.0) / 2.0
        return torch.exp(0.5 * logvar)

