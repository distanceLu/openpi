from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _to_jsonable(value: Any) -> Any:
    try:
        import torch
    except Exception:  # pragma: no cover
        torch = None
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


@dataclass
class SampleRequest:
    observation: Any
    mode: str = "train"
    return_values: bool = True

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self.__dict__)


@dataclass
class SampleResponse:
    action: Any
    chains: Any
    denoise_logprobs: Any
    denoise_means: Any
    denoise_stds: Any
    denoise_timesteps: Any
    denoise_indices: Any
    velocities: Any
    values: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self.__dict__)


@dataclass
class RolloutUpdateRequest:
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

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self.__dict__)


@dataclass
class RolloutUpdateResponse:
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self.__dict__)
