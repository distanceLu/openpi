"""Memory-conscious LoRA adapters for the PyTorch PI0.5 implementation."""

from __future__ import annotations

import math
import dataclasses
from pathlib import Path
from typing import Literal

import safetensors.torch
import torch
from torch import nn


FineTuneMode = Literal["a", "b", "c", "d", "e", "full"]


@dataclasses.dataclass(frozen=True)
class FineTuneSpec:
    rank: int
    alpha: float
    adapt_vision: bool
    unfreeze_action_head: bool = False
    unfreeze_expert_last_n_layers: int = 0
    full_finetune: bool = False


EXPERIMENTS: dict[FineTuneMode, FineTuneSpec] = {
    "a": FineTuneSpec(rank=16, alpha=32.0, adapt_vision=False),
    "b": FineTuneSpec(rank=16, alpha=32.0, adapt_vision=True),
    "c": FineTuneSpec(rank=32, alpha=64.0, adapt_vision=True),
    "d": FineTuneSpec(rank=16, alpha=32.0, adapt_vision=True, unfreeze_action_head=True),
    "e": FineTuneSpec(
        rank=16,
        alpha=32.0,
        adapt_vision=True,
        unfreeze_action_head=True,
        unfreeze_expert_last_n_layers=2,
    ),
    "full": FineTuneSpec(rank=0, alpha=0.0, adapt_vision=False, full_finetune=True),
}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base = base
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for parameter in base.parameters():
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


def _branch(name: str) -> str | None:
    if "gemma_expert." in name:
        return "expert"
    if "paligemma.language_model." in name:
        return "backbone"
    if "paligemma.model.vision_tower." in name:
        return "vision"
    if "paligemma.model.multi_modal_projector." in name:
        return "projector"
    return None


def inject_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    adapt_backbone: bool = True,
    adapt_expert: bool = True,
    adapt_vision: bool = False,
    adapt_projector: bool = True,
) -> list[str]:
    """Freeze the base model and inject LoRA into selected branches.

    The 3090 default intentionally leaves SigLIP frozen. Adapting the multimodal
    projector gives the new cameras a low-cost path into the pretrained language
    space; backbone and Action Expert receive attention/MLP LoRA adapters.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False

    enabled = {
        "backbone": adapt_backbone,
        "expert": adapt_expert,
        "vision": adapt_vision,
        "projector": adapt_projector,
    }
    transformer_targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    replaced = []
    for module_name, module in list(model.named_modules()):
        branch = _branch(f".{module_name}.")
        if branch is None or not enabled[branch]:
            continue
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if branch != "projector" and child_name not in transformer_targets:
                continue
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            setattr(module, child_name, LoRALinear(child, rank, alpha, dropout))
            replaced.append(full_name)

    if not replaced:
        raise RuntimeError("No LoRA targets were found; module names may have changed")
    return replaced


def configure_finetuning(
    model: nn.Module,
    mode: FineTuneMode,
    *,
    dropout: float,
) -> tuple[FineTuneSpec, list[str]]:
    spec = EXPERIMENTS[mode]
    if spec.full_finetune:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return spec, []

    replaced = inject_lora(
        model,
        rank=spec.rank,
        alpha=spec.alpha,
        dropout=dropout,
        adapt_backbone=True,
        adapt_expert=True,
        adapt_vision=spec.adapt_vision,
        adapt_projector=True,
    )
    if spec.unfreeze_action_head:
        for module_name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
            module = getattr(model, module_name, None)
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = True
    if spec.unfreeze_expert_last_n_layers:
        layers = model.paligemma_with_expert.gemma_expert.model.layers
        if spec.unfreeze_expert_last_n_layers > len(layers):
            raise ValueError(
                f"Cannot unfreeze {spec.unfreeze_expert_last_n_layers} expert layers; model has {len(layers)}"
            )
        for layer in layers[-spec.unfreeze_expert_last_n_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
    return spec, replaced


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
        if name in trainable_names
    }


def save_adapter(model: nn.Module, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = adapter_state_dict(model)
    if not state:
        raise RuntimeError("Refusing to save an empty adapter")
    safetensors.torch.save_file(state, destination)


def load_adapter(model: nn.Module, source: str | Path) -> None:
    state = safetensors.torch.load_file(str(source), device="cpu")
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"Unexpected adapter keys: {incompatible.unexpected_keys[:10]}")
