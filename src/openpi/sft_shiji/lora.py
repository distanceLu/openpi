"""Memory-conscious LoRA adapters for the PyTorch PI0.5 implementation."""

from __future__ import annotations

import math
from pathlib import Path

import safetensors.torch
import torch
from torch import nn


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


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
        if ".lora_a" in name or ".lora_b" in name
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
