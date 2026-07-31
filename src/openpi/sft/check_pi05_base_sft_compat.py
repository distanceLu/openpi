from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import safetensors.torch

from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.sft.train_pi05_lora import SFTConfig, inject_lora


def find_model_file(checkpoint: Path) -> Path:
    direct = checkpoint / "model.safetensors"
    if direct.is_file():
        return direct
    matches = sorted(checkpoint.glob("**/model.safetensors"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No model.safetensors found under {checkpoint}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a pi0.5 checkpoint supports current LoRA SFT code")
    parser.add_argument("--checkpoint", type=Path, default=Path("/mnt/data/lcx1/yiqinworkspace/openpi/asset_pi05_base/pytorch"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    model_file = find_model_file(args.checkpoint)
    if model_file.stat().st_size == 0:
        raise ValueError(f"Checkpoint is empty: {model_file}")
    model_config = Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False, dtype="bfloat16")
    device = torch.device(args.device)
    model = PI0Pytorch(model_config).to(device)
    missing, unexpected = safetensors.torch.load_model(model, model_file, device=str(device), strict=False)
    lora_config = SFTConfig(
        initial_checkpoint=args.checkpoint,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    replaced = inject_lora(model, lora_config)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(
        {
            "checkpoint": str(model_file),
            "loaded": True,
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            "lora_wrapped_linear_layers": len(replaced),
            "trainable_lora_parameters": trainable,
            "total_parameters": total,
            "trainable_percent": 100.0 * trainable / total,
            "first_lora_layers": replaced[:10],
        }
    )


if __name__ == "__main__":
    main()
