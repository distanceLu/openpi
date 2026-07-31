from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from openpi.models import model as _model
from openpi.rl.libero_adapter import Pi05LiberoRLAdapter
from openpi.rl.openpi_policy_server import _jsonable_observation_to_torch


class SampleRequest(BaseModel):
    observation: Any
    mode: str = "eval"
    return_values: bool = False


def _resolve_latest_adapter(path: Path) -> Path:
    if path.is_file():
        return path
    if (path / "adapter_model.safetensors").is_file():
        return path
    latest = path / "latest"
    if not latest.is_file():
        raise FileNotFoundError(f"No adapter or latest marker found under {path}")
    checkpoint = path / latest.read_text().strip()
    if not (checkpoint / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Latest SFT checkpoint is incomplete: {checkpoint}")
    return checkpoint


def create_eval_app(
    base_checkpoint: Path,
    sft_output_dir: Path,
    reference_dir: Path,
    assets_dir: Path,
    device: str,
    num_denoise_steps: int,
) -> FastAPI:
    adapter_path = _resolve_latest_adapter(sft_output_dir)
    adapter = Pi05LiberoRLAdapter(
        checkpoint_path=base_checkpoint,
        lora_adapter_path=adapter_path,
        reference_dir=reference_dir,
        assets_dir=assets_dir,
        device=device,
        checkpoint_uses_extra_delta_transform=False,
    )
    model = adapter.load_pi05_model_from_checkpoint()
    model.eval()
    model_device = next(model.parameters()).device
    app = FastAPI(title="PI0.5 native SFT evaluation server")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "inference": "native_sample_actions",
            "base_checkpoint": str(base_checkpoint),
            "sft_adapter": str(adapter_path / "adapter_model.safetensors" if adapter_path.is_dir() else adapter_path),
            "num_denoise_steps": num_denoise_steps,
        }

    @app.post("/sample")
    def sample(request: SampleRequest) -> dict[str, Any]:
        observation_dict = _jsonable_observation_to_torch(request.observation, model_device)
        observation = _model.Observation.from_dict(observation_dict)
        with torch.inference_mode():
            actions = model.sample_actions(
                device=model_device,
                observation=observation,
                num_steps=num_denoise_steps,
            )
        if not torch.isfinite(actions).all():
            raise FloatingPointError("Native PI0.5 sampler produced NaN/Inf actions")
        return {
            "action": actions.detach().cpu().float().tolist(),
            "chains": [],
            "denoise_logprobs": [],
            "denoise_means": [],
            "denoise_stds": [],
            "denoise_timesteps": [],
            "denoise_indices": [],
            "velocities": [],
            "values": [0.0],
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve native PI0.5 inference with the latest SFT LoRA adapter")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--sft-output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-denoise-steps", type=int, default=10)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18001)
    args = parser.parse_args()
    if args.num_denoise_steps <= 0:
        raise ValueError("num-denoise-steps must be positive")
    app = create_eval_app(
        base_checkpoint=args.base_checkpoint,
        sft_output_dir=args.sft_output_dir,
        reference_dir=args.reference_dir,
        assets_dir=args.assets_dir,
        device=args.device,
        num_denoise_steps=args.num_denoise_steps,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
