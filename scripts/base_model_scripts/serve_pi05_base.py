from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

OPENPI_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = OPENPI_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import uvicorn

from openpi.rl.openpi_policy_server import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the untouched pi0.5 base checkpoint for LIBERO evaluation")
    parser.add_argument("--checkpoint", default=str(OPENPI_DIR / "asset_pi05_base/pytorch/model.safetensors"))
    parser.add_argument("--norm-stats-dir", default=str(OPENPI_DIR / "RLinf-Pi05-LIBERO-SFT"))
    parser.add_argument("--assets-dir", default=str(OPENPI_DIR / "assets"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-denoise-steps", type=int, default=10)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Base checkpoint not found: {checkpoint}")
    print(
        {
            "event": "starting_pi05_base_server",
            "checkpoint": str(checkpoint),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": args.device,
            "address": f"http://{args.host}:{args.port}",
        },
        flush=True,
    )
    app = create_app(
        checkpoint_path=str(checkpoint),
        reference_dir=args.norm_stats_dir,
        assets_dir=args.assets_dir,
        device=args.device,
        num_denoise_steps=args.num_denoise_steps,
        sample_method="flow_ode",
        full_model_training=False,
        default_prompt="",
        checkpoint_interval=0,
        tensorboard_log_dir=None,
        monitor_log_dir=str(OPENPI_DIR / "logs/pi05_base_libero_server"),
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
