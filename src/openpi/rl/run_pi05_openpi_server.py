from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import uvicorn

from openpi.rl.openpi_policy_server import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pi0.5 OpenPI policy server")
    parser.add_argument("--checkpoint-path", type=str, default="/mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT/model.safetensors")
    parser.add_argument(
        "--lora-adapter-path",
        type=str,
        default=None,
        help="Optional SFT step directory or adapter_model.safetensors loaded on top of the base checkpoint.",
    )
    parser.add_argument("--reference-dir", type=str, default="/mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT")
    parser.add_argument("--assets-dir", type=str, default="/mnt/data/lcx1/yiqinworkspace/openpi/assets")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-denoise-steps", type=int, default=10)
    parser.add_argument("--sample-method", choices=["flow_ode", "flow_sde", "flow_cps", "flow_noise"], default="flow_noise")
    parser.add_argument("--full-model-training", action="store_true")
    parser.add_argument("--default-prompt", type=str, default="")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--gamma", type=float, default=0.99, help="SMDP discount shared with the LIBERO rollout client.")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/mnt/data/lcx1/yiqinworkspace/openpi/checkpoints/pi05_libero_rl",
        help="Directory for PPO model and optimizer checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save a checkpoint every N successful PPO updates; set 0 to disable.",
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        type=str,
        default="/mnt/data/lcx1/yiqinworkspace/openpi/runs/pi05_libero_rl",
        help="TensorBoard event directory; pass an empty string to disable logging.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ppo-minibatch-size", type=int, default=1)
    parser.add_argument("--monitor-log-dir", type=str, default="/mnt/data/lcx1/yiqinworkspace/openpi/logs")
    parser.add_argument(
        "--deepspeed-config",
        type=str,
        default=None,
        help="DeepSpeed config. Requires a distributed launcher and a rank-synchronized trainer service.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    app = create_app(
        checkpoint_path=args.checkpoint_path,
        lora_adapter_path=args.lora_adapter_path,
        reference_dir=args.reference_dir,
        assets_dir=args.assets_dir,
        device=args.device,
        num_denoise_steps=args.num_denoise_steps,
        sample_method=args.sample_method,
        full_model_training=args.full_model_training,
        default_prompt=args.default_prompt,
        lr=args.lr,
        target_kl=args.target_kl,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        tensorboard_log_dir=args.tensorboard_log_dir or None,
        gamma=args.gamma,
        deepspeed_config=args.deepspeed_config,
        ppo_minibatch_size=args.ppo_minibatch_size,
        monitor_log_dir=args.monitor_log_dir,
    )
    if args.deepspeed_config and os.environ.get("RANK", "0") != "0":
        raise RuntimeError("DeepSpeed HTTP serving requires a rank-0-only server wrapper; use the dedicated trainer launcher.")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
