from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from openpi.rl.libero_adapter import LIBERO_IMAGE_KEYS, Pi05LiberoRLAdapter
from openpi.rl.libero_http_client import OpenPIServerClient


def _checkpoint_step(health: dict[str, Any], explicit_step: int | None) -> int:
    if explicit_step is not None:
        return explicit_step
    adapter_path = Path(str(health["sft_adapter"]))
    for candidate in (adapter_path.parent.name, adapter_path.name):
        if candidate.startswith("step-") and candidate[5:].isdigit():
            return int(candidate[5:])
    raise ValueError("Cannot infer checkpoint step from sft_adapter; provide --checkpoint-step")


def _write_rollout_metrics(args: argparse.Namespace, summary: dict[str, Any], checkpoint_step: int) -> None:
    row = {
        "event": "libero_rollout_validation",
        "step": checkpoint_step,
        "task_suite": args.task_suite,
        "task_ids": args.task_ids,
        "episodes_per_task": args.episodes,
        "completed_episodes": summary["completed_episodes"],
        "successful_episodes": summary["successful_episodes"],
        "success_rate": summary["success_rate"],
        "success_rate_percent": summary["success_rate_percent"],
        "sft_adapter": summary["server"]["sft_adapter"],
        "seed": args.seed,
    }
    for task in summary["tasks"]:
        row[f"task_{task['task_id']}_success_rate"] = task["success_rate"]
    if args.metrics_jsonl is not None:
        args.metrics_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_jsonl.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
    if args.tensorboard_dir is not None:
        writer = SummaryWriter(log_dir=str(args.tensorboard_dir))
        writer.add_scalar(f"rollout/{args.task_suite}/success_rate", summary["success_rate"], checkpoint_step)
        writer.add_scalar(
            f"rollout/{args.task_suite}/success_rate_percent", summary["success_rate_percent"], checkpoint_step
        )
        writer.add_scalar(
            f"rollout/{args.task_suite}/successful_episodes", summary["successful_episodes"], checkpoint_step
        )
        for task in summary["tasks"]:
            writer.add_scalar(
                f"rollout/{args.task_suite}/task_{task['task_id']}_success_rate",
                task["success_rate"],
                checkpoint_step,
            )
        writer.flush()
        writer.close()
    print(json.dumps({"event": "rollout_metrics_logged", **row}), flush=True)


def evaluate_task(
    args: argparse.Namespace,
    client: OpenPIServerClient,
    adapter: Pi05LiberoRLAdapter,
    task_id: int,
) -> dict[str, Any]:
    env = adapter.make_libero_env(
        task_suite_name=args.task_suite,
        task_id=task_id,
        seed=args.seed,
        cycle_init_states=True,
    )
    episodes: list[dict[str, Any]] = []
    try:
        for episode_id in range(args.episodes):
            observation = env.reset()
            if isinstance(observation, tuple):
                observation = observation[0]
            init_state_index = int(observation.get("init_state_index", -1))
            success = False
            env_steps = 0
            policy_calls = 0
            video_writer = None
            if args.video_dir is not None:
                video_path = args.video_dir / f"{args.task_suite}_task{task_id}_episode{episode_id}.mp4"
                video_path.parent.mkdir(parents=True, exist_ok=True)
                video_writer = imageio.get_writer(str(video_path), fps=args.video_fps, codec="libx264")
                frame = adapter._extract_image(observation, LIBERO_IMAGE_KEYS)
                video_writer.append_data(frame)
            while env_steps < args.max_episode_steps and not success:
                model_observation = adapter.env_obs_to_model_obs(observation)
                response = client.sample(model_observation.to_dict(), mode="eval", return_values=False)
                if "action" not in response:
                    raise KeyError(f"Policy response has no action: {sorted(response)}")
                action_chunk = np.asarray(response["action"], dtype=np.float32)
                expected_shape = (1, adapter.action_horizon, adapter.action_dim)
                if action_chunk.shape != expected_shape:
                    raise ValueError(f"Expected action shape {expected_shape}, got {action_chunk.shape}")
                if not np.isfinite(action_chunk).all():
                    raise FloatingPointError("Policy action contains NaN/Inf")
                policy_calls += 1
                executed_actions = []
                for model_action in action_chunk[0, : args.execute_horizon]:
                    env_action = adapter.action_to_env_action(torch.from_numpy(model_action), observation)
                    raw_gripper = float(env_action[-1])
                    if args.binarize_gripper:
                        env_action[-1] = 1.0 if raw_gripper > args.gripper_threshold else -1.0
                    executed_actions.append(env_action.copy())
                    step_result = env.step(env_action)
                    observation = step_result[0]
                    env_steps += 1
                    if video_writer is not None:
                        frame = adapter._extract_image(observation, LIBERO_IMAGE_KEYS)
                        video_writer.append_data(frame)
                    success = bool(env.check_success())
                    if success or env_steps >= args.max_episode_steps:
                        break
                if policy_calls == 1 or policy_calls % args.diagnostics_interval == 0 or success:
                    env_actions = np.asarray(executed_actions, dtype=np.float32)
                    print(
                        json.dumps(
                            {
                                "event": "action_diagnostics",
                                "task_suite": args.task_suite,
                                "task_id": task_id,
                                "episode": episode_id,
                                "env_steps": env_steps,
                                "policy_calls": policy_calls,
                                "model_action_min": float(action_chunk[..., :7].min()),
                                "model_action_max": float(action_chunk[..., :7].max()),
                                "model_gripper": action_chunk[0, : args.execute_horizon, 6].tolist(),
                                "env_action_min": env_actions.min(axis=0).tolist(),
                                "env_action_max": env_actions.max(axis=0).tolist(),
                                "env_gripper": env_actions[:, -1].tolist(),
                                "binarize_gripper": args.binarize_gripper,
                                "success": success,
                            }
                        ),
                        flush=True,
                    )
            if video_writer is not None:
                video_writer.close()
            row = {
                "episode": episode_id,
                "init_state_index": init_state_index,
                "env_steps": env_steps,
                "policy_calls": policy_calls,
                "success": success,
            }
            episodes.append(row)
            print(json.dumps({"event": "episode_complete", "task_suite": args.task_suite, "task_id": task_id, **row}), flush=True)
    finally:
        with contextlib.suppress(Exception):
            env.close()

    successes = sum(int(row["success"]) for row in episodes)
    return {
        "task_suite": args.task_suite,
        "task_id": task_id,
        "completed_episodes": len(episodes),
        "successful_episodes": successes,
        "success_rate": successes / len(episodes),
        "success_rate_percent": 100.0 * successes / len(episodes),
        "episodes": episodes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a PI0.5 SFT LoRA server on LIBERO")
    parser.add_argument("--server-url", default="http://127.0.0.1:18001")
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=240)
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--diagnostics-interval", type=int, default=10)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument(
        "--binarize-gripper",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Map the continuous gripper output to LIBERO's expected {-1, +1} command.",
    )
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.5,
        help="Continuous environment-space gripper value above which the binary command is +1.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--libero-repo-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument(
        "--metrics-jsonl",
        type=Path,
        default=None,
        help="Append downstream LIBERO rollout success metrics for this checkpoint.",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=None,
        help="Write downstream LIBERO rollout success metrics to this TensorBoard run directory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes <= 0 or args.max_episode_steps <= 0 or args.diagnostics_interval <= 0 or args.video_fps <= 0:
        raise ValueError("Episode, step, diagnostics interval, and video FPS values must be positive")
    if not 1 <= args.execute_horizon <= 10:
        raise ValueError("--execute-horizon must be in [1, 10]")
    if args.checkpoint_step is not None and args.checkpoint_step < 0:
        raise ValueError("--checkpoint-step must be non-negative")

    client = OpenPIServerClient(base_url=args.server_url, timeout=args.request_timeout)
    health = client.health()
    if health.get("status") != "ok":
        raise RuntimeError(f"SFT server is unhealthy: {health}")
    if not health.get("sft_adapter"):
        raise RuntimeError(
            "The connected server does not report an SFT adapter. Start openpi.sft.run_pi05_lora_eval_server, not a base-model server."
        )
    print(json.dumps({"event": "sft_server_ready", **health}), flush=True)

    adapter = Pi05LiberoRLAdapter(
        checkpoint_path=args.base_checkpoint,
        reference_dir=args.reference_dir,
        assets_dir=args.assets_dir,
        libero_repo_dir=args.libero_repo_dir,
        device="cpu",
        checkpoint_uses_extra_delta_transform=False,
    )
    tasks = [evaluate_task(args, client, adapter, task_id) for task_id in args.task_ids]
    completed = sum(task["completed_episodes"] for task in tasks)
    successes = sum(task["successful_episodes"] for task in tasks)
    rate = successes / completed if completed else 0.0
    stderr = math.sqrt(rate * (1.0 - rate) / completed) if completed else 0.0
    summary = {
        "server": health,
        "task_suite": args.task_suite,
        "completed_episodes": completed,
        "successful_episodes": successes,
        "success_rate": rate,
        "success_rate_percent": 100.0 * rate,
        "approx_95_percent_ci_percent": [
            100.0 * max(0.0, rate - 1.96 * stderr),
            100.0 * min(1.0, rate + 1.96 * stderr),
        ],
        "tasks": tasks,
    }
    checkpoint_step = _checkpoint_step(health, args.checkpoint_step)
    summary["checkpoint_step"] = checkpoint_step
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    _write_rollout_metrics(args, summary, checkpoint_step)
    print(json.dumps({"event": "evaluation_complete", **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
