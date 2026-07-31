from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OPENPI_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = OPENPI_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import requests
import torch
from tqdm import tqdm

from openpi.rl.libero_adapter import Pi05LiberoRLAdapter


@dataclass
class EpisodeResult:
    success: bool
    env_steps: int
    policy_calls: int
    init_state_index: int
    elapsed_seconds: float
    termination_reason: str


class Pi05BaseClient:
    def __init__(self, server_url: str, timeout: float):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def check_health(self) -> None:
        response = requests.get(f"{self.server_url}/health", timeout=10)
        response.raise_for_status()
        if response.json().get("status") != "ok":
            raise RuntimeError(f"Policy server is unhealthy: {response.json()}")

    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        response = requests.post(
            f"{self.server_url}/sample",
            json={"observation": _jsonable(observation), "mode": "eval", "return_values": False},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if "action" not in payload:
            raise KeyError(f"Policy response is missing action: {sorted(payload)}")
        return np.asarray(payload["action"], dtype=np.float32)


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def run_episode(
    env: Any,
    adapter: Pi05LiberoRLAdapter,
    client: Pi05BaseClient,
    args: argparse.Namespace,
    task_id: int,
    episode_id: int,
) -> EpisodeResult:
    observation = env.reset()
    init_state_index = int(observation.get("init_state_index", -1))
    env_steps = 0
    policy_calls = 0
    started_at = time.monotonic()
    progress = tqdm(
        total=args.max_episode_steps,
        desc=f"task {task_id} episode {episode_id}",
        unit="env_step",
        leave=args.leave_episode_progress,
        dynamic_ncols=True,
    )
    try:
        while env_steps < args.max_episode_steps and policy_calls < args.max_policy_calls:
            inference_started = time.monotonic()
            model_observation = adapter.env_obs_to_model_obs(observation)
            action_chunk = client.infer(model_observation.to_dict())
            inference_seconds = time.monotonic() - inference_started
            policy_calls += 1
            expected_shape = (1, adapter.action_horizon, adapter.action_dim)
            if action_chunk.shape != expected_shape:
                raise ValueError(f"Expected action shape {expected_shape}, got {action_chunk.shape}")
            if not np.isfinite(action_chunk).all():
                raise FloatingPointError("Policy action chunk contains NaN/Inf")

            executed_actions: list[np.ndarray] = []
            success = False
            for model_action in action_chunk[0, : args.execute_horizon]:
                env_action = adapter.action_to_env_action(torch.from_numpy(model_action), observation)
                step_result = env.step(env_action)
                observation = step_result[0]
                executed_actions.append(env_action)
                env_steps += 1
                progress.update(1)
                success = bool(env.check_success())
                if success or env_steps >= args.max_episode_steps:
                    break

            env_actions = np.asarray(executed_actions, dtype=np.float32)
            elapsed = time.monotonic() - started_at
            progress.set_postfix(
                calls=policy_calls,
                success=int(success),
                infer=f"{inference_seconds:.2f}s",
                rate=f"{env_steps / elapsed:.2f}/s" if elapsed else "n/a",
            )
            if policy_calls == 1 or policy_calls % args.log_interval_policy_calls == 0 or success:
                print(
                    json.dumps(
                        {
                            "event": "interaction_diagnostics",
                            "task_suite": "libero_spatial",
                            "task_id": task_id,
                            "episode": episode_id,
                            "init_state_index": init_state_index,
                            "env_steps": env_steps,
                            "max_episode_steps": args.max_episode_steps,
                            "policy_calls": policy_calls,
                            "max_policy_calls": args.max_policy_calls,
                            "progress_percent": round(100.0 * env_steps / args.max_episode_steps, 2),
                            "prompt": adapter._extract_prompt(observation),
                            "observation_keys": sorted(observation),
                            "policy_action_shape": list(action_chunk.shape),
                            "policy_action_min": float(action_chunk.min()),
                            "policy_action_max": float(action_chunk.max()),
                            "policy_action_mean": float(action_chunk.mean()),
                            "env_action_min": float(env_actions.min()),
                            "env_action_max": float(env_actions.max()),
                            "env_action_mean": float(env_actions.mean()),
                            "gripper_actions": env_actions[..., -1].tolist(),
                            "inference_seconds": round(inference_seconds, 4),
                            "episode_elapsed_seconds": round(elapsed, 4),
                            "success": success,
                        }
                    ),
                    flush=True,
                )
            if success:
                return EpisodeResult(True, env_steps, policy_calls, init_state_index, elapsed, "success")

        elapsed = time.monotonic() - started_at
        reason = "max_env_steps" if env_steps >= args.max_episode_steps else "max_policy_calls"
        return EpisodeResult(False, env_steps, policy_calls, init_state_index, elapsed, reason)
    finally:
        progress.close()


def evaluate_task(args: argparse.Namespace, task_id: int, client: Pi05BaseClient) -> dict[str, Any]:
    adapter = Pi05LiberoRLAdapter(
        checkpoint_path=args.checkpoint,
        reference_dir=args.norm_stats_dir,
        assets_dir=args.assets_dir,
        libero_repo_dir=args.libero_repo_dir,
        device="cpu",
        checkpoint_uses_extra_delta_transform=args.extra_delta_transform,
    )
    env = adapter.make_libero_env(
        task_suite_name="libero_spatial",
        task_id=task_id,
        seed=args.seed,
        cycle_init_states=True,
    )
    episodes: list[dict[str, Any]] = []
    task_progress = tqdm(
        range(args.episodes_per_task),
        desc=f"LIBERO Spatial task {task_id}",
        unit="episode",
        position=0,
        dynamic_ncols=True,
    )
    try:
        for episode_id in task_progress:
            episode = run_episode(env, adapter, client, args, task_id, episode_id)
            result = {
                "episode": episode_id,
                "init_state_index": episode.init_state_index,
                "env_steps": episode.env_steps,
                "policy_calls": episode.policy_calls,
                "elapsed_seconds": round(episode.elapsed_seconds, 4),
                "termination_reason": episode.termination_reason,
                "success": episode.success,
            }
            episodes.append(result)
            running_successes = sum(int(row["success"]) for row in episodes)
            running_rate = running_successes / len(episodes)
            task_progress.set_postfix(
                successes=f"{running_successes}/{len(episodes)}",
                success_rate=f"{100.0 * running_rate:.1f}%",
                last_steps=episode.env_steps,
                last_reason=episode.termination_reason,
            )
            print(
                json.dumps(
                    {
                        "event": "episode_complete",
                        "task_suite": "libero_spatial",
                        "task_id": task_id,
                        **result,
                        "task_running_success_rate": running_rate,
                        "task_running_success_rate_percent": 100.0 * running_rate,
                    }
                ),
                flush=True,
            )
    finally:
        task_progress.close()
        with contextlib.suppress(Exception):
            env.close()
    successes = sum(int(row["success"]) for row in episodes)
    return {
        "task_suite": "libero_spatial",
        "task_id": task_id,
        "completed_episodes": len(episodes),
        "successful_episodes": successes,
        "success_rate": successes / len(episodes),
        "success_rate_percent": 100.0 * successes / len(episodes),
        "episodes": episodes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interact with LIBERO Spatial and measure pi0.5 base success rate")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=100,
        help="Maximum LIBERO environment steps in one episode; reaching it counts as failure.",
    )
    parser.add_argument(
        "--max-policy-calls",
        type=int,
        default=20,
        help="Maximum pi0.5 action-chunk inference calls in one episode; reaching it counts as failure.",
    )
    parser.add_argument(
        "--log-interval-policy-calls",
        type=int,
        default=5,
        help="Print interaction/action diagnostics every N policy calls.",
    )
    parser.add_argument(
        "--leave-episode-progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep each completed episode progress bar in the terminal.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--checkpoint", default=str(OPENPI_DIR / "asset_pi05_base/pytorch/model.safetensors"))
    parser.add_argument("--norm-stats-dir", default=str(OPENPI_DIR / "asset_pi05/pytorch/assets"))
    parser.add_argument("--assets-dir", default=str(OPENPI_DIR / "assets"))
    parser.add_argument("--libero-repo-dir", default="/mnt/data/lcx1/yiqinworkspace/AcceRL/LIBERO")
    parser.add_argument(
        "--extra-delta-transform",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use only for checkpoints trained with the legacy extra LIBERO delta transform.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OPENPI_DIR / "logs/pi05_base_libero_spatial/results.json",
    )
    args = parser.parse_args()
    if args.episodes_per_task <= 0 or args.max_episode_steps <= 0 or args.max_policy_calls <= 0:
        parser.error("Episode counts, max steps, and max policy calls must be positive")
    if args.log_interval_policy_calls <= 0:
        parser.error("--log-interval-policy-calls must be positive")
    if not 1 <= args.execute_horizon <= 10:
        parser.error("--execute-horizon must be in [1, 10]")
    if any(not 0 <= task_id < 10 for task_id in args.task_ids):
        parser.error("LIBERO Spatial task IDs must be in [0, 9]")
    return args


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            {
                "event": "evaluation_config",
                "server_url": args.server_url,
                "checkpoint": args.checkpoint,
                "task_suite": "libero_spatial",
                "task_ids": args.task_ids,
                "episodes_per_task": args.episodes_per_task,
                "total_target_episodes": len(args.task_ids) * args.episodes_per_task,
                "execute_horizon": args.execute_horizon,
                "max_episode_steps": args.max_episode_steps,
                "max_policy_calls": args.max_policy_calls,
                "log_interval_policy_calls": args.log_interval_policy_calls,
                "seed": args.seed,
                "output": str(args.output),
            },
            indent=2,
        ),
        flush=True,
    )
    client = Pi05BaseClient(args.server_url, args.request_timeout)
    client.check_health()
    print(json.dumps({"event": "policy_server_health", "status": "ok", "url": args.server_url}), flush=True)
    task_results = [evaluate_task(args, task_id, client) for task_id in args.task_ids]
    completed = sum(result["completed_episodes"] for result in task_results)
    successes = sum(result["successful_episodes"] for result in task_results)
    rate = successes / completed if completed else 0.0
    task_rates = [float(result["success_rate"]) for result in task_results]
    macro_rate = sum(task_rates) / len(task_rates) if task_rates else 0.0
    worst_task_rate = min(task_rates) if task_rates else 0.0
    successful_task_count = sum(rate > 0.0 for rate in task_rates)
    stderr = math.sqrt(rate * (1.0 - rate) / completed) if completed else 0.0
    summary = {
        "checkpoint": args.checkpoint,
        "task_suite": "libero_spatial",
        "completed_episodes": completed,
        "successful_episodes": successes,
        "success_rate": rate,
        "success_rate_percent": 100.0 * rate,
        "macro_success_rate": macro_rate,
        "macro_success_rate_percent": 100.0 * macro_rate,
        "worst_task_success_rate": worst_task_rate,
        "worst_task_success_rate_percent": 100.0 * worst_task_rate,
        "successful_task_count": successful_task_count,
        "total_task_count": len(task_results),
        "approx_95_percent_ci_percent": [
            100.0 * max(0.0, rate - 1.96 * stderr),
            100.0 * min(1.0, rate + 1.96 * stderr),
        ],
        "tasks": task_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "evaluation_complete", **summary}, indent=2), flush=True)
    print(f"Result saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
