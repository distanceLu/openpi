from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

OPENPI_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = OPENPI_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import requests
import torch

from openpi.rl.libero_adapter import Pi05LiberoRLAdapter


class BasePolicyClient:
    def __init__(self, server_url: str, timeout: float = 600.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> None:
        response = requests.get(f"{self.server_url}/health", timeout=10)
        response.raise_for_status()
        if response.json().get("status") != "ok":
            raise RuntimeError(f"Unhealthy policy server: {response.json()}")

    def sample(self, observation: dict[str, Any]) -> np.ndarray:
        payload = {"observation": _to_jsonable(observation), "mode": "eval", "return_values": False}
        response = requests.post(f"{self.server_url}/sample", json=payload, timeout=self.timeout)
        response.raise_for_status()
        result = response.json()
        if "action" not in result:
            raise KeyError(f"Policy response has no action: {sorted(result)}")
        return np.asarray(result["action"], dtype=np.float32)


def _to_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def evaluate_task(args: argparse.Namespace, suite: str, task_id: int) -> dict[str, Any]:
    adapter = Pi05LiberoRLAdapter(
        checkpoint_path=args.checkpoint,
        reference_dir=args.norm_stats_dir,
        assets_dir=args.assets_dir,
        libero_repo_dir=args.libero_repo_dir,
        device="cpu",
        checkpoint_uses_extra_delta_transform=args.extra_delta_transform,
    )
    env = adapter.make_libero_env(
        task_suite_name=suite,
        task_id=task_id,
        seed=args.seed,
        cycle_init_states=True,
    )
    client = BasePolicyClient(args.server_url)
    client.health()
    successes = 0
    episode_rows: list[dict[str, Any]] = []
    try:
        for episode in range(args.episodes):
            observation = env.reset()
            success = False
            steps = 0
            while steps < args.max_episode_steps and not success:
                model_observation = adapter.env_obs_to_model_obs(observation)
                action_chunk = client.sample(model_observation.to_dict())
                expected = (1, adapter.action_horizon, adapter.action_dim)
                if action_chunk.shape != expected:
                    raise ValueError(f"Expected policy action {expected}, got {action_chunk.shape}")
                for model_action in action_chunk[0, : args.execute_horizon]:
                    env_action = adapter.action_to_env_action(torch.from_numpy(model_action), observation)
                    step_result = env.step(env_action)
                    observation = step_result[0]
                    steps += 1
                    success = bool(env.check_success())
                    if success or steps >= args.max_episode_steps:
                        break
            successes += int(success)
            row = {
                "episode": episode,
                "init_state_index": int(observation.get("init_state_index", -1)),
                "steps": steps,
                "success": success,
            }
            episode_rows.append(row)
            print(json.dumps({"event": "episode_complete", "suite": suite, "task_id": task_id, **row}), flush=True)
    finally:
        with contextlib.suppress(Exception):
            env.close()
    return {
        "task_suite": suite,
        "task_id": task_id,
        "completed_episodes": args.episodes,
        "successful_episodes": successes,
        "success_rate": successes / args.episodes,
        "success_rate_percent": 100.0 * successes / args.episodes,
        "episodes": episode_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the untouched pi0.5 base checkpoint on LIBERO")
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--task-suite", required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=240)
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=str(OPENPI_DIR / "asset_pi05_base/pytorch/model.safetensors"))
    parser.add_argument("--norm-stats-dir", default=str(OPENPI_DIR / "RLinf-Pi05-LIBERO-SFT"))
    parser.add_argument("--assets-dir", default=str(OPENPI_DIR / "assets"))
    parser.add_argument("--libero-repo-dir", default="/mnt/data/lcx1/yiqinworkspace/AcceRL/LIBERO")
    parser.add_argument("--extra-delta-transform", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_episode_steps <= 0:
        parser.error("--episodes and --max-episode-steps must be positive")
    if not 1 <= args.execute_horizon <= 10:
        parser.error("--execute-horizon must be between 1 and 10")
    return args


def main() -> None:
    args = parse_args()
    results = [evaluate_task(args, args.task_suite, task_id) for task_id in args.task_ids]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(result) for result in results) + "\n")
    print(json.dumps({"event": "worker_complete", "output": str(args.output), "tasks": len(results)}), flush=True)


if __name__ == "__main__":
    main()
