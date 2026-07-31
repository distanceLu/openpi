from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
from tqdm import tqdm

from openpi.rl.http_protocol import RolloutUpdateRequest
from openpi.rl.libero_adapter import LIBERO_IMAGE_KEYS, Pi05LiberoRLAdapter
from openpi.rl.libero_http_client import OpenPIServerClient


@dataclass
class RolloutEpisode:
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
    old_velocities: list[Any]
    actions: list[list[np.ndarray]]
    infos: list[dict[str, Any]]
    completed_episodes: int
    successful_episodes: int


@dataclass
class RolloutSummary:
    chunks: int = 0
    env_steps: int = 0
    reward_sum: float = 0.0
    completed_episodes: int = 0
    successful_episodes: int = 0

    def add(self, episode: RolloutEpisode) -> None:
        self.chunks += len(episode.rewards)
        self.env_steps += sum(episode.durations)
        self.reward_sum += sum(episode.rewards)
        self.completed_episodes += episode.completed_episodes
        self.successful_episodes += episode.successful_episodes

    def to_dict(self, interrupted: bool = False) -> dict[str, Any]:
        success_rate = self.successful_episodes / self.completed_episodes if self.completed_episodes else 0.0
        return {
            "event": "rollout_summary",
            "interrupted": interrupted,
            "chunks": self.chunks,
            "env_steps": self.env_steps,
            "reward_sum": self.reward_sum,
            "completed_episodes": self.completed_episodes,
            "successful_episodes": self.successful_episodes,
            "success_rate": success_rate,
            "success_rate_percent": 100.0 * success_rate,
        }


def _scalar_policy_value(value: Any, name: str) -> float:
    tensor = np.asarray(value, dtype=np.float32).reshape(-1)
    if tensor.size != 1:
        raise ValueError(f"{name} must have one value for a single-environment rollout, got shape {tensor.shape}")
    return float(tensor[0])


def _resolve_dated_video_path(video_path: Path, task_suite_name: str, task_id: int, seed: int) -> Path:
    today = dt.date.today().isoformat()
    if video_path.suffix.lower() == ".mp4":
        video_root = video_path.parent
        requested_stem = video_path.stem
    else:
        video_root = video_path
        requested_stem = f"{task_suite_name}_task{task_id}_seed{seed}"

    date_dir = video_root / today
    date_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", requested_stem).strip("._-") or "libero_rollout"
    sequence_pattern = re.compile(r"^(\d+)_.*\.mp4$", re.IGNORECASE)
    existing_sequences = [
        int(match.group(1))
        for path in date_dir.glob("*.mp4")
        if (match := sequence_pattern.match(path.name)) is not None
    ]
    sequence = max(existing_sequences, default=0) + 1
    return date_dir / f"{sequence:03d}_{safe_stem}.mp4"


class LiberoRolloutClient:
    """LIBERO-side SMDP sampler; the OpenPI server owns sampling and PPO updates."""

    def __init__(
        self,
        env: Any,
        adapter: Pi05LiberoRLAdapter,
        policy_client: OpenPIServerClient,
        gamma: float,
        execute_horizon: int | None,
        max_episode_steps: int = 240,
        video_path: Path | None = None,
        video_fps: int = 20,
    ):
        self.env, self.adapter, self.policy_client = env, adapter, policy_client
        self.gamma, self.execute_horizon, self._env_obs = gamma, execute_horizon, None
        self.max_episode_steps, self._episode_steps = max_episode_steps, 0
        self._video_path, self._video_fps, self._video_writer = video_path, video_fps, None

    def __enter__(self) -> "LiberoRolloutClient":
        if self._video_path is not None:
            import imageio.v2 as imageio

            self._video_path.parent.mkdir(parents=True, exist_ok=True)
            self._video_writer = imageio.get_writer(str(self._video_path), fps=self._video_fps, codec="libx264")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._video_writer is not None:
            self._video_writer.close()
            self._video_writer = None
            print({"event": "rollout_video_saved", "path": str(self._video_path)}, flush=True)

    def _record_video_frame(self, observation: dict[str, Any]) -> None:
        if self._video_writer is None:
            return
        frame = self.adapter._extract_image(observation, LIBERO_IMAGE_KEYS)
        if frame is None:
            raise ValueError(f"Cannot record video; no agent-view image in observation keys {list(observation)}")
        self._video_writer.append_data(frame)

    def reset(self) -> Any:
        result = self.env.reset()
        self._env_obs = result[0] if isinstance(result, tuple) else result
        self._episode_steps = 0
        self._record_video_frame(self._env_obs)
        return self._env_obs

    def _step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        result = self.env.step(action)
        if len(result) == 5:
            next_obs, reward, terminated, truncated, info = result
        elif len(result) == 4:
            next_obs, reward, done, info = result
            truncated = bool(info.get("TimeLimit.truncated", False)) if isinstance(info, dict) else False
            terminated = bool(done) and not truncated
        else:
            raise ValueError(f"Expected 4 or 5 env.step values, got {len(result)}")
        info = dict(info) if isinstance(info, dict) else {"info": info}
        success = bool(self.env.check_success()) if hasattr(self.env, "check_success") else bool(info.get("success", False))
        info["success"] = success
        # LIBERO is a sparse-success benchmark. Some wrappers keep reward/done
        # at zero when ignore_done=True, so expose the task predicate directly.
        reward = max(float(reward), float(success))
        terminated = bool(terminated) or success
        self._episode_steps += 1
        truncated = bool(truncated) or self._episode_steps >= self.max_episode_steps
        if truncated:
            info["TimeLimit.truncated"] = not terminated
        return next_obs, reward, terminated, truncated, info

    def run_rollout(self, steps: int, mode: str, target_episodes: int | None = None) -> RolloutEpisode:
        if self._env_obs is None:
            self.reset()
        episode = RolloutEpisode(
            observations=[], chains=[], old_logprobs=[], denoise_indices=[], denoise_timesteps=[], rewards=[],
            terminated=[], truncated=[], values=[], next_values=[], durations=[], old_velocities=[], actions=[], infos=[],
            completed_episodes=0, successful_episodes=0,
        )
        progress = tqdm(range(steps), desc="LIBERO rollout", unit="chunk", leave=False)
        for _ in progress:
            model_obs = self.adapter.env_obs_to_model_obs(self._env_obs)
            policy = self.policy_client.sample(model_obs.to_dict(), mode=mode, return_values=True)
            if "action" not in policy:
                raise ValueError(f"Policy response is missing 'action'; available keys: {sorted(policy)}")
            action_chunk = np.asarray(policy["action"], dtype=np.float32)
            expected_shape = (1, self.adapter.action_horizon, self.adapter.action_dim)
            if action_chunk.shape != expected_shape:
                raise ValueError(f"Expected HTTP policy action {expected_shape}, got {action_chunk.shape}")
            action_chunk = action_chunk[0]
            horizon = action_chunk.shape[0]
            execute_horizon = self.execute_horizon or horizon
            if mode == "train" and execute_horizon != 5:
                raise ValueError(
                    f"RLinf pi0.5 LIBERO training requires execute_horizon=5, got {execute_horizon}; "
                    "changing the outer-action duration changes the checkpoint's training interface."
                )
            if not 1 <= execute_horizon <= horizon:
                raise ValueError(f"execute_horizon must be in [1, {horizon}], got {execute_horizon}")
            chunk_reward, executed_actions, info = 0.0, [], {}
            terminated = truncated = False
            next_obs = self._env_obs
            for action_index in range(execute_horizon):
                env_action = self.adapter.action_to_env_action(torch.as_tensor(action_chunk[action_index]), next_obs)
                next_obs, reward, terminated, truncated, info = self._step(env_action)
                self._record_video_frame(next_obs)
                chunk_reward += self.gamma**action_index * reward
                executed_actions.append(env_action)
                if terminated or truncated:
                    break
            duration = len(executed_actions)
            if duration == 0:
                raise RuntimeError("Action chunk executed zero environment actions")
            if not np.isfinite(action_chunk).all() or not np.isfinite(np.asarray(executed_actions)).all():
                raise FloatingPointError("Policy or environment action contains NaN/Inf")
            if len(episode.rewards) == 0:
                env_actions = np.asarray(executed_actions, dtype=np.float32)
                print(
                    {
                        "event": "rollout_diagnostics",
                        "prompt": self.adapter._extract_prompt(self._env_obs),
                        "observation_keys": sorted(self._env_obs.keys()),
                        "policy_action_shape": action_chunk.shape,
                        "policy_action_min": float(action_chunk.min()),
                        "policy_action_max": float(action_chunk.max()),
                        "policy_action_mean": float(action_chunk.mean()),
                        "env_action_min": float(env_actions.min()),
                        "env_action_max": float(env_actions.max()),
                        "env_action_mean": float(env_actions.mean()),
                        "gripper_actions": env_actions[..., -1].tolist(),
                        "info": info,
                    },
                    flush=True,
                )
            if terminated:
                next_value: Any = 0.0
            else:
                next_model_obs = self.adapter.env_obs_to_model_obs(next_obs)
                next_policy = self.policy_client.sample(next_model_obs.to_dict(), mode=mode, return_values=True)
                next_value = _scalar_policy_value(next_policy["values"], "next policy value")
            episode.observations.append(model_obs.to_dict())
            episode.chains.append(policy["chains"])
            episode.old_logprobs.append(policy["denoise_logprobs"])
            episode.denoise_indices.append(policy["denoise_indices"])
            episode.denoise_timesteps.append(policy["denoise_timesteps"])
            episode.rewards.append(chunk_reward)
            episode.terminated.append(terminated)
            episode.truncated.append(truncated)
            episode.values.append(_scalar_policy_value(policy["values"], "policy value"))
            episode.next_values.append(next_value)
            episode.durations.append(duration)
            episode.old_velocities.append(policy["velocities"])
            episode.actions.append(executed_actions)
            episode.infos.append(info if isinstance(info, dict) else {"info": info})
            if terminated or truncated:
                episode.completed_episodes += 1
                episode.successful_episodes += int(bool(info.get("success", False)))
            step_number = len(episode.rewards)
            progress.set_postfix(
                reward=f"{chunk_reward:.3f}",
                mean_reward=f"{np.mean(episode.rewards):.3f}",
                success=f"{float(info.get('success', 0)):.0f}" if isinstance(info, dict) else "n/a",
            )
            print(
                {
                    "event": "rollout_step",
                    "step": step_number,
                    "progress_percent": round(100.0 * step_number / steps, 2),
                    "reward": chunk_reward,
                    "value": episode.values[-1],
                    "next_value": next_value,
                    "duration": duration,
                    "terminated": terminated,
                    "truncated": truncated,
                },
                flush=True,
            )
            self.reset() if terminated or truncated else setattr(self, "_env_obs", next_obs)
            if target_episodes is not None and episode.completed_episodes >= target_episodes:
                break
        return episode

    def submit_rollout(self, episode: RolloutEpisode) -> dict[str, Any]:
        return self.policy_client.update(RolloutUpdateRequest(
            observations=episode.observations, chains=episode.chains, old_logprobs=episode.old_logprobs,
            denoise_indices=episode.denoise_indices, denoise_timesteps=episode.denoise_timesteps, rewards=episode.rewards,
            terminated=episode.terminated, truncated=episode.truncated, values=episode.values, next_values=episode.next_values,
            durations=episode.durations, old_velocities=episode.old_velocities,
        ))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LIBERO SMDP rollouts against an OpenPI policy server")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=128, help="Maximum action-chunk transitions per rollout.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="In evaluation, stop after this many completed episodes; --steps remains a safety cap.",
    )
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--execute-horizon", type=int, default=None, help="Actions executed from each predicted chunk; RLinf pi0.5 LIBERO uses 5.")
    parser.add_argument("--max-episode-steps", type=int, default=240)
    parser.add_argument(
        "--video-path",
        type=Path,
        default=None,
        help="Optional videos root or MP4 name. Output is stored as ROOT/YYYY-MM-DD/NNN_name.mp4.",
    )
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-path", default="/mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT/model.safetensors")
    parser.add_argument("--reference-dir", default="/mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT")
    parser.add_argument("--assets-dir", default="/mnt/data/lcx1/yiqinworkspace/openpi/assets")
    parser.add_argument("--libero-repo-dir", default="/mnt/data/lcx1/yiqinworkspace/AcceRL/LIBERO")
    parser.add_argument(
        "--checkpoint-uses-extra-delta-transform",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match the checkpoint's action convention; SFT trained with extra_delta_transform=False uses --no-checkpoint-uses-extra-delta-transform.",
    )
    parser.add_argument("--submit", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {args.gamma}")
    if args.max_episode_steps <= 0:
        raise ValueError(f"max_episode_steps must be positive, got {args.max_episode_steps}")
    if args.episodes is not None and args.episodes <= 0:
        raise ValueError(f"episodes must be positive, got {args.episodes}")
    if args.episodes is not None and (args.mode != "eval" or args.submit):
        raise ValueError("--episodes is only supported for evaluation without --submit")
    adapter = Pi05LiberoRLAdapter(
        checkpoint_path=args.checkpoint_path,
        reference_dir=args.reference_dir,
        assets_dir=args.assets_dir,
        libero_repo_dir=args.libero_repo_dir,
        device=args.device,
        checkpoint_uses_extra_delta_transform=args.checkpoint_uses_extra_delta_transform,
    )
    policy_client = OpenPIServerClient(base_url=args.server_url)
    if policy_client.health().get("status") != "ok":
        raise RuntimeError(f"OpenPI server health check failed at {args.server_url}")
    env = adapter.make_libero_env(
        task_suite_name=args.task_suite_name,
        task_id=args.task_id,
        seed=args.seed,
        cycle_init_states=args.mode == "train",
    )
    execute_horizon = args.execute_horizon if args.execute_horizon is not None else 5
    video_path = (
        _resolve_dated_video_path(args.video_path, args.task_suite_name, args.task_id, args.seed)
        if args.video_path is not None
        else None
    )
    if video_path is not None:
        print({"event": "rollout_video_path", "path": str(video_path)}, flush=True)
    summary = RolloutSummary()
    interrupted = False
    client = LiberoRolloutClient(
        env,
        adapter,
        policy_client,
        args.gamma,
        execute_horizon,
        max_episode_steps=args.max_episode_steps,
        video_path=video_path,
        video_fps=args.video_fps,
    )
    try:
        with client:
            if not args.submit:
                episode = client.run_rollout(args.steps, args.mode, target_episodes=args.episodes)
                summary.add(episode)
                print({"chunks": len(episode.rewards), "discounted_reward_sum": sum(episode.rewards), "durations": episode.durations})
                return
            if args.mode == "eval":
                raise ValueError("Evaluation must not submit rollouts for PPO updates; remove --submit.")
            completed = 0
            while args.iterations == 0 or completed < args.iterations:
                episode = client.run_rollout(args.steps, args.mode)
                summary.add(episode)
                response = client.submit_rollout(episode)
                completed += 1
                print({"iteration": completed, "chunks": len(episode.rewards), "discounted_reward_sum": sum(episode.rewards), **response}, flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("Rollout interrupted by user.", flush=True)
    finally:
        print(summary.to_dict(interrupted=interrupted), flush=True)
        with contextlib.suppress(Exception):
            env.close()


if __name__ == "__main__":
    main()
