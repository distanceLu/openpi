"""AcceRL asynchronous PPO runner adapted to pi0.5 nested denoising MDPs.

The outer MDP is a LIBERO environment transition.  The inner MDP is the full
pi0.5 denoising chain that produces one action chunk.  Rewards and GAE belong
to outer transitions; PPO log-probability ratios belong to saved inner paths.

The runner retains AcceRL's asynchronous actor/learner split while bounding
off-policy drift through fresh replay sampling, behavior-policy KL, current
value recomputation, and actor/critic learning-rate schedules.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import time
from typing import TYPE_CHECKING, Any

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAY_DEDUP_LOGS", "0")

from accerl_math import compute_smdp_gae
from accerl_math import warmup_cosine_lr_scale
import numpy as np
from pi05_ds_com import InferenceActorCom
from pi05_ds_com import TrainerActorCom
from pi05_utils import Pi05AcceRLConfig
from pi05_utils import build_pi05_actor_critic
from pi05_utils import create_adapter
from pi05_utils import postprocess_action_chunk
from pi05_utils import prepare_inputs_batch
from pi05_utils import prepare_one_obs
from pi05_utils import run_rollout_forward
from pi05_utils import run_training_forward
from pi05_utils import split_parameter_groups
from pi05_utils import trainable_state_dict
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

from openpi.rl.pi05_denoising import normal_entropy
from openpi.rl.pi05_losses import compute_diagonal_gaussian_kl_loss
from openpi.rl.pi05_losses import compute_pi05_ppo_loss
from openpi.rl.pi05_losses import compute_value_loss

if TYPE_CHECKING:
    from openpi.models import model as model_types


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AcceRL pi0.5 nested-MDP PPO on LIBERO")
    repo_root = Path(__file__).resolve().parents[3]
    workspace_root = repo_root.parent
    pi05_checkpoint_dir = repo_root / "asset_pi05" / "pytorch"

    parser.add_argument(
        "--checkpoint-path",
        default=str(pi05_checkpoint_dir),
    )
    parser.add_argument("--lora-adapter-path", default=None)
    parser.add_argument("--reference-dir", default=str(pi05_checkpoint_dir))
    parser.add_argument("--assets-dir", default=str(pi05_checkpoint_dir / "assets"))
    parser.add_argument("--libero-repo-dir", default=str(workspace_root / "AcceRL" / "LIBERO"))
    parser.add_argument("--train-config-name", default="pi05_libero")
    parser.add_argument(
        "--checkpoint-uses-extra-delta-transform",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Whether this checkpoint was trained with the additional LIBERO delta-action transform. "
            "The default asset_pi05/pytorch checkpoint uses the standard openpi convention (False)."
        ),
    )

    parser.add_argument(
        "--benchmark",
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
    )
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--action-chunk-steps", type=int, default=1)

    parser.add_argument("--num-trainer-gpus", type=int, default=1)
    parser.add_argument("--num-inference-actors", type=int, default=1)
    parser.add_argument("--num-rollout-workers", type=int, default=1)
    parser.add_argument(
        "--rollout-python",
        default=str(workspace_root / "clone_env_smoke_test" / "rlinf_env" / "bin" / "python"),
        help="Python executable used by LIBERO RolloutWorkerActor processes.",
    )
    parser.add_argument("--inference-batch", type=int, default=8)
    parser.add_argument("--inference-timeout-ms", type=int, default=10)
    parser.add_argument("--num-cpus", type=int, default=None)
    parser.add_argument("--object-store-memory-gb", type=float, default=0.0)
    parser.add_argument("--broadcast-group-name", default="pi05_policy_broadcast")

    parser.add_argument("--num-denoise-steps", type=int, default=10)
    parser.add_argument(
        "--sample-method",
        choices=["flow_noise"],
        default="flow_noise",
        help="PPO uses flow_noise; deterministic evaluation automatically uses flow_ode.",
    )
    parser.add_argument("--noise-level", type=float, default=0.5)
    parser.add_argument(
        "--compute-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--trainable-scope",
        choices=["rl_heads_only", "lora_and_heads", "action_expert_and_heads", "full_model"],
        default="action_expert_and_heads",
    )

    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--rollout-local-buf", type=int, default=8)
    parser.add_argument("--replay-capacity", type=int, default=512)
    parser.add_argument(
        "--max-policy-lag",
        type=int,
        default=8,
        help="Discard trajectories containing samples older than this many policy versions; 0 disables filtering.",
    )
    parser.add_argument(
        "--max-sample-reuse",
        type=int,
        default=1,
        help="Maximum training selections per trajectory; 0 allows unlimited replay.",
    )
    parser.add_argument("--train-iters", type=int, default=1000)
    parser.add_argument("--policy-lr", type=float, default=1e-5)
    parser.add_argument("--value-lr", type=float, default=1e-4)
    parser.add_argument("--policy-warmup-steps", type=int, default=50)
    parser.add_argument("--value-warmup-steps", type=int, default=50)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--path-logprob-reduce", choices=["mean", "sum"], default="sum")
    parser.add_argument("--log-ratio-clip", type=float, default=20.0)
    parser.add_argument(
        "--recompute-value",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recompute current V(s) and bootstrap values before GAE.",
    )
    parser.add_argument("--use-bf16", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--exp-name", default="pi05_accerl_v1")
    parser.add_argument("--log-dir", default=str(repo_root / "runs" / "AcceRL_pi05"))
    parser.add_argument("--ckpt-dir", default=str(repo_root / "checkpoints" / "AcceRL_pi05"))
    parser.add_argument("--ckpt-every-steps", type=int, default=50)
    parser.add_argument("--log-every-steps", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_pi05_cfg(args: argparse.Namespace) -> Pi05AcceRLConfig:
    cfg = Pi05AcceRLConfig(
        checkpoint_path=args.checkpoint_path,
        lora_adapter_path=args.lora_adapter_path,
        reference_dir=args.reference_dir,
        assets_dir=args.assets_dir,
        libero_repo_dir=args.libero_repo_dir,
        checkpoint_uses_extra_delta_transform=args.checkpoint_uses_extra_delta_transform,
        train_config_name=args.train_config_name,
        num_denoise_steps=args.num_denoise_steps,
        sample_method=args.sample_method,
        noise_level=args.noise_level,
        compute_dtype=args.compute_dtype,
        trainable_scope=args.trainable_scope,
        action_chunk_steps=args.action_chunk_steps,
    )
    cfg.validate()
    return cfg


@dataclass
class Trajectory:
    """A sequence of outer-MDP transitions and their inner denoise paths."""

    observations: list[model_types.Observation[torch.Tensor]]
    chains: np.ndarray
    old_logprobs: np.ndarray
    old_means: np.ndarray
    old_stds: np.ndarray
    denoise_timesteps: np.ndarray
    denoise_indices: np.ndarray
    old_velocities: np.ndarray
    rewards: np.ndarray
    old_values: np.ndarray
    durations: np.ndarray
    bootstrap_value: float
    bootstrap_observation: model_types.Observation[torch.Tensor] | None
    is_terminal: bool
    policy_versions: np.ndarray
    insert_times_ms: np.ndarray
    sample_uses: int = 0

    @property
    def num_steps(self) -> int:
        return len(self.rewards)


@ray.remote
class StatsActor:
    def __init__(self, window_size: int = 100):
        self.returns = deque(maxlen=window_size)
        self.lengths = deque(maxlen=window_size)
        self.successes = deque(maxlen=window_size)
        self.total_episodes = 0
        self.total_env_steps = 0

    def add_episode(self, episode_return: float, episode_length: int, success: bool) -> None:
        self.returns.append(float(episode_return))
        self.lengths.append(int(episode_length))
        self.successes.append(float(success))
        self.total_episodes += 1
        self.total_env_steps += int(episode_length)

    def get_stats(self) -> dict[str, float]:
        return {
            "rollout/return_mean": float(np.mean(self.returns)) if self.returns else 0.0,
            "rollout/episode_length_mean": float(np.mean(self.lengths)) if self.lengths else 0.0,
            "rollout/success_rate": float(np.mean(self.successes)) if self.successes else 0.0,
            "rollout/total_episodes": float(self.total_episodes),
            "rollout/total_env_steps": float(self.total_env_steps),
        }


@ray.remote
class ReplayBufferActor:
    """Freshness-aware trajectory replay for asynchronous PPO."""

    def __init__(self, capacity: int):
        # Keep capacity in trajectories, matching ds_libero_ppo_discrete.py.
        self.trajectories: deque[Trajectory] = deque(maxlen=int(capacity))
        self.total_added = 0
        self.total_sampled = 0
        self.total_stale_discarded = 0

    def add_trajectory(self, trajectory: Trajectory) -> None:
        if trajectory.num_steps <= 0:
            return
        self.trajectories.append(trajectory)
        self.total_added += 1

    def total_steps(self) -> int:
        return sum(trajectory.num_steps for trajectory in self.trajectories)

    def size(self) -> int:
        return len(self.trajectories)

    def sample_trajectories(
        self,
        minimum_steps: int,
        current_policy_version: int,
        max_policy_lag: int,
        max_sample_reuse: int,
    ) -> list[Trajectory]:
        if not self.trajectories:
            return []

        def is_fresh(trajectory: Trajectory) -> bool:
            if max_policy_lag <= 0:
                return True
            oldest_version = int(np.min(trajectory.policy_versions))
            return current_policy_version - oldest_version <= max_policy_lag

        fresh: list[Trajectory] = []
        for trajectory in self.trajectories:
            if is_fresh(trajectory):
                fresh.append(trajectory)
            else:
                self.total_stale_discarded += 1
        self.trajectories = deque(fresh, maxlen=self.trajectories.maxlen)
        if not fresh:
            return []

        selected: list[Trajectory] = []
        selected_steps = 0
        # Prefer newest data; random order is actively harmful when the learner
        # can update faster than rollout workers produce trajectories.
        candidates = sorted(fresh, key=lambda item: int(np.max(item.insert_times_ms)), reverse=True)
        for trajectory in candidates:
            selected.append(trajectory)
            selected_steps += trajectory.num_steps
            if selected_steps >= minimum_steps:
                break
        if selected_steps < minimum_steps:
            return []

        selected_ids = {id(trajectory) for trajectory in selected}
        retained: list[Trajectory] = []
        for trajectory in self.trajectories:
            if id(trajectory) not in selected_ids:
                retained.append(trajectory)
                continue
            trajectory.sample_uses += 1
            self.total_sampled += 1
            if max_sample_reuse <= 0 or trajectory.sample_uses < max_sample_reuse:
                retained.append(trajectory)
        self.trajectories = deque(retained, maxlen=self.trajectories.maxlen)
        return selected

    def get_stats(self) -> dict[str, float]:
        return {
            "replay/trajectories": float(len(self.trajectories)),
            "replay/outer_steps": float(self.total_steps()),
            "replay/total_added": float(self.total_added),
            "replay/total_sampled": float(self.total_sampled),
            "replay/stale_discarded": float(self.total_stale_discarded),
        }


def _reset_env(env: Any) -> dict[str, Any]:
    result = env.reset()
    return result[0] if isinstance(result, tuple) else result


def _step_env(env: Any, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
    result = env.step(action)
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, float(reward), bool(terminated), bool(truncated), info
    observation, reward, done, info = result
    return observation, float(reward), bool(done), False, info


@ray.remote(num_gpus=1)
class InferenceActor(InferenceActorCom):
    def __init__(
        self,
        actor_id: int,
        cfg: Pi05AcceRLConfig,
        inference_batch: int,
        inference_timeout_ms: int,
    ):
        super().__init__()
        self.actor_id = actor_id
        self.cfg = cfg
        self.adapter, self.model, self.nested_mdp = build_pi05_actor_critic(cfg, device="cuda")
        self.model.eval()
        self.policy_version = 0
        self.batch_size = int(inference_batch)
        self.timeout_seconds = float(inference_timeout_ms) / 1000.0
        self.requests: list[tuple[model_types.Observation[torch.Tensor], bool]] = []
        self.promises: list[asyncio.Future[dict[str, Any]]] = []
        self.last_process_time = time.monotonic()
        self.background_task = asyncio.get_event_loop().create_task(self._batch_loop())
        self.background_task.add_done_callback(self._on_background_task_done)
        print(
            f"InferenceActor {actor_id}: pi0.5 loaded on {ray.get_gpu_ids()}, "
            f"dynamic_batch={self.batch_size}, timeout_ms={inference_timeout_ms}",
            flush=True,
        )

    async def request(
        self,
        observation: model_types.Observation[torch.Tensor],
        deterministic: bool = False,
    ) -> dict[str, Any]:
        promise = asyncio.get_running_loop().create_future()
        self.requests.append((observation, deterministic))
        self.promises.append(promise)
        return await promise

    async def _batch_loop(self) -> None:
        while True:
            ready = self.requests and (
                len(self.requests) >= self.batch_size
                or time.monotonic() - self.last_process_time >= self.timeout_seconds
            )
            if not ready:
                await asyncio.sleep(0.0005)
                continue

            batch_count = min(len(self.requests), self.batch_size)
            requests = self.requests[:batch_count]
            promises = self.promises[:batch_count]
            del self.requests[:batch_count]
            del self.promises[:batch_count]
            self.last_process_time = time.monotonic()
            try:
                results = self._run_dynamic_batch(requests)
                for promise, result in zip(promises, results, strict=True):
                    if not promise.done():
                        promise.set_result(result)
            except Exception as exc:
                for promise in promises:
                    if not promise.done():
                        promise.set_exception(exc)

    def _run_dynamic_batch(
        self,
        requests: list[tuple[model_types.Observation[torch.Tensor], bool]],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any] | None] = [None] * len(requests)
        policy_version = self.policy_version
        for deterministic in (False, True):
            request_indices = [index for index, (_, flag) in enumerate(requests) if bool(flag) == deterministic]
            if not request_indices:
                continue
            observation_batch = prepare_inputs_batch(
                self.adapter,
                [requests[index][0] for index in request_indices],
            )
            rollout = run_rollout_forward(
                self.nested_mdp,
                observation_batch,
                deterministic=deterministic,
                return_values=True,
                torch_dtype=self.cfg.compute_dtype,
            )
            for batch_index, request_index in enumerate(request_indices):
                results[request_index] = self._rollout_result(rollout, batch_index, policy_version)
        if any(result is None for result in results):
            raise RuntimeError("Dynamic pi0.5 inference batch produced an incomplete result set")
        return [result for result in results if result is not None]

    @staticmethod
    def _rollout_result(rollout: Any, batch_index: int, policy_version: int) -> dict[str, Any]:
        return {
            "normalized_actions": rollout.actions[batch_index].cpu().float().numpy(),
            "chains": rollout.chains[batch_index].cpu().float().numpy(),
            "old_logprobs": rollout.denoise_logprobs[batch_index].cpu().float().numpy(),
            "old_means": rollout.denoise_means[batch_index].cpu().float().numpy(),
            "old_stds": rollout.denoise_stds[batch_index].cpu().float().numpy(),
            "denoise_timesteps": rollout.denoise_timesteps[batch_index].cpu().float().numpy(),
            "denoise_indices": rollout.denoise_indices[batch_index].cpu().numpy(),
            "old_velocities": rollout.velocities[batch_index].cpu().float().numpy(),
            "value": float(rollout.values[batch_index].cpu().item()),
            "policy_version": policy_version,
        }

    @staticmethod
    def _on_background_task_done(task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        exception = task.exception()
        if exception is not None:
            print(f"InferenceActor dynamic batch loop failed: {exception}", flush=True)

    async def value(self, observation: model_types.Observation[torch.Tensor]) -> float:
        with torch.no_grad():
            return float(self.nested_mdp.compute_value(observation)[0].cpu().item())

    def receive_and_update_weights(self, group_name: str) -> None:
        super().receive_and_update_weights(group_name)
        self.policy_version += 1

    def align_broadcast_dtypes(self, trainer_signature: list[tuple[str, str, tuple[int, ...], str]]) -> None:
        """Match inference parameter dtypes after DeepSpeed applies BF16 casting."""

        parameters = dict(self.model.named_parameters(recurse=True))
        for tensor_kind, name, shape, dtype_name in trainer_signature:
            if tensor_kind != "param":
                continue
            parameter = parameters.get(name)
            if parameter is None or tuple(parameter.shape) != tuple(shape):
                raise RuntimeError(f"Inference broadcast parameter does not match Trainer: {name}")
            dtype = getattr(torch, dtype_name.removeprefix("torch."), None)
            if not isinstance(dtype, torch.dtype):
                raise TypeError(f"Unsupported Trainer broadcast dtype: {dtype_name}")
            if parameter.dtype != dtype:
                parameter.data = parameter.data.to(dtype=dtype)

    def get_model_summary(self) -> dict[str, Any]:
        return {
            "parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "trainable": sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad),
            "policy_version": self.policy_version,
        }


@ray.remote
class RolloutWorkerActor:
    def __init__(
        self,
        inference_actor: Any,
        replay_buffer: Any,
        stats_actor: Any,
        worker_id: int,
        cfg: Pi05AcceRLConfig,
        benchmark: str,
        task_id: int,
        rollout_local_buf: int,
        max_episode_steps: int,
        gamma: float,
        seed: int,
    ):
        self.inference_actor = inference_actor
        self.replay_buffer = replay_buffer
        self.stats_actor = stats_actor
        self.worker_id = worker_id
        self.cfg = cfg
        self.rollout_local_buf = rollout_local_buf
        self.max_episode_steps = max_episode_steps
        self.gamma = gamma
        self.adapter = create_adapter(cfg, device="cpu")
        self.env = self.adapter.make_libero_env(
            task_suite_name=benchmark,
            task_id=task_id,
            seed=seed + worker_id,
            cycle_init_states=True,
        )
        self.task_description = self.adapter.default_prompt

    def run(self) -> None:
        np.random.seed(self.worker_id)
        while True:
            observation = _reset_env(self.env)
            episode_return = 0.0
            episode_length = 0
            success = False
            segment: list[dict[str, Any]] = []

            while episode_length < self.max_episode_steps:
                model_observation = prepare_one_obs(self.adapter, observation, self.task_description)
                inference = ray.get(self.inference_actor.request.remote(model_observation, False))
                env_actions = postprocess_action_chunk(
                    self.adapter,
                    inference["normalized_actions"],
                    observation,
                    self.cfg.action_chunk_steps,
                )

                discounted_chunk_reward = 0.0
                raw_chunk_reward = 0.0
                terminated = False
                truncated = False
                duration = 0
                next_observation = observation
                for env_action in env_actions:
                    next_observation, reward, terminated, truncated, info = _step_env(self.env, env_action)
                    discounted_chunk_reward += (self.gamma**duration) * reward
                    raw_chunk_reward += reward
                    duration += 1
                    episode_length += 1
                    success = success or bool(info.get("success", reward > 0.0))
                    if terminated or truncated or episode_length >= self.max_episode_steps:
                        break

                episode_return += raw_chunk_reward
                segment.append(
                    {
                        "observation": model_observation,
                        "chains": inference["chains"],
                        "old_logprobs": inference["old_logprobs"],
                        "old_means": inference["old_means"],
                        "old_stds": inference["old_stds"],
                        "denoise_timesteps": inference["denoise_timesteps"],
                        "denoise_indices": inference["denoise_indices"],
                        "old_velocities": inference["old_velocities"],
                        "reward": discounted_chunk_reward,
                        "old_value": inference["value"],
                        "duration": duration,
                        "policy_version": inference["policy_version"],
                        "insert_time_ms": int(time.time() * 1000),
                    }
                )
                observation = next_observation
                episode_done = terminated or truncated or episode_length >= self.max_episode_steps

                if episode_done:
                    # True termination has no bootstrap. Gym truncation and
                    # our time limit remain valid MDP states and must bootstrap.
                    is_terminal = bool(terminated)
                    bootstrap_observation = None
                    bootstrap_value = 0.0
                    if not is_terminal:
                        bootstrap_observation = prepare_one_obs(self.adapter, next_observation, self.task_description)
                        bootstrap_value = ray.get(self.inference_actor.value.remote(bootstrap_observation))
                    self.replay_buffer.add_trajectory.remote(
                        _pack_trajectory(
                            segment,
                            bootstrap_value=bootstrap_value,
                            bootstrap_observation=bootstrap_observation,
                            is_terminal=is_terminal,
                        )
                    )
                    segment = []
                elif len(segment) == self.rollout_local_buf + 1:
                    # Match the original AcceRL overlap rule: the newest sample's
                    # rollout value bootstraps the preceding segment, then that
                    # sample is retained as the first item of the next segment.
                    bootstrap_value = float(segment[-1]["old_value"])
                    self.replay_buffer.add_trajectory.remote(
                        _pack_trajectory(
                            segment[:-1],
                            bootstrap_value=bootstrap_value,
                            bootstrap_observation=segment[-1]["observation"],
                            is_terminal=False,
                        )
                    )
                    segment = [segment[-1]]

                if episode_done:
                    break

            self.stats_actor.add_episode.remote(episode_return, episode_length, success)


def _pack_trajectory(
    segment: list[dict[str, Any]],
    bootstrap_value: float,
    bootstrap_observation: model_types.Observation[torch.Tensor] | None,
    is_terminal: bool,
) -> Trajectory:
    return Trajectory(
        observations=[item["observation"] for item in segment],
        chains=np.stack([item["chains"] for item in segment]).astype(np.float32),
        old_logprobs=np.stack([item["old_logprobs"] for item in segment]).astype(np.float32),
        old_means=np.stack([item["old_means"] for item in segment]).astype(np.float32),
        old_stds=np.stack([item["old_stds"] for item in segment]).astype(np.float32),
        denoise_timesteps=np.stack([item["denoise_timesteps"] for item in segment]).astype(np.float32),
        denoise_indices=np.stack([item["denoise_indices"] for item in segment]).astype(np.int64),
        old_velocities=np.stack([item["old_velocities"] for item in segment]).astype(np.float32),
        rewards=np.asarray([item["reward"] for item in segment], dtype=np.float32),
        old_values=np.asarray([item["old_value"] for item in segment], dtype=np.float32),
        durations=np.asarray([item["duration"] for item in segment], dtype=np.int64),
        bootstrap_value=float(bootstrap_value),
        bootstrap_observation=bootstrap_observation,
        is_terminal=bool(is_terminal),
        policy_versions=np.asarray([item["policy_version"] for item in segment], dtype=np.int64),
        insert_times_ms=np.asarray([item["insert_time_ms"] for item in segment], dtype=np.int64),
    )


@ray.remote(num_gpus=1)
class TrainerActor(TrainerActorCom):
    def __init__(
        self,
        rank: int,
        world_size: int,
        replay_buffer: Any,
        cfg: Pi05AcceRLConfig,
        train_batch_size: int,
        accumulation_steps: int,
        policy_lr: float,
        value_lr: float,
        policy_warmup_steps: int,
        value_warmup_steps: int,
        lr_schedule: str,
        train_iters: int,
        gamma: float,
        gae_lambda: float,
        clip_eps: float,
        value_coef: float,
        entropy_coef: float,
        kl_coef: float,
        path_logprob_reduce: str,
        log_ratio_clip: float,
        recompute_value: bool,
        max_policy_lag: int,
        max_sample_reuse: int,
        use_bf16: bool,
    ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.replay_buffer = replay_buffer
        self.cfg = cfg
        self.train_batch_size = train_batch_size
        self.accumulation_steps = accumulation_steps
        self.super_batch_size = train_batch_size * accumulation_steps
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.policy_warmup_steps = policy_warmup_steps
        self.value_warmup_steps = value_warmup_steps
        self.lr_schedule = lr_schedule
        self.train_iters = train_iters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
        self.path_logprob_reduce = path_logprob_reduce
        self.log_ratio_clip = log_ratio_clip
        self.recompute_value = recompute_value
        self.max_policy_lag = max_policy_lag
        self.max_sample_reuse = max_sample_reuse
        self.use_bf16 = use_bf16
        self.global_step = 0
        self.policy_version = 0
        self.model = None
        self.base_model = None
        self.adapter = None
        self.nested_mdp = None

    def get_node_ip(self) -> str:
        return ray.util.get_node_ip_address()

    def setup_deepspeed_group(self, master_addr: str, master_port: int) -> dict[str, int]:
        import deepspeed

        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["LOCAL_RANK"] = "0"
        deepspeed.init_distributed(dist_backend="nccl")

        self.adapter, self.base_model, self.nested_mdp = build_pi05_actor_critic(self.cfg, device="cuda")
        parameter_groups = split_parameter_groups(self.base_model, self.policy_lr, self.value_lr)
        ds_config = {
            "train_micro_batch_size_per_gpu": self.train_batch_size,
            "gradient_accumulation_steps": self.accumulation_steps,
            "optimizer": {"type": "AdamW", "params": {"betas": [0.9, 0.95], "eps": 1e-8}},
            "bf16": {"enabled": self.use_bf16},
            "fp16": {"enabled": False},
            "zero_optimization": {
                "stage": 2,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_scatter": True,
            },
            "gradient_clipping": 1.0,
        }
        self.model, self.optimizer, _, _ = deepspeed.initialize(
            model=self.base_model,
            model_parameters=parameter_groups,
            config=ds_config,
        )
        summary = {
            "parameters": sum(parameter.numel() for parameter in self.base_model.parameters()),
            "trainable": sum(
                parameter.numel() for parameter in self.base_model.parameters() if parameter.requires_grad
            ),
        }
        print(f"TrainerActor {self.rank}: DeepSpeed pi0.5 initialized: {summary}", flush=True)
        return summary

    def run_training_epoch(self) -> dict[str, float]:
        trajectories: list[Trajectory] = []
        while not trajectories:
            trajectories = ray.get(
                self.replay_buffer.sample_trajectories.remote(
                    self.super_batch_size,
                    self.policy_version,
                    self.max_policy_lag,
                    self.max_sample_reuse,
                )
            )
            if trajectories:
                break
            time.sleep(0.5)
        batch = self._flatten_trajectories(trajectories)
        metrics = self._train_super_batch(batch)
        metrics["global_step"] = float(self.global_step)
        metrics["policy_version"] = float(self.policy_version)
        return metrics

    def _compute_current_values(self, observations: list[model_types.Observation[torch.Tensor]]) -> np.ndarray:
        if not observations:
            return np.empty((0,), dtype=np.float32)
        values: list[np.ndarray] = []
        was_training = self.base_model.training
        self.base_model.eval()
        with torch.no_grad():
            for start in range(0, len(observations), self.train_batch_size):
                observation_batch = prepare_inputs_batch(
                    self.adapter, observations[start : start + self.train_batch_size]
                )
                batch_values = self.nested_mdp.compute_value(observation_batch)
                values.append(batch_values.detach().cpu().float().numpy())
        if was_training:
            self.base_model.train()
        return np.concatenate(values, axis=0).astype(np.float32)

    def _compute_gae(
        self,
        trajectory: Trajectory,
        values: np.ndarray | None = None,
        bootstrap_value: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        selected_values = trajectory.old_values if values is None else values
        selected_bootstrap = trajectory.bootstrap_value if bootstrap_value is None else bootstrap_value
        return compute_smdp_gae(
            trajectory.rewards,
            selected_values,
            trajectory.durations,
            bootstrap_value=selected_bootstrap,
            is_terminal=trajectory.is_terminal,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

    def _flatten_trajectories(self, trajectories: list[Trajectory]) -> dict[str, Any]:
        observations: list[model_types.Observation[torch.Tensor]] = []
        arrays: dict[str, list[np.ndarray]] = {
            "chains": [],
            "old_logprobs": [],
            "old_means": [],
            "old_stds": [],
            "denoise_timesteps": [],
            "denoise_indices": [],
            "old_velocities": [],
            "policy_versions": [],
            "insert_times_ms": [],
            "advantages": [],
            "returns": [],
        }
        current_values: list[np.ndarray] | None = None
        current_bootstraps: list[float] | None = None
        if self.recompute_value:
            value_observations = [observation for trajectory in trajectories for observation in trajectory.observations]
            flat_values = self._compute_current_values(value_observations)
            current_values = []
            offset = 0
            for trajectory in trajectories:
                current_values.append(flat_values[offset : offset + trajectory.num_steps])
                offset += trajectory.num_steps

            bootstrap_observations = [
                trajectory.bootstrap_observation
                for trajectory in trajectories
                if not trajectory.is_terminal and trajectory.bootstrap_observation is not None
            ]
            flat_bootstraps = self._compute_current_values(bootstrap_observations)
            bootstrap_index = 0
            current_bootstraps = []
            for trajectory in trajectories:
                if trajectory.is_terminal:
                    current_bootstraps.append(0.0)
                elif trajectory.bootstrap_observation is not None:
                    current_bootstraps.append(float(flat_bootstraps[bootstrap_index]))
                    bootstrap_index += 1
                else:
                    current_bootstraps.append(float(trajectory.bootstrap_value))

        for trajectory_index, trajectory in enumerate(trajectories):
            values = None if current_values is None else current_values[trajectory_index]
            bootstrap = None if current_bootstraps is None else current_bootstraps[trajectory_index]
            advantages, returns = self._compute_gae(trajectory, values, bootstrap)
            observations.extend(trajectory.observations)
            arrays["chains"].append(trajectory.chains)
            arrays["old_logprobs"].append(trajectory.old_logprobs)
            arrays["old_means"].append(trajectory.old_means)
            arrays["old_stds"].append(trajectory.old_stds)
            arrays["denoise_timesteps"].append(trajectory.denoise_timesteps)
            arrays["denoise_indices"].append(trajectory.denoise_indices)
            arrays["old_velocities"].append(trajectory.old_velocities)
            arrays["policy_versions"].append(trajectory.policy_versions)
            arrays["insert_times_ms"].append(trajectory.insert_times_ms)
            arrays["advantages"].append(advantages)
            arrays["returns"].append(returns)

        limit = self.super_batch_size
        flattened = {name: np.concatenate(values, axis=0)[:limit] for name, values in arrays.items()}
        flattened["observations"] = observations[:limit]
        return flattened

    def _update_learning_rates(self) -> dict[str, float]:
        if self.lr_schedule == "constant":
            policy_scale = value_scale = 1.0
        else:
            policy_scale = warmup_cosine_lr_scale(self.global_step, self.policy_warmup_steps, self.train_iters)
            value_scale = warmup_cosine_lr_scale(self.global_step, self.value_warmup_steps, self.train_iters)
        learning_rates = {
            "optim/policy_lr": self.policy_lr * policy_scale,
            "optim/value_lr": self.value_lr * value_scale,
        }
        for parameter_group in self.optimizer.param_groups:
            name = parameter_group.get("name")
            if name == "policy":
                parameter_group["lr"] = learning_rates["optim/policy_lr"]
            elif name == "value":
                parameter_group["lr"] = learning_rates["optim/value_lr"]
        return learning_rates

    def _train_super_batch(self, batch: dict[str, Any]) -> dict[str, float]:
        device = next(self.base_model.parameters()).device
        count = len(batch["observations"])
        learning_rates = self._update_learning_rates()
        order = np.random.permutation(count)
        advantages = torch.from_numpy(batch["advantages"]).to(device=device, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)

        metric_rows: list[dict[str, float]] = []
        # Rollout actors keep the behavior model in eval mode. Gradients still
        # flow in eval mode, and matching dropout/mode semantics is essential
        # for a meaningful current-vs-behavior likelihood ratio.
        self.base_model.eval()
        for start in range(0, count, self.train_batch_size):
            indices_np = order[start : start + self.train_batch_size]
            if len(indices_np) != self.train_batch_size:
                break
            indices = torch.as_tensor(indices_np, device=device, dtype=torch.long)
            observation_batch = prepare_inputs_batch(
                self.adapter,
                [batch["observations"][index] for index in indices_np.tolist()],
            )
            chains = torch.from_numpy(batch["chains"][indices_np]).to(device=device)
            old_logprobs = torch.from_numpy(batch["old_logprobs"][indices_np]).to(device=device)
            old_means = torch.from_numpy(batch["old_means"][indices_np]).to(device=device)
            old_stds = torch.from_numpy(batch["old_stds"][indices_np]).to(device=device)
            denoise_timesteps = torch.from_numpy(batch["denoise_timesteps"][indices_np]).to(device=device)
            denoise_indices = torch.from_numpy(batch["denoise_indices"][indices_np]).to(device=device)
            returns = torch.from_numpy(batch["returns"][indices_np]).to(device=device)

            recomputed = run_training_forward(
                self.nested_mdp,
                observation_batch,
                chains=chains,
                denoise_indices=denoise_indices,
                denoise_timesteps=denoise_timesteps,
                return_values=True,
                torch_dtype=self.cfg.compute_dtype,
            )
            entropy = normal_entropy(recomputed.stds)
            policy_loss, ppo_metrics = compute_pi05_ppo_loss(
                new_logprobs=recomputed.logprobs,
                old_logprobs=old_logprobs,
                advantages=advantages[indices],
                clip_eps=self.clip_eps,
                normalize_advantages=False,
                entropy=entropy,
                mode="path",
                log_ratio_clip=self.log_ratio_clip,
                path_logprob_reduce=self.path_logprob_reduce,
            )
            value_loss = compute_value_loss(recomputed.values, returns)
            behavior_kl = compute_diagonal_gaussian_kl_loss(
                old_means=old_means,
                old_stds=old_stds,
                new_means=recomputed.means,
                new_stds=recomputed.stds,
            )
            entropy_mean = entropy.float().mean()
            total_loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * entropy_mean
                + self.kl_coef * behavior_kl
            )

            update_boundary = self.model.is_gradient_accumulation_boundary()
            self.model.backward(total_loss)
            self.model.step()
            if update_boundary:
                self.global_step += 1
                self.policy_version += 1

            metric_rows.append(
                {
                    "loss/total": float(total_loss.detach().cpu()),
                    "loss/policy": float(policy_loss.detach().cpu()),
                    "loss/value": float(value_loss.detach().cpu()),
                    "loss/behavior_kl": float(behavior_kl.detach().cpu()),
                    "policy/entropy": float(entropy_mean.detach().cpu()),
                    "policy/approx_kl": float(ppo_metrics.approx_kl.detach().cpu()),
                    "policy/clip_fraction": float(ppo_metrics.clip_fraction.detach().cpu()),
                    "policy/ratio_mean": float(ppo_metrics.ratio_mean.detach().cpu()),
                }
            )

        if not metric_rows:
            raise RuntimeError("No complete pi0.5 PPO minibatch was produced")
        metrics = {key: float(np.mean([row[key] for row in metric_rows])) for key in metric_rows[0]}
        metrics.update(learning_rates)
        policy_versions = batch["policy_versions"].astype(np.int64)
        version_lags = np.maximum(self.policy_version - policy_versions, 0)
        sample_ages = np.maximum(int(time.time() * 1000) - batch["insert_times_ms"], 0)
        metrics["rollout/version_lag_mean"] = float(version_lags.mean())
        metrics["rollout/version_lag_p95"] = float(np.quantile(version_lags, 0.95))
        metrics["rollout/version_lag_max"] = float(version_lags.max())
        metrics["rollout/sample_age_ms_mean"] = float(sample_ages.mean())
        metrics["rollout/sample_age_ms_p95"] = float(np.quantile(sample_ages, 0.95))
        metrics["rollout/sample_age_ms_max"] = float(sample_ages.max())
        return metrics

    def save_checkpoint(self, checkpoint_dir: str, step: int) -> str:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        tag = f"step-{step:08d}"
        self.model.save_checkpoint(str(checkpoint_path / "deepspeed"), tag=tag)
        if self.rank == 0:
            torch.save(
                trainable_state_dict(self.base_model),
                checkpoint_path / f"{tag}-trainable.pt",
            )
            (checkpoint_path / "latest.json").write_text(
                json.dumps({"step": step, "tag": tag}, indent=2), encoding="utf-8"
            )
        return str(checkpoint_path / tag)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _validate_first_version_args(args: argparse.Namespace) -> None:
    if args.num_trainer_gpus != 1:
        raise ValueError("The first pi0.5 AcceRL version supports exactly one Trainer GPU")
    if args.num_inference_actors != 1:
        raise ValueError("The first pi0.5 AcceRL version supports exactly one InferenceActor GPU")
    if args.num_rollout_workers < 1:
        raise ValueError("num_rollout_workers must be positive")
    if args.inference_batch <= 0:
        raise ValueError("inference_batch must be positive")
    if args.inference_timeout_ms < 0:
        raise ValueError("inference_timeout_ms must be non-negative")
    if args.train_batch_size <= 0 or args.accumulation_steps <= 0:
        raise ValueError("train_batch_size and accumulation_steps must be positive")
    if args.rollout_local_buf <= 0:
        raise ValueError("rollout_local_buf must be positive")
    if args.replay_capacity <= 0:
        raise ValueError("replay_capacity must be positive")
    if args.max_policy_lag < 0 or args.max_sample_reuse < 0:
        raise ValueError("max_policy_lag and max_sample_reuse must be non-negative")
    if args.kl_coef < 0:
        raise ValueError("kl_coef must be non-negative")
    if not 0 <= args.policy_warmup_steps < args.train_iters:
        raise ValueError("policy_warmup_steps must be in [0, train_iters)")
    if not 0 <= args.value_warmup_steps < args.train_iters:
        raise ValueError("value_warmup_steps must be in [0, train_iters)")
    rollout_python = Path(args.rollout_python)
    if not rollout_python.is_file():
        raise FileNotFoundError(f"Rollout Python executable not found: {rollout_python}")
    if not os.access(rollout_python, os.X_OK):
        raise PermissionError(f"Rollout Python is not executable: {rollout_python}")


def main(args: argparse.Namespace) -> None:
    _validate_first_version_args(args)
    cfg = build_pi05_cfg(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.dry_run:
        print(json.dumps({"args": vars(args), "pi05_config": asdict(cfg)}, indent=2))
        print("Dry run passed: paths and first-version constraints are valid.")
        return

    ray_kwargs: dict[str, Any] = {"ignore_reinit_error": True}
    if args.num_cpus is not None:
        ray_kwargs["num_cpus"] = args.num_cpus
    if args.object_store_memory_gb > 0:
        ray_kwargs["object_store_memory"] = int(args.object_store_memory_gb * 1024**3)
    ray.init(**ray_kwargs)

    run_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.exp_name}"
    log_dir = Path(args.log_dir) / args.benchmark / f"task_{args.task_id}" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=True), encoding="utf-8")
    writer = SummaryWriter(str(log_dir))

    stats_actor = StatsActor.remote()
    replay_buffer = ReplayBufferActor.remote(args.replay_capacity)
    trainer = TrainerActor.remote(
        rank=0,
        world_size=1,
        replay_buffer=replay_buffer,
        cfg=cfg,
        train_batch_size=args.train_batch_size,
        accumulation_steps=args.accumulation_steps,
        policy_lr=args.policy_lr,
        value_lr=args.value_lr,
        policy_warmup_steps=args.policy_warmup_steps,
        value_warmup_steps=args.value_warmup_steps,
        lr_schedule=args.lr_schedule,
        train_iters=args.train_iters,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        kl_coef=args.kl_coef,
        path_logprob_reduce=args.path_logprob_reduce,
        log_ratio_clip=args.log_ratio_clip,
        recompute_value=args.recompute_value,
        max_policy_lag=args.max_policy_lag,
        max_sample_reuse=args.max_sample_reuse,
        use_bf16=args.use_bf16,
    )
    inference_actor = InferenceActor.remote(
        actor_id=0,
        cfg=cfg,
        inference_batch=args.inference_batch,
        inference_timeout_ms=args.inference_timeout_ms,
    )

    broadcast_port = _find_free_port()
    train_port = _find_free_port()
    while train_port == broadcast_port:
        train_port = _find_free_port()
    master_addr = ray.get(trainer.get_node_ip.remote())
    broadcast_participants = [trainer, inference_actor]
    ray.get(
        [
            actor.setup_broadcast_group.remote(
                master_addr,
                broadcast_port,
                args.broadcast_group_name,
                len(broadcast_participants),
                rank,
            )
            for rank, actor in enumerate(broadcast_participants)
        ]
    )
    trainer_summary = ray.get(trainer.setup_deepspeed_group.remote(master_addr, train_port))

    trainer_signature = ray.get(trainer.get_broadcast_signature.remote())
    ray.get(inference_actor.align_broadcast_dtypes.remote(trainer_signature))
    inference_signature = ray.get(inference_actor.get_broadcast_signature.remote())
    if trainer_signature != inference_signature:
        first_mismatch = next(
            (
                (index, left, right)
                for index, (left, right) in enumerate(zip(trainer_signature, inference_signature, strict=False))
                if left != right
            ),
            None,
        )
        raise RuntimeError(
            "Trainer/Inference pi0.5 broadcast signatures differ: "
            f"lengths={len(trainer_signature)}/{len(inference_signature)}, "
            f"first_mismatch={first_mismatch}"
        )

    for dtype_name in ("float32", "bfloat16", "int64"):
        # Every NCCL rank must enter collectives in the same order. Submit one
        # dtype round at a time because Ray async actors may otherwise reorder
        # the independently queued calls.
        ray.get(
            [
                actor.broadcast_sanity_check.remote(args.broadcast_group_name, dtype_name, 8)
                for actor in broadcast_participants
            ]
        )
    ray.get(
        [
            trainer.broadcast_weights.remote(args.broadcast_group_name),
            inference_actor.receive_and_update_weights.remote(args.broadcast_group_name),
        ]
    )

    rollout_runtime_env = {"py_executable": str(Path(args.rollout_python).resolve())}
    rollout_workers = [
        RolloutWorkerActor.options(runtime_env=rollout_runtime_env).remote(
            inference_actor=inference_actor,
            replay_buffer=replay_buffer,
            stats_actor=stats_actor,
            worker_id=worker_id,
            cfg=cfg,
            benchmark=args.benchmark,
            task_id=args.task_id,
            rollout_local_buf=args.rollout_local_buf,
            max_episode_steps=args.max_episode_steps,
            gamma=args.gamma,
            seed=args.seed,
        )
        for worker_id in range(args.num_rollout_workers)
    ]
    for worker in rollout_workers:
        worker.run.remote()
    print(
        f"AcceRL pi0.5 started: trainer={trainer_summary}, workers={len(rollout_workers)}, log_dir={log_dir}",
        flush=True,
    )

    try:
        global_step = 0
        while global_step < args.train_iters:
            started_at = time.time()
            metrics = ray.get(trainer.run_training_epoch.remote())
            global_step = int(metrics["global_step"])
            ray.get(
                [
                    trainer.broadcast_weights.remote(args.broadcast_group_name),
                    inference_actor.receive_and_update_weights.remote(args.broadcast_group_name),
                ]
            )
            metrics["system/train_and_sync_seconds"] = time.time() - started_at
            metrics.update(ray.get(stats_actor.get_stats.remote()))
            metrics.update(ray.get(replay_buffer.get_stats.remote()))

            if global_step % args.log_every_steps == 0:
                print(
                    f"step={global_step} total={metrics['loss/total']:.5f} "
                    f"policy={metrics['loss/policy']:.5f} value={metrics['loss/value']:.5f} "
                    f"kl={metrics['loss/behavior_kl']:.5f} "
                    f"lag={metrics['rollout/version_lag_mean']:.1f} "
                    f"return={metrics['rollout/return_mean']:.3f}",
                    flush=True,
                )
                for name, value in metrics.items():
                    writer.add_scalar(name, value, global_step)
                writer.flush()

            if global_step > 0 and global_step % args.ckpt_every_steps == 0:
                ray.get(trainer.save_checkpoint.remote(args.ckpt_dir, global_step))
    finally:
        for worker in rollout_workers:
            ray.kill(worker, no_restart=True)
        writer.close()
        ray.shutdown()


if __name__ == "__main__":
    main(parse_args())
