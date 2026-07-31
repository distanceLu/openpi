from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch

from openpi.rl.pi05_nested_mdp import Pi05NestedMDP
from openpi.rl.rollout_buffer import Pi05RolloutBuffer


class EnvLike(Protocol):
    def reset(self) -> Any: ...

    def step(self, action: Any) -> Any: ...


EnvObservationToModelObservation = Callable[[Any], Any]
ActionToEnvAction = Callable[[torch.Tensor, Any], Any]


@dataclass
class Pi05RolloutCollectorConfig:
    rollout_steps: int = 128
    mode: str = "train"
    return_values: bool = True
    action_horizon: int | None = None
    execute_horizon: int | None = None
    gamma: float = 0.99
    stop_chunk_on_done: bool = True


class Pi05RolloutCollector:
    """Collect SMDP transitions, one per pi0.5 action chunk.

    The policy emits `[B, H, action_dim]`; only the prefix of length
    `execute_horizon` is executed as `[execute_horizon, env_action_dim=7]`.
    Training defaults to the full chunk, while evaluation may use a shorter
    prefix for receding-horizon visual feedback.
    """

    def __init__(self, env: EnvLike, nested_mdp: Pi05NestedMDP, env_obs_to_model_obs: EnvObservationToModelObservation,
                 action_to_env_action: ActionToEnvAction, config: Pi05RolloutCollectorConfig | None = None):
        self.env, self.nested_mdp = env, nested_mdp
        self.env_obs_to_model_obs, self.action_to_env_action = env_obs_to_model_obs, action_to_env_action
        self.config = config or Pi05RolloutCollectorConfig()
        self._env_obs: Any = None

    def reset(self) -> Any:
        reset_output = self.env.reset()
        self._env_obs = reset_output[0] if isinstance(reset_output, tuple) else reset_output
        return self._env_obs

    def collect(self, buffer: Pi05RolloutBuffer) -> dict[str, float]:
        if self._env_obs is None:
            self.reset()
        reward_sum = 0.0
        episodes = successes = 0
        for _ in range(self.config.rollout_steps):
            model_obs = self.env_obs_to_model_obs(self._env_obs)
            rollout = self.nested_mdp.sample_inner_mdp(model_obs, mode=self.config.mode, return_values=True)
            next_obs, reward, terminated, truncated, info, duration = self._execute_action_prefix(rollout.actions.detach().cpu())
            if terminated:
                next_value = 0.0
            else:
                next_model_obs = self.env_obs_to_model_obs(next_obs)
                next_value = self.nested_mdp.compute_value(next_model_obs).detach().cpu().reshape(-1)
            buffer.add(
                observation=model_obs, chains=rollout.chains.detach().cpu(), old_logprobs=rollout.denoise_logprobs.detach().cpu(),
                denoise_indices=rollout.denoise_indices.detach().cpu(), denoise_timesteps=rollout.denoise_timesteps.detach().cpu(),
                reward=reward, terminated=terminated, truncated=truncated, value=rollout.values.detach().cpu().reshape(-1),
                next_value=next_value, duration=duration, old_velocities=rollout.velocities.detach().cpu(),
            )
            reward_sum += reward
            if terminated or truncated:
                episodes += 1
                successes += int(bool(info.get("success", False))) if isinstance(info, dict) else 0
                self.reset()
            else:
                self._env_obs = next_obs
        return {"rollout/reward_sum": reward_sum, "rollout/episodes": float(episodes), "rollout/successes": float(successes)}

    def _execute_action_prefix(self, action_chunk: torch.Tensor) -> tuple[Any, float, bool, bool, dict[str, Any], int]:
        horizon = action_chunk.shape[1] if action_chunk.dim() == 3 else action_chunk.shape[0]
        execute_horizon = self.config.execute_horizon or horizon
        if not 1 <= execute_horizon <= horizon:
            raise ValueError(f"execute_horizon must be in [1, {horizon}], got {execute_horizon}")
        total_reward = 0.0
        next_obs, info = self._env_obs, {}
        terminated = truncated = False
        for step in range(execute_horizon):
            action_step = action_chunk[:, step] if action_chunk.dim() == 3 else action_chunk[step]
            raw_step = self.env.step(self.action_to_env_action(action_step, self._env_obs))
            if len(raw_step) == 5:
                next_obs, reward, terminated, truncated, info = raw_step
            elif len(raw_step) == 4:
                next_obs, reward, done, info = raw_step
                truncated = bool(info.get("TimeLimit.truncated", False)) if isinstance(info, dict) else False
                terminated = bool(done) and not truncated
            else:
                raise ValueError(f"Expected Gym/Gymnasium step return with 4 or 5 values, got {len(raw_step)}")
            total_reward += self.config.gamma**step * float(reward)
            if (terminated or truncated) and self.config.stop_chunk_on_done:
                return next_obs, total_reward, terminated, truncated, info, step + 1
        return next_obs, total_reward, terminated, truncated, info, execute_horizon
