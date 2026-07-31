"""Prototype pi0.5 Flow-PPO style RL training with fake rollouts.

This script is intentionally lightweight and self-contained:
- loads the pi05_libero checkpoint
- builds a fake environment that emits synthetic observations and rewards
- collects rollouts into a trajectory buffer
- computes a pi0.5 flow-matching loss via `model.compute_loss`
- computes PPO-style policy and value losses with GAE
- updates actor and critic with minibatch SGD over multiple PPO epochs
- logs metrics to TensorBoard
- saves and restores checkpoints

This is a research/prototyping scaffold, not a production RL trainer.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from torch.utils.tensorboard import SummaryWriter

from openpi.models.model import Observation, restore_params
from openpi.shared import download
from openpi.training import config as cfg


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    rollout_horizon: int = 4
    batch_size: int = 1
    train_iters: int = 20
    ppo_epochs: int = 2
    clip_ratio: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    flow_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 0.5
    entropy_bonus: float = 0.0
    actor_lr: float = 1e-5
    critic_lr: float = 1e-4
    save_interval: int = 5
    log_interval: int = 1
    max_grad_norm: float = 1.0
    output_dir: str = "./checkpoints/pi05_fake_flowppo"
    tensorboard_dir: str = "./tensorboard/pi05_fake_flowppo"
    resume: bool = True


@dataclass
class RolloutBuffer:
    observations: list[Observation]
    actions: list[jnp.ndarray]
    rewards: list[jnp.ndarray]
    values: list[jnp.ndarray]
    dones: list[jnp.ndarray]
    old_logprobs: list[jnp.ndarray]
    returns: list[jnp.ndarray] | None = None
    advantages: list[jnp.ndarray] | None = None


@dataclass
class ReplayBuffer:
    observations: list[Observation]
    actions: list[jnp.ndarray]
    rewards: list[jnp.ndarray]
    values: list[jnp.ndarray]
    dones: list[jnp.ndarray]
    old_logprobs: list[jnp.ndarray]
    advantages: list[jnp.ndarray]
    returns: list[jnp.ndarray]


class Critic(nnx.Module):
    def __init__(self, hidden_dim: int, rngs: nnx.Rngs):
        self.encoder = nnx.Linear(6, hidden_dim, rngs=rngs)
        self.hidden = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.value = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, observation: Observation) -> jnp.ndarray:
        state = jnp.asarray(observation.state).reshape(observation.state.shape[0], -1)
        x = self.encoder(state)
        x = nnx.swish(x)
        x = self.hidden(x)
        x = nnx.swish(x)
        return self.value(x).squeeze(-1)


class FakeEnv:
    def __init__(self, obs_spec, act_spec, seed: int):
        self.obs_spec = obs_spec
        self.act_spec = act_spec
        self.key = jax.random.key(seed)

    def _random_value(self, spec, key):
        shape = tuple(spec.shape)
        dtype = spec.dtype
        if dtype == jnp.float32:
            return jax.random.uniform(key, shape, minval=-1.0, maxval=1.0, dtype=jnp.float32)
        if dtype == jnp.int32:
            return jax.random.randint(key, shape, 0, 100, dtype=jnp.int32)
        if dtype == jnp.bool_:
            return jnp.ones(shape, dtype=jnp.bool_)
        raise TypeError(f"Unsupported dtype in spec: {dtype}")

    def sample_obs(self, key) -> Observation:
        keys = jax.random.split(key, 5)
        obs_dict = {
            "image": {
                name: self._random_value(spec, keys[0])
                for name, spec in self.obs_spec.images.items()
            },
            "image_mask": {
                name: self._random_value(spec, keys[1])
                for name, spec in self.obs_spec.image_masks.items()
            },
            "state": self._random_value(self.obs_spec.state, keys[2]),
            "tokenized_prompt": self._random_value(self.obs_spec.tokenized_prompt, keys[3]),
            "tokenized_prompt_mask": self._random_value(self.obs_spec.tokenized_prompt_mask, keys[4]),
        }
        return jax.tree.map(jnp.asarray, Observation.from_dict(obs_dict))

    def rollout(self, policy, critic, horizon: int) -> RolloutBuffer:
        observations = []
        actions = []
        rewards = []
        values = []
        dones = []
        old_logprobs = []
        for step in range(horizon):
            self.key, obs_key, act_key, rew_key = jax.random.split(self.key, 4)
            obs = self.sample_obs(obs_key)
            act = policy.sample_actions(act_key, obs)
            value = critic(obs)
            target = jax.random.normal(rew_key, act.shape)
            reward = jnp.exp(-jnp.mean(jnp.square(act - target), axis=(-1, -2)))
            old_logprob = -0.5 * jnp.sum(jnp.square(act - target), axis=(-1, -2))
            done = jnp.asarray(step + 1 == horizon, dtype=jnp.bool_)
            observations.append(obs)
            actions.append(act)
            rewards.append(reward)
            values.append(value)
            dones.append(done)
            old_logprobs.append(old_logprob)
        return RolloutBuffer(observations, actions, rewards, values, dones, old_logprobs)

    def action_logprob(self, actions: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        scale = 1.0
        return -0.5 * jnp.sum(jnp.square((actions - target) / scale), axis=(-1, -2))


def _stack(xs: list[jnp.ndarray]) -> jnp.ndarray:
    return jnp.concatenate(xs, axis=0)


def _tree_stack(buffer_list: list[Observation]) -> Observation:
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *buffer_list)


def _compute_gae(rewards, values, dones, gamma: float, lam: float):
    rewards = jnp.asarray(rewards)
    values = jnp.asarray(values)
    dones = jnp.asarray(dones)
    advantages = []
    gae = jnp.zeros_like(values[-1])
    next_value = jnp.zeros_like(values[-1])
    for t in range(rewards.shape[0] - 1, -1, -1):
        mask = 1.0 - dones[t].astype(jnp.float32)
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages.append(gae)
        next_value = values[t]
    advantages = advantages[::-1]
    returns = [adv + val for adv, val in zip(advantages, values, strict=True)]
    return jnp.stack(advantages), jnp.stack(returns)


def _flatten_replay(buffer: RolloutBuffer, advantages, returns) -> ReplayBuffer:
    return ReplayBuffer(
        observations=buffer.observations,
        actions=buffer.actions,
        rewards=buffer.rewards,
        values=buffer.values,
        dones=buffer.dones,
        old_logprobs=buffer.old_logprobs,
        advantages=list(jnp.asarray(advantages)),
        returns=list(jnp.asarray(returns)),
    )


def _iter_minibatches(replay: ReplayBuffer, minibatch_size: int, seed: int):
    num_samples = len(replay.observations)
    perm = np.random.default_rng(seed).permutation(num_samples)
    for start in range(0, num_samples, minibatch_size):
        mb = perm[start : start + minibatch_size]
        yield [replay.observations[i] for i in mb], replay.actions[mb], replay.old_logprobs[mb], replay.advantages[mb], replay.returns[mb]


def _save_checkpoint(path: Path, *, model_state, critic_state, actor_opt_state, critic_opt_state, step: int, config: TrainConfig) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "config": dataclasses.asdict(config),
        "model_state": model_state.to_pure_dict(),
        "critic_state": critic_state.to_pure_dict(),
        "actor_opt_state": actor_opt_state,
        "critic_opt_state": critic_opt_state,
    }
    with open(path / "checkpoint.pkl", "wb") as f:
        pickle.dump(payload, f)
    with open(path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"step": step, "timestamp": time.time()}, f, indent=2)


def _load_checkpoint(path: Path):
    ckpt_file = path / "checkpoint.pkl"
    if not ckpt_file.exists():
        return None
    with open(ckpt_file, "rb") as f:
        return pickle.load(f)


def _critic_loss_fn(critic: Critic, observations, returns) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    values = jnp.stack([critic(obs) for obs in observations], axis=0)
    loss = jnp.mean(jnp.square(values - returns))
    return loss, {"loss/value": loss, "value/pred": jnp.mean(values), "value/target": jnp.mean(returns)}


def _policy_and_flow_loss_fn(model, critic: Critic, observations, actions, old_logprobs, advantages, returns, cfg_: TrainConfig) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    flow_losses = []
    policy_losses = []
    entropy_terms = []
    value_losses = []
    rewards = []

    for idx, (obs, old_lp, adv, ret, act) in enumerate(zip(observations, old_logprobs, advantages, returns, actions, strict=True)):
        target_key = jax.random.fold_in(jax.random.key(cfg_.seed), idx + 17)
        target_actions = jax.random.normal(target_key, act.shape)
        flow_loss = jnp.mean(model.compute_loss(target_key, obs, target_actions, train=True))
        sampled_actions = model.sample_actions(target_key, obs)
        new_logprob = -0.5 * jnp.sum(jnp.square(sampled_actions - target_actions), axis=(-1, -2))
        ratio = jnp.exp(new_logprob - old_lp)
        clipped = jnp.clip(ratio, 1.0 - cfg_.clip_ratio, 1.0 + cfg_.clip_ratio)
        policy_loss = -jnp.mean(jnp.minimum(ratio * adv, clipped * adv))
        value_pred = critic(obs)
        value_loss = jnp.mean(jnp.square(value_pred - ret))
        entropy = -jnp.mean(new_logprob)
        flow_losses.append(flow_loss)
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        entropy_terms.append(entropy)
        rewards.append(jnp.mean(ret))

    flow_loss = jnp.mean(jnp.stack(flow_losses))
    policy_loss = jnp.mean(jnp.stack(policy_losses))
    value_loss = jnp.mean(jnp.stack(value_losses))
    entropy = jnp.mean(jnp.stack(entropy_terms))
    reward = jnp.mean(jnp.stack(rewards))

    total_loss = (
        cfg_.flow_loss_weight * flow_loss
        + cfg_.policy_loss_weight * policy_loss
        + cfg_.value_loss_weight * value_loss
        - cfg_.entropy_bonus * entropy
    )
    metrics = {
        "loss/flow": flow_loss,
        "loss/policy": policy_loss,
        "loss/value": value_loss,
        "loss/entropy": entropy,
        "reward/mean": reward,
        "loss/total": total_loss,
    }
    return total_loss, metrics


def main() -> None:
    cfg_train = TrainConfig()
    output_dir = Path(cfg_train.output_dir)
    tb_dir = Path(cfg_train.tensorboard_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    openpi_cache = os.getenv("OPENPI_DATA_HOME")
    if openpi_cache:
        print(f"OPENPI_DATA_HOME={openpi_cache}")

    train_config = cfg.get_config("pi05_libero")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
    params_dir = checkpoint_dir / "params"

    params = restore_params(params_dir, dtype=jnp.bfloat16)
    model = train_config.model.load(params)
    critic = Critic(hidden_dim=256, rngs=nnx.Rngs(jax.random.key(123)))

    model_graphdef, model_state = nnx.split(model)
    critic_graphdef, critic_state = nnx.split(critic)
    actor_tx = optax.chain(optax.clip_by_global_norm(cfg_train.max_grad_norm), optax.adam(cfg_train.actor_lr))
    critic_tx = optax.chain(optax.clip_by_global_norm(cfg_train.max_grad_norm), optax.adam(cfg_train.critic_lr))
    actor_opt_state = actor_tx.init(model_state.to_pure_dict())
    critic_opt_state = critic_tx.init(critic_state.to_pure_dict())

    start_step = 0
    ckpt = _load_checkpoint(output_dir) if cfg_train.resume else None
    if ckpt is not None:
        start_step = int(ckpt["step"])
        model_state.replace_by_pure_dict(ckpt["model_state"])
        critic_state.replace_by_pure_dict(ckpt["critic_state"])
        actor_opt_state = ckpt["actor_opt_state"]
        critic_opt_state = ckpt["critic_opt_state"]
        print(f"Resumed from checkpoint step={start_step}")

    obs_spec, act_spec = train_config.model.inputs_spec(batch_size=cfg_train.batch_size)
    fake_env = FakeEnv(obs_spec, act_spec, seed=cfg_train.seed)

    print("JAX devices:", jax.devices())
    print("JAX backend:", jax.default_backend())
    print("Starting fake pi0.5 Flow-PPO training...")

    for step in range(start_step, cfg_train.train_iters):
        model = nnx.merge(model_graphdef, model_state)
        critic = nnx.merge(critic_graphdef, critic_state)
        rollout = fake_env.rollout(model, critic, cfg_train.rollout_horizon)

        advantages, returns = _compute_gae(
            jnp.stack(rollout.rewards, axis=0),
            jnp.stack(rollout.values, axis=0),
            jnp.stack(rollout.dones, axis=0),
            cfg_train.gamma,
            cfg_train.gae_lambda,
        )
        replay = _flatten_replay(rollout, advantages, returns)

        (v_loss, v_metrics), v_grads = jax.value_and_grad(_critic_loss_fn, has_aux=True)(
            critic, replay.observations, returns
        )
        critic_updates, critic_opt_state = critic_tx.update(v_grads, critic_opt_state, critic_state.to_pure_dict())
        critic_state.replace_by_pure_dict(optax.apply_updates(critic_state.to_pure_dict(), critic_updates))

        epoch_losses = []
        epoch_metrics: dict[str, list[float]] = {}
        for epoch in range(cfg_train.ppo_epochs):
            for mb_obs, mb_actions, mb_old_logprobs, mb_advantages, mb_returns in _iter_minibatches(
                replay,
                minibatch_size=max(1, len(replay.observations) // 2),
                seed=cfg_train.seed + step + epoch,
            ):
                (loss, metrics), grads = jax.value_and_grad(_policy_and_flow_loss_fn, has_aux=True)(
                    model,
                    critic,
                    mb_obs,
                    mb_actions,
                    mb_old_logprobs,
                    mb_advantages,
                    mb_returns,
                    cfg_train,
                )
                actor_updates, actor_opt_state = actor_tx.update(grads, actor_opt_state, model_state.to_pure_dict())
                model_state.replace_by_pure_dict(optax.apply_updates(model_state.to_pure_dict(), actor_updates))
                epoch_losses.append(float(loss))
                for k, v in metrics.items():
                    epoch_metrics.setdefault(k, []).append(float(v))

        model = nnx.merge(model_graphdef, model_state)
        critic = nnx.merge(critic_graphdef, critic_state)

        if step % cfg_train.log_interval == 0:
            writer.add_scalar("loss/total", float(np.mean(epoch_losses)), step)
            writer.add_scalar("loss/flow", float(np.mean(epoch_metrics["loss/flow"])), step)
            writer.add_scalar("loss/policy", float(np.mean(epoch_metrics["loss/policy"])), step)
            writer.add_scalar("loss/value", float(np.mean(epoch_metrics["loss/value"])), step)
            writer.add_scalar("reward/mean", float(np.mean(epoch_metrics["reward/mean"])), step)
            writer.add_scalar("debug/critic_loss", float(v_loss), step)
            writer.add_scalar("debug/value_pred", float(v_metrics["value/pred"]), step)
            writer.add_scalar("debug/value_target", float(v_metrics["value/target"]), step)
            writer.flush()

        print(
            f"step={step} total={float(np.mean(epoch_losses)):.6f} "
            f"flow={float(np.mean(epoch_metrics['loss/flow'])):.6f} policy={float(np.mean(epoch_metrics['loss/policy'])):.6f} "
            f"value={float(np.mean(epoch_metrics['loss/value'])):.6f} reward={float(np.mean(epoch_metrics['reward/mean'])):.6f}"
        )

        if (step + 1) % cfg_train.save_interval == 0 or step == cfg_train.train_iters - 1:
            ckpt_path = output_dir / f"step_{step + 1:06d}"
            _save_checkpoint(
                ckpt_path,
                model_state=model_state,
                critic_state=critic_state,
                actor_opt_state=actor_opt_state,
                critic_opt_state=critic_opt_state,
                step=step + 1,
                config=cfg_train,
            )
            print(f"Saved checkpoint to {ckpt_path}")

    writer.close()
    print("Training loop finished.")


if __name__ == "__main__":
    main()
