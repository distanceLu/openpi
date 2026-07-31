"""Run a single forward pass for the pi0.5 LIBERO policy.

This script avoids the websocket server path and directly loads the pi05_libero
checkpoint, builds a synthetic observation that matches the model input spec,
and runs one forward/inference call.
"""

from __future__ import annotations

import dataclasses
import os

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.model import Observation, restore_params
from openpi.training import config as cfg


LOCAL_CACHE_CANDIDATES = [
    "/mnt/data/lcx1/yiqinworkspace/cache",
    os.getenv("OPENPI_DATA_HOME", ""),
]


def _random_value_from_spec(x):
    shape = tuple(x.shape)
    dtype = x.dtype
    if dtype == jnp.float32:
        return np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    if dtype == jnp.int32:
        return np.random.randint(0, 100, size=shape, dtype=np.int32)
    if dtype == jnp.bool_:
        return np.ones(shape, dtype=np.bool_)
    raise TypeError(f"Unsupported dtype in spec: {dtype}")


def _make_obs_dict(obs_spec):
    return {
        "image": jax.tree.map(_random_value_from_spec, obs_spec.images),
        "image_mask": jax.tree.map(_random_value_from_spec, obs_spec.image_masks),
        "state": _random_value_from_spec(obs_spec.state),
        "tokenized_prompt": _random_value_from_spec(obs_spec.tokenized_prompt),
        "tokenized_prompt_mask": _random_value_from_spec(obs_spec.tokenized_prompt_mask),
    }


def main() -> None:
    local_root = "/mnt/data/lcx1/yiqinworkspace/pi05_weights"
    os.makedirs(local_root, exist_ok=True)
    print(f"Using local checkpoint root: {local_root}")

    train_config = cfg.get_config("pi05_libero")

    from openpi.shared import download

    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
    params_dir = checkpoint_dir / "params"
    print(f"Checkpoint cache directory: {checkpoint_dir}")
    print(f"Loading checkpoint from: {params_dir}")

    print(f"Loading checkpoint from: {params_dir}")
    params = restore_params(params_dir, dtype=jnp.bfloat16)
    model = train_config.model.load(params)

    print("JAX devices:", jax.devices())
    print("JAX backend:", jax.default_backend())

    obs_spec, act_spec = train_config.model.inputs_spec(batch_size=1)
    obs_dict = _make_obs_dict(obs_spec)

    observation = Observation.from_dict(obs_dict)
    observation = jax.tree.map(jnp.asarray, observation)
    rng = jax.random.key(0)

    print("Running pi0.5 forward pass...")
    actions = model.sample_actions(rng, observation)

    actions_np = np.asarray(actions)
    print("=== forward success ===")
    print(f"model_config = {train_config.name} (pi05={train_config.model.pi05})")
    print(f"expected_action_shape = {tuple(act_spec.shape)}")
    print(f"actual_action_shape   = {actions_np.shape}")
    print(f"action_dtype          = {actions_np.dtype}")
    print(f"action_preview        = {actions_np[0, 0, : min(8, actions_np.shape[-1])]}")


if __name__ == "__main__":
    main()
