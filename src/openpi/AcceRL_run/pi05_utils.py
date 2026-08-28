"""pi0.5 model glue used by the AcceRL Ray actors.

This module intentionally contains model-specific code only. AcceRL owns the
outer environment MDP and distributed execution; ``Pi05NestedMDP`` owns the
inner denoising MDP used to assign PPO log-probabilities to a pi0.5 action.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from openpi.models import model as model_types
from openpi.rl.libero_adapter import Pi05LiberoRLAdapter
from openpi.rl.pi05_action_head import Pi05GaussianActionHead
from openpi.rl.pi05_nested_mdp import Pi05NestedMDP
from openpi.rl.pi05_nested_mdp import Pi05NestedMDPConfig
from openpi.rl.pi05_types import Pi05LogProbOutput
from openpi.rl.pi05_types import Pi05RolloutOutput

Pi05TrainableScope = Literal[
    "rl_heads_only",
    "lora_and_heads",
    "action_expert_and_heads",
    "full_model",
]


@dataclass(frozen=True)
class Pi05AcceRLConfig:
    checkpoint_path: str
    reference_dir: str
    assets_dir: str
    libero_repo_dir: str
    checkpoint_uses_extra_delta_transform: bool
    train_config_name: str = "pi05_libero"
    lora_adapter_path: str | None = None
    num_denoise_steps: int = 10
    sample_method: str = "flow_noise"
    noise_level: float = 0.5
    compute_dtype: str = "bfloat16"
    trainable_scope: Pi05TrainableScope = "action_expert_and_heads"
    action_chunk_steps: int = 1

    def validate(self) -> None:
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"pi0.5 checkpoint not found: {self.checkpoint_path}")
        if self.lora_adapter_path is not None and not Path(self.lora_adapter_path).exists():
            raise FileNotFoundError(f"pi0.5 LoRA adapter not found: {self.lora_adapter_path}")
        if self.num_denoise_steps <= 0:
            raise ValueError("num_denoise_steps must be positive")
        if self.action_chunk_steps <= 0:
            raise ValueError("action_chunk_steps must be positive")
        validate_pi05_sample_method(self.sample_method, self.noise_level, ppo_training=True)
        if self.trainable_scope not in {
            "rl_heads_only",
            "lora_and_heads",
            "action_expert_and_heads",
            "full_model",
        }:
            raise ValueError(f"Unsupported pi0.5 trainable_scope: {self.trainable_scope}")
        if self.trainable_scope == "lora_and_heads" and self.lora_adapter_path is None:
            raise ValueError("trainable_scope='lora_and_heads' requires lora_adapter_path")
        resolve_torch_dtype(self.compute_dtype)
        if not isinstance(self.checkpoint_uses_extra_delta_transform, bool):
            raise TypeError("checkpoint_uses_extra_delta_transform must be explicitly set to bool")


def create_adapter(cfg: Pi05AcceRLConfig, device: str = "cpu") -> Pi05LiberoRLAdapter:
    """Create the shared LIBERO transform/checkpoint adapter."""

    return Pi05LiberoRLAdapter(
        train_config_name=cfg.train_config_name,
        checkpoint_path=cfg.checkpoint_path,
        lora_adapter_path=cfg.lora_adapter_path,
        reference_dir=cfg.reference_dir,
        assets_dir=cfg.assets_dir,
        libero_repo_dir=cfg.libero_repo_dir,
        device=device,
        checkpoint_uses_extra_delta_transform=cfg.checkpoint_uses_extra_delta_transform,
    )


def validate_pi05_sample_method(
    sample_method: str,
    noise_level: float,
    *,
    ppo_training: bool,
) -> None:
    supported_methods = {"flow_ode", "flow_sde", "flow_cps", "flow_noise"}
    if sample_method not in supported_methods:
        raise ValueError(f"Unsupported pi0.5 sample_method: {sample_method}")
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative")
    if sample_method in {"flow_sde", "flow_cps"} and noise_level <= 0:
        raise ValueError(f"{sample_method} requires noise_level > 0")
    if ppo_training and sample_method != "flow_noise":
        raise ValueError(
            "AcceRL PPO rollout and logprob recomputation require sample_method='flow_noise'; "
            "flow_ode is reserved for deterministic evaluation, while flow_sde/flow_cps are evaluation ablations"
        )


def freeze_modules(modules: tuple[torch.nn.Module, ...]) -> None:
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(False)


def unfreeze_modules(modules: tuple[torch.nn.Module, ...]) -> None:
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)


def configure_trainable_parameters(
    model: torch.nn.Module,
    trainable_scope: Pi05TrainableScope,
    *,
    has_lora_adapter: bool = False,
) -> None:
    """Select pi0.5 PPO parameters through explicit module references."""

    if trainable_scope == "full_model":
        unfreeze_modules((model,))
        return

    freeze_modules((model,))
    if trainable_scope == "rl_heads_only":
        unfreeze_modules(_pi05_rl_head_modules(model))
    elif trainable_scope == "lora_and_heads":
        if not has_lora_adapter:
            raise ValueError("trainable_scope='lora_and_heads' requires an injected LoRA adapter")
        unfreeze_modules(_pi05_rl_head_modules(model))
        _unfreeze_lora_parameters(model)
    elif trainable_scope == "action_expert_and_heads":
        unfreeze_modules((*_pi05_action_expert_modules(model), *_pi05_rl_head_modules(model)))
    else:
        raise ValueError(f"Unsupported pi0.5 trainable_scope: {trainable_scope}")


def _pi05_action_expert_modules(model: torch.nn.Module) -> tuple[torch.nn.Module, ...]:
    try:
        modules = (
            model.paligemma_with_expert.gemma_expert,
            model.action_in_proj,
            model.action_out_proj,
            model.time_mlp_in,
            model.time_mlp_out,
        )
    except AttributeError as exc:
        raise AttributeError(f"pi0.5 action expert is missing required module: {exc}") from exc
    if not all(isinstance(module, torch.nn.Module) for module in modules):
        raise TypeError("Every pi0.5 action-expert component must be a torch.nn.Module")
    return modules


def _pi05_rl_head_modules(model: torch.nn.Module) -> tuple[torch.nn.Module, ...]:
    value_head = getattr(model, "value_head", None)
    action_head = getattr(model, "rl_action_head", None)
    modules = (value_head,) if action_head is None else (action_head, value_head)
    if not all(isinstance(module, torch.nn.Module) for module in modules):
        raise TypeError("pi0.5 value_head and any configured rl_action_head must be torch.nn.Module instances")
    return modules


def _unfreeze_lora_parameters(model: torch.nn.Module) -> None:
    """Enable injected LoRA tensors by their concrete module type, not parameter names."""

    from openpi.sft.train_pi05_lora import LoRALinear

    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_a.requires_grad_(True)
            module.lora_b.requires_grad_(True)


def load_pi05_policy(
    cfg: Pi05AcceRLConfig,
    device: str = "cuda",
) -> tuple[Pi05LiberoRLAdapter, torch.nn.Module]:
    """Load the adapter and pretrained pi0.5 policy without constructing RL heads."""

    cfg.validate()
    adapter = create_adapter(cfg, device=device)
    model = adapter.load_pi05_model_from_checkpoint()
    if not getattr(model, "pi05", False):
        raise ValueError("Loaded checkpoint is not a pi0.5 policy")
    return adapter, model


def build_pi05_rl_heads(
    model: torch.nn.Module,
    cfg: Pi05AcceRLConfig,
) -> Pi05NestedMDP:
    """Attach the stochastic denoising head and outer-MDP value head."""

    if cfg.sample_method == "flow_noise":
        if hasattr(model, "rl_action_head"):
            raise ValueError("model.rl_action_head already exists; refusing to replace a loaded RL head")
        model.rl_action_head = Pi05GaussianActionHead(
            input_dim=model.action_out_proj.in_features,
            action_dim=model.config.action_dim,
        ).to(device=next(model.parameters()).device, dtype=torch.float32)
    elif hasattr(model, "rl_action_head"):
        raise ValueError("Pi05GaussianActionHead is only valid when sample_method='flow_noise'")

    return Pi05NestedMDP(
        model,
        Pi05NestedMDPConfig(
            num_denoise_steps=cfg.num_denoise_steps,
            sample_method=cfg.sample_method,
            noise_level=cfg.noise_level,
            deterministic_eval=True,
            learned_action_head=cfg.sample_method == "flow_noise",
            require_trainable_heads=False,
            require_value_head=True,
        ),
    )


def build_pi05_actor_critic(
    cfg: Pi05AcceRLConfig,
    device: str = "cuda",
) -> tuple[Pi05LiberoRLAdapter, torch.nn.Module, Pi05NestedMDP]:
    """Load pi0.5 and attach the stochastic actor head and outer critic."""

    adapter, model = load_pi05_policy(cfg, device=device)
    nested_mdp = build_pi05_rl_heads(model, cfg)
    configure_trainable_parameters(
        model,
        cfg.trainable_scope,
        has_lora_adapter=cfg.lora_adapter_path is not None,
    )
    check_pi05_action_contract(cfg, adapter, model, nested_mdp)
    return adapter, model, nested_mdp


def check_pi05_action_contract(
    cfg: Pi05AcceRLConfig,
    adapter: Pi05LiberoRLAdapter,
    model: torch.nn.Module,
    nested_mdp: Pi05NestedMDP | None = None,
) -> None:
    """Validate the checkpoint, transforms, action shapes, and denoising policy contract."""

    if not getattr(model, "pi05", False):
        raise ValueError("AcceRL pi0.5 requires a model constructed with pi05=True")
    model_horizon = int(model.config.action_horizon)
    model_action_dim = int(model.config.action_dim)
    if model_horizon != int(adapter.action_horizon):
        raise ValueError(f"Action horizon mismatch: model={model_horizon}, adapter={adapter.action_horizon}")
    if model_action_dim != int(adapter.action_dim):
        raise ValueError(f"Action dimension mismatch: model={model_action_dim}, adapter={adapter.action_dim}")
    if not 1 <= cfg.action_chunk_steps <= model_horizon:
        raise ValueError(f"action_chunk_steps must be in [1, {model_horizon}], got {cfg.action_chunk_steps}")
    if adapter.checkpoint_uses_extra_delta_transform != cfg.checkpoint_uses_extra_delta_transform:
        raise ValueError(
            "Adapter delta-action convention differs from Pi05AcceRLConfig: "
            f"adapter={adapter.checkpoint_uses_extra_delta_transform}, "
            f"config={cfg.checkpoint_uses_extra_delta_transform}"
        )

    required_norm_dims = {"actions": int(adapter.env_action_dim), "state": 8}
    use_quantiles = bool(adapter.data_config.use_quantile_norm)
    stat_names = ("q01", "q99") if use_quantiles else ("mean", "std")
    for key, minimum_dim in required_norm_dims.items():
        if key not in adapter.norm_stats:
            raise KeyError(f"Required pi0.5 normalization key {key!r} is missing")
        stats = adapter.norm_stats[key]
        stat_values: dict[str, np.ndarray] = {}
        for stat_name in stat_names:
            value = getattr(stats, stat_name, None)
            value_array = None if value is None else np.asarray(value)
            if value_array is None or value_array.ndim != 1 or value_array.shape[0] < minimum_dim:
                shape = None if value is None else np.asarray(value).shape
                raise ValueError(
                    f"norm_stats[{key!r}].{stat_name} must be 1D with at least {minimum_dim} values, got {shape}"
                )
            if not np.isfinite(value_array).all():
                raise ValueError(f"norm_stats[{key!r}].{stat_name} contains NaN/Inf")
            stat_values[stat_name] = value_array
        if stat_values[stat_names[0]].shape != stat_values[stat_names[1]].shape:
            raise ValueError(f"norm_stats[{key!r}] statistic shapes do not match")
        if use_quantiles:
            if np.any(stat_values["q99"][:minimum_dim] <= stat_values["q01"][:minimum_dim]):
                raise ValueError(f"norm_stats[{key!r}] requires q99 > q01 for used dimensions")
        elif np.any(stat_values["std"][:minimum_dim] <= 0):
            raise ValueError(f"norm_stats[{key!r}].std must be positive for used dimensions")

    if cfg.sample_method == "flow_noise" and not isinstance(
        getattr(model, "rl_action_head", None), Pi05GaussianActionHead
    ):
        raise ValueError("flow_noise requires model.rl_action_head to be Pi05GaussianActionHead")
    if not hasattr(model, "value_head"):
        raise ValueError("PPO requires a value_head attached to the pi0.5 model")
    if nested_mdp is not None:
        if nested_mdp.model is not model:
            raise ValueError("Pi05NestedMDP is not bound to the model being validated")
        if nested_mdp.config.num_denoise_steps != cfg.num_denoise_steps:
            raise ValueError("Nested-MDP denoise-step count differs from Pi05AcceRLConfig")
        if nested_mdp.config.sample_method != cfg.sample_method:
            raise ValueError("Nested-MDP sample method differs from Pi05AcceRLConfig")


def prepare_one_obs(
    adapter: Pi05LiberoRLAdapter,
    observation: dict[str, Any],
    task_description: str,
) -> model_types.Observation[torch.Tensor]:
    """Convert one raw LIBERO observation into a CPU pi0.5 Observation."""

    raw_observation = dict(observation)
    raw_observation["task_description"] = task_description
    return observation_to_cpu(adapter.env_obs_to_model_obs(raw_observation))


def prepare_inputs_batch(
    adapter: Pi05LiberoRLAdapter,
    observations: list[model_types.Observation[torch.Tensor]],
) -> model_types.Observation[torch.Tensor]:
    """Collate already-preprocessed observations for trainer replay batches."""

    if not observations:
        raise ValueError("Cannot collate an empty pi0.5 observation batch")
    batch = adapter.collate_observations(observations)
    _check_observation_batch(batch, expected_batch_size=len(observations))
    return batch


def prepare_obs_batch(
    adapter: Pi05LiberoRLAdapter,
    observations: list[dict[str, Any]],
    task_descriptions: list[str],
) -> model_types.Observation[torch.Tensor]:
    """Convert a batch of raw LIBERO observations into one pi0.5 Observation batch."""

    if not observations:
        raise ValueError("Cannot prepare an empty observation batch")
    if len(observations) != len(task_descriptions):
        raise ValueError(
            "observations and task_descriptions must have equal length, got "
            f"{len(observations)} and {len(task_descriptions)}"
        )
    prepared = [
        prepare_one_obs(adapter, observation, task_description)
        for observation, task_description in zip(observations, task_descriptions, strict=True)
    ]
    batch = prepare_inputs_batch(adapter, prepared)
    _check_observation_batch(batch, expected_batch_size=len(observations))
    return batch


def observation_to_cpu(
    observation: model_types.Observation[torch.Tensor],
) -> model_types.Observation[torch.Tensor]:
    """Detach an Observation before it crosses a Ray object-store boundary."""

    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if torch.is_tensor(value):
            return value.detach().cpu().contiguous()
        return value

    return model_types.Observation.from_dict(convert(observation.to_dict()))


def resolve_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        resolved = dtype
    else:
        dtype_by_name = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        try:
            resolved = dtype_by_name[dtype]
        except KeyError as exc:
            raise ValueError(f"Unsupported pi0.5 compute dtype: {dtype!r}") from exc
    if resolved not in {torch.bfloat16, torch.float16, torch.float32}:
        raise ValueError(f"Unsupported pi0.5 compute dtype: {resolved}")
    return resolved


def _autocast_ctx(device: torch.device, dtype: str | torch.dtype):
    compute_dtype = resolve_torch_dtype(dtype)
    if device.type == "cuda" and compute_dtype in {torch.bfloat16, torch.float16}:
        return torch.autocast(device_type="cuda", dtype=compute_dtype)
    return nullcontext()


def run_rollout_forward(
    nested_mdp: Pi05NestedMDP,
    observations: model_types.Observation[torch.Tensor],
    *,
    deterministic: bool = False,
    return_values: bool = True,
    torch_dtype: str | torch.dtype = "bfloat16",
) -> Pi05RolloutOutput:
    """Run the inner denoising MDP behind one stable AcceRL inference interface."""

    batch_size = _check_observation_batch(observations)
    with _autocast_ctx(nested_mdp.device, torch_dtype):
        output = nested_mdp.sample_inner_mdp(
            observations,
            mode="eval" if deterministic else "train",
            return_values=return_values,
        )
    model = nested_mdp.model
    expected_action_shape = (batch_size, model.config.action_horizon, model.config.action_dim)
    expected_chain_shape = (
        batch_size,
        nested_mdp.config.num_denoise_steps + 1,
        model.config.action_horizon,
        model.config.action_dim,
    )
    expected_step_shape = (
        batch_size,
        nested_mdp.config.num_denoise_steps,
        model.config.action_horizon,
        model.config.action_dim,
    )
    _require_shape("rollout.actions", output.actions, expected_action_shape)
    _require_shape("rollout.chains", output.chains, expected_chain_shape)
    for name, tensor in (
        ("denoise_logprobs", output.denoise_logprobs),
        ("denoise_means", output.denoise_means),
        ("denoise_stds", output.denoise_stds),
        ("velocities", output.velocities),
    ):
        _require_shape(f"rollout.{name}", tensor, expected_step_shape)
        _require_finite(f"rollout.{name}", tensor)
    expected_schedule_shape = (batch_size, nested_mdp.config.num_denoise_steps)
    _require_shape("rollout.denoise_timesteps", output.denoise_timesteps, expected_schedule_shape)
    _require_shape("rollout.denoise_indices", output.denoise_indices, expected_schedule_shape)
    _require_finite("rollout.denoise_timesteps", output.denoise_timesteps)
    if torch.any((output.denoise_timesteps < 0) | (output.denoise_timesteps > 1)):
        raise ValueError("rollout.denoise_timesteps must be in [0, 1]")
    if torch.any((output.denoise_indices < 0) | (output.denoise_indices >= nested_mdp.config.num_denoise_steps)):
        raise ValueError(f"rollout.denoise_indices must be in [0, {nested_mdp.config.num_denoise_steps - 1}]")
    _require_finite("rollout.actions", output.actions)
    _require_finite("rollout.chains", output.chains)
    effective_sample_method = (
        "flow_ode" if deterministic and nested_mdp.config.deterministic_eval else nested_mdp.config.sample_method
    )
    _check_transition_stds(
        "rollout.denoise_stds",
        output.denoise_stds,
        sample_method=effective_sample_method,
    )
    if return_values:
        if output.values is None:
            raise ValueError("Rollout requested values but the nested MDP returned None")
        _require_shape("rollout.values", output.values, (batch_size,))
        _require_finite("rollout.values", output.values)
    return output


def run_training_forward(
    nested_mdp: Pi05NestedMDP,
    observations: model_types.Observation[torch.Tensor],
    *,
    chains: torch.Tensor | np.ndarray,
    denoise_indices: torch.Tensor | np.ndarray | None = None,
    denoise_timesteps: torch.Tensor | np.ndarray | None = None,
    return_values: bool = True,
    torch_dtype: str | torch.dtype = "bfloat16",
) -> Pi05LogProbOutput:
    """Recompute current-policy log-probabilities for saved rollout denoise paths."""

    batch_size = _check_observation_batch(observations)
    device = nested_mdp.device
    chains_tensor = _as_tensor(chains, device=device, dtype=torch.float32)
    expected_chain_shape = (
        batch_size,
        nested_mdp.config.num_denoise_steps + 1,
        nested_mdp.model.config.action_horizon,
        nested_mdp.model.config.action_dim,
    )
    _require_shape("training.chains", chains_tensor, expected_chain_shape)
    _require_finite("training.chains", chains_tensor)
    indices_tensor = None if denoise_indices is None else _as_tensor(denoise_indices, device=device, dtype=torch.long)
    timesteps_tensor = (
        None if denoise_timesteps is None else _as_tensor(denoise_timesteps, device=device, dtype=torch.float32)
    )
    expected_index_shape = (batch_size, nested_mdp.config.num_denoise_steps)
    if indices_tensor is not None:
        _require_shape("training.denoise_indices", indices_tensor, expected_index_shape)
        if torch.any((indices_tensor < 0) | (indices_tensor >= nested_mdp.config.num_denoise_steps)):
            raise ValueError(f"training.denoise_indices must be in [0, {nested_mdp.config.num_denoise_steps - 1}]")
    if timesteps_tensor is not None:
        _require_shape("training.denoise_timesteps", timesteps_tensor, expected_index_shape)
        _require_finite("training.denoise_timesteps", timesteps_tensor)
        if torch.any((timesteps_tensor < 0) | (timesteps_tensor > 1)):
            raise ValueError("training.denoise_timesteps must be in [0, 1]")

    with _autocast_ctx(device, torch_dtype):
        output = nested_mdp.recompute_logprobs(
            observation=observations,
            chains=chains_tensor,
            denoise_indices=indices_tensor,
            denoise_timesteps=timesteps_tensor,
            return_values=return_values,
        )
    expected_step_shape = (
        batch_size,
        nested_mdp.config.num_denoise_steps,
        nested_mdp.model.config.action_horizon,
        nested_mdp.model.config.action_dim,
    )
    for name, tensor in (
        ("logprobs", output.logprobs),
        ("means", output.means),
        ("stds", output.stds),
        ("velocities", output.velocities),
    ):
        _require_shape(f"training.{name}", tensor, expected_step_shape)
        _require_finite(f"training.{name}", tensor)
    _check_transition_stds(
        "training.stds",
        output.stds,
        sample_method=nested_mdp.config.sample_method,
    )
    if return_values:
        if output.values is None:
            raise ValueError("Training forward requested values but the nested MDP returned None")
        _require_shape("training.values", output.values, (batch_size,))
        _require_finite("training.values", output.values)
    return output


def _check_observation_batch(
    observations: model_types.Observation[torch.Tensor],
    expected_batch_size: int | None = None,
) -> int:
    if observations.state.ndim != 2:
        raise ValueError(f"Observation state must be [B, D], got {tuple(observations.state.shape)}")
    _require_finite("observation.state", observations.state)
    batch_size = int(observations.state.shape[0])
    if expected_batch_size is not None and batch_size != expected_batch_size:
        raise ValueError(f"Observation batch size must be {expected_batch_size}, got {batch_size}")
    if batch_size <= 0:
        raise ValueError("Observation batch must be non-empty")
    if not observations.images:
        raise ValueError("Observation batch contains no images")
    for key, image in observations.images.items():
        if image.ndim != 4 or image.shape[0] != batch_size:
            raise ValueError(f"Observation image {key!r} must be [B, C, H, W], got {tuple(image.shape)}")
        _require_finite(f"observation.images[{key!r}]", image)
        image_mask = observations.image_masks.get(key)
        if image_mask is None or image_mask.ndim != 1 or image_mask.shape[0] != batch_size:
            shape = None if image_mask is None else tuple(image_mask.shape)
            raise ValueError(f"Observation image mask {key!r} must be [B], got {shape}")
    if observations.tokenized_prompt is None or observations.tokenized_prompt_mask is None:
        raise ValueError("pi0.5 Observation requires tokenized prompt and prompt mask")
    if observations.tokenized_prompt.shape[0] != batch_size:
        raise ValueError("Prompt batch size differs from state batch size")
    if observations.tokenized_prompt_mask.shape != observations.tokenized_prompt.shape:
        raise ValueError("Prompt mask shape differs from tokenized prompt shape")
    return batch_size


def _as_tensor(value: torch.Tensor | np.ndarray, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _require_shape(name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != tuple(expected):
        raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contains NaN/Inf")


def _check_transition_stds(name: str, tensor: torch.Tensor, *, sample_method: str) -> None:
    if torch.any(tensor < 0):
        raise FloatingPointError(f"{name} must be non-negative")
    if sample_method != "flow_ode" and torch.any(tensor <= 0):
        raise FloatingPointError(f"{name} must be strictly positive for {sample_method}")


def postprocess_action_chunk(
    adapter: Pi05LiberoRLAdapter,
    normalized_actions: torch.Tensor | np.ndarray,
    env_observation: dict[str, Any],
    max_steps: int,
) -> np.ndarray:
    """Convert one normalized pi0.5 action chunk into executable LIBERO actions."""

    if not 1 <= max_steps <= adapter.action_horizon:
        raise ValueError(f"max_steps must be in [1, {adapter.action_horizon}], got {max_steps}")

    if torch.is_tensor(normalized_actions):
        action_chunk = normalized_actions.detach().cpu().float().numpy()
    else:
        action_chunk = np.asarray(normalized_actions, dtype=np.float32)
    if action_chunk.ndim == 3:
        if action_chunk.shape[0] != 1:
            raise ValueError(f"Expected one rollout action batch, got {action_chunk.shape}")
        action_chunk = action_chunk[0]
    if action_chunk.ndim != 2:
        raise ValueError(f"Expected normalized actions [H, D], got {action_chunk.shape}")
    if action_chunk.shape[0] != adapter.action_horizon:
        raise ValueError(f"Normalized action horizon must be {adapter.action_horizon}, got {action_chunk.shape[0]}")
    if action_chunk.shape[1] != adapter.action_dim:
        raise ValueError(f"Normalized action dimension must be {adapter.action_dim}, got {action_chunk.shape[1]}")
    if not np.isfinite(action_chunk).all():
        raise FloatingPointError("Normalized pi0.5 action chunk contains NaN/Inf")

    action_count = min(max_steps, action_chunk.shape[0])
    env_actions = np.stack(
        [
            adapter.action_to_env_action(torch.from_numpy(action_chunk[index]), env_observation)
            for index in range(action_count)
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    if env_actions.shape != (action_count, adapter.env_action_dim):
        raise ValueError(
            "Postprocessed action shape mismatch: expected "
            f"{(action_count, adapter.env_action_dim)}, got {env_actions.shape}"
        )
    if not np.isfinite(env_actions).all():
        raise FloatingPointError("Transformed LIBERO action chunk contains NaN/Inf")
    return env_actions


def action_chunk_to_env(
    adapter: Pi05LiberoRLAdapter,
    normalized_actions: torch.Tensor | np.ndarray,
    env_observation: dict[str, Any],
    max_steps: int,
) -> np.ndarray:
    """Compatibility alias for :func:`postprocess_action_chunk`."""

    return postprocess_action_chunk(adapter, normalized_actions, env_observation, max_steps)


def split_parameter_groups(
    model: torch.nn.Module,
    policy_lr: float,
    value_lr: float,
) -> list[dict[str, Any]]:
    """Create named DeepSpeed groups for policy and critic parameters."""

    value_head = getattr(model, "value_head", None)
    if not isinstance(value_head, torch.nn.Module):
        raise TypeError("PPO parameter grouping requires model.value_head to be a torch.nn.Module")
    value_parameter_ids = {id(parameter) for parameter in value_head.parameters()}
    policy_parameters: list[torch.nn.Parameter] = []
    value_parameters: list[torch.nn.Parameter] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in value_parameter_ids:
            value_parameters.append(parameter)
        else:
            policy_parameters.append(parameter)
    if not policy_parameters:
        raise RuntimeError("No trainable pi0.5 policy parameters were selected")
    if not value_parameters:
        raise RuntimeError("No trainable pi0.5 value-head parameters were selected")
    return [
        {"params": policy_parameters, "name": "policy", "lr": policy_lr},
        {"params": value_parameters, "name": "value", "lr": value_lr},
    ]


def trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return a compact checkpoint containing every parameter updated by PPO."""

    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items() if name in trainable_names}
