from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from openpi import transforms
from openpi.models import model as _model
from openpi.shared import normalize as _normalize
from openpi.training import config as _config


LIBERO_IMAGE_KEYS = ("agentview_image", "image", "observation/image", "base_0_rgb")
LIBERO_WRIST_IMAGE_KEYS = ("robot0_eye_in_hand_image", "wrist_image", "observation/wrist_image", "left_wrist_0_rgb")
LIBERO_STATE_KEYS = (
    "state",
    "observation/state",
    "robot_state",
    "proprio",
)
LIBERO_STATE_PART_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)


@dataclasses.dataclass
class Pi05LiberoRLAdapter:
    """Glue code between Libero raw env data and openpi pi0.5 RL modules."""

    train_config_name: str = "pi05_libero"
    checkpoint_path: str | Path | None = None
    lora_adapter_path: str | Path | None = None
    reference_dir: str | Path = "/mnt/data/lcx1/yiqinworkspace/openpi/RLinf-Pi05-LIBERO-SFT"
    assets_dir: str | Path = "/mnt/data/lcx1/yiqinworkspace/openpi/assets"
    libero_repo_dir: str | Path = "/mnt/data/lcx1/yiqinworkspace/AcceRL/LIBERO"
    asset_id: str = "physical-intelligence/libero"
    device: str = "cuda"
    default_prompt: str = ""
    action_horizon: int = 10
    action_dim: int = 32
    env_action_dim: int = 7
    checkpoint_uses_extra_delta_transform: bool = True

    def __post_init__(self) -> None:
        self.train_config = _config.get_config(self.train_config_name)
        self.data_config = self.train_config.data.create(self.train_config.assets_dirs, self.train_config.model)
        self.norm_stats = self._load_norm_stats()
        data_inputs = list(self.data_config.data_transforms.inputs)
        data_outputs = list(self.data_config.data_transforms.outputs)
        if self.checkpoint_uses_extra_delta_transform:
            delta_action_mask = transforms.make_bool_mask(6, -1)
            data_inputs.append(transforms.DeltaActions(delta_action_mask))
            data_outputs.insert(0, transforms.AbsoluteActions(delta_action_mask))
        self.input_transform = transforms.compose(
            [
                transforms.InjectDefaultPrompt(self.default_prompt),
                *data_inputs,
                transforms.Normalize(self.norm_stats, use_quantiles=self.data_config.use_quantile_norm),
                *self.data_config.model_transforms.inputs,
            ]
        )
        self.output_transform = transforms.compose(
            [
                *self.data_config.model_transforms.outputs,
                transforms.Unnormalize(self.norm_stats, use_quantiles=self.data_config.use_quantile_norm),
                *data_outputs,
            ]
        )

    def make_libero_env(self, task_suite_name: str = "libero_spatial", task_id: int = 0, **kwargs: Any) -> Any:
        """Create a Libero environment from the configured Libero repository."""

        import sys

        libero_repo_dir = Path(self.libero_repo_dir)
        if str(libero_repo_dir) not in sys.path:
            sys.path.insert(0, str(libero_repo_dir))

        try:
            from libero.libero import benchmark
            from libero.libero.envs import OffScreenRenderEnv
        except ImportError as exc:
            raise ImportError(
                f"Could not import LIBERO from {libero_repo_dir}. "
                "Use the Libero environment and ensure the repo path is on PYTHONPATH."
            ) from exc

        benchmark_dict = benchmark.get_benchmark_dict()
        if task_suite_name not in benchmark_dict:
            raise KeyError(f"Unknown Libero task suite {task_suite_name!r}; available: {list(benchmark_dict)}")
        task_suite = benchmark_dict[task_suite_name]()
        task = task_suite.get_task(task_id)
        task_language = str(task.language).strip()
        if not task_language:
            raise ValueError(f"LIBERO task {task_suite_name}/{task_id} has an empty language instruction")
        self.default_prompt = task_language
        bddl_file = Path(task.bddl_file)
        if not bddl_file.exists():
            fallback_candidates = [
                libero_repo_dir / "libero" / "libero" / "bddl_files" / task_suite_name / bddl_file.name,
                libero_repo_dir / "libero" / "bddl_files" / task_suite_name / bddl_file.name,
                libero_repo_dir / "bddl_files" / task_suite_name / bddl_file.name,
            ]
            for fallback in fallback_candidates:
                if fallback.exists():
                    bddl_file = fallback
                    break
        seed = kwargs.pop("seed", 0) if "seed" in kwargs else 0
        cycle_init_states = bool(kwargs.pop("cycle_init_states", False))
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=kwargs.pop("camera_heights", 256),
            camera_widths=kwargs.pop("camera_widths", 256),
            ignore_done=kwargs.pop("ignore_done", True),
            **kwargs,
        )
        if hasattr(env, "seed"):
            env.seed(seed)

        try:
            init_states = task_suite.get_task_init_states(task_id)
        except Exception as exc:
            # LIBERO's loader predates PyTorch 2.6, where torch.load changed its
            # default to weights_only=True. These repository-provided benchmark
            # states contain NumPy arrays and therefore require the legacy mode.
            if "Weights only load failed" not in str(exc):
                raise
            import os

            from libero.libero import get_libero_path

            init_states_path = os.path.join(
                get_libero_path("init_states"),
                task.problem_folder,
                task.init_states_file,
            )
            init_states = torch.load(init_states_path, weights_only=False)
        if len(init_states) == 0:
            raise ValueError(f"LIBERO task {task_suite_name}/{task_id} has no benchmark initial states")
        initial_state_index = seed % len(init_states)
        reset_count = 0
        original_reset = env.reset

        def reset_to_benchmark_state() -> Any:
            nonlocal reset_count
            original_reset()
            state_index = (initial_state_index + reset_count) % len(init_states) if cycle_init_states else initial_state_index
            observation = env.set_init_state(np.asarray(init_states[state_index]))
            reset_count += 1
            for _ in range(15):
                zero_action = np.zeros(self.env_action_dim, dtype=np.float32)
                zero_action[-1] = -1.0
                observation, *_ = env.step(zero_action)
            if isinstance(observation, tuple):
                observation = observation[0]
            observation = dict(observation)
            observation["task_description"] = task_language
            observation["init_state_index"] = state_index
            return observation

        env.reset = reset_to_benchmark_state
        print(
            {
                "event": "libero_env_initialized",
                "task_suite": task_suite_name,
                "task_id": task_id,
                "task_name": task.name,
                "task_language": task_language,
                "seed": seed,
                "init_state_index": initial_state_index,
                "cycle_init_states": cycle_init_states,
                "camera_resolution": [256, 256],
                "bddl_file": str(bddl_file),
            },
            flush=True,
        )
        return env

    def load_pi05_model_from_checkpoint(self) -> torch.nn.Module:
        """Load a pi0.5 PyTorch checkpoint and move it to the configured device."""

        checkpoint = self._resolve_checkpoint_path()
        if checkpoint is None:
            raise FileNotFoundError(
                "Cannot find a pi0.5 checkpoint under the configured path/assets. "
                "Expected a .safetensors file or checkpoint directory containing model.safetensors/params. "
                f"checkpoint_path={self.checkpoint_path}, assets_dir={self.assets_dir}"
            )

        if checkpoint.name == "model.safetensors" or checkpoint.suffix == ".safetensors":
            model = self.train_config.model.load_pytorch(self.train_config, str(checkpoint))
            model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        elif (checkpoint / "model.safetensors").exists():
            model = self.train_config.model.load_pytorch(self.train_config, str(checkpoint / "model.safetensors"))
            model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        elif (checkpoint / "params").exists():
            raise FileNotFoundError(
                "Found a JAX params checkpoint, but this RL implementation uses the PyTorch pi0.5 model. "
                "Please provide a .safetensors PyTorch checkpoint."
            )
        else:
            raise FileNotFoundError(f"Unsupported checkpoint layout: {checkpoint}")

        if self.lora_adapter_path is not None:
            self._load_lora_adapter(model, Path(self.lora_adapter_path))
        model.to(self.device)
        model.train()
        return model

    @staticmethod
    def _load_lora_adapter(model: torch.nn.Module, adapter_path: Path) -> None:
        """Inject the SFT LoRA topology and load adapter-only safetensors."""
        import safetensors.torch

        if adapter_path.is_dir():
            adapter_file = adapter_path / "adapter_model.safetensors"
            config_file = adapter_path / "adapter_config.json"
        else:
            adapter_file = adapter_path
            config_file = adapter_path.parent / "adapter_config.json"
        if not adapter_file.is_file():
            raise FileNotFoundError(f"LoRA adapter weights not found: {adapter_file}")
        if not config_file.is_file():
            raise FileNotFoundError(f"LoRA adapter config not found: {config_file}")

        adapter_config = json.loads(config_file.read_text())
        from openpi.sft.train_pi05_lora import SFTConfig, inject_lora

        lora_config = SFTConfig(
            lora_rank=int(adapter_config["lora_rank"]),
            lora_alpha=float(adapter_config["lora_alpha"]),
            lora_dropout=0.0,
        )
        replaced = inject_lora(model, lora_config)
        adapter_state = safetensors.torch.load_file(str(adapter_file), device="cpu")
        expected = {name for name, _ in model.named_parameters() if ".lora_" in name}
        missing = sorted(expected - set(adapter_state))
        unexpected = sorted(set(adapter_state) - expected)
        if missing or unexpected:
            raise RuntimeError(
                f"LoRA checkpoint topology mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        incompatible = model.load_state_dict(adapter_state, strict=False)
        non_lora_missing = [name for name in incompatible.missing_keys if ".lora_" in name]
        if non_lora_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Failed to load LoRA adapter: missing={non_lora_missing[:10]}, "
                f"unexpected={incompatible.unexpected_keys[:10]}"
            )
        print(
            {"event": "lora_adapter_loaded", "path": str(adapter_file), "layers": len(replaced)},
            flush=True,
        )

    def env_obs_to_model_obs(self, env_obs: dict[str, Any]) -> _model.Observation[torch.Tensor]:
        """Convert one raw Libero observation dict to a batched torch Observation."""

        raw = {
            "observation/state": self._extract_state(env_obs),
            "observation/image": self._extract_image(env_obs, LIBERO_IMAGE_KEYS),
            "observation/wrist_image": self._extract_image(env_obs, LIBERO_WRIST_IMAGE_KEYS, allow_missing=True),
            "prompt": self._extract_prompt(env_obs),
        }
        if raw["observation/wrist_image"] is None:
            raw["observation/wrist_image"] = np.zeros_like(raw["observation/image"])

        transformed = self.input_transform(raw)
        batched = self._numpy_tree_to_torch_batch(transformed)
        return _model.Observation.from_dict(batched)

    def action_to_env_action(self, action_tensor: torch.Tensor, env_obs: dict[str, Any] | None = None) -> np.ndarray:
        """Convert a normalized model action chunk to one LIBERO environment action."""

        action = action_tensor.detach().cpu().float().numpy()
        if action.ndim == 3:
            action = action[0]
        # Output transforms expect an action chunk: [action_horizon, action_dim].
        # A single receding-horizon action arrives as [action_dim], so retain its time axis.
        if action.ndim == 1:
            action = action[None, :]
        if action.ndim != 2:
            raise ValueError(f"Expected model action [H, action_dim] or [action_dim], got {action.shape}")
        action = action[: self.action_horizon]

        # The shared norm-stats tree contains both ``actions`` and ``state``. Unnormalize
        # runs in strict mode, so retain the current state even though LiberoOutputs drops it.
        state = self._extract_state(env_obs) if env_obs is not None else np.zeros(8, dtype=np.float32)
        outputs = self.output_transform({"actions": action, "state": state})
        env_actions = np.asarray(outputs["actions"], dtype=np.float32)
        if env_actions.ndim == 2:
            env_actions = env_actions[0]
        env_action = env_actions[: self.env_action_dim]
        if not np.isfinite(env_action).all():
            raise FloatingPointError("Transformed LIBERO action contains NaN/Inf")
        return env_action

    def collate_observations(self, observations: list[_model.Observation[torch.Tensor]]) -> _model.Observation[torch.Tensor]:
        if not observations:
            raise ValueError("Cannot collate an empty observation list")
        data = [obs.to_dict() for obs in observations]
        collated = _tree_stack(data)
        return _model.Observation.from_dict(collated)

    def _load_norm_stats(self) -> dict[str, transforms.NormStats]:
        candidates = [
            Path(self.assets_dir) / "pi05_libero" / self.asset_id / "norm_stats.json",
            Path(self.assets_dir) / self.asset_id / "norm_stats.json",
            Path(self.reference_dir) / "physical-intelligence" / "libero" / "norm_stats.json",
            Path(self.reference_dir) / "norm_stats.json",
        ]
        for norm_stats_path in candidates:
            if norm_stats_path.exists():
                if norm_stats_path.is_file():
                    data = json.loads(norm_stats_path.read_text())
                    if "norm_stats" in data:
                        data = data["norm_stats"]
                    return {
                        key: _normalize.NormStats(
                            mean=np.asarray(value["mean"], dtype=np.float32),
                            std=np.asarray(value["std"], dtype=np.float32),
                            q01=np.asarray(value["q01"], dtype=np.float32) if value.get("q01") is not None else None,
                            q99=np.asarray(value["q99"], dtype=np.float32) if value.get("q99") is not None else None,
                        )
                        for key, value in data.items()
                    }
                return _normalize.load(norm_stats_path)
        raise FileNotFoundError(
            "Cannot find norm_stats.json under assets_dir/reference_dir. "
            f"Tried: {[str(path) for path in candidates]}"
        )

    def _resolve_checkpoint_path(self) -> Path | None:
        candidates: list[Path] = []
        if self.checkpoint_path is not None:
            candidates.append(Path(self.checkpoint_path))
        candidates.append(Path(self.reference_dir) / "model.safetensors")
        candidates.extend(Path(self.reference_dir).glob("**/model.safetensors"))
        candidates.extend(Path(self.reference_dir).glob("**/*.safetensors"))
        candidates.extend(Path(self.assets_dir).glob("**/model.safetensors"))
        candidates.extend(Path(self.assets_dir).glob("**/*.safetensors"))
        candidates.extend(path.parent for path in Path(self.assets_dir).glob("**/params"))
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _extract_prompt(self, env_obs: dict[str, Any]) -> str:
        for key in ("prompt", "task_description", "language_instruction", "task"):
            if key in env_obs:
                value = env_obs[key]
                return value[0] if isinstance(value, list) else str(value)
        return self.default_prompt

    @staticmethod
    def _extract_image(env_obs: dict[str, Any], keys: Iterable[str], allow_missing: bool = False) -> np.ndarray | None:
        for key in keys:
            if key in env_obs:
                image = _to_hwc_uint8(env_obs[key])
                # RLinf's pi0.5 LIBERO SFT pipeline rotates both simulator cameras
                # by 180 degrees before tokenization.
                return np.ascontiguousarray(image[::-1, ::-1])
        if allow_missing:
            return None
        raise KeyError(f"Cannot find image key. Tried: {list(keys)}; available: {list(env_obs)}")

    @staticmethod
    def _extract_state(env_obs: dict[str, Any]) -> np.ndarray:
        for key in LIBERO_STATE_KEYS:
            if key in env_obs:
                state = np.asarray(env_obs[key], dtype=np.float32).reshape(-1)
                return _pad_or_trim(state, 8)
        if all(key in env_obs for key in LIBERO_STATE_PART_KEYS):
            from robosuite.utils.transform_utils import quat2axisangle

            eef_pos = np.asarray(env_obs["robot0_eef_pos"], dtype=np.float32).reshape(3)
            eef_quat = np.asarray(env_obs["robot0_eef_quat"], dtype=np.float32).reshape(4)
            eef_axis_angle = np.asarray(quat2axisangle(eef_quat), dtype=np.float32).reshape(3)
            gripper_qpos = np.asarray(env_obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
            return _pad_or_trim(np.concatenate([eef_pos, eef_axis_angle, gripper_qpos]), 8)
        raise KeyError(f"Cannot find state in Libero observation. Available keys: {list(env_obs)}")

    @staticmethod
    def _numpy_tree_to_torch_batch(tree: Any) -> Any:
        if isinstance(tree, dict):
            return {key: Pi05LiberoRLAdapter._numpy_tree_to_torch_batch(value) for key, value in tree.items()}
        if isinstance(tree, np.ndarray):
            return torch.from_numpy(np.asarray(tree)).unsqueeze(0)
        if isinstance(tree, np.generic):
            return torch.from_numpy(np.asarray(tree)).unsqueeze(0)
        return tree


def _to_hwc_uint8(image: Any) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if np.issubdtype(image.dtype, np.floating):
        if image.min() >= -1.0 and image.max() <= 1.0:
            image = (image + 1.0) * 127.5
        elif image.max() <= 1.0:
            image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _pad_or_trim(x: np.ndarray, target_dim: int) -> np.ndarray:
    if x.shape[0] == target_dim:
        return x.astype(np.float32)
    if x.shape[0] > target_dim:
        return x[:target_dim].astype(np.float32)
    return np.pad(x, (0, target_dim - x.shape[0])).astype(np.float32)


def _tree_stack(items: list[Any]) -> Any:
    first = items[0]
    if first is None:
        if not all(item is None for item in items):
            raise ValueError("Cannot collate a mixture of None and non-None values")
        return None
    if isinstance(first, dict):
        return {key: _tree_stack([item[key] for item in items]) for key in first}
    if torch.is_tensor(first):
        return torch.cat(items, dim=0)
    raise TypeError(f"Unsupported collate leaf type: {type(first)!r}")
