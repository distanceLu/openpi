"""Run PI0.5 LoRA inference and save one video per camera for each HDF5 trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from PIL import ImageDraw
import safetensors.torch
import torch
from torch.utils.data import default_collate

from openpi import transforms
from openpi.models import tokenizer as tokenizer_api
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.sft_shiji.dataset import SymmetricActionUnnormalizer
from openpi.sft_shiji.dataset import batch_to_model
from openpi.sft_shiji.hdf5_dataset import HDF5Trajectory
from openpi.sft_shiji.hdf5_dataset import list_trajectory_files
from openpi.sft_shiji.lora import EXPERIMENTS
from openpi.sft_shiji.lora import configure_finetuning
from openpi.sft_shiji.lora import load_adapter
from openpi.shared import normalize as normalize_api

CAMERA_OUTPUT_NAMES = {
    "base_0_rgb": "camera_paper_aruco",
    "left_wrist_0_rgb": "camera_pool",
    "right_wrist_0_rgb": "camera_pool1",
}


class _InferenceTransform:
    def __init__(
        self,
        prompt: str,
        norm_stats,
        max_token_len: int,
        action_horizon: int,
        *,
        use_robot_state: bool = False,
    ):
        self.prompt = prompt
        self.use_robot_state = use_robot_state
        self.action_horizon = action_horizon
        self.resize = transforms.ResizeImages(224, 224)
        state_norm_stats = {"state": norm_stats["state"]} if use_robot_state else None
        self.normalizer = transforms.Normalize(state_norm_stats, use_quantiles=True, strict=False)
        action_q01 = np.asarray(norm_stats["actions"].q01, dtype=np.float32)
        action_q99 = np.asarray(norm_stats["actions"].q99, dtype=np.float32)
        self.symmetric_action_norm = bool(np.allclose(action_q01, -action_q99, rtol=1e-5, atol=1e-8))
        if self.symmetric_action_norm:
            self.action_scale = action_q99
            if np.any(self.action_scale <= 0) or not np.isfinite(self.action_scale).all():
                raise ValueError(f"Invalid symmetric action scales: {self.action_scale}")
            self.action_unnormalizer = SymmetricActionUnnormalizer(norm_stats["actions"])
        else:
            self.action_scale = None
            self.legacy_action_normalizer = transforms.Normalize(
                {"actions": norm_stats["actions"]}, use_quantiles=True
            )
            self.action_unnormalizer = transforms.Unnormalize(
                {"actions": norm_stats["actions"]}, use_quantiles=True
            )
        self.tokenize = transforms.TokenizePrompt(
            tokenizer_api.PaligemmaTokenizer(max_token_len), discrete_state_input=use_robot_state
        )
        self.pad = transforms.PadStatesAndActions(32)

    def __call__(self, data: dict) -> dict:
        data = dict(data)
        data["prompt"] = self.prompt
        data = self.resize(data)
        if not self.use_robot_state:
            data["state"] = np.zeros_like(data["state"], dtype=np.float32)
        data = self.normalizer(data)
        data["state"] = np.asarray(data["state"], dtype=np.float32)
        if self.symmetric_action_norm:
            data["actions"] = np.clip(
                np.asarray(data["actions"], dtype=np.float32) / self.action_scale,
                -1.0,
                1.0,
            )
        else:
            data["actions"] = self.legacy_action_normalizer(
                {"actions": np.asarray(data["actions"], dtype=np.float32)}
            )["actions"]
        return self.pad(self.tokenize(data))

    def unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        values = np.asarray(actions)
        if self.symmetric_action_norm:
            values = np.clip(values, -1.0, 1.0)
        return np.asarray(self.action_unnormalizer({"actions": values})["actions"], dtype=np.float32)


def _model_file(checkpoint: Path) -> Path:
    path = checkpoint if checkpoint.suffix == ".safetensors" else checkpoint / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"Base PI0.5 checkpoint not found: {path}")
    return path


def _find_training_config(adapter: Path) -> tuple[Path, dict] | None:
    """Load the training configuration stored beside an adapter checkpoint."""
    for parent in (adapter.parent, *adapter.parents):
        config_path = parent / "train_config.json"
        if config_path.is_file():
            return config_path, json.loads(config_path.read_text())
    return None


def _resolve_training_behavior(
    adapter: Path,
    use_robot_state: bool | None,
    mask_non_z_actions: bool | None,
) -> tuple[bool, bool, Path | None]:
    config_result = _find_training_config(adapter)
    if config_result is None:
        if use_robot_state is None or mask_non_z_actions is None:
            raise FileNotFoundError(
                "Cannot infer training behavior: train_config.json was not found beside the adapter. "
                "Pass both --use-robot-state/--no-use-robot-state and "
                "--mask-non-z-actions/--no-mask-non-z-actions."
            )
        return use_robot_state, mask_non_z_actions, None

    config_path, config = config_result
    trained_use_robot_state = bool(config.get("use_robot_state", False))
    trained_mask_non_z_actions = bool(config.get("mask_non_z_actions", False))
    requested = {
        "use_robot_state": use_robot_state,
        "mask_non_z_actions": mask_non_z_actions,
    }
    trained = {
        "use_robot_state": trained_use_robot_state,
        "mask_non_z_actions": trained_mask_non_z_actions,
    }
    conflicts = [
        f"{name}: requested={requested[name]!r}, trained={trained[name]!r}"
        for name in requested
        if requested[name] is not None and requested[name] != trained[name]
    ]
    if conflicts:
        raise ValueError(
            "Inference options must match the training configuration in "
            f"{config_path}: " + "; ".join(conflicts)
        )
    return trained_use_robot_state, trained_mask_non_z_actions, config_path


def _load_model(
    base_checkpoint: Path,
    adapter: Path,
    device: torch.device,
    action_horizon: int,
    max_token_len: int,
    finetune_mode: str,
    *,
    use_robot_state: bool,
):
    config = Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=action_horizon,
        max_token_len=max_token_len,
        discrete_state_input=use_robot_state,
        dtype="bfloat16",
        pytorch_compile_mode=None,
    )
    model = PI0Pytorch(config)
    safetensors.torch.load_model(model, _model_file(base_checkpoint), device="cpu", strict=False)
    if finetune_mode not in EXPERIMENTS:
        raise ValueError(f"Unknown finetune mode {finetune_mode!r}; choose from {sorted(EXPERIMENTS)}")
    configure_finetuning(model, finetune_mode, dropout=0.0)
    load_adapter(model, adapter)
    model.to(device).eval()
    return model


def _move(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move(child, device) for key, child in value.items()}
    return value


def _even_rgb(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.uint8)
    height, width = frame.shape[:2]
    return np.ascontiguousarray(frame[: height - height % 2, : width - width % 2, :3])


def _open_video(path: Path, fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return imageio.get_writer(
            path,
            format="FFMPEG",
            mode="I",
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=2,
            ffmpeg_params=["-movflags", "+faststart"],
        )
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(
            "MP4 encoder is unavailable. Install imageio-ffmpeg in the uv environment: "
            "uv add imageio-ffmpeg"
        ) from error


def _comparison_panel(predicted: np.ndarray, target: np.ndarray, height: int, timestep: int) -> np.ndarray:
    width = 560
    image = Image.new("RGB", (width, height), (247, 248, 250))
    draw = ImageDraw.Draw(image)
    draw.text((18, 12), f"Predicted (red) vs ground truth (blue) z | timestep {timestep}", fill=(25, 30, 38))
    top = 42
    row_bottom = height - 18
    center = (top + row_bottom) // 2
    plot_left, plot_right = 78, width - 18
    horizon = predicted.shape[0]
    draw.text((18, center - 7), "z", fill=(35, 40, 48))
    draw.rectangle((plot_left, top, plot_right, row_bottom), outline=(190, 195, 202))
    draw.line((plot_left, center, plot_right, center), fill=(215, 218, 223))
    extent = max(float(np.max(np.abs(predicted[:, 2]))), float(np.max(np.abs(target[:, 2]))), 1e-8) * 1.1

    def points(values: np.ndarray) -> list[tuple[int, int]]:
        return [
            (
                int(plot_left + index * (plot_right - plot_left) / max(1, horizon - 1)),
                int(center - value / extent * max(1, row_bottom - top) / 2),
            )
            for index, value in enumerate(values)
        ]

    predicted_points = points(predicted[:, 2])
    target_points = points(target[:, 2])
    if horizon > 1:
        draw.line(target_points, fill=(45, 105, 190), width=3)
        draw.line(predicted_points, fill=(210, 55, 55), width=3)
    else:
        for point, color in ((target_points[0], (45, 105, 190)), (predicted_points[0], (210, 55, 55))):
            draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill=color)
    return np.asarray(image, dtype=np.uint8)


def _comparison_frame(frame: np.ndarray, predicted: np.ndarray, target: np.ndarray, timestep: int) -> np.ndarray:
    frame = _even_rgb(frame)
    panel = _comparison_panel(predicted, target, frame.shape[0], timestep)
    return _even_rgb(np.concatenate((frame, panel), axis=1))


def _z_error_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    motion_threshold: float,
) -> dict[str, float]:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[-1] != 6:
        raise ValueError(f"Expected matching [N, 6] arrays, got {predicted.shape} and {target.shape}")
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("Predicted and target actions must contain only finite values")

    if not np.isfinite(motion_threshold) or motion_threshold <= 0:
        raise ValueError(f"motion_threshold must be finite and positive, got {motion_threshold}")

    predicted_z = predicted[:, 2]
    target_z = target[:, 2]
    error = predicted_z - target_z
    absolute_error = np.abs(error)
    absolute_target = np.abs(target_z)
    epsilon = np.finfo(np.float64).eps

    z_rmse = float(np.sqrt(np.mean(np.square(error))))
    zero_rmse = float(np.sqrt(np.mean(np.square(target_z))))
    moving = absolute_target >= motion_threshold
    stationary = ~moving
    metrics = {
        "z_mae": float(np.mean(absolute_error)),
        "z_rmse": z_rmse,
        "z_wape_percent": float(
            100.0 * np.sum(absolute_error) / max(float(np.sum(absolute_target)), epsilon)
        ),
        "z_thresholded_relative_percent": float(
            100.0 * np.mean(absolute_error / np.maximum(absolute_target, motion_threshold))
        ),
        "mean_abs_target_z": float(np.mean(absolute_target)),
        "zero_baseline_z_rmse": zero_rmse,
        "z_advantage_vs_zero_percent": 100.0 * (1.0 - z_rmse / max(zero_rmse, epsilon)),
        "motion_threshold": motion_threshold,
        "moving_fraction": float(np.mean(moving)),
    }
    if np.any(moving):
        moving_error = error[moving]
        metrics.update({
            "moving_z_mae": float(np.mean(np.abs(moving_error))),
            "moving_z_rmse": float(np.sqrt(np.mean(np.square(moving_error)))),
            "moving_z_wape_percent": float(
                100.0 * np.sum(np.abs(moving_error))
                / max(float(np.sum(absolute_target[moving])), epsilon)
            ),
        })
    if np.any(stationary):
        metrics.update({
            "stationary_z_mae": float(np.mean(absolute_error[stationary])),
            "stationary_false_motion_rate": float(
                np.mean(np.abs(predicted_z[stationary]) >= motion_threshold)
            ),
        })
    return metrics


def _print_prediction_comparison(
    episode_id: str,
    timestep: int,
    predicted: np.ndarray,
    target: np.ndarray,
    metrics: dict[str, float],
) -> None:
    threshold = metrics["motion_threshold"]
    z_thresholded_relative_percent = (
        100.0
        * np.abs(predicted[:, 2] - target[:, 2])
        / np.maximum(np.abs(target[:, 2]), threshold)
    )
    array_options = {"precision": 6, "suppress_small": False, "max_line_width": 200}
    print(f"\n{episode_id}: timestep={timestep} action column=z")
    print(f"  predicted_z:\n{np.array2string(predicted[:, 2], **array_options)}")
    print(f"  ground_truth_z:\n{np.array2string(target[:, 2], **array_options)}")
    print(
        "  z_thresholded_relative_error_percent:\n"
        f"{np.array2string(z_thresholded_relative_percent, **array_options)}"
    )
    print(
        "  aggregate: "
        f"z_wape={metrics['z_wape_percent']:.2f}% "
        f"z_mae={metrics['z_mae']:.6f} "
        f"z_rmse={metrics['z_rmse']:.6f} "
        f"zero_rmse={metrics['zero_baseline_z_rmse']:.6f} "
        f"advantage_vs_zero={metrics['z_advantage_vs_zero_percent']:.2f}%"
    )


def _infer_trajectory(
    trajectory_path: Path,
    output_root: Path,
    model: PI0Pytorch,
    transform: _InferenceTransform,
    device: torch.device,
    fps: float,
    diffusion_steps: int,
    action_mask: torch.Tensor | None,
    motion_threshold: float,
) -> None:
    episode_output = output_root / trajectory_path.stem
    episode_output.mkdir(parents=True, exist_ok=True)
    temporary_videos = {
        model_key: episode_output / f".{camera_name}_prediction_vs_ground_truth.tmp.mp4"
        for model_key, camera_name in CAMERA_OUTPUT_NAMES.items()
    }
    final_videos = {
        model_key: episode_output / f"{camera_name}_prediction_vs_ground_truth.mp4"
        for model_key, camera_name in CAMERA_OUTPUT_NAMES.items()
    }
    for path in temporary_videos.values():
        path.unlink(missing_ok=True)
    writers = {model_key: _open_video(path, fps) for model_key, path in temporary_videos.items()}
    predictions = []
    ground_truth = []
    per_timestep_metrics = []
    timestamps = []
    completed = False
    try:
        with HDF5Trajectory(trajectory_path) as trajectory, torch.inference_mode():
            if trajectory.action_shape != (transform.action_horizon, 6):
                raise ValueError(
                    f"HDF5 action shape must be ({transform.action_horizon}, 6), "
                    f"got {trajectory.action_shape}"
                )
            if trajectory.action_mode != "relative_pose" or "command_delta" not in trajectory.action_definition:
                raise ValueError(
                    "Inference expects consecutive relative-pose deltas, got "
                    f"mode={trajectory.action_mode!r}, definition={trajectory.action_definition!r}"
                )
            for timestep in range(trajectory.length):
                raw = trajectory.read_step(timestep)
                sample = transform(raw)
                batch = _move(default_collate([sample]), device)
                observation, _ = batch_to_model(batch)
                action = model.sample_actions(
                    device, observation, num_steps=diffusion_steps, action_mask=action_mask
                )
                normalized_action = action[0].float().cpu().numpy()
                predicted = transform.unnormalize_actions(normalized_action)[..., :6]
                if action_mask is not None:
                    predicted[..., ~action_mask[:6].cpu().numpy().astype(bool)] = 0.0
                target = np.asarray(raw["actions"], dtype=np.float32).copy()
                if action_mask is not None:
                    target[..., ~action_mask[:6].cpu().numpy().astype(bool)] = 0.0
                if timestep == 0:
                    print(
                        f"{trajectory_path.stem}: action dimensions "
                        f"model_output={tuple(action.shape)} "
                        f"normalized_episode_action={tuple(normalized_action.shape)} "
                        f"predicted_physical={tuple(predicted.shape)} "
                        f"ground_truth={tuple(target.shape)}"
                    )
                if predicted.shape != target.shape:
                    raise ValueError(
                        f"Action shape mismatch at timestep {timestep}: "
                        f"prediction={predicted.shape}, ground_truth={target.shape}"
                    )
                for model_key, writer in writers.items():
                    writer.append_data(
                        _comparison_frame(raw["image"][model_key], predicted, target, timestep)
                    )
                metrics = _z_error_metrics(predicted, target, motion_threshold)
                predictions.append(predicted)
                ground_truth.append(target)
                per_timestep_metrics.append(metrics)
                timestamps.append(timestep)
                _print_prediction_comparison(
                    trajectory_path.stem, timestep, predicted, target, metrics
                )
                if timestep % 25 == 0 or timestep + 1 == trajectory.length:
                    print(f"{trajectory_path.stem}: progress={timestep + 1}/{trajectory.length}")
        completed = True
    finally:
        for writer in writers.values():
            writer.close()
        if not completed:
            for path in temporary_videos.values():
                path.unlink(missing_ok=True)
    if not predictions:
        raise ValueError(f"No predictions produced for {trajectory_path}")
    for model_key, temporary in temporary_videos.items():
        temporary.replace(final_videos[model_key])
    prediction_array = np.asarray(predictions, dtype=np.float32)
    target_array = np.asarray(ground_truth, dtype=np.float32)
    trajectory_metrics = _z_error_metrics(
        prediction_array.reshape(-1, 6), target_array.reshape(-1, 6), motion_threshold
    )
    print(
        f"{trajectory_path.stem}: TRAJECTORY SUMMARY "
        f"z_wape={trajectory_metrics['z_wape_percent']:.2f}% "
        f"z_mae={trajectory_metrics['z_mae']:.6f} "
        f"z_rmse={trajectory_metrics['z_rmse']:.6f} "
        f"zero_rmse={trajectory_metrics['zero_baseline_z_rmse']:.6f} "
        f"advantage_vs_zero={trajectory_metrics['z_advantage_vs_zero_percent']:.2f}%"
    )
    np.save(episode_output / "predicted_actions.npy", prediction_array)
    np.save(episode_output / "ground_truth_actions.npy", target_array)
    np.save(episode_output / "timesteps.npy", np.asarray(timestamps, dtype=np.int64))
    (episode_output / "relative_error_metrics.json").write_text(json.dumps({
        "definition": {
            "z_wape_percent": "100 * sum(abs(pred_z - target_z)) / sum(abs(target_z))",
            "z_advantage_vs_zero_percent": "100 * (1 - model_z_rmse / zero_baseline_z_rmse)",
            "z_thresholded_relative_percent": (
                "100 * mean(abs(pred_z - target_z) / max(abs(target_z), motion_threshold))"
            ),
            "moving": "abs(target_z) >= motion_threshold",
            "stationary_false_motion_rate": (
                "mean(abs(pred_z) >= motion_threshold for stationary targets)"
            ),
            "aggregation": "episode timesteps and action-horizon steps are flattened",
        },
        "action_space": "physical relative-pose z delta after quantile unnormalization",
        "units": {"z": "same length unit as tool_pose.csv"},
        "trained_action_axes": ["z"],
        "disabled_action_axes": ["x", "y", "rx", "ry", "rz"],
        "motion_threshold": motion_threshold,
        "trajectory": trajectory_metrics,
        "per_timestep": per_timestep_metrics,
    }, indent=2, ensure_ascii=False))
    (episode_output / "inference_manifest.json").write_text(json.dumps({
        "episode_id": trajectory_path.stem,
        "source_hdf5": str(trajectory_path),
        "frames_per_camera": len(predictions),
        "fps": fps,
        "predicted_action_shape": list(prediction_array.shape),
        "predicted_action_space": "physical relative-pose z delta after quantile unnormalization",
        "ground_truth_action_shape": list(target_array.shape),
        "trained_action_axes": ["z"],
        "disabled_action_axes": ["x", "y", "rx", "ry", "rz"],
        "motion_threshold": motion_threshold,
        "relative_error_metrics": trajectory_metrics,
        "camera_videos": {camera_name: path.name for (_model_key, camera_name), path in zip(
            CAMERA_OUTPUT_NAMES.items(), final_videos.values(), strict=True
        )},
    }, indent=2, ensure_ascii=False))
    print(f"saved episode outputs: {episode_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PI0.5 LoRA inference with three per-camera videos")
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="follow the demonstrated robot tool trajectory")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=1e-4,
        help="Physical |z| threshold separating moving and stationary targets.",
    )
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--max-token-len", type=int, default=200)
    parser.add_argument(
        "--finetune-mode",
        choices=sorted(EXPERIMENTS),
        default=None,
        help="Override the adapter's training mode; omitted means inherit it.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--use-robot-state",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the adapter's training setting; omitted means inherit it.",
    )
    parser.add_argument(
        "--mask-non-z-actions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the adapter's training setting; omitted means inherit it.",
    )
    args = parser.parse_args()
    if not np.isfinite(args.motion_threshold) or args.motion_threshold <= 0:
        raise ValueError("--motion-threshold must be finite and positive")
    resolved_use_robot_state, resolved_mask_non_z_actions, config_path = _resolve_training_behavior(
        args.adapter,
        args.use_robot_state,
        args.mask_non_z_actions,
    )
    args.use_robot_state = resolved_use_robot_state
    args.mask_non_z_actions = resolved_mask_non_z_actions
    if config_path is not None:
        training_config = json.loads(config_path.read_text())
        trained_finetune_mode = str(training_config.get("finetune_mode", "a"))
        if args.finetune_mode is not None and args.finetune_mode != trained_finetune_mode:
            raise ValueError(
                f"--finetune-mode={args.finetune_mode!r} disagrees with training mode "
                f"{trained_finetune_mode!r} in {config_path}"
            )
        args.finetune_mode = trained_finetune_mode
        print(
            "inference behavior inherited from training config: "
            f"finetune_mode={args.finetune_mode}, "
            f"use_robot_state={args.use_robot_state}, "
            f"mask_non_z_actions={args.mask_non_z_actions}, "
            f"config={config_path}"
        )
    elif args.finetune_mode is None:
        raise ValueError("--finetune-mode is required when train_config.json is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PI0.5 inference")
    if args.hdf5.is_dir():
        trajectory_files = list_trajectory_files(args.hdf5)
    elif args.hdf5.is_file():
        trajectory_files = [args.hdf5]
    else:
        raise FileNotFoundError(args.hdf5)
    if not trajectory_files:
        raise FileNotFoundError(f"No HDF5 trajectory found: {args.hdf5}")

    with HDF5Trajectory(trajectory_files[0]) as first_trajectory:
        first_action_shape = first_trajectory.action_shape
        first_action_mode = first_trajectory.action_mode
        first_action_definition = first_trajectory.action_definition
        first_use_robot_state = first_trajectory.use_robot_state
    for trajectory_path in trajectory_files:
        with HDF5Trajectory(trajectory_path) as trajectory:
            if (
                trajectory.action_shape != first_action_shape
                or trajectory.action_mode != first_action_mode
                or trajectory.action_definition != first_action_definition
                or trajectory.use_robot_state != first_use_robot_state
            ):
                raise ValueError(
                    f"Inconsistent action metadata in {trajectory_path}: "
                    f"shape={trajectory.action_shape}, mode={trajectory.action_mode!r}, "
                    f"definition={trajectory.action_definition!r}, use_robot_state={trajectory.use_robot_state}; "
                    f"expected shape={first_action_shape}, mode={first_action_mode!r}, "
                    f"definition={first_action_definition!r}, use_robot_state={first_use_robot_state}"
                )
    if first_use_robot_state != args.use_robot_state:
        print(
            f"warning: adapter use_robot_state={args.use_robot_state}, HDF5 use_robot_state="
            f"{first_use_robot_state}; the stored zero state will be tokenized as constant state input"
        )
    if len(first_action_shape) != 2 or first_action_shape[1] != 6:
        raise ValueError(f"Expected HDF5 actions shaped [horizon, 6], got {first_action_shape}")
    inferred_horizon = first_action_shape[0]
    if args.action_horizon is not None and args.action_horizon != inferred_horizon:
        raise ValueError(
            f"--action-horizon={args.action_horizon} disagrees with HDF5 horizon={inferred_horizon}"
        )
    if first_action_mode != "relative_pose" or "command_delta" not in first_action_definition:
        raise ValueError(
            "Inference expects consecutive relative-pose deltas, got "
            f"mode={first_action_mode!r}, definition={first_action_definition!r}"
        )
    action_horizon = inferred_horizon

    norm_stats_dir = args.norm_stats.parent if args.norm_stats.is_file() else args.norm_stats
    norm_stats = normalize_api.load(norm_stats_dir)
    if tuple(np.asarray(norm_stats["actions"].q01).shape) != (6,):
        raise ValueError("Action normalization statistics must have exactly 6 dimensions")
    transform = _InferenceTransform(
        args.prompt,
        norm_stats,
        args.max_token_len,
        action_horizon,
        use_robot_state=args.use_robot_state,
    )
    model = _load_model(
        args.base_checkpoint,
        args.adapter,
        device,
        action_horizon,
        args.max_token_len,
        args.finetune_mode,
        use_robot_state=args.use_robot_state,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    action_mask = None
    if args.mask_non_z_actions:
        action_mask = torch.zeros(32, dtype=torch.float32, device=device)
        action_mask[2] = 1.0
    for trajectory_path in trajectory_files:
        _infer_trajectory(
            trajectory_path,
            args.output_dir,
            model,
            transform,
            device,
            args.fps,
            args.diffusion_steps,
            action_mask,
            args.motion_threshold,
        )


if __name__ == "__main__":
    main()
