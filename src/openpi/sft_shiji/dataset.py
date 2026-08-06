"""Timestamp-aligned dataset for the 2026-08-03 real-robot demonstrations."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable, Sequence
import csv
import dataclasses
import json
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
import torch

from openpi import transforms
from openpi.models import model as model_api
from openpi.models import tokenizer as tokenizer_api
from openpi.shared import normalize as normalize_api

CAMERA_TO_MODEL_KEY = {
    "camera_paper_aruco": "base_0_rgb",
    "camera_pool": "left_wrist_0_rgb",
    "camera_pool1": "right_wrist_0_rgb",
}


@dataclasses.dataclass(frozen=True)
class TimedPath:
    timestamp: int
    path: Path


@dataclasses.dataclass(frozen=True)
class AlignedSample:
    episode_id: str
    timestamp: int
    images: dict[str, Path]
    state: np.ndarray
    actions: np.ndarray

    @property
    def recording(self) -> str:
        """Backward-compatible alias: one recording time folder is one episode."""
        return self.episode_id


@dataclasses.dataclass(frozen=True)
class AlignmentConfig:
    action_horizon: int = 10
    control_hz: float = 10.0
    sample_hz: float = 10.0
    # The recorded timestamps advance in microseconds (about 50,000 per 20 Hz frame).
    timestamp_ticks_per_second: int = 1_000_000
    camera_tolerance_ms: float = 100.0
    state_tolerance_ms: float = 30.0
    action_mode: Literal["absolute_pose", "relative_pose"] = "relative_pose"

    @property
    def control_period(self) -> int:
        return round(self.timestamp_ticks_per_second / self.control_hz)

    @property
    def sample_period(self) -> int:
        return round(self.timestamp_ticks_per_second / self.sample_hz)

    @property
    def camera_tolerance(self) -> int:
        return round(self.camera_tolerance_ms * self.timestamp_ticks_per_second / 1000.0)

    @property
    def state_tolerance(self) -> int:
        return round(self.state_tolerance_ms * self.timestamp_ticks_per_second / 1000.0)


def _timestamp_from_image(path: Path) -> int:
    try:
        return int(path.name.split(".", 1)[0])
    except ValueError as error:
        raise ValueError(f"Image filename does not start with an integer timestamp: {path}") from error


def _camera_stream(directory: Path) -> list[TimedPath]:
    frames = [TimedPath(_timestamp_from_image(path), path) for path in directory.glob("*.jpg")]
    frames.extend(TimedPath(_timestamp_from_image(path), path) for path in directory.glob("*.jpeg"))
    frames.extend(TimedPath(_timestamp_from_image(path), path) for path in directory.glob("*.png"))
    return sorted(frames, key=lambda frame: (frame.timestamp, frame.path.name))



def _read_robot_poses(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: dict[int, np.ndarray] = {}
    with path.open(newline="") as stream:
        for row in csv.reader(stream):
            if len(row) < 7 or row[0].strip().lower() == "timestamp":
                continue
            try:
                timestamp = int(float(row[0]))
                pose = np.asarray([float(value) for value in row[1:7]], dtype=np.float32)
            except ValueError:
                continue
            if np.isfinite(pose).all():
                rows[timestamp] = pose
    if not rows:
        raise ValueError(f"No valid robot poses in {path}")
    timestamps = np.asarray(sorted(rows), dtype=np.int64)
    poses = np.stack([rows[int(timestamp)] for timestamp in timestamps])
    return timestamps, poses


def _nearest_index(timestamps: Sequence[int] | np.ndarray, target: int) -> tuple[int, int]:
    index = bisect_left(timestamps, target)
    candidates = []
    if index < len(timestamps):
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    if not candidates:
        raise IndexError("Cannot align against an empty timestamp stream")
    best = min(candidates, key=lambda candidate: abs(int(timestamps[candidate]) - target))
    return best, abs(int(timestamps[best]) - target)


def _first_at_or_after_index(timestamps: Sequence[int] | np.ndarray, target: int) -> tuple[int, int]:
    """Return the first frame at or after target; never leak an earlier camera frame."""
    index = bisect_left(timestamps, target)
    if index >= len(timestamps):
        raise IndexError("No camera frame exists at or after the target timestamp")
    return index, int(timestamps[index]) - target


def _subsample_anchor(frames: Iterable[TimedPath], period: int) -> list[TimedPath]:
    selected = []
    next_timestamp: int | None = None
    for frame in frames:
        if next_timestamp is None or frame.timestamp >= next_timestamp:
            selected.append(frame)
            next_timestamp = frame.timestamp + period
    return selected


def align_recording(recording_dir: Path, config: AlignmentConfig) -> list[AlignedSample]:
    """Align one date-time recording folder as one indivisible demonstration group."""
    streams = {name: _camera_stream(recording_dir / name) for name in CAMERA_TO_MODEL_KEY}
    missing = [name for name, stream in streams.items() if not stream]
    if missing:
        raise FileNotFoundError(f"{recording_dir.name} is missing camera images for {missing}")

    robot_timestamps, robot_poses = _read_robot_poses(recording_dir / "robot_state" / "tool_pose.csv")
    stream_timestamps = {
        name: [frame.timestamp for frame in stream]
        for name, stream in streams.items()
    }
    anchors = _subsample_anchor(streams["camera_paper_aruco"], config.sample_period)
    aligned: list[AlignedSample] = []

    for anchor in anchors:
        images: dict[str, Path] = {"base_0_rgb": anchor.path}
        valid = True
        for camera_name in ("camera_pool", "camera_pool1"):
            try:
                index, delay = _first_at_or_after_index(stream_timestamps[camera_name], anchor.timestamp)
            except IndexError:
                valid = False
                break
            if delay > config.camera_tolerance:
                valid = False
                break
            images[CAMERA_TO_MODEL_KEY[camera_name]] = streams[camera_name][index].path
        if not valid:
            continue

        state_index, state_error = _nearest_index(robot_timestamps, anchor.timestamp)
        if state_error > config.state_tolerance:
            continue
        state = robot_poses[state_index].copy()

        action_rows = []
        for step in range(config.action_horizon):
            # Predict executable future targets, starting at the next control tick.
            action_timestamp = anchor.timestamp + (step + 1) * config.control_period
            action_index, action_error = _nearest_index(robot_timestamps, action_timestamp)
            if action_error > config.state_tolerance:
                valid = False
                break
            action_rows.append(robot_poses[action_index])
        if not valid:
            continue
        actions = np.stack(action_rows).astype(np.float32)
        if config.action_mode == "relative_pose":
            actions = actions - state[None, :]
            actions[:, 3:] = (actions[:, 3:] + np.pi) % (2.0 * np.pi) - np.pi

        aligned.append(
            AlignedSample(
                episode_id=recording_dir.name,
                timestamp=anchor.timestamp,
                images=images,
                state=state,
                actions=actions,
            )
        )
    return aligned


def build_aligned_samples(root: str | Path, config: AlignmentConfig) -> list[AlignedSample]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {root}")
    samples = []
    errors = []
    for recording_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        try:
            samples.extend(align_recording(recording_dir, config))
        except (FileNotFoundError, ValueError) as error:
            errors.append(f"{recording_dir.name}: {error}")
    if not samples:
        details = "\n".join(errors[:10])
        raise ValueError(f"No aligned samples found under {root}.\n{details}")
    return samples


def split_by_recording(
    samples: Sequence[AlignedSample], validation_recording: str | Path
) -> tuple[list[AlignedSample], list[AlignedSample]]:
    """Reserve one whole time-folder episode; individual aligned steps are never split across sets."""
    validation_name = Path(validation_recording).name
    episode_ids = sorted({sample.episode_id for sample in samples})
    if validation_name not in episode_ids:
        raise ValueError(f"Validation episode {validation_name!r} not found; available episodes: {episode_ids}")
    train = [sample for sample in samples if sample.episode_id != validation_name]
    validation = [sample for sample in samples if sample.episode_id == validation_name]
    if not train or not validation:
        raise ValueError("Fixed recording split produced an empty training or validation set")
    return train, validation


def compute_norm_stats(samples: Sequence[AlignedSample]) -> dict[str, normalize_api.NormStats]:
    if not samples:
        raise ValueError("Cannot compute normalization statistics from no samples")
    states = np.stack([sample.state for sample in samples]).astype(np.float64)
    actions = np.concatenate([sample.actions for sample in samples], axis=0).astype(np.float64)

    def statistics(values: np.ndarray) -> normalize_api.NormStats:
        return normalize_api.NormStats(
            mean=values.mean(axis=0),
            std=values.std(axis=0),
            q01=np.quantile(values, 0.01, axis=0),
            q99=np.quantile(values, 0.99, axis=0),
        )

    return {"state": statistics(states), "actions": statistics(actions)}


def save_alignment_report(
    destination: Path,
    all_samples: Sequence[AlignedSample],
    train_samples: Sequence[AlignedSample],
    validation_samples: Sequence[AlignedSample],
    config: AlignmentConfig,
) -> None:
    counts: dict[str, int] = {}
    for sample in all_samples:
        counts[sample.episode_id] = counts.get(sample.episode_id, 0) + 1
    payload = {
        "alignment": dataclasses.asdict(config),
        "camera_mapping": CAMERA_TO_MODEL_KEY,
        "camera_match_policy": "first frame at or after camera_paper_aruco timestamp",
        "state_source": "robot_state/tool_pose.csv",
        "paper_state_usage": "not used for model input",
        "session_meta_usage": "not used for model input; metadata only",
        "episode_definition": "one HH-MM-SS time folder containing all three camera streams and state CSV files",
        "total_aligned_steps": len(all_samples),
        "train_aligned_steps": len(train_samples),
        "validation_aligned_steps": len(validation_samples),
        "aligned_steps_per_episode": counts,
        "train_episodes": sorted({sample.episode_id for sample in train_samples}),
        "validation_episodes": sorted({sample.episode_id for sample in validation_samples}),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


class RealRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: Sequence[AlignedSample],
        prompt: str,
        norm_stats: dict[str, normalize_api.NormStats],
        *,
        model_action_dim: int = 32,
        max_token_len: int = 200,
        use_paper_state: bool = False,
    ):
        if not prompt.strip():
            raise ValueError("A non-empty task prompt is required")
        self.samples = list(samples)
        self.prompt = prompt
        self.use_paper_state = use_paper_state
        self.normalizer = transforms.Normalize(norm_stats, use_quantiles=True, strict=False)
        self.resize = transforms.ResizeImages(224, 224)
        self.tokenize = transforms.TokenizePrompt(
            tokenizer_api.PaligemmaTokenizer(max_token_len), discrete_state_input=True
        )
        self.pad = transforms.PadStatesAndActions(model_action_dim)

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _read_image(path: Path) -> np.ndarray:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        data = {
            "image": {key: self._read_image(path) for key, path in sample.images.items()},
            "image_mask": dict.fromkeys(CAMERA_TO_MODEL_KEY.values(), np.True_),
            "state": sample.state.copy(),
            "actions": sample.actions.copy(),
            "prompt": self.prompt,
        }
        data = self.resize(data)
        data = self.normalizer(data)
        data["state"] = np.asarray(data["state"], dtype=np.float32)
        data["actions"] = np.asarray(data["actions"], dtype=np.float32)
        data = self.tokenize(data)
        return self.pad(data)


def batch_to_model(batch: dict) -> tuple[model_api.Observation, torch.Tensor]:
    observation = model_api.Observation.from_dict(batch)
    return observation, batch["actions"]
