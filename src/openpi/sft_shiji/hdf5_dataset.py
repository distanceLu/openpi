"""HDF5-backed trajectory storage with strict episode and timestep ordering."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from openpi.sft_shiji.dataset import CAMERA_TO_MODEL_KEY
from openpi.sft_shiji.dataset import RealRobotDataset
from openpi.shared import normalize as normalize_api


def list_trajectory_files(root: str | Path) -> list[Path]:
    return sorted(Path(root).glob("*.hdf5"))


def trajectory_length(path: str | Path) -> int:
    with h5py.File(path, "r") as file:
        return int(file["timestamps/anchor"].shape[0])


def compute_hdf5_norm_stats(
    paths: Sequence[str | Path],
    *,
    action_abs_quantile: float = 0.99,
) -> dict[str, normalize_api.NormStats]:
    states = []
    actions = []
    for path in paths:
        with h5py.File(path, "r") as file:
            states.append(np.asarray(file["observations/state"], dtype=np.float64))
            trajectory_actions = np.asarray(file["actions/trajectory"], dtype=np.float64)
            if "actions/valid_mask" in file:
                valid_mask = np.asarray(file["actions/valid_mask"], dtype=np.bool_)
                if valid_mask.shape != trajectory_actions.shape[:2]:
                    raise ValueError(
                        f"Invalid action mask shape in {path}: {valid_mask.shape}; "
                        f"expected {trajectory_actions.shape[:2]}"
                    )
                actions.append(trajectory_actions[valid_mask])
            else:
                actions.append(trajectory_actions.reshape(-1, 6))
    if not states:
        raise ValueError("No HDF5 trajectories were found")
    if not 0.5 < action_abs_quantile <= 1.0:
        raise ValueError("action_abs_quantile must be in (0.5, 1.0]")

    def statistics(values: np.ndarray) -> normalize_api.NormStats:
        return normalize_api.NormStats(
            mean=values.mean(axis=0),
            std=values.std(axis=0),
            q01=np.quantile(values, 0.01, axis=0),
            q99=np.quantile(values, 0.99, axis=0),
        )

    state_stats = statistics(np.concatenate(states))
    action_values = np.concatenate(actions)
    action_stats = statistics(action_values)
    action_scale = np.quantile(np.abs(action_values), action_abs_quantile, axis=0)
    action_scale = np.maximum(action_scale, 1e-6)
    action_stats = normalize_api.NormStats(
        mean=action_stats.mean,
        std=action_stats.std,
        q01=-action_scale,
        q99=action_scale,
    )
    return {"state": state_stats, "actions": action_stats}


class HDF5Trajectory:
    """Read one complete trajectory without flattening across episode boundaries."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file: h5py.File | None = None

    def __enter__(self) -> HDF5Trajectory:
        self._file = h5py.File(self.path, "r")
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def episode_id(self) -> str:
        return self.path.stem

    @property
    def length(self) -> int:
        return int(self._require_file()["timestamps/anchor"].shape[0])

    @property
    def action_shape(self) -> tuple[int, ...]:
        return tuple(self._require_file()["actions/trajectory"].shape[1:])

    @property
    def action_mode(self) -> str:
        value = self._require_file()["metadata"].attrs.get("action_mode", "unknown")
        return value.decode() if isinstance(value, bytes) else str(value)

    @property
    def action_definition(self) -> str:
        value = self._require_file()["metadata"].attrs.get("action_definition", "unknown")
        return value.decode() if isinstance(value, bytes) else str(value)

    @property
    def use_robot_state(self) -> bool:
        return bool(self._require_file()["metadata"].attrs.get("use_robot_state", False))

    def _require_file(self) -> h5py.File:
        if self._file is None:
            raise RuntimeError("HDF5Trajectory must be used inside a context manager")
        return self._file

    @staticmethod
    def _decode(blob: np.ndarray) -> np.ndarray:
        with Image.open(BytesIO(np.asarray(blob, dtype=np.uint8).tobytes())) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def read_step(self, index: int) -> dict:
        file = self._require_file()
        if not 0 <= index < self.length:
            raise IndexError(index)
        images = {}
        for camera_name, model_key in CAMERA_TO_MODEL_KEY.items():
            source_index = int(file[f"observations/image_index/{camera_name}"][index])
            images[model_key] = self._decode(file[f"observations/images/{camera_name}"][source_index])
        actions = np.asarray(file["actions/trajectory"][index], dtype=np.float32)
        action_valid_mask = (
            np.asarray(file["actions/valid_mask"][index], dtype=np.bool_)
            if "actions/valid_mask" in file
            else np.ones(actions.shape[0], dtype=np.bool_)
        )
        return {
            "image": images,
            "image_mask": dict.fromkeys(CAMERA_TO_MODEL_KEY.values(), np.True_),
            "state": np.asarray(file["observations/state"][index], dtype=np.float32),
            "actions": actions,
            "action_valid_mask": action_valid_mask,
        }


class HDF5EpisodeDataset:
    """Transform one HDF5 trajectory while preserving its timestep order."""

    def __init__(
        self,
        prompt: str,
        norm_stats,
        max_token_len: int = 200,
        *,
        use_robot_state: bool = False,
    ):
        self.transform = RealRobotDataset(
            [],
            prompt,
            norm_stats,
            max_token_len=max_token_len,
            use_robot_state=use_robot_state,
        )

    def iter_trajectory(self, path: str | Path) -> Iterator[tuple[str, int, dict]]:
        with HDF5Trajectory(path) as trajectory:
            for index in range(trajectory.length):
                yield trajectory.episode_id, index, self.transform.transform_sample(trajectory.read_step(index))

    def iter_trajectory_with_motion(
        self,
        path: str | Path,
        motion_threshold: float,
    ) -> Iterator[tuple[str, int, dict, bool]]:
        with HDF5Trajectory(path) as trajectory:
            for index in range(trajectory.length):
                raw = trajectory.read_step(index)
                is_moving = bool(np.any(
                    np.abs(raw["actions"][raw["action_valid_mask"], 2]) >= motion_threshold
                ))
                yield (
                    trajectory.episode_id,
                    index,
                    self.transform.transform_sample(raw),
                    is_moving,
                )
