"""Validate ordered HDF5 trajectories without loading PI0.5 or a GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from openpi.sft_shiji.dataset import CAMERA_TO_MODEL_KEY
from openpi.sft_shiji.hdf5_dataset import list_trajectory_files


def inspect_trajectory(path: Path) -> dict:
    with h5py.File(path, "r") as file:
        anchors = np.asarray(file["timestamps/anchor"], dtype=np.int64)
        if anchors.size == 0 or np.any(np.diff(anchors) <= 0):
            raise ValueError(f"{path.name}: anchor timestamps are not strictly increasing")
        states = file["observations/state"]
        actions = file["actions/trajectory"]
        if states.shape[0] != anchors.size or actions.shape[0] != anchors.size:
            raise ValueError(f"{path.name}: state/action length differs from the unified timeline")
        cameras = {}
        for camera_name in CAMERA_TO_MODEL_KEY:
            indices = np.asarray(file[f"observations/image_index/{camera_name}"], dtype=np.int64)
            source_timestamps = np.asarray(file[f"timestamps/{camera_name}/source"], dtype=np.int64)
            offsets = np.asarray(file[f"timestamps/{camera_name}/offset_ticks"], dtype=np.int64)
            if len(indices) != len(anchors) or len(offsets) != len(anchors):
                raise ValueError(f"{path.name}/{camera_name}: aligned length mismatch")
            if np.any(indices < 0) or np.any(indices >= len(source_timestamps)):
                raise ValueError(f"{path.name}/{camera_name}: source image index is out of range")
            selected_timestamps = source_timestamps[indices]
            actual_offsets = selected_timestamps - anchors
            if not np.array_equal(np.abs(actual_offsets), offsets):
                raise ValueError(f"{path.name}/{camera_name}: stored camera offsets are inconsistent")
            cameras[camera_name] = {
                "all_source_images": int(file[f"observations/images/{camera_name}"].shape[0]),
                "aligned_timesteps": len(indices),
                "reused_timesteps": int(np.count_nonzero(np.diff(indices) == 0)),
                "maximum_absolute_offset_ms": float(
                    np.abs(actual_offsets).max()
                    * 1000
                    / int(file["metadata"].attrs["timestamp_ticks_per_second"])
                ),
                "future_frame_count": int(np.count_nonzero(actual_offsets > 0)),
            }
        return {
            "episode_id": path.stem,
            "timesteps": int(anchors.size),
            "start_timestamp": int(anchors[0]),
            "end_timestamp": int(anchors[-1]),
            "action_shape": list(actions.shape),
            "cameras": cameras,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect sequential HDF5 trajectory alignment")
    parser.add_argument(
        "--hdf5-root",
        type=Path,
        default=Path("/mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/sft_shiji/hdf5_data/2026-08-03"),
    )
    args = parser.parse_args()
    files = list_trajectory_files(args.hdf5_root)
    if not files:
        raise FileNotFoundError(f"No HDF5 trajectories under {args.hdf5_root}")
    trajectories = [inspect_trajectory(path) for path in files]
    print(json.dumps({
        "trajectory_count": len(trajectories),
        "total_timesteps": sum(item["timesteps"] for item in trajectories),
        "episode_order": [item["episode_id"] for item in trajectories],
        "shuffle_episodes": False,
        "shuffle_timesteps": False,
        "camera_policy": "three views share one fixed timeline; missing updates reuse only the previous frame",
        "trajectories": trajectories,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
