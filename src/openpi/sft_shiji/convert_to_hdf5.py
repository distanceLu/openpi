"""Convert command-delta recordings into ordered three-camera HDF5 trajectories."""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import h5py
import numpy as np

from openpi.sft_shiji.dataset import CAMERA_TO_MODEL_KEY
from openpi.sft_shiji.dataset import AlignmentConfig
from openpi.sft_shiji.dataset import _camera_stream
from openpi.sft_shiji.dataset import _nearest_index

_DELTA_COLUMNS = ("delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz")
_STATE_COLUMNS = (
    "actual_tcp_x", "actual_tcp_y", "actual_tcp_z",
    "actual_tcp_rx", "actual_tcp_ry", "actual_tcp_rz",
)


def _read_bytes(path: Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype=np.uint8)


def _read_command_deltas(path: Path) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    rows: list[tuple[int, str, np.ndarray, np.ndarray]] = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"pool_timestamp", "pool_image", *_DELTA_COLUMNS, *_STATE_COLUMNS}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            try:
                timestamp = int(float(row["pool_timestamp"]))
                state = np.asarray([float(row[name]) for name in _STATE_COLUMNS], dtype=np.float32)
                delta = np.asarray([float(row[name]) for name in _DELTA_COLUMNS], dtype=np.float32)
            except (TypeError, ValueError):
                continue
            if np.isfinite(state).all() and np.isfinite(delta).all():
                rows.append((timestamp, row["pool_image"], state, delta))
    if not rows:
        raise ValueError(f"No valid command deltas in {path}")
    rows.sort(key=lambda row: row[0])
    timestamps = np.asarray([row[0] for row in rows], dtype=np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{path} has duplicate or non-increasing pool timestamps")
    return (
        timestamps,
        [row[1] for row in rows],
        np.stack([row[2] for row in rows]),
        np.stack([row[3] for row in rows]),
    )


def _derive_command_deltas(recording: Path, generated_destination: Path) -> Path:
    source = recording / "robot_state" / "robot_command_state.csv"
    recorded_destination = recording / "robot_state" / "command_delta.csv"
    if recorded_destination.is_file():
        return recorded_destination
    if not source.is_file():
        raise FileNotFoundError(f"{recording.name} has neither command_delta.csv nor robot_command_state.csv")

    pool_frames = _camera_stream(recording / "camera_pool")
    if not pool_frames:
        raise FileNotFoundError(f"{recording.name} has no camera_pool images")
    command_rows: list[dict[str, str]] = []
    command_timestamps: list[int] = []
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"timestamp", *_STATE_COLUMNS}
        required.update(f"command_tcp_{axis}" for axis in ("x", "y", "z", "rx", "ry", "rz"))
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{source} is missing columns: {sorted(missing)}")
        for row in reader:
            try:
                timestamp = int(float(row["timestamp"]))
                values = [float(row[name]) for name in _STATE_COLUMNS]
                values.extend(float(row[f"command_tcp_{axis}"]) for axis in ("x", "y", "z", "rx", "ry", "rz"))
            except (TypeError, ValueError):
                continue
            if np.isfinite(values).all():
                command_timestamps.append(timestamp)
                command_rows.append(row)
    if not command_rows:
        raise ValueError(f"No valid command states in {source}")

    destination = generated_destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["frame_idx", "pool_timestamp", "pool_image", *_STATE_COLUMNS, *_DELTA_COLUMNS]
    temporary = destination.with_suffix(".tmp.csv")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for frame_index, frame in enumerate(pool_frames):
            actual_index, _ = _nearest_index(command_timestamps, frame.timestamp)
            if frame_index + 1 < len(pool_frames):
                next_timestamp = pool_frames[frame_index + 1].timestamp
                command_index, _ = _nearest_index(command_timestamps, next_timestamp)
            else:
                command_index = actual_index
            actual = command_rows[actual_index]
            command = command_rows[command_index]
            actual_pose = np.asarray([float(actual[name]) for name in _STATE_COLUMNS], dtype=np.float64)
            command_pose = np.asarray(
                [float(command[f"command_tcp_{axis}"]) for axis in ("x", "y", "z", "rx", "ry", "rz")],
                dtype=np.float64,
            )
            delta = command_pose - actual_pose
            delta[3:] = (delta[3:] + np.pi) % (2.0 * np.pi) - np.pi
            writer.writerow({
                "frame_idx": frame_index,
                "pool_timestamp": frame.timestamp,
                "pool_image": frame.path.name,
                **dict(zip(_STATE_COLUMNS, actual_pose, strict=True)),
                **dict(zip(_DELTA_COLUMNS, delta, strict=True)),
            })
    temporary.replace(destination)
    return destination


def _recording_id(recording: Path, data_root: Path) -> str:
    relative = recording.relative_to(data_root)
    return "__".join(relative.parts)


def _recording_directories(data_root: Path, recording_dates: set[str]) -> list[Path]:
    recordings = sorted(
        path.parent.parent
        for path in data_root.glob("**/robot_state/robot_command_state.csv")
        if "infer" not in path.parent.parent.name.lower()
    )
    if not recording_dates:
        return recordings
    return [
        recording
        for recording in recordings
        if any(part in recording_dates for part in recording.relative_to(data_root).parts)
    ]


def _latest_at_or_before_index(timestamps: np.ndarray, target: int) -> int | None:
    index = int(np.searchsorted(timestamps, target, side="right")) - 1
    return index if index >= 0 else None


def convert_recording(
    recording: Path,
    destination: Path,
    config: AlignmentConfig,
    max_camera_offset_ms: float,
    max_action_offset_ms: float,
    *,
    use_robot_state: bool = False,
    allow_future_camera_frames: bool = False,
) -> int:
    if "infer" in recording.name.lower():
        raise ValueError("inference recording is excluded")
    streams = {name: _camera_stream(recording / name) for name in CAMERA_TO_MODEL_KEY}
    missing = [name for name, stream in streams.items() if not stream]
    if missing:
        raise FileNotFoundError(f"{recording.name} is missing camera images for {missing}")

    generated_delta_path = destination.parent / "generated_command_delta" / f"{destination.stem}.csv"
    command_delta_path = _derive_command_deltas(recording, generated_delta_path)
    command_timestamps, pool_filenames, recorded_states, deltas = _read_command_deltas(command_delta_path)
    max_offset = round(max_camera_offset_ms * config.timestamp_ticks_per_second / 1000.0)
    max_action_offset = round(max_action_offset_ms * config.timestamp_ticks_per_second / 1000.0)
    stream_timestamps = {
        name: np.asarray([frame.timestamp for frame in stream], dtype=np.int64)
        for name, stream in streams.items()
    }
    pool_by_name = {frame.path.name: index for index, frame in enumerate(streams["camera_pool"])}

    valid_times: list[int] = []
    command_indices: list[int] = []
    selected_indices: dict[str, list[int]] = {name: [] for name in streams}
    selected_offsets: dict[str, list[int]] = {name: [] for name in streams}
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    action_indices: list[np.ndarray] = []
    action_timestamps: list[np.ndarray] = []
    action_offsets: list[np.ndarray] = []

    if len(command_timestamps) < config.action_horizon:
        raise ValueError(
            f"{recording.name} has {len(command_timestamps)} command rows, fewer than "
            f"action_horizon={config.action_horizon}"
        )
    for row_index in range(len(command_timestamps)):
        timestamp = int(command_timestamps[row_index])
        target_action_timestamps = timestamp + np.arange(config.action_horizon) * config.control_period
        matched_action_indices = []
        matched_action_offsets = []
        for target_timestamp in target_action_timestamps:
            action_index, action_offset = _nearest_index(command_timestamps, int(target_timestamp))
            matched_action_indices.append(action_index)
            matched_action_offsets.append(action_offset)
        if (
            any(offset > max_action_offset for offset in matched_action_offsets)
            or any(
                current >= following
                for current, following in itertools.pairwise(matched_action_indices)
            )
        ):
            continue
        camera_matches: dict[str, tuple[int, int]] = {}
        pool_index = pool_by_name.get(pool_filenames[row_index])
        if pool_index is None:
            pool_index, pool_error = _nearest_index(stream_timestamps["camera_pool"], timestamp)
        else:
            pool_error = abs(int(stream_timestamps["camera_pool"][pool_index]) - timestamp)
        camera_matches["camera_pool"] = (pool_index, pool_error)
        causal_match_missing = False
        for name in ("camera_paper_aruco", "camera_pool1"):
            if allow_future_camera_frames:
                camera_matches[name] = _nearest_index(stream_timestamps[name], timestamp)
            else:
                index = _latest_at_or_before_index(stream_timestamps[name], timestamp)
                if index is None:
                    causal_match_missing = True
                    break
                camera_matches[name] = (index, timestamp - int(stream_timestamps[name][index]))
        if causal_match_missing or any(offset > max_offset for _, offset in camera_matches.values()):
            continue

        action = deltas[np.asarray(matched_action_indices, dtype=np.int64)].copy()
        state = recorded_states[row_index].copy() if use_robot_state else np.zeros(6, dtype=np.float32)
        valid_times.append(timestamp)
        command_indices.append(row_index)
        states.append(state)
        actions.append(action)
        action_indices.append(np.asarray(matched_action_indices, dtype=np.int64))
        action_timestamps.append(command_timestamps[matched_action_indices].copy())
        action_offsets.append(np.asarray(matched_action_offsets, dtype=np.int64))
        for name, (index, offset) in camera_matches.items():
            selected_indices[name].append(index)
            selected_offsets[name].append(offset)

    if not valid_times:
        raise ValueError(f"{recording.name} has no valid three-camera command-delta chunks")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.hdf5")
    temporary.unlink(missing_ok=True)
    encoded_image = h5py.vlen_dtype(np.dtype("uint8"))
    string = h5py.string_dtype("utf-8")
    with h5py.File(temporary, "w") as file:
        metadata = file.create_group("metadata")
        metadata.attrs.update({
            "format_version": 3,
            "episode_id": destination.stem,
            "source_directory": str(recording),
            "number_of_steps": len(valid_times),
            "timestamp_ticks_per_second": config.timestamp_ticks_per_second,
            "action_horizon": config.action_horizon,
            "action_mode": "relative_pose",
            "action_definition": "command_delta_csv: fixed-rate rows from current anchor",
            "action_sampling": "fixed-rate nearest command_delta rows from anchor, including anchor",
            "action_frequency_hz": config.control_hz,
            "action_period_ms": 1000.0 / config.control_hz,
            "max_action_offset_ms": max_action_offset_ms,
            "action_source": str(command_delta_path),
            "state_source": "command_delta.csv:actual_tcp" if use_robot_state else "disabled_zero_vector",
            "use_robot_state": use_robot_state,
            "max_camera_offset_ms": max_camera_offset_ms,
            "camera_alignment": (
                "nearest image to command_delta pool_timestamp; camera_pool prefers pool_image"
                if allow_future_camera_frames
                else "latest side-camera image at or before command_delta pool_timestamp; camera_pool uses pool_image"
            ),
            "future_camera_frames_allowed": allow_future_camera_frames,
            "infer_filter": "recording directory names containing infer are excluded",
        })
        file.create_dataset("timestamps/anchor", data=np.asarray(valid_times, dtype=np.int64))
        file.create_dataset("metadata/command_delta_row", data=np.asarray(command_indices, dtype=np.int64))
        file.create_dataset("metadata/action_command_delta_row", data=np.stack(action_indices))
        file.create_dataset("timestamps/action/source", data=np.stack(action_timestamps))
        file.create_dataset("timestamps/action/offset_ticks", data=np.stack(action_offsets))
        image_group = file.create_group("observations/images")
        for camera_name, indices in selected_indices.items():
            frames = streams[camera_name]
            encoded = image_group.create_dataset(camera_name, (len(frames),), dtype=encoded_image)
            source_timestamps = file.create_dataset(
                f"timestamps/{camera_name}/source", (len(frames),), dtype=np.int64
            )
            source_filenames = file.create_dataset(
                f"metadata/{camera_name}_source_filename", (len(frames),), dtype=string
            )
            for image_index, frame in enumerate(frames):
                encoded[image_index] = _read_bytes(frame.path)
                source_timestamps[image_index] = frame.timestamp
                source_filenames[image_index] = frame.path.name
            file.create_dataset(
                f"observations/image_index/{camera_name}", data=np.asarray(indices, dtype=np.int64)
            )
            file.create_dataset(
                f"timestamps/{camera_name}/offset_ticks",
                data=np.asarray(selected_offsets[camera_name], dtype=np.int64),
            )
        file.create_dataset("observations/state", data=np.stack(states), compression="lzf")
        file.create_dataset("actions/trajectory", data=np.stack(actions), compression="lzf")
        file.create_dataset("validity/valid_step", data=np.ones(len(valid_times), dtype=np.bool_))
        file.flush()
    temporary.replace(destination)
    return len(valid_times)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert command-delta recordings to HDF5 trajectories")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/data/lcx2/yanjieworkspace/data_collect"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/sft_shiji/hdf5_data/2026-08-05-06-command-delta-causal-10hz-v2"),
    )
    parser.add_argument("--max-camera-offset-ms", type=float, default=100.0)
    parser.add_argument("--max-action-offset-ms", type=float, default=50.0)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument(
        "--recording-dates",
        default="2026-08-05,2026-08-06",
        help="Comma-separated recording date directory names; empty includes every date.",
    )
    parser.add_argument("--use-robot-state", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--allow-future-camera-frames",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use nearest frames that may occur after the action timestamp; disabled avoids future-image leakage.",
    )
    args = parser.parse_args()
    if args.action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    if args.max_camera_offset_ms < 0:
        raise ValueError("max_camera_offset_ms must be non-negative")
    if args.max_action_offset_ms < 0 or args.control_hz <= 0:
        raise ValueError("max_action_offset_ms must be non-negative and control_hz must be positive")
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")
    config = AlignmentConfig(
        action_horizon=args.action_horizon,
        control_hz=args.control_hz,
        action_mode="relative_pose",
    )
    converted = 0
    recording_dates = {date.strip() for date in args.recording_dates.split(",") if date.strip()}
    recordings = _recording_directories(args.data_root, recording_dates)
    for recording in recordings:
        episode_id = _recording_id(recording, args.data_root)
        destination = args.output_root / f"{episode_id}.hdf5"
        try:
            steps = convert_recording(
                recording,
                destination,
                config,
                args.max_camera_offset_ms,
                args.max_action_offset_ms,
                use_robot_state=args.use_robot_state,
                allow_future_camera_frames=args.allow_future_camera_frames,
            )
            print(f"converted {episode_id}: {steps} ordered steps -> {destination}")
            converted += 1
        except (FileNotFoundError, ValueError) as error:
            print(f"skipped {recording.name}: {error}")
    if converted == 0:
        raise RuntimeError(f"No trajectories were converted from {args.data_root}")


if __name__ == "__main__":
    main()
