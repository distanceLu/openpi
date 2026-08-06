"""Inspect fixed-recording timestamp alignment without loading PI0.5 or a GPU."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from openpi.sft_shiji.dataset import AlignmentConfig
from openpi.sft_shiji.dataset import build_aligned_samples
from openpi.sft_shiji.dataset import split_by_recording


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/home/shugen/yanjie/ros2_ws/data_collect/2026-08-03"))
    parser.add_argument("--validation-recording", default="17-01-09")
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--timestamp-ticks-per-second", type=int, default=1_000_000)
    parser.add_argument("--camera-tolerance-ms", type=float, default=100.0)
    parser.add_argument("--state-tolerance-ms", type=float, default=30.0)
    parser.add_argument("--action-mode", choices=("absolute_pose", "relative_pose"), default="relative_pose")
    args = parser.parse_args()
    alignment = AlignmentConfig(
        action_horizon=args.action_horizon,
        control_hz=args.control_hz,
        sample_hz=args.sample_hz,
        timestamp_ticks_per_second=args.timestamp_ticks_per_second,
        camera_tolerance_ms=args.camera_tolerance_ms,
        state_tolerance_ms=args.state_tolerance_ms,
        action_mode=args.action_mode,
    )
    samples = build_aligned_samples(args.data_root, alignment)
    train, validation = split_by_recording(samples, args.validation_recording)
    counts = {}
    for sample in samples:
        counts[sample.episode_id] = counts.get(sample.episode_id, 0) + 1
    first = samples[0]
    camera_timestamps = {key: int(path.name.split(".", 1)[0]) for key, path in first.images.items()}
    print(json.dumps({
        "alignment": dataclasses.asdict(alignment),
        "episode_definition": "one HH-MM-SS folder with three cameras, robot_state and paper_state",
        "validation_episode": Path(args.validation_recording).name,
        "aligned_steps": len(samples),
        "train_aligned_steps": len(train),
        "validation_aligned_steps": len(validation),
        "aligned_steps_per_episode": counts,
        "state_source": "robot_state/tool_pose.csv only",
        "session_meta_usage": "not used as model input; episode recording metadata only",
        "paper_state_usage": "not used as model input",
        "first_aligned_step": {
            "episode_id": first.episode_id,
            "timestamp": first.timestamp,
            "images": {key: str(value) for key, value in first.images.items()},
            "camera_timestamps": camera_timestamps,
            "pool_frames_not_earlier": all(
                camera_timestamps[key] >= first.timestamp
                for key in ("left_wrist_0_rgb", "right_wrist_0_rgb")
            ),
            "robot_state": first.state.tolist(),
            "actions_shape": list(first.actions.shape),
            "first_action": first.actions[0].tolist(),
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
