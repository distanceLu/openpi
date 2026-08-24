"""Render predicted and demonstrated action trajectories beside each camera video."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from openpi.sft_shiji.hdf5_dataset import HDF5Trajectory

CAMERAS = {
    "base_0_rgb": "camera_paper_aruco",
    "left_wrist_0_rgb": "camera_pool",
    "right_wrist_0_rgb": "camera_pool1",
}


def _even_rgb(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.uint8)
    height, width = frame.shape[:2]
    return np.ascontiguousarray(frame[: height - height % 2, : width - width % 2, :3])


def _panel(predicted: np.ndarray, target: np.ndarray, width: int, height: int, step: int) -> np.ndarray:
    panel = Image.new("RGB", (width, height), (245, 247, 250))
    draw = ImageDraw.Draw(panel)
    left, top, right, bottom = 42, 42, width - 20, height - 42
    draw.rectangle((left, top, right, bottom), outline=(80, 88, 100), width=1)
    draw.line((left, (top + bottom) // 2, right, (top + bottom) // 2), fill=(190, 195, 202), width=1)
    draw.line(((left + right) // 2, top, (left + right) // 2, bottom), fill=(190, 195, 202), width=1)
    draw.text((left, 12), "Predicted / demonstrated future XY", fill=(25, 30, 38))
    draw.text((left, bottom + 12), f"timestep {step}", fill=(70, 77, 88))
    extent = max(float(np.max(np.abs(predicted[:, :2]))), float(np.max(np.abs(target[:, :2]))), 1e-4) * 1.15

    def point(value: np.ndarray) -> tuple[int, int]:
        x = int((left + right) / 2 + value[0] / extent * (right - left) / 2)
        y = int((top + bottom) / 2 - value[1] / extent * (bottom - top) / 2)
        return x, y

    pred_points = [point(value) for value in predicted]
    target_points = [point(value) for value in target]
    if len(pred_points) > 1:
        draw.line(pred_points, fill=(210, 55, 55), width=3)
    if len(target_points) > 1:
        draw.line(target_points, fill=(45, 105, 190), width=3)
    draw.ellipse((left - 4, (top + bottom) // 2 - 4, left + 4, (top + bottom) // 2 + 4), fill=(25, 30, 38))
    draw.line((left + 8, height - 19, left + 35, height - 19), fill=(210, 55, 55), width=3)
    draw.text((left + 42, height - 27), "predicted", fill=(45, 50, 60))
    draw.line((left + 125, height - 19, left + 152, height - 19), fill=(45, 105, 190), width=3)
    draw.text((left + 159, height - 27), "demonstrated", fill=(45, 50, 60))
    return np.asarray(panel, dtype=np.uint8)


def _writer(path: Path, fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
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


def render_episode(hdf5_path: Path, prediction_path: Path, output_dir: Path, fps: float, panel_width: int) -> None:
    predictions = np.load(prediction_path)
    with HDF5Trajectory(hdf5_path) as trajectory:
        if predictions.shape[0] != trajectory.length:
            raise ValueError(f"Prediction length {predictions.shape[0]} != trajectory length {trajectory.length}")
        writers = {}
        try:
            first = trajectory.read_step(0)
            for model_key, camera_name in CAMERAS.items():
                frame = _even_rgb(first["image"][model_key])
                writers[model_key] = _writer(output_dir / f"{camera_name}_with_actions.mp4", fps)
                writers[model_key].append_data(np.concatenate([frame, _panel(predictions[0], first["actions"], panel_width, frame.shape[0], 0)], axis=1))
            for timestep in range(1, trajectory.length):
                sample = trajectory.read_step(timestep)
                for model_key, writer in writers.items():
                    frame = _even_rgb(sample["image"][model_key])
                    panel = _panel(predictions[timestep], sample["actions"], panel_width, frame.shape[0], timestep)
                    writer.append_data(np.concatenate([frame, panel], axis=1))
        finally:
            for writer in writers.values():
                writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Add predicted and demonstrated XY action trajectories to camera videos")
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--panel-width", type=int, default=420)
    args = parser.parse_args()
    render_episode(args.hdf5, args.prediction, args.output_dir, args.fps, args.panel_width)
    print(f"saved action-overlay videos: {args.output_dir}")


if __name__ == "__main__":
    main()
