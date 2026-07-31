"""Dataset adapter for the flat PT SFT data exported from LIBERO."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch


_TASK_PATTERN = re.compile(r"(?:^|[_-])task[_-]?(\d+)(?:[_-]|$)", re.IGNORECASE)
_EPISODE_PATTERN = re.compile(r"(?:^|[_-])ep(?:isode)?[_-]?(\d+)(?:[_-]|$)", re.IGNORECASE)


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, torch.Tensor):
        value = value.item()
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {name}: {value!r}") from error


def _id_from_filename(path: Path, pattern: re.Pattern[str], name: str) -> int:
    match = pattern.search(path.stem)
    if match is None:
        raise ValueError(f"Cannot infer {name} from {path.name}; provide it in metadata.json or the PT payload")
    return int(match.group(1))


class LiberoHDF5Dataset(torch.utils.data.Dataset):
    """Read flat ``.pt`` episodes and split each task at episode level.

    The historical class name is retained so existing training imports remain
    compatible. Task IDs come from metadata first, then the PT payload, and only
    then the filename. No task-directory layout is required.
    """

    def __init__(
        self,
        root: str | Path,
        action_horizon: int,
        default_prompt: str | None = None,
        *,
        split: str = "train",
        validation_fraction: float = 0.1,
        split_seed: int = 0,
        cache_size: int = 4,
    ):
        self.root = Path(root)
        self.action_horizon = action_horizon
        self.default_prompt = default_prompt
        self.split = split
        self.cache_size = max(1, cache_size)
        self._cache: OrderedDict[Path, dict[str, Any]] = OrderedDict()
        if split not in {"train", "validation", "all"}:
            raise ValueError(f"Unknown split {split!r}")
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError(f"validation_fraction must be between 0 and 1, got {validation_fraction}")
        if action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}")

        records = self._load_records()
        selected = self._split_records(records, validation_fraction, split_seed)
        self.task_ids = tuple(sorted({record["task_id"] for record in selected}))
        self.task_episode_counts = self._task_episode_counts(selected)
        self.index: list[tuple[dict[str, Any], int]] = []
        for record in selected:
            self.index.extend((record, frame) for frame in record["frame_ids"])
        if not self.index:
            raise ValueError(f"No samples selected for split={split!r} under {self.root}")

    def _load_records(self) -> list[dict[str, Any]]:
        metadata_path = self.root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Expected flat PT dataset metadata at {metadata_path}")
        payload = json.loads(metadata_path.read_text())
        entries = payload.get("metadata", payload) if isinstance(payload, dict) else payload
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"metadata.json contains no samples: {metadata_path}")

        grouped: dict[Path, dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict) or "path" not in entry:
                raise ValueError("Every metadata entry must be an object containing 'path'")
            path = Path(entry["path"])
            if not path.is_absolute():
                path = self.root / path
            if not path.is_file():
                raise FileNotFoundError(f"Metadata references missing PT episode: {path}")
            task_id = _integer(entry["task_id"], "task_id") if "task_id" in entry else _id_from_filename(
                path, _TASK_PATTERN, "task_id"
            )
            episode = _integer(entry["episode"], "episode") if "episode" in entry else _id_from_filename(
                path, _EPISODE_PATTERN, "episode"
            )
            frame_values = entry.get("frame_ids", [])
            frames = [_integer(frame, "frame_id") for frame in frame_values]
            record = grouped.setdefault(
                path,
                {
                    "path": path,
                    "task_id": task_id,
                    "episode": episode,
                    "instruction": entry.get("instruction"),
                    "valid_frames": _integer(entry.get("valid_frames", 0), "valid_frames"),
                    "frame_ids": [],
                },
            )
            if record["task_id"] != task_id or record["episode"] != episode:
                raise ValueError(f"Inconsistent task/episode metadata for {path}")
            record["frame_ids"].extend(frames)

        records = []
        for record in grouped.values():
            frames = sorted(set(record["frame_ids"]))
            if not frames:
                frames = list(range(record["valid_frames"]))
            if not frames:
                raise ValueError(f"No valid frames listed for {record['path']}")
            if frames[0] < 0 or frames[-1] >= record["valid_frames"]:
                raise IndexError(f"Frame metadata exceeds valid_frames for {record['path']}")
            record["frame_ids"] = frames
            records.append(record)
        return sorted(records, key=lambda item: (item["task_id"], item["episode"], str(item["path"])))

    def _split_records(self, records: list[dict[str, Any]], fraction: float, seed: int) -> list[dict[str, Any]]:
        if self.split == "all":
            return records
        by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_task[record["task_id"]].append(record)

        selected: list[dict[str, Any]] = []
        for task_id in sorted(by_task):
            task_records = by_task[task_id]
            if len(task_records) < 2:
                raise ValueError(
                    f"Task {task_id} has only {len(task_records)} episode; an episode-isolated 90/10 split requires at least 2"
                )
            rng = np.random.default_rng(seed + task_id * 1_000_003)
            order = rng.permutation(len(task_records)).tolist()
            validation_count = min(len(task_records) - 1, max(1, int(round(len(task_records) * fraction))))
            validation_indices = set(order[:validation_count])
            for index, record in enumerate(task_records):
                is_validation = index in validation_indices
                if (self.split == "validation") == is_validation:
                    selected.append(record)
        return selected

    @staticmethod
    def _task_episode_counts(records: list[dict[str, Any]]) -> dict[int, int]:
        counts: dict[int, int] = defaultdict(int)
        for record in records:
            counts[record["task_id"]] += 1
        return dict(sorted(counts.items()))

    def _episode(self, path: Path) -> dict[str, Any]:
        cached = self._cache.pop(path, None)
        if cached is not None:
            self._cache[path] = cached
            return cached
        episode = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(episode, dict):
            raise TypeError(f"Expected a dictionary in {path}, got {type(episode).__name__}")
        self._cache[path] = episode
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return episode

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, Any]:
        record, frame = self.index[item]
        episode = self._episode(record["path"])
        payload_task = episode.get("task_id", record["task_id"])
        if _integer(payload_task, "task_id") != record["task_id"]:
            raise ValueError(f"PT task_id disagrees with metadata for {record['path']}")

        video = _as_numpy(episode["video"])
        wrist_video = _as_numpy(episode["wrist_video"])
        state = _as_numpy(episode["proprio"])
        actions = _as_numpy(episode.get("actions_continuous", episode.get("actions")))
        if actions.ndim != 2:
            raise ValueError(f"Expected [T, action_dim] actions in {record['path']}, got {actions.shape}")
        if not (len(video) == len(wrist_video) == len(state) == len(actions)):
            raise ValueError(f"Modalities have inconsistent frame counts in {record['path']}")

        end = min(len(actions), frame + self.action_horizon)
        action_chunk = np.asarray(actions[frame:end], dtype=np.float32)
        if len(action_chunk) < self.action_horizon:
            padding = np.repeat(action_chunk[-1:], self.action_horizon - len(action_chunk), axis=0)
            action_chunk = np.concatenate((action_chunk, padding), axis=0)
        prompt = episode.get("instruction") or record.get("instruction") or self.default_prompt
        if not prompt:
            raise ValueError(f"No language instruction for task {record['task_id']} in {record['path']}")

        return {
            "image": np.ascontiguousarray(video[frame]),
            "wrist_image": np.ascontiguousarray(wrist_video[frame]),
            "state": np.asarray(state[frame], dtype=np.float32),
            "actions": action_chunk,
            "prompt": str(prompt),
        }
