from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import torch


class TrainingMonitor:
    """Rank-aware CSV and TensorBoard monitor for the OpenPI RL process."""

    def __init__(self, log_dir: str | Path | None, enabled: bool = True):
        self.log_dir = Path(log_dir) if log_dir else None
        self.enabled = enabled and self.log_dir is not None
        self._csv_file = None
        self._csv_writer = None
        self._writer = None
        if not self.enabled:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.log_dir / "train_metrics.csv"
        self._csv_file = csv_path.open("a", newline="", encoding="utf-8")
        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))

    @staticmethod
    def _scalar(value: Any) -> float | str:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return float(value.detach().float().mean().item())
            return float(value.detach().float().item())
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value) if math.isfinite(float(value)) else str(value)
        return str(value)

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        if not self.enabled:
            return
        values = {key: self._scalar(value) for key, value in metrics.items()}
        values["step"] = step
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(values), extrasaction="ignore")
            self._csv_writer.writeheader()
        else:
            known = set(self._csv_writer.fieldnames or ())
            if set(values) - known:
                self._csv_file.close()
                self._csv_file = (self.log_dir / "train_metrics.csv").open("a", newline="", encoding="utf-8")
                self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=sorted(known | set(values)))
        self._csv_writer.writerow(values)
        self._csv_file.flush()
        for key, value in values.items():
            if key != "step" and isinstance(value, (int, float)):
                self._writer.add_scalar(key, value, step)
        self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        if self._csv_file is not None:
            self._csv_file.close()
