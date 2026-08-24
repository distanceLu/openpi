"""Record real-robot writing success rates and select a deployment checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def _wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    if trials <= 0:
        return 0.0
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = rate + z * z / (2.0 * trials)
    margin = z * math.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials))
    return (center - margin) / denominator


def main() -> None:
    parser = argparse.ArgumentParser(description="Record PI0.5 real-robot brush-writing evaluation")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/data/lcx1/yiqinworkspace/openpi/src/openpi/sft_shiji/output"),
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint folder name, for example step-00000510")
    parser.add_argument("--successes", type=int, required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--note", default="")
    parser.add_argument("--minimum-trials", type=int, default=5)
    args = parser.parse_args()
    if args.trials <= 0 or not 0 <= args.successes <= args.trials:
        raise ValueError("Require trials > 0 and 0 <= successes <= trials")
    checkpoint_dir = args.output_dir / args.checkpoint
    if not (checkpoint_dir / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Adapter checkpoint does not exist: {checkpoint_dir}")

    metrics_path = args.output_dir / "real_robot_success.jsonl"
    records = []
    if metrics_path.is_file():
        records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    trainer_state = json.loads(trainer_state_path.read_text()) if trainer_state_path.is_file() else {}
    record = {
        "evaluation_index": len(records) + 1,
        "timestamp": datetime.now().astimezone().isoformat(),
        "checkpoint": args.checkpoint,
        "step": int(trainer_state.get("step", 0)),
        "successes": args.successes,
        "trials": args.trials,
        "success_rate": args.successes / args.trials,
        "wilson_lower_bound_95": _wilson_lower_bound(args.successes, args.trials),
        "validation_loss": trainer_state.get("validation_loss", trainer_state.get("best_validation_loss")),
        "note": args.note,
    }
    records.append(record)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records))

    writer = SummaryWriter(log_dir=str(args.output_dir / "tensorboard"))
    writer.add_scalar("real_robot/success_rate", record["success_rate"], record["step"])
    writer.add_scalar("real_robot/wilson_lower_bound_95", record["wilson_lower_bound_95"], record["step"])
    writer.add_scalar("real_robot/trials", record["trials"], record["step"])
    writer.close()

    eligible = [item for item in records if item["trials"] >= args.minimum_trials]
    if eligible:
        # Confidence-aware success is primary; validation loss only breaks effectively equal scores.
        best = max(
            eligible,
            key=lambda item: (
                item["wilson_lower_bound_95"],
                item["success_rate"],
                -float(item["validation_loss"]) if item.get("validation_loss") is not None else float("-inf"),
            ),
        )
        (args.output_dir / "deployment_best").write_text(best["checkpoint"])
        print(json.dumps({"recorded": record, "deployment_best": best}, indent=2, ensure_ascii=False))
    else:
        print(json.dumps({
            "recorded": record,
            "deployment_best": None,
            "reason": f"No checkpoint has at least {args.minimum_trials} real-robot trials",
        }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
