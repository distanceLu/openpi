from __future__ import annotations

import argparse
import os
import threading
from typing import Any

import torch
import uvicorn

from openpi.rl.distributed_trainer import update_from_payload, worker_loop
from openpi.rl.openpi_policy_server import create_app


class RankZeroCoordinator:
    def __init__(self, state: Any, minibatch_size: int):
        self.state = state
        self.minibatch_size = minibatch_size
        self.lock = threading.Lock()

    def submit(self, request: Any) -> dict[str, Any]:
        payload = request.model_dump()
        with self.lock:
            box = [payload]
            torch.distributed.broadcast_object_list(box, src=0)
            metrics = update_from_payload(
                self.state, payload, self.minibatch_size,
                rank=0, world_size=torch.distributed.get_world_size(),
            )
            torch.distributed.barrier()
            self.state.update_count += 1
            metrics["pi05_rl/update_count"] = self.state.update_count
            if self.state.tensorboard_writer is not None:
                for name, value in metrics.items():
                    self.state.tensorboard_writer.add_scalar(name, value, self.state.update_count)
                self.state.tensorboard_writer.flush()
            if self.state.checkpoint_interval > 0 and self.state.update_count % self.state.checkpoint_interval == 0:
                self.state.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_dir = self.state.checkpoint_dir / f"deepspeed_update_{self.state.update_count:08d}"
                if self.state.engine is None:
                    raise RuntimeError("Distributed coordinator requires a DeepSpeed engine for checkpointing")
                self.state.engine.save_checkpoint(str(checkpoint_dir), tag=f"update_{self.state.update_count:08d}")
                metrics["pi05_rl/checkpoint_path"] = str(checkpoint_dir)
            print({"event": "distributed_ppo_update", **metrics}, flush=True)
            return {"metrics": metrics}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--local_rank", type=int, default=None, help="Injected by the DeepSpeed launcher.")
    p.add_argument("--deepspeed-config", required=True)
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--reference-dir", required=True)
    p.add_argument("--assets-dir", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--monitor-log-dir", default=None)
    p.add_argument("--tensorboard-log-dir", default=None)
    p.add_argument("--ppo-minibatch-size", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--num-denoise-steps", type=int, default=10)
    p.add_argument("--sample-method", default="flow_noise")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--target-kl", type=float, default=0.03)
    p.add_argument("--checkpoint-interval", type=int, default=10)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    return p


def main() -> None:
    args = parser().parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = args.local_rank if args.local_rank is not None else int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    app = create_app(
        checkpoint_path=args.checkpoint_path, reference_dir=args.reference_dir, assets_dir=args.assets_dir,
        device=f"cuda:{local_rank}", num_denoise_steps=args.num_denoise_steps,
        sample_method=args.sample_method, lr=args.lr, target_kl=args.target_kl,
        checkpoint_dir=args.checkpoint_dir, checkpoint_interval=args.checkpoint_interval,
        tensorboard_log_dir=args.tensorboard_log_dir, gamma=args.gamma,
        deepspeed_config=args.deepspeed_config, ppo_minibatch_size=args.ppo_minibatch_size,
        monitor_log_dir=args.monitor_log_dir if rank == 0 else None,
    )
    state = app.state.pi05
    state.engine = state.trainer.engine
    if rank == 0:
        app.state.pi05_coordinator = RankZeroCoordinator(state, args.ppo_minibatch_size)
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        worker_loop(state, args.ppo_minibatch_size)


if __name__ == "__main__":
    main()
