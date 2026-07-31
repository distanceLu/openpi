"""Run a single forward pass for the torch pi0.5 policy.

This script avoids the websocket server path and directly loads the pi05_libero
checkpoint, builds a synthetic observation that matches the model input spec,
and runs one forward/inference call with the PyTorch implementation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from openpi.models.model import Observation
from openpi.training import config as cfg

LOCAL_CACHE_ROOT = Path("/mnt/data/lcx1/yiqinworkspace/cache")
LOCAL_CHECKPOINT_ROOTS = (
    LOCAL_CACHE_ROOT / "openpi" / "openpi-assets" / "checkpoints",
    LOCAL_CACHE_ROOT / "openpi-assets" / "checkpoints",
    LOCAL_CACHE_ROOT / "checkpoints",
    LOCAL_CACHE_ROOT,
)
DEFAULT_CHECKPOINT_DIR = Path("/mnt/data/lcx1/yiqinworkspace/cache/hf/hub/models--RLinf--RLinf-Pi05-LIBERO-SFT") / "snapshots" / "45ccfcc4e28634f1576ebf78cab0fbe2fd82432d"


def _resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _find_checkpoint_file(root: Path) -> Path:
    candidates: list[Path] = []
    if root.is_file():
        return root
    for suffix in ("*.safetensors", "*.pt", "*.pth"):
        candidates.extend(sorted(root.rglob(suffix)))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found under {root}")
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple checkpoint files found under {root}: {candidates}")
    return candidates[0]


def _random_image(batch_size: int, device: torch.device) -> torch.Tensor:
    # Match torch pi0.5 vision encoder input: [B, 3, H, W], float32 in [-1, 1]
    return torch.empty((batch_size, 3, 224, 224), dtype=torch.float32, device=device).uniform_(-1.0, 1.0)


def _random_state(batch_size: int, action_dim: int, device: torch.device) -> torch.Tensor:
    return torch.empty((batch_size, action_dim), dtype=torch.float32, device=device).uniform_(-1.0, 1.0)


def _random_prompt(batch_size: int, max_token_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    # A realistic prompt tensor: integer token ids with a mask, padded to max length.
    tokenized_prompt = torch.zeros((batch_size, max_token_len), dtype=torch.int32, device=device)
    tokenized_prompt_mask = torch.zeros((batch_size, max_token_len), dtype=torch.bool, device=device)
    prompt_len = min(16, max_token_len)
    tokenized_prompt[:, :prompt_len] = torch.randint(1, 32000, (batch_size, prompt_len), dtype=torch.int32, device=device)
    tokenized_prompt_mask[:, :prompt_len] = True
    return tokenized_prompt, tokenized_prompt_mask


def make_synthetic_observation(train_config, batch_size: int, device: torch.device) -> Observation:
    obs_spec, _ = train_config.model.inputs_spec(batch_size=batch_size)

    image = {key: _random_image(batch_size, device) for key in obs_spec.images.keys()}
    image_mask = {key: torch.ones((batch_size,), dtype=torch.bool, device=device) for key in obs_spec.image_masks.keys()}
    state = _random_state(batch_size, train_config.model.action_dim, device)
    tokenized_prompt, tokenized_prompt_mask = _random_prompt(batch_size, train_config.model.max_token_len, device)

    return Observation.from_dict(
        {
            "image": image,
            "image_mask": image_mask,
            "state": state,
            "tokenized_prompt": tokenized_prompt,
            "tokenized_prompt_mask": tokenized_prompt_mask,
        }
    )


def _find_torch_checkpoint_path(config_name: str, checkpoint_arg: str | None) -> Path:
    candidates: list[Path] = []

    if checkpoint_arg:
        user_path = Path(checkpoint_arg).expanduser().resolve()
        candidates.append(user_path)

    local_roots = [base / config_name for base in LOCAL_CHECKPOINT_ROOTS]
    for root in local_roots:
        if root.exists():
            candidates.append(root)

    for root in candidates:
        if root.is_file():
            return root
        if root.is_dir():
            try:
                return _find_checkpoint_file(root)
            except (FileNotFoundError, RuntimeError):
                continue

    raise FileNotFoundError(
        "Could not find a local checkpoint. Tried: "
        + ", ".join(str(path) for path in candidates)
        + f". Please place weights under {LOCAL_CACHE_ROOT}/openpi/openpi-assets/checkpoints/{config_name} or pass --checkpoint explicitly."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single torch pi0.5 forward pass.")
    parser.add_argument("--config", default="pi05_libero", help="Training config name.")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Local checkpoint file or directory.",
    )
    parser.add_argument("--device", default=None, help="Override device (cpu, cuda, cuda:0, mps).")
    parser.add_argument("--batch-size", type=int, default=1, help="Synthetic batch size.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for synthetic inputs.")
    args = parser.parse_args()

    openpi_cache = os.getenv("OPENPI_DATA_HOME")
    if openpi_cache:
        print(f"OPENPI_DATA_HOME={openpi_cache}")

    torch.manual_seed(args.seed)

    train_config = cfg.get_config(args.config)
    device = _resolve_device(args.device)
    print("PyTorch version:", torch.__version__)
    print("Selected device:", device)

    checkpoint_path = _find_torch_checkpoint_path(train_config.name, args.checkpoint)
    print(f"Loading checkpoint from: {checkpoint_path}")
    model = train_config.model.load_pytorch(train_config, str(checkpoint_path))
    model = model.to(device)
    model.eval()

    _, act_spec = train_config.model.inputs_spec(batch_size=args.batch_size)
    observation = make_synthetic_observation(train_config, args.batch_size, device)

    print("Running torch pi0.5 forward pass...")
    with torch.inference_mode():#将单次推理改成了多次推理，实验,后面若要单次推理去掉for循环
        while True:
            actions = model.sample_actions(device, observation)
            if actions is not None:
                break

    actions_np = actions.detach().cpu().numpy()
    print("=== forward success ===")
    print(f"model_config = {train_config.name} (pi05={train_config.model.pi05})")
    print(f"expected_action_shape = {tuple(act_spec.shape)}")
    print(f"actual_action_shape   = {actions_np.shape}")
    print(f"action_dtype          = {actions_np.dtype}")
    print(f"action_preview        = {actions_np[0, 0, : min(8, actions_np.shape[-1])]}")


if __name__ == "__main__":
    main()
