"""Select the AcceRL policy backend without changing either implementation.

Examples:
    python run.py --backend pi05 --task-id 0
    python run.py --backend openvla --benchmark libero_spatial

The backend defaults to pi0.5. All unrecognized arguments are forwarded to the
selected backend's original command-line parser.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend", choices=("pi05", "openvla"), default="pi05")
    launcher_args, backend_args = parser.parse_known_args()

    script_dir = Path(__file__).resolve().parent
    target = {
        "pi05": script_dir / "ds_libero_ppo_pi05.py",
        "openvla": script_dir / "ds_libero_ppo_discrete.py",
    }[launcher_args.backend]

    sys.argv = [str(target), *backend_args]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
