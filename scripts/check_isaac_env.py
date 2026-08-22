#!/usr/bin/env python3
"""Read-only audit of the known-good Isaac Sim environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


EXPECTED = {
    "numpy": "1.26.0",
    "torch": "2.7.0+cu128",
    "torchvision": "0.22.0+cu128",
    "torchaudio": "2.7.0+cu128",
    "opencv-python": "4.10.0.84",
    "cvxpy": "1.5.3",
    "osqp": "0.6.7.post3",
    "transformers": "5.13.1",
    "warp-lang": "1.8.2",
}


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (result.stdout or result.stderr).strip()
    return text or None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    packages = {name: package_version(name) for name in EXPECTED}
    isaac_packages = {
        name: package_version(name)
        for name in ("isaacsim", "isaacsim-kernel", "isaacsim-core", "isaacsim-robot")
    }
    mismatches = {
        name: {"expected": expected, "actual": packages[name]}
        for name, expected in EXPECTED.items()
        if packages[name] != expected
    }
    report = {
        "schema_version": 1,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": platform.platform(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "packages": packages,
        "isaac_packages": isaac_packages,
        "expected_mismatches": mismatches,
        "nvidia_smi": command_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
        ),
        "important_environment": {
            key: os.environ.get(key)
            for key in ("CUDA_HOME", "LD_LIBRARY_PATH", "PYTHONPATH", "ROS_DISTRO")
        },
    }

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")

    if report["conda_default_env"] != "env_isaaclab":
        print("WARNING: CONDA_DEFAULT_ENV is not env_isaaclab", file=sys.stderr)
    if mismatches:
        print("WARNING: package versions differ from the recorded working environment", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

