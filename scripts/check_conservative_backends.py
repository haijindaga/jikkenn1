#!/usr/bin/env python3
"""Fail-fast dependency/source audit for conservative mapping backends."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys


EXPECTED_NVBLOX_TORCH_SERIES = "0.0.10"
EXPECTED_ISAAC_ROS_NVBLOX_COMMIT = "a0dbb2a06475dc8fa0dbdf5b919ec53973843d17"
EXPECTED_NVBLOX_CORE_COMMIT = "24eee4948768682fa1ffb969b881efee4fca29c2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--isaac-ros-nvblox-source",
        type=Path,
        help="Optional checkout to audit for backend B",
    )
    return parser.parse_args()


def distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def audit_isaac_ros_source(source: Path) -> dict:
    node_source = source / "nvblox_ros" / "src" / "lib" / "nvblox_node.cpp"
    service_source = source / "nvblox_msgs" / "srv" / "EsdfAndGradients.srv"
    if not node_source.is_file() or not service_source.is_file():
        raise FileNotFoundError("path is not an isaac_ros_nvblox source checkout")
    node_text = node_source.read_text(encoding="utf-8")
    service_text = service_source.read_text(encoding="utf-8")
    try:
        commit = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = "unknown"
    core_source = source / "nvblox_ros" / "nvblox_core"
    try:
        core_commit = subprocess.run(
            ["git", "-C", str(core_source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        core_commit = "unknown"
    return {
        "path": str(source),
        "commit": commit,
        "commit_matches_pin": commit == EXPECTED_ISAAC_ROS_NVBLOX_COMMIT,
        "nvblox_core_commit": core_commit,
        "nvblox_core_commit_matches_pin": core_commit == EXPECTED_NVBLOX_CORE_COMMIT,
        "full_aabb_missing_blocks_are_marked": all(
            token in node_text
            for token in (
                "missing_block_indices",
                "markBlocksForUpdate",
                "BlocksToUpdateType::kEsdf",
            )
        ),
        "service_supports_aabb_clear": all(
            token in service_text
            for token in ("aabbs_to_clear_min_m", "aabbs_to_clear_size_m")
        ),
        "service_supports_sphere_clear": all(
            token in service_text
            for token in ("spheres_to_clear_center_m", "spheres_to_clear_radius_m")
        ),
    }


def main() -> int:
    args = parse_args()
    report = {
        "python": sys.version,
        "backend_a": {
            "nvblox_torch_expected_series": EXPECTED_NVBLOX_TORCH_SERIES,
            "nvblox_torch_actual": distribution_version("nvblox-torch"),
            "scipy": distribution_version("scipy"),
            "torch": distribution_version("torch"),
        },
    }
    try:
        from nvblox_torch.mapper import Mapper, QueryType
        from nvblox_torch.mapper_params import (
            MapperParams,
            ProjectiveIntegratorParams,
            ViewCalculatorParams,
        )
        from nvblox_torch.sensor import Sensor

        report["backend_a"]["required_api_available"] = all(
            item is not None
            for item in (
                Mapper,
                QueryType.TSDF,
                MapperParams,
                ProjectiveIntegratorParams,
                ViewCalculatorParams,
                Sensor,
            )
        )
    except Exception as error:  # The audit must report binary-load failures cleanly.
        report["backend_a"]["required_api_available"] = False
        report["backend_a"]["api_import_error"] = repr(error)
    if args.isaac_ros_nvblox_source is not None:
        report["backend_b"] = audit_isaac_ros_source(
            args.isaac_ros_nvblox_source.resolve()
        )
    print(json.dumps(report, indent=2))
    backend_a = report["backend_a"]
    ok = (
        backend_a["nvblox_torch_actual"] is not None
        and backend_a["nvblox_torch_actual"].startswith(EXPECTED_NVBLOX_TORCH_SERIES)
        and backend_a["scipy"] is not None
        and backend_a["torch"] is not None
        and backend_a["required_api_available"]
    )
    if "backend_b" in report:
        ok = ok and all(
            value
            for key, value in report["backend_b"].items()
            if key
            in {
                "commit_matches_pin",
                "nvblox_core_commit_matches_pin",
                "full_aabb_missing_blocks_are_marked",
                "service_supports_aabb_clear",
                "service_supports_sphere_clear",
            }
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
