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
EXPECTED_TORCH_VERSION = "2.9.1+cu128"
EXPECTED_TORCHVISION_VERSION = "0.24.1+cu128"
EXPECTED_NPP_VERSION = "12.3.3.65"
EXPECTED_ISAAC_ROS_NVBLOX_COMMIT = "a0dbb2a06475dc8fa0dbdf5b919ec53973843d17"
EXPECTED_NVBLOX_CORE_COMMIT = "24eee4948768682fa1ffb969b881efee4fca29c2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("a", "b", "both"),
        default="a",
        help="Select which isolated backend contract to audit",
    )
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
        gitlink_output = subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "ls-tree",
                "HEAD",
                "nvblox_ros/nvblox_core",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        core_gitlink_commit = gitlink_output.split()[2]
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
        core_gitlink_commit = "unknown"
    try:
        if not (core_source / "CMakeLists.txt").is_file():
            raise FileNotFoundError(core_source)
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
        "nvblox_core_gitlink_commit": core_gitlink_commit,
        "nvblox_core_gitlink_matches_pin": (
            core_gitlink_commit == EXPECTED_NVBLOX_CORE_COMMIT
        ),
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
    }
    if args.backend in {"a", "both"}:
        torch_actual = distribution_version("torch")
        torchvision_actual = distribution_version("torchvision")
        npp_actual = distribution_version("nvidia-npp-cu12")
        required_versions_match = (
            torch_actual == EXPECTED_TORCH_VERSION
            and torchvision_actual == EXPECTED_TORCHVISION_VERSION
            and npp_actual == EXPECTED_NPP_VERSION
        )
        report["backend_a"] = {
            "nvblox_torch_expected_series": EXPECTED_NVBLOX_TORCH_SERIES,
            "nvblox_torch_actual": distribution_version("nvblox-torch"),
            "scipy": distribution_version("scipy"),
            "torch_expected": EXPECTED_TORCH_VERSION,
            "torch_actual": torch_actual,
            "torchvision_expected": EXPECTED_TORCHVISION_VERSION,
            "torchvision_actual": torchvision_actual,
            "nvidia_npp_cu12_expected": EXPECTED_NPP_VERSION,
            "nvidia_npp_cu12_actual": npp_actual,
            "required_versions_match": required_versions_match,
        }
        if not required_versions_match:
            report["backend_a"]["required_api_available"] = False
            report["backend_a"]["api_import_error"] = (
                "Skipped native import because Backend A versions do not match the "
                "official nvblox v0.0.10 x86_64/CUDA 12 build set"
            )
        else:
            try:
                import torch
                from nvblox_torch.mapper import Mapper, QueryType
                from nvblox_torch.mapper_params import (
                    MapperParams,
                    ProjectiveIntegratorParams,
                    ViewCalculatorParams,
                )
                from nvblox_torch.sensor import Sensor

                report["backend_a"]["torch_cuda_runtime"] = torch.version.cuda
                report["backend_a"]["torch_cxx11_abi"] = bool(
                    torch._C._GLIBCXX_USE_CXX11_ABI
                )
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
            except Exception as error:  # Report binary-load failures cleanly.
                report["backend_a"]["required_api_available"] = False
                report["backend_a"]["api_import_error"] = repr(error)
    if args.backend in {"b", "both"}:
        if args.isaac_ros_nvblox_source is None:
            raise ValueError("--backend b requires --isaac-ros-nvblox-source")
        report["backend_b"] = audit_isaac_ros_source(
            args.isaac_ros_nvblox_source.resolve()
        )
    print(json.dumps(report, indent=2))
    ok = True
    if "backend_a" in report:
        backend_a = report["backend_a"]
        ok = ok and (
            backend_a["nvblox_torch_actual"] is not None
            and backend_a["nvblox_torch_actual"].startswith(
                EXPECTED_NVBLOX_TORCH_SERIES
            )
            and backend_a["scipy"] is not None
            and backend_a["required_versions_match"]
            and backend_a["required_api_available"]
        )
    if "backend_b" in report:
        ok = ok and all(
            value
            for key, value in report["backend_b"].items()
            if key
            in {
                "commit_matches_pin",
                "nvblox_core_gitlink_matches_pin",
                "nvblox_core_commit_matches_pin",
                "full_aabb_missing_blocks_are_marked",
                "service_supports_aabb_clear",
                "service_supports_sphere_clear",
            }
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
