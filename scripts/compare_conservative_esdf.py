#!/usr/bin/env python3
"""Compare backend A and B conservative ESDF outputs on the same voxel grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-a", type=Path, required=True)
    parser.add_argument("--backend-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--surface-band", type=float, default=0.02)
    return parser.parse_args()


def load_backend(path: Path) -> tuple[dict, np.ndarray]:
    report = json.loads((path / "esdf_check.json").read_text(encoding="utf-8"))
    esdf = np.load(path / "esdf_features.npy").astype(np.float32, copy=False)
    expected_shape = tuple(report["grid"]["shape_xyz"])
    if esdf.shape != expected_shape:
        raise ValueError(f"{path}: ESDF shape {esdf.shape} != report {expected_shape}")
    if report.get("safe_to_plan") is not False:
        raise ValueError(f"{path}: comparison expects the pre-target-clear safety gate")
    return report, esdf


def grids_match(grid_a: dict, grid_b: dict) -> bool:
    """Compare serialized grids while tolerating float32 ROS metadata."""
    if tuple(grid_a["shape_xyz"]) != tuple(grid_b["shape_xyz"]):
        return False
    for key in (
        "voxel_size_m",
        "extent_m",
        "center_robot_base_m",
        "min_corner_robot_base_m",
    ):
        if not np.allclose(grid_a[key], grid_b[key], atol=1e-7, rtol=0.0):
            return False
    return True


def input_fingerprints_match(report_a: dict, report_b: dict) -> bool:
    fingerprint_a = report_a.get("input_fingerprint_sha256")
    fingerprint_b = report_b.get("input_fingerprint_sha256")
    return (
        isinstance(fingerprint_a, str)
        and len(fingerprint_a) == 64
        and fingerprint_a == fingerprint_b
    )


def compare_fields(
    esdf_a: np.ndarray, esdf_b: np.ndarray, *, surface_band_m: float
) -> tuple[dict, dict[str, np.ndarray]]:
    if esdf_a.shape != esdf_b.shape:
        raise ValueError("backend ESDF shapes must match")
    if surface_band_m < 0.0 or not np.isfinite(surface_band_m):
        raise ValueError("surface_band_m must be finite and non-negative")
    free_a = esdf_a > 0.0
    free_b = esdf_b > 0.0
    agreement = free_a == free_b
    both_free = free_a & free_b
    both_blocked = ~free_a & ~free_b
    a_only_free = free_a & ~free_b
    b_only_free = free_b & ~free_a
    away_from_surfaces = (np.abs(esdf_a) > surface_band_m) & (
        np.abs(esdf_b) > surface_band_m
    )
    comparable = both_free & away_from_surfaces
    absolute_difference = np.abs(esdf_a - esdf_b)
    metrics = {
        "sign_agreement_fraction": float(agreement.mean()),
        "both_free_fraction": float(both_free.mean()),
        "both_blocked_fraction": float(both_blocked.mean()),
        "backend_a_only_free_fraction": float(a_only_free.mean()),
        "backend_b_only_free_fraction": float(b_only_free.mean()),
        "comparable_free_voxels": int(comparable.sum()),
        "mean_abs_distance_difference_m": (
            float(absolute_difference[comparable].mean()) if comparable.any() else None
        ),
        "p95_abs_distance_difference_m": (
            float(np.percentile(absolute_difference[comparable], 95.0))
            if comparable.any()
            else None
        ),
    }
    masks = {
        "sign_agreement_mask": agreement,
        "backend_a_only_free_mask": a_only_free,
        "backend_b_only_free_mask": b_only_free,
    }
    return metrics, masks


def main() -> int:
    args = parse_args()
    report_a, esdf_a = load_backend(args.backend_a)
    report_b, esdf_b = load_backend(args.backend_b)
    if not grids_match(report_a["grid"], report_b["grid"]):
        raise ValueError("backend grid metadata differs")
    if not input_fingerprints_match(report_a, report_b):
        raise ValueError("backend inputs differ; refusing an invalid A/B comparison")
    metrics, masks = compare_fields(
        esdf_a, esdf_b, surface_band_m=args.surface_band
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    for name, mask in masks.items():
        np.save(output / f"{name}.npy", mask)
    checks = {
        "backend_a_checks_passed": all(report_a["automatic_checks"].values()),
        "backend_b_checks_passed": all(report_b["automatic_checks"].values()),
        "same_grid": True,
        "same_input_fingerprint": True,
        "both_keep_target_blocked": bool(
            report_a["unknown_environment_contract"]["target_is_currently_blocked"]
            and report_b["unknown_environment_contract"]["target_is_currently_blocked"]
        ),
    }
    comparison = {
        "status": "success" if all(checks.values()) else "failed_checks",
        "backend_a": str(args.backend_a),
        "backend_b": str(args.backend_b),
        "grid": report_a["grid"],
        "surface_band_excluded_from_distance_statistics_m": args.surface_band,
        "metrics": metrics,
        "automatic_checks": checks,
        "safe_to_plan": False,
        "next_gate": (
            "Inspect sign disagreement masks and choose the target-clear proxy before "
            "creating any motion planner"
        ),
    }
    report_path = output / "comparison.json"
    report_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(f"sign agreement: {metrics['sign_agreement_fraction']:.6f}")
    print(f"saved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
