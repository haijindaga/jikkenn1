#!/usr/bin/env python3
"""Locate whether start-state map collisions originate in filtered depth.

This is a read-only diagnostic. It does not modify the ESDF, planner, robot
model, or trajectory artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--prepared-map", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view-index", type=int, default=0)
    parser.add_argument("--point-batch-size", type=int, default=32768)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from panda_handover.map_diagnostics import (
        backproject_depth_to_frame,
        nearest_point_to_spheres,
    )

    view = args.prepared_map / "views" / f"camera_{args.view_index}"
    mapping_depth = np.load(view / "mapping_depth_m.npy", allow_pickle=False)
    robot_mask = np.load(view / "robot_mask.npy", allow_pickle=False).astype(bool)
    intrinsics = np.load(args.capture / "intrinsics.npy", allow_pickle=False)
    transform = np.load(
        args.capture / "T_robot_base_camera.npy", allow_pickle=False
    )
    spheres = np.load(
        args.plan / "start_robot_spheres.npy", allow_pickle=False
    ).reshape(-1, 4)
    costs = np.load(
        args.plan / "start_penetration_cost.npy", allow_pickle=False
    ).reshape(-1)
    if mapping_depth.shape != robot_mask.shape:
        raise ValueError("mapping depth and robot mask shapes differ")
    if len(costs) != len(spheres):
        raise ValueError("start collision costs and sphere count differ")

    prepared_report = json.loads(
        (args.prepared_map / "esdf_check.json").read_text(encoding="utf-8")
    )
    threshold = float(
        prepared_report["parameters"]["robot_distance_threshold_m"]
    )
    points, pixels_vu = backproject_depth_to_frame(
        mapping_depth, intrinsics, transform
    )
    distances, clearances, nearest_indices = nearest_point_to_spheres(
        points, spheres, point_batch_size=args.point_batch_size
    )
    hit = costs > 0.0
    hit_clearance = clearances[hit]
    tolerance = 1e-4
    inside_sphere = hit_clearance < -tolerance
    inside_expected_buffer = hit_clearance < threshold - tolerance

    nearest_pixels = pixels_vu[nearest_indices]
    nearest_points = points[nearest_indices]
    nearest_mask_values = robot_mask[
        nearest_pixels[:, 0], nearest_pixels[:, 1]
    ]
    depth_nonzero_under_robot_mask = int(
        np.count_nonzero(
            np.isfinite(mapping_depth[robot_mask])
            & (mapping_depth[robot_mask] > 0.0)
        )
    )

    if bool(inside_sphere.any()):
        diagnosis = "filtered_depth_contains_points_inside_robot_collision_spheres"
    elif bool(inside_expected_buffer.any()):
        diagnosis = "filtered_depth_contains_points_inside_segmenter_expected_buffer"
    else:
        diagnosis = "filtered_depth_does_not_explain_esdf_start_collision"

    colliding_indices = np.flatnonzero(hit)
    hit_rows = []
    for local_index, sphere_index in enumerate(colliding_indices):
        point_index = int(nearest_indices[sphere_index])
        hit_rows.append(
            {
                "sphere_index": int(sphere_index),
                "collision_cost_m": float(costs[sphere_index]),
                "sphere_xyzr_robot_base_m": spheres[sphere_index].tolist(),
                "nearest_mapping_point_robot_base_m": nearest_points[
                    sphere_index
                ].tolist(),
                "nearest_pixel_vu": nearest_pixels[sphere_index].tolist(),
                "nearest_point_index": point_index,
                "centre_distance_m": float(distances[sphere_index]),
                "surface_clearance_m": float(hit_clearance[local_index]),
                "inside_collision_sphere": bool(inside_sphere[local_index]),
                "inside_expected_segmentation_buffer": bool(
                    inside_expected_buffer[local_index]
                ),
            }
        )

    checks = {
        "mapping_depth_is_zero_under_robot_mask": (
            depth_nonzero_under_robot_mask == 0
        ),
        "nearest_mapping_pixels_are_not_robot_masked": bool(
            np.all(~nearest_mask_values)
        ),
        "pointcloud_is_finite": bool(np.isfinite(points).all()),
        "at_least_one_start_sphere_is_reported_colliding": bool(hit.any()),
    }
    report = {
        "status": "success" if all(checks.values()) else "failed_checks",
        "diagnosis": diagnosis,
        "reference": {
            "robot_segmentation": (
                "cuRobo RobotSegmenter collision-sphere distance mask"
            ),
            "pixel_convention": "pixel centres at (u + 0.5, v + 0.5)",
        },
        "inputs": {
            "capture": str(args.capture),
            "prepared_map": str(args.prepared_map),
            "plan": str(args.plan),
            "view_index": args.view_index,
        },
        "parameters": {
            "robot_distance_threshold_m": threshold,
            "comparison_tolerance_m": tolerance,
        },
        "counts": {
            "mapping_points": int(len(points)),
            "robot_mask_pixels": int(robot_mask.sum()),
            "mapping_depth_nonzero_under_robot_mask": (
                depth_nonzero_under_robot_mask
            ),
            "robot_spheres": int(len(spheres)),
            "colliding_robot_spheres": int(hit.sum()),
            "colliding_spheres_with_mapping_point_inside_sphere": int(
                inside_sphere.sum()
            ),
            "colliding_spheres_with_mapping_point_inside_expected_buffer": int(
                inside_expected_buffer.sum()
            ),
        },
        "clearance_m": {
            "colliding_sphere_min": float(hit_clearance.min()),
            "colliding_sphere_median": float(np.median(hit_clearance)),
            "colliding_sphere_max": float(hit_clearance.max()),
        },
        "colliding_spheres": hit_rows,
        "automatic_checks": checks,
        "safety": {
            "map_modified": False,
            "planner_modified": False,
            "trajectory_executed": False,
            "safe_to_execute": False,
        },
        "next_gate": (
            "If filtered depth enters the expected robot buffer, inspect the "
            "RobotSegmenter/planner sphere-model match. Otherwise inspect Backend A "
            "TSDF coordinates and dense-grid loading before changing mask thresholds."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "robot_map_overlap_check.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "diagnosis": diagnosis,
        "counts": report["counts"],
        "clearance_m": report["clearance_m"],
        "automatic_checks": checks,
    }, indent=2))
    print(f"saved: {report_path}")
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
