#!/usr/bin/env python3
"""Visualize every saved GraspGenX candidate with explicit filter states.

This viewer is read-only. It reuses GraspGenX's official Viser helpers and
saved pipeline artifacts; it does not regenerate, rescore, or reclassify a
candidate with a new heuristic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--filtered", type=Path, required=True)
    parser.add_argument(
        "--plan-trials",
        type=Path,
        help="Optional cuRobo trial directory containing grasp_lift_trial_plans.json",
    )
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="0 shows every candidate; a positive value keeps highest scores",
    )
    parser.add_argument("--scene-point-size", type=float, default=0.0025)
    parser.add_argument("--target-point-size", type=float, default=0.004)
    parser.add_argument("--grasp-line-width", type=float, default=1.5)
    return parser.parse_args()


def _load_plan_attempts(directory: Path | None) -> list[dict]:
    if directory is None:
        return []
    path = directory / "grasp_lift_trial_plans.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing cuRobo trial manifest: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    attempts = report.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("cuRobo trial manifest has no attempts list")
    return attempts


def main() -> int:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be in 1..65535")
    if args.max_candidates < 0:
        raise ValueError("--max-candidates cannot be negative")
    if args.scene_point_size <= 0.0 or args.target_point_size <= 0.0:
        raise ValueError("point sizes must be positive")
    if args.grasp_line_width <= 0.0:
        raise ValueError("--grasp-line-width must be positive")

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from panda_handover.geometry import transform_points
    from panda_handover.grasp_visualization import (
        CUROBO_PLAN_SUCCESS,
        STATE_COLORS_RGB,
        classify_candidate_states,
        representative_gripper_mesh,
        resolve_saved_gripper_identity,
        select_representative_candidate,
        state_counts,
        verify_saved_world_grasps,
    )

    try:
        from graspgenx.utils.viser_utils import (
            create_visualizer,
            visualize_mesh,
            visualize_pointcloud,
            visualize_x_grasp,
        )
        from graspgenx.x_grippers import (
            resolve_gripper_info,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Run this viewer with the official GraspGenX uv environment"
        ) from exc

    grasps_camera = np.load(
        args.candidates / "grasps_camera.npy", allow_pickle=False
    )
    grasps_world = np.load(
        args.candidates / "grasps_world.npy", allow_pickle=False
    )
    scores = np.load(args.candidates / "scores.npy", allow_pickle=False).reshape(-1)
    collision_free_mask = np.load(
        args.filtered / "collision_free_mask.npy", allow_pickle=False
    ).reshape(-1)
    kept_candidate_indices = np.load(
        args.filtered / "kept_candidate_indices.npy", allow_pickle=False
    ).reshape(-1)
    T_world_camera = np.load(
        args.capture / "T_world_camera.npy", allow_pickle=False
    )
    candidate_report = json.loads(
        (args.candidates / "graspgenx_check.json").read_text(encoding="utf-8")
    )
    filter_report = json.loads(
        (args.filtered / "collision_filter_check.json").read_text(encoding="utf-8")
    )
    gripper_name = resolve_saved_gripper_identity(candidate_report, filter_report)
    candidate_count = len(grasps_camera)
    if grasps_world.shape != grasps_camera.shape:
        raise ValueError("camera/world grasp arrays have different shapes")
    if scores.shape != (candidate_count,):
        raise ValueError("scores and grasp candidates have different lengths")
    if collision_free_mask.shape != (candidate_count,):
        raise ValueError("collision mask and raw candidates have different lengths")
    expected_kept_indices = np.flatnonzero(collision_free_mask)
    if not np.array_equal(kept_candidate_indices, expected_kept_indices):
        raise ValueError(
            "kept_candidate_indices.npy disagrees with collision_free_mask.npy"
        )
    filtered_world_path = args.filtered / "grasps_world.npy"
    if filtered_world_path.is_file():
        filtered_world = np.load(filtered_world_path, allow_pickle=False)
        if not np.allclose(
            filtered_world,
            grasps_world[kept_candidate_indices],
            atol=1e-5,
            rtol=0.0,
        ):
            raise ValueError(
                "filtered world grasps do not match raw candidates at kept indices"
            )
    verify_saved_world_grasps(grasps_camera, grasps_world, T_world_camera)

    plan_attempts = _load_plan_attempts(args.plan_trials)
    states = classify_candidate_states(collision_free_mask, plan_attempts)
    counts = state_counts(states)

    points_camera = np.load(
        args.capture / "points_camera.npy", allow_pickle=False
    ).reshape(-1, 3)
    rgb = np.load(args.capture / "rgb.npy", allow_pickle=False).reshape(-1, 3)
    target_mask = np.load(
        args.segmentation / "union_mask.npy", allow_pickle=False
    ).reshape(-1).astype(bool)
    if not (len(points_camera) == len(rgb) == len(target_mask)):
        raise ValueError("capture points, RGB and segmentation mask are misaligned")
    valid = np.isfinite(points_camera).all(axis=1) & (points_camera[:, 2] > 0.0)
    points_world = transform_points(T_world_camera, points_camera[valid])
    rgb_valid = rgb[valid]
    target_valid = target_mask[valid]

    order = np.argsort(-scores, kind="stable")
    if args.max_candidates:
        order = order[: min(args.max_candidates, len(order))]
    representative_index, representative_reason = select_representative_candidate(
        scores,
        collision_free_mask,
        states,
    )

    gripper = resolve_gripper_info(gripper_name)
    representative_mesh = representative_gripper_mesh(gripper)
    vis = create_visualizer(port=args.port)
    vis.scene.set_up_direction((0.0, 0.0, 1.0))
    vis.scene.world_axes.visible = True
    vis.gui.add_markdown(
        "\n".join(
            [
                "## Grasp candidates",
                "- **Red**: static collision rejected",
                "- **Green**: static collision-free, not planned",
                "- **Yellow**: cuRobo planning rejected",
                "- **Blue**: cuRobo plan succeeded",
                "- **Magenta**: planner/infrastructure error",
                (
                    "- **Light-blue collision mesh**: "
                    f"candidate {representative_index} "
                    f"({representative_reason})"
                    if representative_index is not None
                    else f"- **Light-blue collision mesh**: {representative_reason}"
                ),
                "",
                "All geometry is displayed in the saved Isaac world frame (+Z up).",
            ]
        )
    )
    visualize_pointcloud(
        vis,
        "scene/surrounding",
        points_world[~target_valid],
        rgb_valid[~target_valid],
        size=args.scene_point_size,
    )
    visualize_pointcloud(
        vis,
        "scene/target",
        points_world[target_valid],
        rgb_valid[target_valid],
        size=args.target_point_size,
    )

    for candidate_index in order:
        candidate_index = int(candidate_index)
        state = str(states[candidate_index])
        color = list(STATE_COLORS_RGB[state])
        visualize_x_grasp(
            vis,
            (
                f"candidates/{state}/"
                f"candidate_{candidate_index:03d}_score_{scores[candidate_index]:.4f}"
            ),
            grasps_world[candidate_index],
            color=color,
            gripper_info=gripper,
            linewidth=(
                args.grasp_line_width * 2.5
                if candidate_index == representative_index
                or state == CUROBO_PLAN_SUCCESS
                else args.grasp_line_width
            ),
        )
    if representative_index is not None and representative_index in set(order.tolist()):
        visualize_mesh(
            vis,
            "selection/representative_gripper_mesh",
            representative_mesh,
            color=[80, 200, 255],
            transform=grasps_world[representative_index],
        )

    print(
        json.dumps(
            {
                "frame": "isaac_world_z_up",
                "candidate_count": candidate_count,
                "displayed_candidate_count": int(len(order)),
                "state_counts": counts,
                "representative_candidate": representative_index,
                "representative_candidate_reason": representative_reason,
                "gripper": gripper_name,
                "representative_gripper_geometry": (
                    "official GraspGenX XGripperInfo.collision_mesh"
                ),
                "representative_gripper_frame": "saved GraspGenX grasp frame",
                "candidate_generation_changed": False,
                "candidate_classification_changed": False,
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"Open http://localhost:{args.port}", flush=True)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())
