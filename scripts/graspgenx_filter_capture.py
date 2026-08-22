#!/usr/bin/env python3
"""Apply GraspGenX's official scene point-cloud collision filter.

Run with the official GraspGenX ``uv`` environment after grasp candidate
generation.  This is a static, observed-point-cloud prefilter; it is not a
trajectory or final safety check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collision-threshold", type=float, default=0.02)
    parser.add_argument("--max-scene-points", type=int, default=8192)
    parser.add_argument("--num-collision-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--viser-port", type=int, default=8081)
    parser.add_argument("--max-visualized-grasps", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.collision_threshold <= 0:
        raise ValueError("--collision-threshold must be positive")
    if args.max_scene_points <= 0:
        raise ValueError("--max-scene-points must be positive")
    if args.num_collision_samples <= 0:
        raise ValueError("--num-collision-samples must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_visualized_grasps < 0:
        raise ValueError("--max-visualized-grasps cannot be negative")
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    try:
        import trimesh
        from graspgenx.utils.collision_filter import filter_colliding_grasps
        from graspgenx.x_grippers import resolve_gripper_info
    except ImportError as exc:
        raise RuntimeError(
            "Run this script with the official GraspGenX uv environment."
        ) from exc

    from panda_handover.grasp_candidates import (
        save_collision_filter_results,
        split_target_from_scene,
    )

    points_camera = np.load(args.capture / "points_camera.npy")
    rgb = np.load(args.capture / "rgb.npy")
    T_world_camera = np.load(args.capture / "T_world_camera.npy")
    union_mask = np.load(args.segmentation / "union_mask.npy")
    grasps = np.load(args.candidates / "grasps_camera.npy")
    scores = np.load(args.candidates / "scores.npy")
    branch_tags = json.loads((args.candidates / "branch_tags.json").read_text())

    surrounding, surrounding_rgb, target, target_rgb = split_target_from_scene(
        points_camera, union_mask, rgb
    )
    scene_count_before = len(surrounding)

    # The official demo uses np.random.choice for the 8192-point cap and
    # trimesh.sample.sample_surface for 2000 gripper samples.  Seed the same
    # operations so an experiment can be exactly repeated.
    np.random.seed(args.random_seed)
    if len(surrounding) > args.max_scene_points:
        indices = np.random.choice(
            len(surrounding), args.max_scene_points, replace=False
        )
        collision_scene = surrounding[indices]
    else:
        collision_scene = surrounding

    gripper = resolve_gripper_info("franka_panda")
    surface_points, _ = trimesh.sample.sample_surface(
        gripper.collision_mesh, args.num_collision_samples
    )
    surface_points = np.asarray(surface_points, dtype=np.float32)
    started = time.monotonic()
    collision_free_mask = filter_colliding_grasps(
        scene_pc=collision_scene,
        grasp_poses=grasps,
        collision_threshold=args.collision_threshold,
        gripper_surface_points=surface_points,
        batch_size=args.batch_size,
        device=None if args.device == "auto" else args.device,
    )
    elapsed_ms = (time.monotonic() - started) * 1000.0

    parameters = {
        "collision_threshold_m": args.collision_threshold,
        "max_scene_points": args.max_scene_points,
        "num_collision_samples": args.num_collision_samples,
        "batch_size": args.batch_size,
        "random_seed": args.random_seed,
        "device": args.device,
        "elapsed_ms": elapsed_ms,
    }
    report = save_collision_filter_results(
        args.output,
        grasps_camera=grasps,
        scores=scores,
        branch_tags=branch_tags,
        collision_free_mask=collision_free_mask,
        T_world_camera=T_world_camera,
        collision_scene_camera=collision_scene,
        scene_point_count_before_downsampling=scene_count_before,
        parameters=parameters,
    )
    print(
        "collision filter: "
        f"{report['candidates']['collision_free']}/{report['candidates']['before']} free",
        flush=True,
    )
    print(f"saved: {args.output / 'collision_filter_check.json'}", flush=True)

    if args.visualize:
        from graspgenx.utils.viser_utils import (
            create_visualizer,
            get_color_from_score,
            visualize_mesh,
            visualize_pointcloud,
            visualize_x_grasp,
        )

        vis = create_visualizer(port=args.viser_port)
        visualize_pointcloud(
            vis,
            "scene/surrounding",
            surrounding,
            surrounding_rgb,
            size=0.0025,
        )
        visualize_pointcloud(vis, "scene/target", target, target_rgb, size=0.004)
        kept_grasps = grasps[collision_free_mask]
        kept_scores = scores[collision_free_mask]
        score_colors = get_color_from_score(kept_scores, use_255_scale=True)
        limit = min(args.max_visualized_grasps, len(kept_grasps))
        best_index = int(np.argmax(kept_scores)) if len(kept_scores) else None
        for index in range(limit):
            color = [0, 100, 255] if index == best_index else score_colors[index]
            visualize_x_grasp(
                vis,
                f"collision_free/grasp_{index:03d}",
                kept_grasps[index],
                color=color,
                gripper_info=gripper,
                linewidth=5.0 if index == best_index else 1.5,
            )
        if best_index is not None:
            visualize_mesh(
                vis,
                "collision_free/top_gripper_collision_mesh",
                gripper.collision_mesh,
                color=[0, 100, 255],
                transform=kept_grasps[best_index],
            )
        print(f"Open http://localhost:{args.viser_port}", flush=True)
        while True:
            time.sleep(1)

    return 0 if report["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
