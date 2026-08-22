#!/usr/bin/env python3
"""Inspect a saved cuRobo occupied-voxel map in cuRobo's official Viser viewer.

This process is visualization-only. It never constructs a planner or executes
a robot action.
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
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--robot", default="franka.yml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--point-size", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from panda_handover.curobo_bridge import select_named_joint_positions

    points = np.load(args.map / "occupied_points_robot_base.npy").astype(
        np.float32, copy=False
    )
    colors = np.load(args.map / "occupied_colors.npy").astype(np.uint8, copy=False)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"occupied point cloud must be non-empty Nx3, got {points.shape}")
    if colors.shape != points.shape:
        raise ValueError(f"occupied colors {colors.shape} do not match points {points.shape}")

    saved_positions = np.load(args.capture / "panda_joint_positions.npy").astype(
        np.float32, copy=False
    )
    robot_report = json.loads(
        (args.capture / "robot_state.json").read_text(encoding="utf-8")
    )
    saved_names = tuple(robot_report["joint_names"])

    import torch
    from curobo.types import ContentPath, JointState
    from curobo.viewer import ViserVisualizer

    visualizer = ViserVisualizer(
        content_path=ContentPath(robot_config_file=args.robot),
        add_robot_to_scene=True,
        connect_ip=args.host,
        connect_port=args.port,
        add_control_frames=False,
        visualize_robot_spheres=False,
    )
    viewer_joint_names = tuple(visualizer.joint_names)
    viewer_positions = select_named_joint_positions(
        saved_names, saved_positions, viewer_joint_names
    )
    viewer_state = JointState.from_position(
        torch.from_numpy(viewer_positions)
        .to(device="cuda", dtype=torch.float32)
        .unsqueeze(0),
        joint_names=list(viewer_joint_names),
    )
    visualizer.set_joint_state(viewer_state)
    visualizer.add_point_cloud(
        pointcloud=points,
        colors=colors,
        point_size=args.point_size,
        name="/occupied_surface_voxels",
    )

    print(f"occupied points: {points.shape[0]}")
    print(f"viewer: http://localhost:{args.port}")
    print("inspection only; no planner or robot action is running")
    print("Press Ctrl+C to stop")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
