#!/usr/bin/env python3
"""Physically close the Panda gripper and replay a cuRobo grasp/lift in Isaac Sim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from panda_handover.scene_layout import DEFAULT_TABLETOP_LAYOUT
from panda_handover.trajectory_replay import (
    load_grasp_lift_replay,
    sample_positions_at_physics_rate,
)


LAYOUT = DEFAULT_TABLETOP_LAYOUT
PHYSICS_DT_S = 1.0 / 60.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--settle-frames", type=int, default=60)
    parser.add_argument("--close-frames", type=int, default=60)
    parser.add_argument("--hold-frames", type=int, default=180)
    parser.add_argument(
        "--simulation-only",
        action="store_true",
        help="Required acknowledgement: this command controls only an Isaac Sim robot",
    )
    args = parser.parse_args()
    if not args.simulation_only:
        parser.error("--simulation-only is required")
    if min(args.settle_frames, args.close_frames, args.hold_frames) < 0:
        parser.error("frame counts must be non-negative")
    return args


args = parse_args()
replay = load_grasp_lift_replay(args.capture, args.plan)

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

try:
    import numpy as np
    from PIL import Image

    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.sensors.camera import Camera

    from panda_handover.geometry import look_at_quaternion_world

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=PHYSICS_DT_S,
        rendering_dt=1.0 / 30.0,
    )
    world.scene.add_default_ground_plane(z_position=LAYOUT.ground_z_m)
    panda = world.scene.add(
        Franka(
            prim_path="/World/Panda",
            name="panda",
            position=np.asarray(LAYOUT.robot_base_position_m),
        )
    )
    world.scene.add(
        FixedCuboid(
            prim_path="/World/Table",
            name="table",
            position=np.asarray(LAYOUT.table_center_m),
            scale=np.asarray(LAYOUT.table_size_m),
            color=np.array([0.45, 0.32, 0.20]),
        )
    )
    target = world.scene.add(
        DynamicCuboid(
            prim_path="/World/TestObject",
            name="test_object",
            position=np.asarray(LAYOUT.target_center_m),
            scale=np.asarray(LAYOUT.target_size_m),
            color=np.array([0.1, 0.5, 0.9]),
        )
    )
    world.scene.add(
        FixedCuboid(
            prim_path="/World/Obstacle",
            name="obstacle",
            position=np.asarray(LAYOUT.obstacle_center_m),
            scale=np.asarray(LAYOUT.obstacle_size_m),
            color=np.array([0.9, 0.2, 0.1]),
        )
    )

    camera_position = np.asarray(LAYOUT.camera_position_m, dtype=np.float64)
    camera_target = np.asarray(LAYOUT.camera_target_m, dtype=np.float64)
    camera_orientation = look_at_quaternion_world(camera_position, camera_target)
    camera = Camera(
        prim_path="/World/replay_camera",
        position=camera_position,
        orientation=camera_orientation,
        frequency=30,
        resolution=(640, 480),
    )
    world.reset()
    camera.initialize()
    camera.set_world_pose(camera_position, camera_orientation, camera_axes="world")

    def save_rgb(label: str) -> str | None:
        frame = camera.get_current_frame()
        rgba = frame.get("rgba")
        if rgba is None or np.asarray(rgba).size <= 1:
            rgba = frame.get("rgb")
        if rgba is None or np.asarray(rgba).size <= 1:
            return None
        rgb = np.asarray(rgba)[..., :3]
        if np.issubdtype(rgb.dtype, np.floating):
            scale = 255.0 if float(np.nanmax(rgb)) <= 1.0 else 1.0
            rgb = np.clip(rgb * scale, 0, 255).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8, copy=False)
        path = output / f"{label}.png"
        Image.fromarray(rgb).save(path)
        return str(path)

    for _ in range(args.settle_frames):
        world.step(render=True)
    target_settled_position, target_settled_orientation = target.get_world_pose()
    target_settled_position = np.asarray(target_settled_position, dtype=np.float64)
    saved_frames: list[str] = []
    first_frame = save_rgb("00_settled")
    if first_frame:
        saved_frames.append(first_frame)

    isaac_names = tuple(str(name) for name in panda.dof_names)
    if len(set(isaac_names)) != len(isaac_names):
        raise RuntimeError("Isaac Panda DOF names are not unique")
    index_by_name = {name: index for index, name in enumerate(isaac_names)}
    missing = [name for name in replay.joint_names if name not in index_by_name]
    if missing:
        raise RuntimeError(f"planned joints are missing from Isaac Panda: {missing}")
    arm_indices = np.asarray(
        [index_by_name[name] for name in replay.joint_names], dtype=np.int64
    )
    finger_names = ("panda_finger_joint1", "panda_finger_joint2")
    if any(name not in index_by_name for name in finger_names):
        raise RuntimeError("Isaac Panda finger joint names changed")
    finger_indices = np.asarray([index_by_name[name] for name in finger_names], dtype=np.int64)
    capture_by_name = {
        name: replay.capture_joint_positions[index]
        for index, name in enumerate(replay.capture_joint_names)
    }
    expected_start = np.asarray(
        [capture_by_name[name] for name in replay.joint_names], dtype=np.float64
    )
    open_fingers = np.asarray(
        [capture_by_name[name] for name in finger_names], dtype=np.float64
    )
    actual_start = np.asarray(panda.get_joint_positions(), dtype=np.float64)[arm_indices]
    start_error = np.abs(actual_start - expected_start)
    if not np.all(start_error <= 2e-3):
        raise RuntimeError(
            "Isaac Panda did not reproduce the capture start state; "
            f"maximum error={float(start_error.max()):.6g}"
        )

    measurements: dict[str, np.ndarray] = {}
    commands: dict[str, np.ndarray] = {}
    durations: dict[str, float] = {}

    def execute_phase(phase: str, finger_target: np.ndarray) -> None:
        phase_time, phase_commands = sample_positions_at_physics_rate(
            replay.phase_positions[phase],
            replay.phase_segment_dt_s[phase],
            PHYSICS_DT_S,
        )
        phase_measured = np.empty_like(phase_commands)
        all_indices = np.concatenate((arm_indices, finger_indices))
        for index, arm_target in enumerate(phase_commands):
            panda.apply_action(
                ArticulationAction(
                    joint_positions=np.concatenate((arm_target, finger_target)),
                    joint_indices=all_indices,
                )
            )
            world.step(render=True)
            phase_measured[index] = np.asarray(
                panda.get_joint_positions(), dtype=np.float64
            )[arm_indices]
            if not np.isfinite(phase_measured[index]).all():
                raise RuntimeError(f"non-finite Panda state during {phase}")
        commands[phase] = phase_commands
        measurements[phase] = phase_measured
        durations[phase] = float(phase_time[-1])
        frame_path = save_rgb(f"{len(saved_frames):02d}_{phase}_end")
        if frame_path:
            saved_frames.append(frame_path)

    execute_phase("approach", open_fingers)
    execute_phase("grasp", open_fingers)

    grasp_arm_target = commands["grasp"][-1]
    all_indices = np.concatenate((arm_indices, finger_indices))
    closed_finger_target = np.zeros(2, dtype=np.float64)
    for _ in range(args.close_frames):
        panda.apply_action(
            ArticulationAction(
                joint_positions=np.concatenate((grasp_arm_target, closed_finger_target)),
                joint_indices=all_indices,
            )
        )
        world.step(render=True)
    measured_fingers_after_close = np.asarray(
        panda.get_joint_positions(), dtype=np.float64
    )[finger_indices]
    frame_path = save_rgb("03_gripper_closed")
    if frame_path:
        saved_frames.append(frame_path)

    target_before_lift_position, target_before_lift_orientation = target.get_world_pose()
    target_before_lift_position = np.asarray(target_before_lift_position, dtype=np.float64)
    execute_phase("lift", closed_finger_target)
    target_after_lift_position, target_after_lift_orientation = target.get_world_pose()
    target_after_lift_position = np.asarray(target_after_lift_position, dtype=np.float64)

    lift_arm_target = commands["lift"][-1]
    for _ in range(args.hold_frames):
        panda.apply_action(
            ArticulationAction(
                joint_positions=np.concatenate((lift_arm_target, closed_finger_target)),
                joint_indices=all_indices,
            )
        )
        world.step(render=True)
    target_held_position, target_held_orientation = target.get_world_pose()
    target_held_position = np.asarray(target_held_position, dtype=np.float64)
    measured_fingers_held = np.asarray(panda.get_joint_positions(), dtype=np.float64)[
        finger_indices
    ]
    frame_path = save_rgb("05_lift_held")
    if frame_path:
        saved_frames.append(frame_path)

    for phase in ("approach", "grasp", "lift"):
        np.save(output / f"{phase}_commanded_joint_positions.npy", commands[phase])
        np.save(output / f"{phase}_measured_joint_positions.npy", measurements[phase])
        np.save(
            output / f"{phase}_tracking_error_rad.npy",
            measurements[phase] - commands[phase],
        )
    np.save(output / "target_settled_position_world.npy", target_settled_position)
    np.save(output / "target_before_lift_position_world.npy", target_before_lift_position)
    np.save(output / "target_after_lift_position_world.npy", target_after_lift_position)
    np.save(output / "target_held_position_world.npy", target_held_position)

    object_lift_m = float(target_after_lift_position[2] - target_before_lift_position[2])
    held_object_lift_m = float(target_held_position[2] - target_before_lift_position[2])
    minimum_clear_lift_m = float(LAYOUT.target_size_m[2])
    phase_max_errors = {
        phase: float(np.max(np.abs(measurements[phase] - commands[phase])))
        for phase in commands
    }
    physical_pick_observed = bool(
        object_lift_m >= minimum_clear_lift_m
        and held_object_lift_m >= minimum_clear_lift_m
    )
    report = {
        "status": "success" if physical_pick_observed else "physical_pick_not_observed",
        "reference": {
            "controller": "Isaac Sim 5.1 ArticulationAction position targets",
            "dynamic_target": "Isaac Sim DynamicCuboid with default physical properties",
            "finger_close": (
                "Isaac Sim 5.1 articulation controller example: finger joints 7 and 8 to 0"
            ),
            "source_plan": str(args.plan / "grasp_lift_plan_check.json"),
        },
        "inputs": {
            "capture": str(args.capture),
            "plan": str(args.plan),
        },
        "replay": {
            "physics_dt_s": PHYSICS_DT_S,
            "phase_duration_s": durations,
            "phase_command_count": {
                phase: int(value.shape[0]) for phase, value in commands.items()
            },
            "phase_maximum_tracking_error_rad": phase_max_errors,
            "close_frames": args.close_frames,
            "hold_frames": args.hold_frames,
            "open_finger_targets_rad": open_fingers.tolist(),
            "measured_fingers_after_close_rad": measured_fingers_after_close.tolist(),
            "measured_fingers_held_rad": measured_fingers_held.tolist(),
            "saved_review_frames": saved_frames,
        },
        "physical_object": {
            "settled_position_world_m": target_settled_position.tolist(),
            "before_lift_position_world_m": target_before_lift_position.tolist(),
            "after_lift_position_world_m": target_after_lift_position.tolist(),
            "held_position_world_m": target_held_position.tolist(),
            "lift_during_motion_m": object_lift_m,
            "lift_after_hold_m": held_object_lift_m,
            "minimum_clear_lift_evidence_m": minimum_clear_lift_m,
            "physical_pick_observed": physical_pick_observed,
        },
        "automatic_checks": {
            "capture_start_state_reproduced": bool(np.all(start_error <= 2e-3)),
            "all_arm_commands_finite": bool(
                all(np.isfinite(value).all() for value in commands.values())
            ),
            "all_arm_measurements_finite": bool(
                all(np.isfinite(value).all() for value in measurements.values())
            ),
            "finger_measurements_finite": bool(
                np.isfinite(measured_fingers_after_close).all()
                and np.isfinite(measured_fingers_held).all()
            ),
            "object_lifted_by_at_least_one_object_height": physical_pick_observed,
        },
        "safety": {
            "simulation_only": True,
            "dynamic_target_used": True,
            "object_not_fixed_to_gripper": True,
            "physical_gripper_close_commanded": True,
            "physical_contact_monitoring_automated": False,
            "first_lift_held_object_collision_checked_by_curobo": False,
            "manual_review_required": True,
            "safe_for_real_robot_execution": False,
        },
    }
    report_path = output / "grasp_lift_replay_check.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"object lift: {object_lift_m:.6g} m; held after {args.hold_frames} frames: "
        f"{held_object_lift_m:.6g} m"
    )
    print(f"saved: {report_path}")
    if not physical_pick_observed:
        raise SystemExit(2)
except Exception as exc:
    if isinstance(exc, SystemExit):
        raise
    failure_traceback = traceback.format_exc()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "grasp_lift_replay_check.json").write_text(
        json.dumps(
            {
                "status": "failure",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": failure_traceback,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(failure_traceback, file=sys.stderr, flush=True)
    raise
finally:
    sys.stdout.flush()
    sys.stderr.flush()
    simulation_app.close()
