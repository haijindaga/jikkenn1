#!/usr/bin/env python3
"""Replay a validated cuRobo pre-grasp trajectory in the matching Isaac scene.

This is a simulation-only visual review gate.  It does not plan or execute the
final contact approach, close the gripper, attach the object, or lift it.
"""

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
    load_pregrasp_replay,
    sample_positions_at_physics_rate,
)


LAYOUT = DEFAULT_TABLETOP_LAYOUT
PHYSICS_DT_S = 1.0 / 60.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scene-usd",
        type=Path,
        help="Open the authored USD used by capture instead of the legacy block scene",
    )
    parser.add_argument("--panda-prim", default="/World/Panda")
    parser.add_argument("--target-prim", default="/World/Objects/Target")
    parser.add_argument("--camera-prim", default="/World/camera_0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--settle-frames", type=int, default=60)
    parser.add_argument("--hold-frames", type=int, default=300)
    parser.add_argument(
        "--simulation-only",
        action="store_true",
        help="Required acknowledgement: this command controls only an Isaac Sim robot",
    )
    args = parser.parse_args()
    if not args.simulation_only:
        parser.error("--simulation-only is required")
    if args.settle_frames < 0 or args.hold_frames < 0:
        parser.error("frame counts must be non-negative")
    return args


args = parse_args()
replay = load_pregrasp_replay(args.capture, args.plan)
scene_usd = None
if args.scene_usd is not None:
    scene_usd = args.scene_usd.expanduser().resolve()
    if not scene_usd.is_file():
        raise FileNotFoundError(f"authored scene USD does not exist: {scene_usd}")
    if scene_usd.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError("--scene-usd must end in .usd, .usda, or .usdc")

    scene_layout_path = args.capture / "scene_layout.json"
    if not scene_layout_path.is_file():
        raise FileNotFoundError(
            "authored-scene replay requires the capture scene report: "
            f"{scene_layout_path}"
        )
    scene_layout_report = json.loads(scene_layout_path.read_text(encoding="utf-8"))
    scene_source = scene_layout_report.get("scene_source", {})
    if scene_source.get("kind") != "authored_usd_scene":
        raise ValueError(
            "--scene-usd was provided, but the capture was not recorded from an "
            "authored USD scene"
        )
    recorded_scene_value = scene_source.get("scene_usd")
    if not recorded_scene_value:
        raise ValueError("capture scene report has no authored scene_usd path")
    recorded_scene = Path(recorded_scene_value).expanduser().resolve()
    if recorded_scene != scene_usd:
        raise ValueError(
            "replay scene does not match the scene recorded by capture: "
            f"{scene_usd} != {recorded_scene}"
        )

# Isaac requires SimulationApp construction before importing other Isaac modules.
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

try:
    import numpy as np
    from PIL import Image

    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.core.experimental.utils import stage as stage_utils
    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.sensors.camera import Camera

    from panda_handover.geometry import look_at_quaternion_world

    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    if scene_usd is not None:
        stage_opened, stage = stage_utils.open_stage(str(scene_usd))
        if not stage_opened or stage is None:
            raise RuntimeError(f"Isaac Sim could not open authored scene: {scene_usd}")
        required_prim_paths = (
            args.panda_prim,
            args.target_prim,
            args.camera_prim,
        )
        missing_prims = [
            prim_path
            for prim_path in required_prim_paths
            if not stage.GetPrimAtPath(prim_path).IsValid()
        ]
        if missing_prims:
            raise RuntimeError(
                "authored scene is missing required prims: " + ", ".join(missing_prims)
            )

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=PHYSICS_DT_S,
        rendering_dt=1.0 / 30.0,
    )
    if scene_usd is None:
        world.scene.add_default_ground_plane(z_position=LAYOUT.ground_z_m)
        panda = world.scene.add(
            Franka(
                prim_path=args.panda_prim,
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
        world.scene.add(
            FixedCuboid(
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
    else:
        panda = world.scene.add(
            SingleArticulation(
                prim_path=args.panda_prim,
                name="panda",
            )
        )
        camera = Camera(
            prim_path=args.camera_prim,
            frequency=30,
            resolution=(640, 480),
        )

    world.reset()
    camera.initialize()
    if scene_usd is None:
        camera.set_world_pose(camera_position, camera_orientation, camera_axes="world")

    def save_camera_rgb(path: Path) -> bool:
        frame = camera.get_current_frame()
        rgba = frame.get("rgba")
        if rgba is None or np.asarray(rgba).size <= 1:
            rgba = frame.get("rgb")
        if rgba is None or np.asarray(rgba).size <= 1:
            return False
        rgb = np.asarray(rgba)[..., :3]
        if np.issubdtype(rgb.dtype, np.floating):
            scale = 255.0 if float(np.nanmax(rgb)) <= 1.0 else 1.0
            rgb = np.clip(rgb * scale, 0, 255).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8, copy=False)
        Image.fromarray(rgb, mode="RGB").save(path)
        return True

    for _ in range(args.settle_frames):
        world.step(render=True)

    isaac_names = tuple(str(name) for name in panda.dof_names)
    if len(set(isaac_names)) != len(isaac_names):
        raise RuntimeError("Isaac Panda DOF names are not unique")
    name_to_isaac_index = {name: index for index, name in enumerate(isaac_names)}
    missing = [name for name in replay.joint_names if name not in name_to_isaac_index]
    if missing:
        raise RuntimeError(f"planned joints are missing from Isaac Panda: {missing}")
    active_indices = np.asarray(
        [name_to_isaac_index[name] for name in replay.joint_names], dtype=np.int64
    )
    capture_by_name = {
        name: replay.capture_joint_positions[index]
        for index, name in enumerate(replay.capture_joint_names)
    }
    expected_start = np.asarray(
        [capture_by_name[name] for name in replay.joint_names], dtype=np.float64
    )
    actual_start = np.asarray(panda.get_joint_positions(), dtype=np.float64)[active_indices]
    start_error = np.abs(actual_start - expected_start)
    if not np.all(start_error <= 2e-3):
        raise RuntimeError(
            "Isaac Panda did not reproduce the captured start state; "
            f"maximum error={float(start_error.max()):.6g}"
        )

    replay_time, commanded = sample_positions_at_physics_rate(
        replay.positions, replay.segment_dt_s, PHYSICS_DT_S
    )
    measured = np.empty_like(commanded)
    frame_indices = set(
        np.linspace(0, commanded.shape[0] - 1, 5, dtype=np.int64).tolist()
    )
    saved_frames: list[str] = []
    for index, target in enumerate(commanded):
        panda.apply_action(
            ArticulationAction(
                joint_positions=target,
                joint_indices=active_indices,
            )
        )
        world.step(render=True)
        measured[index] = np.asarray(panda.get_joint_positions(), dtype=np.float64)[
            active_indices
        ]
        if not np.all(np.isfinite(measured[index])):
            raise RuntimeError(f"Isaac returned a non-finite joint state at command {index}")
        if index in frame_indices:
            frame_path = output / f"replay_{index:04d}.png"
            if save_camera_rgb(frame_path):
                saved_frames.append(str(frame_path))

    final_target = commanded[-1]
    for _ in range(args.hold_frames):
        panda.apply_action(
            ArticulationAction(
                joint_positions=final_target,
                joint_indices=active_indices,
            )
        )
        world.step(render=True)

    held_final = np.asarray(panda.get_joint_positions(), dtype=np.float64)[active_indices]
    if not np.all(np.isfinite(held_final)):
        raise RuntimeError("Isaac returned a non-finite final held joint state")
    held_final_error = held_final - final_target
    held_frame_path = output / "replay_final_held.png"
    if save_camera_rgb(held_frame_path):
        saved_frames.append(str(held_frame_path))

    tracking_error = measured - commanded
    np.save(output / "replay_time_s.npy", replay_time)
    np.save(output / "commanded_joint_positions.npy", commanded)
    np.save(output / "measured_joint_positions.npy", measured)
    np.save(output / "tracking_error_rad.npy", tracking_error)
    np.save(output / "held_final_joint_positions.npy", held_final)
    np.save(output / "held_final_error_rad.npy", held_final_error)
    report = {
        "status": "success",
        "reference": {
            "controller": "Isaac Sim 5.1 ArticulationAction position targets",
            "source_plan": str(args.plan / "pregrasp_plan_check.json"),
        },
        "inputs": {
            "capture": str(args.capture),
            "plan": str(args.plan),
            "scene_usd": str(scene_usd) if scene_usd is not None else None,
            "scene_kind": (
                "authored_usd_scene" if scene_usd is not None else "legacy_block_scene"
            ),
            "panda_prim": args.panda_prim,
            "target_prim": args.target_prim if scene_usd is not None else "/World/TestObject",
            "camera_prim": args.camera_prim if scene_usd is not None else "/World/replay_camera",
            "source_waypoints": int(replay.positions.shape[0]),
        },
        "replay": {
            "physics_dt_s": PHYSICS_DT_S,
            "duration_s": float(replay_time[-1]),
            "command_count": int(commanded.shape[0]),
            "joint_names": list(replay.joint_names),
            "isaac_joint_indices": active_indices.tolist(),
            "maximum_start_state_error": float(start_error.max(initial=0.0)),
            "maximum_tracking_error": float(np.max(np.abs(tracking_error))),
            "rms_tracking_error": float(np.sqrt(np.mean(np.square(tracking_error)))),
            "hold_frames": args.hold_frames,
            "maximum_held_final_error": float(np.max(np.abs(held_final_error))),
            "rms_held_final_error": float(
                np.sqrt(np.mean(np.square(held_final_error)))
            ),
            "saved_review_frames": saved_frames,
        },
        "automatic_checks": {
            "replay_scene_matches_capture": True,
            "capture_start_state_reproduced": bool(np.all(start_error <= 2e-3)),
            "all_commands_finite": bool(np.isfinite(commanded).all()),
            "all_measurements_finite": bool(np.isfinite(measured).all()),
            "held_final_measurement_is_finite": bool(np.isfinite(held_final).all()),
            "final_command_matches_saved_plan": bool(
                np.allclose(commanded[-1], replay.positions[-1], atol=1e-12, rtol=0.0)
            ),
        },
        "safety": {
            "simulation_only": True,
            "position_targets_applied_without_waypoint_teleportation": True,
            "final_approach_executed": False,
            "gripper_closed": False,
            "object_attached": False,
            "physical_contact_monitoring_automated": False,
            "manual_review_required": True,
            "safe_for_real_robot_execution": False,
        },
    }
    report_path = output / "replay_check.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"replayed {commanded.shape[0]} position targets over {replay_time[-1]:.3f} s")
    print(f"maximum tracking error: {report['replay']['maximum_tracking_error']:.6g}")
    print(f"saved: {report_path}")
except Exception as exc:
    failure_traceback = traceback.format_exc()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "replay_check.json").write_text(
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
