#!/usr/bin/env python3
"""Plan approach, contact approach, and a short lift with cuRobo ``plan_grasp``.

This simulation-only gate consumes a previously reviewed single-view pre-grasp
artifact.  The target remains absent from the observed collision mesh, while
the Panda, table, and surrounding observed geometry remain collision checked.
The saved trajectories are plans only; Isaac Sim performs the physical gripper
close and decides whether the dynamic object was actually retained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


CUROBO_COMMIT = "057a96ffb1088531535f9915154f9d0dabd62428"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--pregrasp-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot", default="franka.yml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lift-offset", type=float, default=0.15)
    parser.add_argument("--max-attempts", type=int, default=2)
    return parser.parse_args()


def _cpu_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _trajectory_field(trajectory: Any, name: str, *, required: bool) -> np.ndarray | None:
    value = getattr(trajectory, name, None)
    if value is None:
        if required:
            raise RuntimeError(f"planned trajectory has no {name}")
        return None
    array = _cpu_numpy(value)
    if array.ndim > 2:
        if any(size != 1 for size in array.shape[:-2]):
            raise RuntimeError(
                f"trajectory {name} has multiple batch/seed entries: {array.shape}"
            )
        array = array.reshape(array.shape[-2:])
    if array.ndim != 2:
        raise RuntimeError(f"trajectory {name} must be HxD, got {array.shape}")
    return array.astype(np.float32, copy=False)


def _get_active_trajectory(planner: Any, trajectory: Any) -> tuple[Any, list[str]]:
    raw_position = np.asarray(_cpu_numpy(getattr(trajectory, "position", None)))
    raw_names = getattr(trajectory, "joint_names", None)
    if raw_position.ndim < 2 or raw_names is None:
        raise RuntimeError("full trajectory must contain positions and joint_names")
    full_names = [str(name) for name in raw_names]
    if len(full_names) != raw_position.shape[-1] or len(set(full_names)) != len(full_names):
        raise RuntimeError("full trajectory joint_names do not match position columns")
    active = planner.trajopt_solver.get_active_js(trajectory)
    active_names = [str(name) for name in (active.joint_names or ())]
    if active_names != list(planner.joint_names):
        raise RuntimeError("cuRobo active trajectory joint order changed")
    return active, full_names


def _save_phase(output: Path, phase: str, planner: Any, trajectory: Any) -> dict[str, Any]:
    active, full_names = _get_active_trajectory(planner, trajectory)
    position = _trajectory_field(active, "position", required=True)
    velocity = _trajectory_field(active, "velocity", required=True)
    acceleration = _trajectory_field(active, "acceleration", required=True)
    jerk = _trajectory_field(active, "jerk", required=False)
    assert position is not None and velocity is not None and acceleration is not None
    if position.shape[0] < 2:
        raise RuntimeError(f"{phase} trajectory has fewer than two waypoints")
    arrays = (position, velocity, acceleration) + ((jerk,) if jerk is not None else ())
    if not all(np.isfinite(array).all() for array in arrays):
        raise RuntimeError(f"{phase} trajectory contains non-finite values")
    dt = getattr(active, "dt", None)
    if dt is None:
        raise RuntimeError(f"{phase} trajectory has no timing")
    np.save(output / f"{phase}_trajectory_position.npy", position)
    np.save(output / f"{phase}_trajectory_velocity.npy", velocity)
    np.save(output / f"{phase}_trajectory_acceleration.npy", acceleration)
    if jerk is not None:
        np.save(output / f"{phase}_trajectory_jerk.npy", jerk)
    np.save(output / f"{phase}_trajectory_dt_s.npy", _cpu_numpy(dt))
    return {
        "waypoints": int(position.shape[0]),
        "active_joint_names": list(planner.joint_names),
        "full_joint_names": full_names,
        "start": position[0],
        "end": position[-1],
    }


def _load_reviewed_pregrasp(directory: Path) -> tuple[dict[str, Any], np.ndarray]:
    report_path = directory / "pregrasp_plan_check.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checks = report.get("automatic_checks", {})
    safety = report.get("safety", {})
    parameters = report.get("parameters", {})
    if report.get("status") != "success" or not checks or not all(checks.values()):
        raise ValueError("pre-grasp plan did not pass its automatic gate")
    if parameters.get("scene_backend") != "observed_pointcloud_mesh":
        raise ValueError("grasp planning currently accepts only the reviewed observed mesh")
    if safety.get("simulation_only") is not True:
        raise ValueError("grasp planning is restricted to the simulation-only route")
    if safety.get("final_approach_planned") or safety.get("trajectory_executed"):
        raise ValueError("unexpected scope in source pre-grasp artifact")
    transforms = np.load(
        directory / "grasp_transforms_robot_base.npy", allow_pickle=False
    ).astype(np.float32, copy=False)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4) or len(transforms) == 0:
        raise ValueError("source grasp transforms must have shape (N,4,4)")
    return report, transforms


def main() -> int:
    args = parse_args()
    if not np.isfinite(args.lift_offset) or args.lift_offset <= 0.0:
        raise ValueError("--lift-offset must be positive and finite")
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from panda_handover.curobo_bridge import select_named_joint_positions
    from panda_handover.curobo_planning import rotation_matrix_to_quaternion_wxyz
    from panda_handover.geometry import transform_points

    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "apply_curobo_voxel_round_fix.py"),
            "--check-only",
        ],
        check=True,
    )
    pregrasp_report, grasp_transforms = _load_reviewed_pregrasp(args.pregrasp_plan)
    mesh_path = args.pregrasp_plan / "observed_scene_mesh.obj"
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)

    segmentation_report_path = args.segmentation / "segmentation_check.json"
    segmentation_report = json.loads(segmentation_report_path.read_text(encoding="utf-8"))
    if (
        segmentation_report.get("automatic_checks_passed") is not True
        or int(segmentation_report.get("valid_3d_pixels", 0)) < 100
    ):
        raise ValueError("SAM3 target point cloud did not pass its capture gate")
    target_world = np.load(
        args.segmentation / "points_world.npy", allow_pickle=False
    ).astype(np.float32, copy=False)
    if target_world.ndim != 2 or target_world.shape[1] != 3:
        raise ValueError("SAM3 points_world.npy must have shape (N,3)")
    if len(target_world) < 100 or not np.isfinite(target_world).all():
        raise ValueError("SAM3 target point cloud is too small or non-finite")
    T_world_robot_base = np.load(
        args.capture / "T_world_robot_base.npy", allow_pickle=False
    )
    target_robot_base = transform_points(
        np.linalg.inv(T_world_robot_base), target_world
    ).astype(np.float32, copy=False)

    robot_report = json.loads(
        (args.capture / "robot_state.json").read_text(encoding="utf-8")
    )
    captured_names = tuple(str(name) for name in robot_report["joint_names"])
    captured_positions = np.load(
        args.capture / "panda_joint_positions.npy", allow_pickle=False
    ).astype(np.float32, copy=False)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("cuRobo grasp planning requires CUDA")
    from curobo._src.geom.collision.buffer_collision import CollisionBuffer
    from curobo._src.geom.types import Mesh, SceneCfg
    from curobo._src.motion.motion_planner import MotionPlanner
    from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
    from curobo._src.state.state_joint import JointState
    from curobo._src.state.state_joint_trajectory_ops import trim_joint_state_trajectory
    from curobo._src.types.device_cfg import DeviceCfg
    from curobo._src.types.pose import Pose
    from curobo._src.types.tool_pose import GoalToolPose

    device_cfg = DeviceCfg(device=torch.device(args.device), dtype=torch.float32)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    scene_mesh = Mesh(
        name="observed_scene_without_robot_or_target",
        file_path=str(mesh_path.resolve()),
    )
    planner_cfg = MotionPlannerCfg.create(
        robot=args.robot,
        scene_model=SceneCfg(mesh=[scene_mesh]),
        device_cfg=device_cfg,
        num_ik_seeds=16,
        num_trajopt_seeds=2,
        optimizer_collision_activation_distance=0.01,
        use_cuda_graph=False,
        random_seed=123,
        max_goalset=len(grasp_transforms),
    )
    planner = MotionPlanner(planner_cfg)
    if planner.tool_frames != ["panda_hand"]:
        raise RuntimeError(f"reviewed Franka tool frame changed: {planner.tool_frames}")
    planner.warmup(enable_graph=True, num_warmup_iterations=2)

    start_positions = select_named_joint_positions(
        captured_names, captured_positions, planner.joint_names
    ).astype(np.float32, copy=False)
    current_state = JointState.from_position(
        torch.from_numpy(start_positions).to(device_cfg.device).unsqueeze(0),
        joint_names=planner.joint_names,
    )
    quaternions = rotation_matrix_to_quaternion_wxyz(
        grasp_transforms[:, :3, :3]
    )
    grasp_goals = GoalToolPose(
        tool_frames=planner.tool_frames,
        position=torch.from_numpy(grasp_transforms[:, :3, 3]).to(device_cfg.device)[
            None, None, None, :, :
        ],
        quaternion=torch.from_numpy(quaternions).to(device_cfg.device)[
            None, None, None, :, :
        ],
    )

    # Exact workaround reported in cuRobo Issue #663: one official goalset
    # plan_pose call before plan_grasp.  Do not patch or fork cuRobo internals.
    planner.reset_seed()
    preflight = planner.plan_pose(
        current_state=current_state,
        goal_tool_poses=grasp_goals,
        max_attempts=args.max_attempts,
    )
    preflight_success = bool(
        preflight is not None
        and preflight.success is not None
        and preflight.success.any().item()
    )
    if not preflight_success:
        raise RuntimeError("cuRobo Issue #663 preflight goalset plan did not succeed")

    started = time.monotonic()
    result = planner.plan_grasp(
        grasp_poses=grasp_goals,
        current_state=current_state,
        grasp_approach_axis="z",
        grasp_approach_offset=-float(
            pregrasp_report["parameters"]["approach_offset_m"]
        ),
        grasp_approach_in_tool_frame=True,
        grasp_lift_axis="z",
        grasp_lift_offset=args.lift_offset,
        grasp_lift_in_tool_frame=False,
        plan_approach_to_grasp=True,
        plan_grasp_to_lift=True,
        # Target geometry was removed from the observed scene.  Keep Panda hand
        # and finger collisions active against the table and red obstacle.
        disable_collision_links=[],
    )
    elapsed_s = time.monotonic() - started
    success = bool(
        result is not None
        and result.success is not None
        and result.success.any().item()
    )
    if not success:
        status_text = str(getattr(result, "status", "plan_grasp returned no result"))
        raise RuntimeError(f"cuRobo plan_grasp failed: {status_text}")

    raw_phase_sources = {
        "approach": (
            result.approach_interpolated_trajectory,
            result.approach_interpolated_last_tstep,
        ),
        "grasp": (
            result.grasp_interpolated_trajectory,
            result.grasp_interpolated_last_tstep,
        ),
        "lift": (
            result.lift_interpolated_trajectory,
            result.lift_interpolated_last_tstep,
        ),
    }
    phase_sources = {}
    raw_phase_waypoints = {}
    for name, (trajectory, last_tstep) in raw_phase_sources.items():
        if trajectory is None or last_tstep is None:
            raise RuntimeError(f"successful plan_grasp omitted {name} trajectory metadata")
        last_values = _cpu_numpy(last_tstep).reshape(-1)
        if last_values.size != 1:
            raise RuntimeError(f"{name} has ambiguous interpolated_last_tstep")
        raw_phase_waypoints[name] = int(np.asarray(_cpu_numpy(trajectory.position)).shape[-2])
        # Use cuRobo's own TrajOptSolverResult.get_interpolated_plan operation
        # rather than the heuristic workaround proposed in Issue #692.
        phase_sources[name] = trim_joint_state_trajectory(
            trajectory, 0, last_tstep.reshape(-1)[0]
        )
    phase_reports = {
        name: _save_phase(output, name, planner, trajectory)
        for name, trajectory in phase_sources.items()
    }
    continuity_tolerance = 2e-3
    continuity_checks = {
        "approach_starts_at_capture": bool(
            np.allclose(
                phase_reports["approach"]["start"],
                start_positions,
                atol=continuity_tolerance,
                rtol=0.0,
            )
        ),
        "grasp_starts_at_approach_end": bool(
            np.allclose(
                phase_reports["grasp"]["start"],
                phase_reports["approach"]["end"],
                atol=continuity_tolerance,
                rtol=0.0,
            )
        ),
        "lift_starts_at_grasp_end": bool(
            np.allclose(
                phase_reports["lift"]["start"],
                phase_reports["grasp"]["end"],
                atol=continuity_tolerance,
                rtol=0.0,
            )
        ),
    }
    if not all(continuity_checks.values()):
        raise RuntimeError(f"plan_grasp phase continuity failed: {continuity_checks}")

    selected_rank = int(_cpu_numpy(result.goalset_index).reshape(-1)[0])
    if not 0 <= selected_rank < len(grasp_transforms):
        raise RuntimeError("cuRobo returned an invalid grasp goalset index")

    # Prepare the target attachment only after the official lift.  During the
    # first tabletop lift, the object starts in intentional contact with its
    # support surface; cuRobo has no per-pair allowed-collision matrix here.
    target_mesh = Mesh.from_pointcloud(
        target_robot_base,
        pitch=float(pregrasp_report["parameters"]["observed_mesh"]["pitch_m"]),
        name="sam3_target_for_attachment",
    )
    target_mesh.save_as_mesh(str(output / "sam3_target_attachment_mesh.obj"))
    lift_end = JointState.from_position(
        torch.from_numpy(phase_reports["lift"]["end"])
        .to(device_cfg.device)
        .unsqueeze(0),
        joint_names=planner.joint_names,
    )
    identity_pose = Pose.from_list(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device_cfg=device_cfg
    )
    planner.attachment_manager.attach(
        joint_states=lift_end,
        obstacles=[target_mesh],
        link_name="attached_object",
        num_spheres=4,
        world_objects_pose_offset=identity_pose,
    )
    attached_indices = (
        planner.attachment_manager.kinematics_params.get_sphere_index_from_link_name(
            "attached_object"
        )
    )
    attached_local_spheres = _cpu_numpy(
        planner.attachment_manager.kinematics_params.link_spheres[
            0, attached_indices, :
        ]
    ).astype(np.float32, copy=False)
    np.save(output / "attached_object_spheres_panda_hand.npy", attached_local_spheres)

    lifted_kinematics = planner.compute_kinematics(lift_end)
    if lifted_kinematics.robot_spheres is None:
        raise RuntimeError("cuRobo returned no lifted robot collision spheres")
    collision_buffer = CollisionBuffer.from_shape(
        lifted_kinematics.robot_spheres.shape, device_cfg
    )
    collision_buffer.zero_()
    lifted_cost = planner.scene_collision_checker.get_sphere_collision(
        lifted_kinematics,
        collision_buffer,
        torch.tensor([1.0], device=device_cfg.device, dtype=torch.float32),
        torch.tensor([0.0], device=device_cfg.device, dtype=torch.float32),
    )
    torch.cuda.synchronize(device_cfg.device)
    lifted_cost_np = _cpu_numpy(lifted_cost).astype(np.float32, copy=False)
    np.save(output / "lift_end_attached_penetration_cost.npy", lifted_cost_np)

    source_indices = np.load(
        args.pregrasp_plan / "source_candidate_indices.npy", allow_pickle=False
    )
    candidate_scores = np.load(
        args.pregrasp_plan / "candidate_scores.npy", allow_pickle=False
    )
    report = {
        "status": "success",
        "reference": {
            "curobo_commit": CUROBO_COMMIT,
            "planner": "MotionPlanner.plan_grasp",
            "planner_source": (
                "curobo/_src/motion/motion_planner.py::MotionPlanner.plan_grasp"
            ),
            "issue_663_preflight": "https://github.com/NVlabs/curobo/issues/663",
            "issue_692_padding": "https://github.com/NVlabs/curobo/issues/692",
            "padding_trim": (
                "official trim_joint_state_trajectory using interpolated_last_tstep"
            ),
            "attachment": "AttachmentManager.attach with Mesh.from_pointcloud",
            "isaac_execution_precedent": "Isaac Sim 5.1 Franka Pick and Place",
        },
        "inputs": {
            "capture": str(args.capture),
            "segmentation": str(args.segmentation),
            "pregrasp_plan": str(args.pregrasp_plan),
            "target_point_count": int(len(target_robot_base)),
        },
        "parameters": {
            "robot": args.robot,
            "device": args.device,
            "approach_axis": "panda_hand +Z with negative offset",
            "approach_offset_m": float(
                pregrasp_report["parameters"]["approach_offset_m"]
            ),
            "lift_axis": "robot-base/world +Z",
            "lift_offset_m": args.lift_offset,
            "disable_collision_links": [],
            "target_absent_from_observed_scene": True,
            "attachment_sphere_count": 4,
        },
        "result": {
            "planner_reported_success": True,
            "planner_status": str(result.status),
            "selected_goalset_rank": selected_rank,
            "selected_source_candidate_index": int(source_indices[selected_rank]),
            "selected_graspgenx_score": float(candidate_scores[selected_rank]),
            "wall_time_s": elapsed_s,
            "trajectory_active_joint_names": list(planner.joint_names),
            "phases": {
                name: {
                    "waypoints": values["waypoints"],
                    "raw_padded_waypoints": raw_phase_waypoints[name],
                    "full_joint_names": values["full_joint_names"],
                }
                for name, values in phase_reports.items()
            },
        },
        "automatic_checks": {
            "issue_663_preflight_succeeded": preflight_success,
            "planner_reported_success": True,
            **continuity_checks,
            "lift_end_attached_spheres_are_finite": bool(
                np.isfinite(attached_local_spheres).all()
            ),
            "lift_end_attached_object_not_penetrating_observed_scene": bool(
                not np.any(lifted_cost_np > 0.0)
            ),
        },
        "safety": {
            "simulation_only": True,
            "unknown_space_assumed_free": True,
            "robot_world_collision_enabled_during_all_planned_phases": True,
            "robot_self_collision_enabled": True,
            "target_removed_from_world_collision_scene": True,
            "final_approach_planned": True,
            "gripper_close_is_physics_execution_not_curobo_plan": True,
            "lift_planned": True,
            "held_object_collision_checked_during_first_lift": False,
            "reason_first_lift_attachment_deferred": (
                "target begins in intentional contact with the support surface and "
                "this cuRobo API has no per-pair allowed-collision matrix"
            ),
            "attachment_prepared_and_checked_at_lift_end": True,
            "trajectory_executed": False,
            "manual_review_required": True,
            "safe_for_real_robot_execution": False,
        },
        "next_gate": (
            "Replay all three phases in Isaac Sim with a DynamicCuboid target, close "
            "the physical Franka gripper at the grasp boundary, and measure target lift."
        ),
    }
    if not all(report["automatic_checks"].values()):
        report["status"] = "automatic_check_failed"
    report_path = output / "grasp_lift_plan_check.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    planner.destroy()
    print(json.dumps(report, indent=2))
    print(f"saved: {report_path}")
    return 0 if report["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
