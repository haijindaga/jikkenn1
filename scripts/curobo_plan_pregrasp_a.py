#!/usr/bin/env python3
"""Plan only the collision-enabled motion to a reviewed pre-grasp pose.

The scene can use either cuRobo V2's dense ESDF ``VoxelGrid`` or its official
``Mesh.from_pointcloud`` representation of observed occupied surfaces.  It
deliberately does not enter the target region, close the gripper, attach an
object, lift, or execute anything in Isaac Sim.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-backend",
        choices=("backend_a_esdf", "observed_pointcloud_mesh"),
        default="backend_a_esdf",
    )
    parser.add_argument("--esdf", type=Path)
    parser.add_argument(
        "--prepared-map",
        type=Path,
        help="curobo_map_capture.py output used by observed_pointcloud_mesh",
    )
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--voxel-fix-report",
        type=Path,
        default=Path("outputs/curobo_voxel_fix_check.json"),
    )
    parser.add_argument("--robot", default="franka.yml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--approach-offset", type=float, default=0.15)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--unknown-policy",
        choices=("blocked", "free"),
        default=None,
        help="must exactly match the Backend A map; free is simulation-only",
    )
    return parser.parse_args()


def _verify_imported_curobo_source(project_root: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "apply_curobo_voxel_round_fix.py"),
            "--check-only",
        ],
        check=True,
    )


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
    # cuRobo V2 TrajOptSolverResult documents trajectory tensors as
    # (batch, return_seeds, horizon, dof).  plan_pose currently returns one
    # batch and one selected seed, but keep this fail-closed if either becomes
    # ambiguous instead of silently flattening multiple trajectories.
    if array.ndim > 2:
        if any(size != 1 for size in array.shape[:-2]):
            raise RuntimeError(
                f"trajectory {name} has multiple batch/seed entries: {array.shape}"
            )
        array = array.reshape(array.shape[-2:])
    if array.ndim != 2:
        raise RuntimeError(f"trajectory {name} must be HxD, got {array.shape}")
    return array.astype(np.float32, copy=False)


def _json_tensor(value: Any) -> Any:
    if value is None:
        return None
    array = np.asarray(_cpu_numpy(value))
    if array.size == 1:
        return array.reshape(-1)[0].item()
    return array.tolist()


def _get_active_trajectory(planner: Any, trajectory: Any) -> tuple[Any, list[str]]:
    """Use cuRobo's name-aware full-to-active JointState conversion."""
    raw_position = np.asarray(_cpu_numpy(getattr(trajectory, "position", None)))
    raw_joint_names = getattr(trajectory, "joint_names", None)
    if raw_position.ndim < 2:
        raise RuntimeError(f"full trajectory position has invalid shape {raw_position.shape}")
    if raw_joint_names is None:
        raise RuntimeError("full trajectory has no joint_names")
    full_joint_names = [str(name) for name in raw_joint_names]
    if len(full_joint_names) != raw_position.shape[-1]:
        raise RuntimeError(
            "full trajectory joint_names do not match its position columns: "
            f"{len(full_joint_names)} vs {raw_position.shape[-1]}"
        )
    if len(set(full_joint_names)) != len(full_joint_names):
        raise RuntimeError("full trajectory joint_names contain duplicates")

    active = planner.trajopt_solver.get_active_js(trajectory)
    active_joint_names = [str(name) for name in (active.joint_names or ())]
    if active_joint_names != list(planner.joint_names):
        raise RuntimeError(
            "cuRobo active trajectory joint order does not match the planner"
        )
    active_position = np.asarray(_cpu_numpy(active.position))
    if active_position.shape[-1] != len(active_joint_names):
        raise RuntimeError("active trajectory columns do not match its joint_names")
    return active, full_joint_names


def _resolved_report_view_matches(report: dict[str, Any], capture: Path) -> bool:
    expected = capture.resolve()
    for view in report.get("views", []):
        if isinstance(view, dict) and isinstance(view.get("capture"), str):
            if Path(view["capture"]).resolve() == expected:
                return True
    return False


def main() -> int:
    args = parse_args()
    if args.max_candidates <= 0 or args.max_attempts <= 0:
        raise ValueError("--max-candidates and --max-attempts must be positive")
    if args.scene_backend == "backend_a_esdf":
        if args.esdf is None or args.prepared_map is not None:
            raise ValueError("backend_a_esdf requires --esdf and forbids --prepared-map")
    elif args.prepared_map is None or args.esdf is not None:
        raise ValueError(
            "observed_pointcloud_mesh requires --prepared-map and forbids --esdf"
        )
    elif args.unknown_policy is not None:
        raise ValueError(
            "observed_pointcloud_mesh fixes unobserved space as free; "
            "do not pass --unknown-policy"
        )
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from panda_handover.curobo_bridge import select_named_joint_positions
    from panda_handover.curobo_planning import (
        classify_pregrasp_failure,
        load_backend_a_esdf,
        load_singleview_observed_pointcloud,
        prepare_pregrasp_goalset,
        rotation_matrix_to_quaternion_wxyz,
        summarize_ik_result_arrays,
        validate_voxel_fix_report,
    )

    # Keep both scene paths on the exact reviewed cuRobo checkout.  The voxel
    # regression is additionally required before real ESDF values reach it.
    _verify_imported_curobo_source(project_root)
    voxel_fix_report = None
    esdf = None
    observed_scene = None
    if args.scene_backend == "backend_a_esdf":
        voxel_fix_report = validate_voxel_fix_report(args.voxel_fix_report)
        assert args.esdf is not None
        esdf = load_backend_a_esdf(
            args.esdf, expected_unknown_policy=args.unknown_policy or "blocked"
        )
        if not _resolved_report_view_matches(esdf.report, args.capture):
            raise ValueError("--capture is not one of the views integrated into Backend A")
    else:
        assert args.prepared_map is not None
        observed_scene = load_singleview_observed_pointcloud(
            args.prepared_map, args.capture
        )

    collision_report_path = args.candidates / "collision_filter_check.json"
    if not collision_report_path.is_file():
        raise FileNotFoundError(
            "pre-grasp planning requires GraspGenX collision-filtered candidates"
        )
    collision_report = json.loads(collision_report_path.read_text(encoding="utf-8"))
    collision_safety = collision_report.get("safety", {})
    if collision_report.get("status") != "success" or (
        collision_safety.get("static_gripper_pose_vs_observed_scene_checked") is not True
    ):
        raise ValueError("GraspGenX static collision filter did not pass")
    if collision_safety.get("safe_to_execute") is not False:
        raise ValueError("unexpected executable candidate artifact")

    panda_hand_world = np.load(
        args.candidates / "panda_hand_world.npy", allow_pickle=False
    )
    scores = np.load(args.candidates / "scores.npy", allow_pickle=False)
    kept_path = args.candidates / "kept_candidate_indices.npy"
    kept_indices = np.load(kept_path, allow_pickle=False) if kept_path.is_file() else None
    T_world_robot_base = np.load(
        args.capture / "T_world_robot_base.npy", allow_pickle=False
    )
    goalset = prepare_pregrasp_goalset(
        panda_hand_world,
        scores,
        T_world_robot_base,
        approach_offset_m=args.approach_offset,
        max_candidates=args.max_candidates,
        candidate_indices=kept_indices,
    )

    robot_report = json.loads(
        (args.capture / "robot_state.json").read_text(encoding="utf-8")
    )
    captured_joint_names = tuple(robot_report["joint_names"])
    captured_joint_positions = np.load(
        args.capture / "panda_joint_positions.npy", allow_pickle=False
    ).astype(np.float32, copy=False)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("cuRobo pre-grasp planning requires CUDA")
    from curobo._src.geom.types import Mesh, SceneCfg, VoxelGrid
    from curobo._src.geom.collision.buffer_collision import CollisionBuffer
    from curobo._src.motion.motion_planner import MotionPlanner
    from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
    from curobo._src.solver.solver_ik import IKSolver
    from curobo._src.state.state_joint import JointState
    from curobo._src.types.device_cfg import DeviceCfg
    from curobo._src.types.tool_pose import GoalToolPose

    device_cfg = DeviceCfg(device=torch.device(args.device), dtype=torch.float32)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    scene_mesh = None
    if args.scene_backend == "backend_a_esdf":
        assert esdf is not None
        features_gpu = torch.from_numpy(esdf.features_m).to(
            device=device_cfg.device, dtype=torch.float16
        ).contiguous()
        # Recompute dimensions from integer shape, rather than trusting serialized
        # float products. The reviewed cuRobo patch still guards the GPU conversion.
        grid_dims = [float(count) * esdf.voxel_size_m for count in esdf.shape_xyz]
        voxel_grid = VoxelGrid(
            name="backend_a_conservative_esdf",
            pose=[*esdf.center_robot_base_m, 1.0, 0.0, 0.0, 0.0],
            dims=grid_dims,
            voxel_size=esdf.voxel_size_m,
            feature_tensor=features_gpu,
            feature_dtype=torch.float16,
        )
        scene = SceneCfg(voxel=[voxel_grid])
    else:
        assert observed_scene is not None
        scene_mesh = Mesh.from_pointcloud(
            observed_scene.points_robot_base_m,
            pitch=observed_scene.voxel_size_m,
            name="observed_scene_without_robot_or_target",
        )
        vertices = np.asarray(scene_mesh.vertices)
        faces = np.asarray(scene_mesh.faces)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 4:
            raise RuntimeError("cuRobo pointcloud mesh has invalid vertices")
        if faces.size < 3 or faces.size % 3 != 0:
            raise RuntimeError("cuRobo pointcloud mesh has invalid triangle indices")
        if np.any(faces < 0) or np.any(faces >= vertices.shape[0]):
            raise RuntimeError("cuRobo pointcloud mesh contains out-of-range faces")
        scene_mesh.save_as_mesh(str(output / "observed_scene_mesh.obj"))
        scene = SceneCfg(mesh=[scene_mesh])
    planner_cfg = MotionPlannerCfg.create(
        robot=args.robot,
        scene_model=scene,
        device_cfg=device_cfg,
        num_ik_seeds=16,
        num_trajopt_seeds=2,
        optimizer_collision_activation_distance=0.01,
        use_cuda_graph=False,
        random_seed=123,
        max_goalset=len(goalset.scores),
    )
    planner = MotionPlanner(planner_cfg)
    if planner.tool_frames != ["panda_hand"]:
        raise RuntimeError(f"reviewed Franka tool frame changed: {planner.tool_frames}")
    planner.warmup(enable_graph=True, num_warmup_iterations=2)

    start_positions = select_named_joint_positions(
        captured_joint_names, captured_joint_positions, planner.joint_names
    ).astype(np.float32, copy=False)
    current_state = JointState.from_position(
        torch.from_numpy(start_positions).to(device_cfg.device).unsqueeze(0),
        joint_names=planner.joint_names,
    )
    pregrasp = goalset.pregrasp_robot_base
    quaternions = rotation_matrix_to_quaternion_wxyz(pregrasp[:, :3, :3])
    positions_gpu = torch.from_numpy(pregrasp[:, :3, 3]).to(device_cfg.device)
    quaternions_gpu = torch.from_numpy(quaternions).to(device_cfg.device)
    goals = GoalToolPose(
        tool_frames=planner.tool_frames,
        position=positions_gpu[None, None, None, :, :],
        quaternion=quaternions_gpu[None, None, None, :, :],
    )

    # Query the same collision spheres and scene used by cuRobo's optimizers.
    # Zero activation measures actual overlap; 10 mm matches the optimizer cost.
    start_kinematics = planner.compute_kinematics(current_state)
    start_spheres = start_kinematics.robot_spheres
    if start_spheres is None:
        raise RuntimeError("cuRobo Franka model returned no collision spheres")
    collision_buffer = CollisionBuffer.from_shape(start_spheres.shape, device_cfg)
    collision_weight = torch.tensor([1.0], device=device_cfg.device, dtype=torch.float32)

    def query_start_collision(activation_distance_m: float) -> np.ndarray:
        collision_buffer.zero_()
        values = planner.scene_collision_checker.get_sphere_collision(
            start_kinematics,
            collision_buffer,
            collision_weight,
            torch.tensor(
                [activation_distance_m], device=device_cfg.device, dtype=torch.float32
            ),
        )
        torch.cuda.synchronize(device_cfg.device)
        return _cpu_numpy(values).astype(np.float32, copy=False)

    start_penetration_cost = query_start_collision(0.0)
    start_optimizer_cost = query_start_collision(0.01)

    # Reproduce the exact first stage of MotionPlanner._plan_pose_goalset.  Reset
    # before planning so this diagnostic does not consume or change planner seeds.
    planner.reset_seed()
    world_ik_result = planner.ik_solver.solve_pose(
        goals,
        return_seeds=planner.trajopt_solver.config.num_seeds,
        current_state=current_state,
    )
    world_ik_summary = summarize_ik_result_arrays(
        _cpu_numpy(world_ik_result.success),
        feasible=(
            _cpu_numpy(world_ik_result.feasible)
            if world_ik_result.feasible is not None
            else None
        ),
        position_error=(
            _cpu_numpy(world_ik_result.position_error)
            if world_ik_result.position_error is not None
            else None
        ),
        rotation_error=(
            _cpu_numpy(world_ik_result.rotation_error)
            if world_ik_result.rotation_error is not None
            else None
        ),
        goalset_index=(
            _cpu_numpy(world_ik_result.goalset_index)
            if world_ik_result.goalset_index is not None
            else None
        ),
    )
    planner.reset_seed()

    # If collision-aware IK has no solution, run the official IK solver again
    # without a world scene. Joint limits and self-collision remain enabled. This
    # is a diagnosis-only control and can never produce an executable artifact.
    free_world_ik_summary = None
    free_world_ik_solver = None
    if world_ik_summary["success_count"] == 0:
        free_world_cfg = MotionPlannerCfg.create(
            robot=args.robot,
            scene_model=None,
            device_cfg=device_cfg,
            num_ik_seeds=16,
            num_trajopt_seeds=2,
            optimizer_collision_activation_distance=0.01,
            use_cuda_graph=False,
            random_seed=123,
            max_goalset=len(goalset.scores),
        )
        free_world_ik_solver = IKSolver(free_world_cfg.ik_solver_config, None)
        free_world_result = free_world_ik_solver.solve_pose(
            goals,
            return_seeds=2,
            current_state=current_state,
        )
        free_world_ik_summary = summarize_ik_result_arrays(
            _cpu_numpy(free_world_result.success),
            feasible=(
                _cpu_numpy(free_world_result.feasible)
                if free_world_result.feasible is not None
                else None
            ),
            position_error=(
                _cpu_numpy(free_world_result.position_error)
                if free_world_result.position_error is not None
                else None
            ),
            rotation_error=(
                _cpu_numpy(free_world_result.rotation_error)
                if free_world_result.rotation_error is not None
                else None
            ),
            goalset_index=(
                _cpu_numpy(free_world_result.goalset_index)
                if free_world_result.goalset_index is not None
                else None
            ),
        )

    np.save(output / "grasp_transforms_robot_base.npy", goalset.grasp_robot_base)
    np.save(output / "pregrasp_transforms_robot_base.npy", pregrasp)
    np.save(output / "candidate_scores.npy", goalset.scores)
    np.save(output / "source_candidate_indices.npy", goalset.candidate_indices)
    np.save(output / "start_joint_positions.npy", start_positions)
    np.save(output / "start_robot_spheres.npy", _cpu_numpy(start_spheres))
    np.save(output / "start_penetration_cost.npy", start_penetration_cost)
    np.save(output / "start_optimizer_collision_cost.npy", start_optimizer_cost)

    started = time.monotonic()
    result = planner.plan_pose(
        current_state=current_state,
        goal_tool_poses=goals,
        max_attempts=args.max_attempts,
    )
    elapsed_s = time.monotonic() - started
    planner_success = bool(
        result is not None
        and result.success is not None
        and result.success.any().item()
    )
    start_penetrating_sphere_count = int(np.count_nonzero(start_penetration_cost > 0.0))
    start_activation_sphere_count = int(np.count_nonzero(start_optimizer_cost > 0.0))
    selected_goalset = None
    trajectory_checks: dict[str, bool] = {}
    trajectory_waypoints = 0
    full_trajectory_joint_names = None
    if planner_success:
        if result.goalset_index is None and len(goalset.scores) == 1:
            selected_goalset = 0
        elif result.goalset_index is None:
            raise RuntimeError("successful goalset plan has no goalset_index")
        else:
            selected_goalset = int(_cpu_numpy(result.goalset_index).reshape(-1)[0])
        if not 0 <= selected_goalset < len(goalset.scores):
            raise RuntimeError("cuRobo returned an invalid goalset index")
        full_trajectory = result.get_interpolated_plan()
        trajectory, full_trajectory_joint_names = _get_active_trajectory(
            planner, full_trajectory
        )
        position = _trajectory_field(trajectory, "position", required=True)
        velocity = _trajectory_field(trajectory, "velocity", required=True)
        acceleration = _trajectory_field(trajectory, "acceleration", required=True)
        jerk = _trajectory_field(trajectory, "jerk", required=False)
        assert position is not None and velocity is not None and acceleration is not None
        trajectory_waypoints = int(position.shape[0])
        trajectory_checks = {
            "planner_reported_success": True,
            "selected_goalset_index_valid": True,
            "trajectory_has_at_least_two_waypoints": trajectory_waypoints >= 2,
            "trajectory_position_is_finite": bool(np.isfinite(position).all()),
            "trajectory_velocity_is_finite": bool(np.isfinite(velocity).all()),
            "trajectory_acceleration_is_finite": bool(np.isfinite(acceleration).all()),
            "trajectory_starts_at_capture": bool(
                np.allclose(position[0], start_positions, atol=2e-3, rtol=0.0)
            ),
        }
        if jerk is not None:
            trajectory_checks["trajectory_jerk_is_finite"] = bool(np.isfinite(jerk).all())
        if not all(trajectory_checks.values()):
            planner_success = False
        np.save(output / "trajectory_position.npy", position)
        np.save(output / "trajectory_velocity.npy", velocity)
        np.save(output / "trajectory_acceleration.npy", acceleration)
        if jerk is not None:
            np.save(output / "trajectory_jerk.npy", jerk)
        if getattr(trajectory, "dt", None) is not None:
            np.save(output / "trajectory_dt_s.npy", _cpu_numpy(trajectory.dt))

    failure_stage = classify_pregrasp_failure(
        planner_success=planner_success,
        world_ik_success_count=int(world_ik_summary["success_count"]),
        free_world_ik_success_count=(
            int(free_world_ik_summary["success_count"])
            if free_world_ik_summary is not None
            else None
        ),
        start_penetrating_sphere_count=start_penetrating_sphere_count,
        planner_returned_result=result is not None,
    )
    status = "success" if planner_success else "no_safe_pregrasp_plan"
    if args.scene_backend == "backend_a_esdf":
        assert esdf is not None and voxel_fix_report is not None
        scene_reference = {
            "implementation": "cuRobo V2 MotionPlanner.plan_pose with ESDF VoxelGrid",
            "official_test": "curobo/tests/_src/motion/test_motion_planner_esdf.py",
            "issue_699": "https://github.com/NVlabs/curobo/issues/699",
            "voxel_fix_report": str(args.voxel_fix_report),
            "voxel_fix_source_sha256": voxel_fix_report["reference"][
                "patched_source_sha256"
            ],
        }
        scene_inputs = {
            "esdf": str(args.esdf),
            "backend_a_input_fingerprint_sha256": esdf.report.get(
                "input_fingerprint_sha256"
            ),
        }
        unknown_policy = esdf.unknown_policy
        scene_diagnostics_key = "start_state_vs_conservative_esdf"
        mesh_report = None
    else:
        assert observed_scene is not None and scene_mesh is not None
        scene_reference = {
            "implementation": (
                "cuRobo V2 MotionPlanner.plan_pose with "
                "Mesh.from_pointcloud observed surfaces"
            ),
            "official_api_test": "curobo/tests/_src/geom/test_types.py::test_mesh_from_pointcloud",
            "official_scene_test": "curobo/tests/_src/motion/test_motion_planner.py::test_update_world_with_scene",
            "precedent": "NVlabs/VoLoAgent curobo_planner.py create_mesh_scene",
        }
        scene_inputs = {
            "prepared_map": str(args.prepared_map),
            "occupied_surface_points": int(
                observed_scene.points_robot_base_m.shape[0]
            ),
        }
        unknown_policy = "free_outside_observed_mesh"
        scene_diagnostics_key = "start_state_vs_observed_pointcloud_mesh"
        mesh_report = {
            "pitch_m": observed_scene.voxel_size_m,
            "vertices": int(np.asarray(scene_mesh.vertices).shape[0]),
            "triangles": int(np.asarray(scene_mesh.faces).size // 3),
            "saved_obj": str(output / "observed_scene_mesh.obj"),
        }
    report = {
        "status": status,
        "reference": {
            "curobo_commit": "057a96ffb1088531535f9915154f9d0dabd62428",
            **scene_reference,
        },
        "inputs": {
            "capture": str(args.capture),
            "candidates": str(args.candidates),
            **scene_inputs,
        },
        "frames": {
            "map": "panda_link0 robot base",
            "goal": "panda_hand",
            "candidate_input": "world",
        },
        "parameters": {
            "robot": args.robot,
            "device": args.device,
            "approach_axis": "negative panda_hand Z",
            "approach_offset_m": args.approach_offset,
            "candidate_count": len(goalset.scores),
            "num_ik_seeds": 16,
            "num_trajopt_seeds": 2,
            "max_attempts": args.max_attempts,
            "optimizer_collision_activation_distance_m": 0.01,
            "random_seed": 123,
            "use_cuda_graph": False,
            "scene_backend": args.scene_backend,
            "unknown_policy": unknown_policy,
            "planning_mode": (
                "conservative"
                if unknown_policy == "blocked"
                else "optimistic_sim"
            ),
            "observed_mesh": mesh_report,
        },
        "result": {
            "planner_reported_success": bool(
                result is not None
                and result.success is not None
                and result.success.any().item()
            ),
            "selected_goalset_rank": selected_goalset,
            "selected_source_candidate_index": (
                int(goalset.candidate_indices[selected_goalset])
                if selected_goalset is not None
                else None
            ),
            "selected_graspgenx_score": (
                float(goalset.scores[selected_goalset])
                if selected_goalset is not None
                else None
            ),
            "trajectory_waypoints": trajectory_waypoints,
            "trajectory_active_joint_names": list(planner.joint_names),
            "trajectory_full_joint_names": full_trajectory_joint_names,
            "wall_time_s": elapsed_s,
            "curobo_total_time_s": _json_tensor(
                getattr(result, "total_time", None) if result is not None else None
            ),
            "curobo_solve_time_s": _json_tensor(
                getattr(result, "solve_time", None) if result is not None else None
            ),
            "position_error_m": _json_tensor(
                getattr(result, "position_error", None) if result is not None else None
            ),
            "rotation_error_rad": _json_tensor(
                getattr(result, "rotation_error", None) if result is not None else None
            ),
            "failure_stage": failure_stage,
        },
        "diagnostics": {
            "collision_aware_ik": world_ik_summary,
            "ik_without_world_scene_control": free_world_ik_summary,
            scene_diagnostics_key: {
                "robot_collision_sphere_count": int(start_spheres.shape[-2]),
                "penetrating_sphere_count_activation_0m": start_penetrating_sphere_count,
                "active_sphere_count_activation_0_01m": start_activation_sphere_count,
                "maximum_penetration_cost_m": float(np.max(start_penetration_cost)),
                "maximum_optimizer_collision_cost_m": float(np.max(start_optimizer_cost)),
            },
        },
        "automatic_checks": trajectory_checks,
        "safety": {
            "input_map_declared_safe_to_plan": False,
            "pregrasp_only_scope_gate_passed": True,
            "unknown_space_blocked": unknown_policy == "blocked",
            "unknown_space_assumed_free": unknown_policy != "blocked",
            "target_region_blocked": unknown_policy == "blocked",
            "simulation_only": unknown_policy != "blocked",
            "robot_world_collision_enabled_for_entire_saved_trajectory": True,
            "robot_self_collision_enabled": True,
            "static_graspgenx_filter_required": True,
            "final_approach_planned": False,
            "gripper_close_planned": False,
            "object_attached": False,
            "lift_planned": False,
            "trajectory_executed": False,
            "safe_to_execute": False,
            "manual_review_required": True,
        },
        "next_gate": (
            "If planning failed, review diagnostics before changing the scene or solver. "
            "If it succeeded, inspect the saved pre-grasp trajectory in Isaac Sim. "
            "Final linear contact approach requires a separately reviewed non-target "
            "collision sweep."
        ),
    }
    report_path = output / "pregrasp_plan_check.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if free_world_ik_solver is not None:
        free_world_ik_solver.destroy()
    planner.destroy()
    print(json.dumps(report, indent=2))
    print(f"saved: {report_path}")
    return 0 if planner_success else 2


if __name__ == "__main__":
    raise SystemExit(main())
