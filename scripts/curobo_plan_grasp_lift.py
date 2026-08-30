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
SIMULATION_FINGER_SUPPORT_CONTACT_LIMIT_M = 0.001
SIMULATION_FINGER_CONTACT_LINKS = frozenset(
    {"panda_leftfinger", "panda_rightfinger"}
)


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
    parser.add_argument(
        "--allow-reviewed-support-contact-preflight",
        action="store_true",
        help=(
            "Simulation-only opt-in: retry an exact-grasp preflight with only "
            "finger/support contacts disabled after strict per-candidate review"
        ),
    )
    parser.add_argument(
        "--handover-goal-position-robot-base-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help=(
            "Optional fixed panda_hand transport goal in panda_link0 metres. "
            "Omitting this keeps the existing grasp/lift-only scope."
        ),
    )
    parser.add_argument(
        "--handover-goal-quaternion-wxyz",
        type=float,
        nargs=4,
        metavar=("W", "X", "Y", "Z"),
        help="Optional handover orientation; omitted preserves the grasp orientation",
    )
    return parser.parse_args()


def _cpu_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _restore_triangle_faces(vertices: Any, faces: Any) -> np.ndarray:
    """Restore cuRobo's flattened point-cloud faces for trimesh consumption."""
    vertex_array = np.asarray(vertices)
    face_array = np.asarray(faces)
    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3 or len(vertex_array) == 0:
        raise RuntimeError(
            f"attachment mesh vertices must be non-empty Nx3, got {vertex_array.shape}"
        )
    if not np.issubdtype(face_array.dtype, np.integer):
        raise RuntimeError(f"attachment mesh faces must be integer, got {face_array.dtype}")
    if face_array.ndim == 1:
        if face_array.size == 0 or face_array.size % 3 != 0:
            raise RuntimeError(
                "flattened attachment mesh faces must be non-empty and divisible by 3"
            )
        face_array = face_array.reshape(-1, 3)
    elif face_array.ndim != 2 or face_array.shape[1] != 3 or len(face_array) == 0:
        raise RuntimeError(
            f"attachment mesh faces must be non-empty Nx3, got {face_array.shape}"
        )
    if int(face_array.min()) < 0 or int(face_array.max()) >= len(vertex_array):
        raise RuntimeError("attachment mesh face index is outside the vertex array")
    return face_array.astype(np.int64, copy=False)


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
        "position": position,
        "start": position[0],
        "end": position[-1],
    }


def _success_any(result: Any) -> bool:
    success = getattr(result, "success", None)
    return bool(success is not None and success.any().item())


def _classify_preflight_failure(candidate_diagnostics: list[dict[str, Any]]) -> str:
    """Classify exact-grasp preflight failure without changing planner settings."""
    world_success_count = sum(
        int(item["collision_aware_ik"]["success_count"])
        for item in candidate_diagnostics
    )
    if world_success_count > 0:
        return "trajectory_optimization_failed_after_collision_aware_ik"

    free_world_summaries = [
        item["ik_without_world_scene_control"]
        for item in candidate_diagnostics
        if item.get("ik_without_world_scene_control") is not None
    ]
    if not free_world_summaries:
        return "collision_aware_ik_failed_without_control_measurement"
    if any(int(summary["success_count"]) > 0 for summary in free_world_summaries):
        return "world_collision_rejects_exact_grasp_ik"
    return "exact_grasp_ik_fails_even_without_world_collision"


def _review_exact_grasp_support_candidate(
    candidate: dict[str, Any],
    *,
    support_surface_z_m: float,
    support_height_tolerance_m: float,
    map_discretization_allowance_m: float,
    maximum_physical_penetration_m: float = SIMULATION_FINGER_SUPPORT_CONTACT_LIMIT_M,
) -> dict[str, Any]:
    """Narrowly review a free-world exact grasp against the measured support."""
    collision = candidate.get("free_world_solution_vs_observed_scene")
    penetration = collision.get("actual_penetration") if collision else None
    contacts = penetration.get("contacts", []) if penetration else []
    effective_cost_limit_m = (
        maximum_physical_penetration_m + map_discretization_allowance_m
    )
    checks = {
        "free_world_ik_succeeded": int(
            (candidate.get("ik_without_world_scene_control") or {}).get(
                "success_count", 0
            )
        )
        > 0,
        "actual_contact_exists_for_review": bool(contacts),
        "only_finger_links_contact_environment": bool(contacts)
        and all(
            str(contact["link_name"]) in SIMULATION_FINGER_CONTACT_LINKS
            for contact in contacts
        ),
        "nearest_points_match_support_height": bool(contacts)
        and all(
            abs(
                float(
                    contact["nearest_observed_source_point"][
                        "point_robot_base_m"
                    ][2]
                )
                - support_surface_z_m
            )
            <= support_height_tolerance_m
            for contact in contacts
        ),
        "collision_cost_within_voxel_aware_limit": bool(contacts)
        and max(float(contact["collision_cost_m"]) for contact in contacts)
        <= effective_cost_limit_m,
        "physical_sphere_support_penetration_within_1mm": bool(contacts)
        and all(
            support_surface_z_m
            - (
                float(contact["sphere_robot_base_xyz_radius_m"][2])
                - float(contact["sphere_robot_base_xyz_radius_m"][3])
            )
            <= maximum_physical_penetration_m
            for contact in contacts
        ),
    }
    return {
        "policy": (
            "simulation-only exact-grasp support review; only finger contacts "
            "at the validated tabletop are eligible"
        ),
        "support_surface_z_m": float(support_surface_z_m),
        "support_height_tolerance_m": float(support_height_tolerance_m),
        "maximum_physical_penetration_m": float(maximum_physical_penetration_m),
        "map_discretization_allowance_m": float(map_discretization_allowance_m),
        "effective_collision_cost_limit_m": float(effective_cost_limit_m),
        "checks": checks,
        "accepted": bool(all(checks.values())),
        "safe_for_real_robot_execution": False,
    }


def _load_validated_support_surface_z(
    capture: Path,
    T_world_robot_base: np.ndarray,
    *,
    expected_tabletop_world_z_m: float,
) -> float:
    """Validate the capture and express the reviewed tabletop plane in base z."""
    report_path = capture / "scene_layout.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "success":
        raise ValueError("capture scene layout did not pass validation")
    scene_kind = report.get("scene_source", {}).get("kind")
    if scene_kind not in {"authored_usd_scene", "generated_legacy_smoke_scene"}:
        raise ValueError("capture scene layout has an unsupported scene source")
    if not np.isfinite(expected_tabletop_world_z_m):
        raise ValueError("reviewed tabletop height must be finite")
    transform = np.asarray(T_world_robot_base, dtype=np.float64)
    if transform.shape != (4, 4) or not np.allclose(
        transform[:3, :3], np.eye(3), atol=1e-5, rtol=0.0
    ):
        raise ValueError("support-contact review requires an axis-aligned robot base")
    # The root table prim AABB is not a reliable top-plane measurement for every
    # referenced USD hierarchy. The project scene contract fixes the robot mount
    # and tabletop plane at world z=0 and the RGB-D map verifies it independently.
    return float(expected_tabletop_world_z_m - transform[2, 3])


def _goalset_index(result: Any) -> int | None:
    value = getattr(result, "goalset_index", None)
    if value is None:
        return None
    flat = _cpu_numpy(value).reshape(-1)
    return int(flat[0]) if flat.size else None


def _normalized_quaternion(value: Any) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("handover quaternion must contain four finite wxyz values")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        raise ValueError("handover quaternion norm is zero")
    quaternion /= norm
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion.astype(np.float32)


def _collision_sphere_link_names(
    kinematics_params: Any, sphere_count: int
) -> tuple[list[str], dict[str, list[int]]]:
    """Resolve cuRobo collision-sphere ownership through its official API."""
    link_index_map = getattr(kinematics_params, "link_name_to_idx_map", None)
    if not isinstance(link_index_map, dict) or not link_index_map:
        raise RuntimeError("cuRobo kinematics has no link_name_to_idx_map")
    sphere_links: list[str | None] = [None] * sphere_count
    link_sphere_indices: dict[str, list[int]] = {}
    for link_name in link_index_map:
        indices = _cpu_numpy(
            kinematics_params.get_sphere_index_from_link_name(link_name)
        ).astype(np.int64).reshape(-1)
        if indices.size == 0:
            continue
        if int(indices.min()) < 0 or int(indices.max()) >= sphere_count:
            raise RuntimeError(f"sphere index for {link_name} is outside the robot model")
        index_list = [int(index) for index in indices]
        link_sphere_indices[str(link_name)] = index_list
        for index in index_list:
            previous = sphere_links[index]
            if previous is not None and previous != str(link_name):
                raise RuntimeError(
                    f"collision sphere {index} belongs to both {previous} and {link_name}"
                )
            sphere_links[index] = str(link_name)
    missing = [index for index, name in enumerate(sphere_links) if name is None]
    if missing:
        raise RuntimeError(f"cuRobo collision spheres have no owning link: {missing}")
    return [str(name) for name in sphere_links], link_sphere_indices


def _nearest_surface_point(
    points: np.ndarray, center: np.ndarray, radius: float
) -> dict[str, Any]:
    """Report the nearest measured surface point without inventing a class label."""
    cloud = np.asarray(points, dtype=np.float32)
    query = np.asarray(center, dtype=np.float32)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) == 0:
        raise RuntimeError(f"diagnostic point cloud must be non-empty Nx3, got {cloud.shape}")
    if query.shape != (3,) or not np.isfinite(query).all():
        raise RuntimeError("diagnostic sphere center must be a finite xyz vector")
    if not np.isfinite(radius) or radius < 0.0:
        raise RuntimeError("diagnostic sphere radius must be finite and non-negative")
    distance_squared = np.einsum("ij,ij->i", cloud - query, cloud - query)
    nearest_index = int(np.argmin(distance_squared))
    center_distance = float(np.sqrt(distance_squared[nearest_index]))
    return {
        "point_index": nearest_index,
        "point_robot_base_m": cloud[nearest_index].astype(float).tolist(),
        "center_distance_m": center_distance,
        "sphere_surface_clearance_m": center_distance - float(radius),
    }


def _phase_contact_diagnostics(
    phase_name: str,
    costs: np.ndarray,
    spheres: np.ndarray,
    sphere_link_names: list[str],
    observed_points: np.ndarray,
    target_points: np.ndarray,
) -> dict[str, Any]:
    """Describe positive full-robot collision costs without relaxing the gate."""
    cost_values = np.asarray(costs, dtype=np.float32)
    sphere_values = np.asarray(spheres, dtype=np.float32)
    if sphere_values.shape[:-1] != cost_values.shape or sphere_values.shape[-1] != 4:
        raise RuntimeError(
            f"{phase_name} collision costs {cost_values.shape} and spheres "
            f"{sphere_values.shape} do not align"
        )
    if cost_values.shape[-1] != len(sphere_link_names):
        raise RuntimeError(f"{phase_name} sphere-link mapping length changed")
    contacts = []
    for raw_index in np.argwhere(cost_values > 0.0):
        index = tuple(int(value) for value in raw_index)
        sphere_index = index[-1]
        sphere = sphere_values[index]
        observed_nearest = _nearest_surface_point(
            observed_points, sphere[:3], float(sphere[3])
        )
        target_nearest = _nearest_surface_point(
            target_points, sphere[:3], float(sphere[3])
        )
        observed_point = np.asarray(
            observed_nearest["point_robot_base_m"], dtype=np.float32
        )
        observed_to_target = _nearest_surface_point(
            target_points, observed_point, 0.0
        )
        contacts.append(
            {
                "tensor_index": list(index),
                "waypoint_index": index[0],
                "sphere_index": sphere_index,
                "link_name": sphere_link_names[sphere_index],
                "collision_cost_m": float(cost_values[index]),
                "sphere_robot_base_xyz_radius_m": sphere.astype(float).tolist(),
                "nearest_observed_source_point": observed_nearest,
                "nearest_sam3_target_point": target_nearest,
                "nearest_observed_point_to_sam3_target": observed_to_target,
            }
        )
    return {
        "phase": phase_name,
        "cost_shape": list(cost_values.shape),
        "positive_count": len(contacts),
        "maximum_collision_cost_m": float(np.max(cost_values)),
        "contacts": contacts,
    }


def _review_transient_finger_support_contact(
    phase_reports: dict[str, dict[str, Any]],
    *,
    support_surface_z_m: float,
    support_height_tolerance_m: float,
    maximum_collision_cost_m: float = SIMULATION_FINGER_SUPPORT_CONTACT_LIMIT_M,
    map_discretization_allowance_m: float = 0.0,
) -> dict[str, Any]:
    """Apply the reviewed simulation-only support-contact acceptance policy.

    The policy does not alter cuRobo collision costs. It accepts only a
    contiguous finger-only contact suffix at the end of grasp followed by a
    contiguous prefix at the start of lift, with no approach contact and a
    collision-free lift endpoint.
    """
    required_phases = {"approach", "grasp", "lift"}
    if set(phase_reports) != required_phases:
        raise RuntimeError(
            f"contact policy requires {sorted(required_phases)}, got "
            f"{sorted(phase_reports)}"
        )
    if not np.isfinite(maximum_collision_cost_m) or maximum_collision_cost_m <= 0.0:
        raise ValueError("maximum support-contact cost must be positive and finite")
    if (
        not np.isfinite(map_discretization_allowance_m)
        or map_discretization_allowance_m < 0.0
    ):
        raise ValueError("map discretization allowance must be finite and non-negative")
    if not np.isfinite(support_surface_z_m):
        raise ValueError("support surface height must be finite")
    if not np.isfinite(support_height_tolerance_m) or support_height_tolerance_m <= 0.0:
        raise ValueError("support height tolerance must be positive and finite")

    contacts = {
        phase: list(phase_reports[phase].get("contacts", ()))
        for phase in required_phases
    }
    grasp_waypoint_count = int(phase_reports["grasp"]["cost_shape"][0])
    lift_waypoint_count = int(phase_reports["lift"]["cost_shape"][0])
    grasp_indices = sorted({int(item["waypoint_index"]) for item in contacts["grasp"]})
    lift_indices = sorted({int(item["waypoint_index"]) for item in contacts["lift"]})

    def is_contiguous(values: list[int]) -> bool:
        return not values or values == list(range(values[0], values[-1] + 1))

    all_contacts = contacts["approach"] + contacts["grasp"] + contacts["lift"]
    effective_collision_cost_limit_m = (
        maximum_collision_cost_m + map_discretization_allowance_m
    )
    checks = {
        "approach_is_strictly_clear": not contacts["approach"],
        "contact_exists_for_review": bool(contacts["grasp"] or contacts["lift"]),
        "only_finger_links_contact_environment": bool(all_contacts)
        and all(
            str(item["link_name"]) in SIMULATION_FINGER_CONTACT_LINKS
            for item in all_contacts
        ),
        "nearest_environment_points_match_support_height": bool(all_contacts)
        and all(
            abs(
                float(
                    item["nearest_observed_source_point"]["point_robot_base_m"][2]
                )
                - support_surface_z_m
            )
            <= support_height_tolerance_m
            for item in all_contacts
        ),
        "maximum_collision_cost_within_voxel_aware_limit": bool(all_contacts)
        and max(float(item["collision_cost_m"]) for item in all_contacts)
        <= effective_collision_cost_limit_m,
        "physical_sphere_support_penetration_within_1mm": bool(all_contacts)
        and all(
            support_surface_z_m
            - (
                float(item["sphere_robot_base_xyz_radius_m"][2])
                - float(item["sphere_robot_base_xyz_radius_m"][3])
            )
            <= maximum_collision_cost_m
            for item in all_contacts
        ),
        "grasp_contacts_form_terminal_suffix": bool(grasp_indices)
        and is_contiguous(grasp_indices)
        and grasp_indices[-1] == grasp_waypoint_count - 1,
        "lift_contacts_form_initial_prefix": bool(lift_indices)
        and is_contiguous(lift_indices)
        and lift_indices[0] == 0,
        "lift_endpoint_is_clear": not lift_indices
        or lift_indices[-1] < lift_waypoint_count - 1,
    }
    return {
        "policy": (
            "simulation-only reviewed transient finger/support contact; collision "
            "costs are retained and no planner tolerance is changed"
        ),
        "maximum_collision_cost_m": float(maximum_collision_cost_m),
        "map_discretization_allowance_m": float(map_discretization_allowance_m),
        "effective_collision_cost_limit_m": float(
            effective_collision_cost_limit_m
        ),
        "support_surface_z_m": float(support_surface_z_m),
        "support_height_tolerance_m": float(support_height_tolerance_m),
        "accepted_links": sorted(SIMULATION_FINGER_CONTACT_LINKS),
        "grasp_contact_waypoints": grasp_indices,
        "lift_contact_waypoints": lift_indices,
        "checks": checks,
        "accepted": bool(all(checks.values())),
        "safe_for_real_robot_execution": False,
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
    if (
        args.handover_goal_quaternion_wxyz is not None
        and args.handover_goal_position_robot_base_m is None
    ):
        raise ValueError("a handover quaternion requires a handover goal position")
    handover_goal_position = None
    if args.handover_goal_position_robot_base_m is not None:
        handover_goal_position = np.asarray(
            args.handover_goal_position_robot_base_m, dtype=np.float32
        )
        if not np.isfinite(handover_goal_position).all():
            raise ValueError("handover goal position must contain finite values")
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from panda_handover.curobo_bridge import select_named_joint_positions
    from panda_handover.curobo_planning import (
        load_singleview_observed_pointcloud,
        rotation_matrix_to_quaternion_wxyz,
        summarize_ik_result_arrays,
    )
    from panda_handover.geometry import transform_points
    from panda_handover.scene_layout import DEFAULT_TABLETOP_LAYOUT

    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "apply_curobo_voxel_round_fix.py"),
            "--check-only",
        ],
        check=True,
    )
    pregrasp_report, grasp_transforms = _load_reviewed_pregrasp(args.pregrasp_plan)
    source_indices = np.load(
        args.pregrasp_plan / "source_candidate_indices.npy", allow_pickle=False
    ).reshape(-1)
    candidate_scores = np.load(
        args.pregrasp_plan / "candidate_scores.npy", allow_pickle=False
    ).reshape(-1)
    if len(source_indices) != len(grasp_transforms):
        raise ValueError("source candidate indices do not match grasp transforms")
    if len(candidate_scores) != len(grasp_transforms):
        raise ValueError("candidate scores do not match grasp transforms")
    prepared_map_value = pregrasp_report.get("inputs", {}).get("prepared_map")
    if not isinstance(prepared_map_value, str) or not prepared_map_value:
        raise ValueError("source pre-grasp report has no prepared_map provenance")
    prepared_map = Path(prepared_map_value)
    observed_scene = load_singleview_observed_pointcloud(prepared_map, args.capture)

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
    support_surface_z_m = (
        _load_validated_support_surface_z(
            args.capture,
            T_world_robot_base,
            expected_tabletop_world_z_m=DEFAULT_TABLETOP_LAYOUT.table_top_z_m,
        )
        if args.allow_reviewed_support_contact_preflight
        else None
    )
    map_discretization_allowance_m = float(observed_scene.voxel_size_m * 0.5)
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
    from curobo._src.solver.solver_ik import IKSolver
    from curobo._src.state.state_joint import JointState
    from curobo._src.state.state_joint_trajectory_ops import trim_joint_state_trajectory
    from curobo._src.types.device_cfg import DeviceCfg
    from curobo._src.types.pose import Pose
    from curobo._src.types.tool_pose import GoalToolPose

    device_cfg = DeviceCfg(device=torch.device(args.device), dtype=torch.float32)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    # Recreate the exact in-memory representation used by the successful
    # pre-grasp planner.  Do not round-trip Mesh.from_pointcloud through OBJ:
    # that adds a trimesh parser/exporter boundary that cuRobo does not need.
    scene_mesh = Mesh.from_pointcloud(
        observed_scene.points_robot_base_m,
        pitch=observed_scene.voxel_size_m,
        name="observed_scene_without_robot_or_target",
    )
    planner_cfg = MotionPlannerCfg.create(
        robot=args.robot,
        scene_model=SceneCfg(mesh=[scene_mesh]),
        device_cfg=device_cfg,
        max_goalset=len(grasp_transforms),
    )
    planner = MotionPlanner(planner_cfg)
    if planner.tool_frames != ["panda_hand"]:
        raise RuntimeError(f"reviewed Franka tool frame changed: {planner.tool_frames}")
    contact_collision_links = list(
        planner.kinematics.config.kinematics_config.grasp_contact_link_names or ()
    )
    if not contact_collision_links:
        raise RuntimeError("reviewed Franka config has no grasp_contact_link_names")
    # Match GraspGenX's official end-to-end planner initialization.  Its
    # public-cuRobo compatibility note explicitly warns that overriding the
    # default seeds/tolerances and use_cuda_graph=False makes approach/grasp
    # TrajOpt fail with "Planning to grasp pose failed".
    planner.warmup(enable_graph=False, num_warmup_iterations=1)

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
    strict_preflight_success = preflight_success
    support_contact_preflight = None
    if not preflight_success:
        def summarize_ik_result(result: Any) -> dict[str, Any]:
            return summarize_ik_result_arrays(
                _cpu_numpy(result.success),
                feasible=(
                    _cpu_numpy(result.feasible)
                    if result.feasible is not None
                    else None
                ),
                position_error=(
                    _cpu_numpy(result.position_error)
                    if result.position_error is not None
                    else None
                ),
                rotation_error=(
                    _cpu_numpy(result.rotation_error)
                    if result.rotation_error is not None
                    else None
                ),
                goalset_index=(
                    _cpu_numpy(result.goalset_index)
                    if result.goalset_index is not None
                    else None
                ),
            )

        # Diagnose each exact grasp independently. This never saves an executable
        # trajectory and leaves all reviewed planner settings unchanged.
        candidate_diagnostics: list[dict[str, Any]] = []
        free_world_solver = None
        sphere_link_names = None
        for rank in range(len(grasp_transforms)):
            candidate_goal = GoalToolPose(
                tool_frames=planner.tool_frames,
                position=torch.from_numpy(
                    grasp_transforms[rank : rank + 1, :3, 3]
                ).to(device_cfg.device)[None, None, None, :, :],
                quaternion=torch.from_numpy(quaternions[rank : rank + 1]).to(
                    device_cfg.device
                )[None, None, None, :, :],
            )
            planner.reset_seed()
            world_ik_result = planner.ik_solver.solve_pose(
                candidate_goal,
                return_seeds=planner.trajopt_solver.config.num_seeds,
                current_state=current_state,
            )
            world_ik_summary = summarize_ik_result(world_ik_result)
            free_world_summary = None
            free_world_solution_collision = None
            if int(world_ik_summary["success_count"]) == 0:
                if free_world_solver is None:
                    free_world_cfg = MotionPlannerCfg.create(
                        robot=args.robot,
                        scene_model=None,
                        device_cfg=device_cfg,
                        max_goalset=1,
                    )
                    free_world_solver = IKSolver(
                        free_world_cfg.ik_solver_config, None
                    )
                free_world_result = free_world_solver.solve_pose(
                    candidate_goal,
                    return_seeds=planner.trajopt_solver.config.num_seeds,
                    current_state=current_state,
                )
                free_world_summary = summarize_ik_result(free_world_result)
                free_success = _cpu_numpy(free_world_result.success).astype(
                    bool, copy=False
                ).reshape(-1)
                if np.any(free_success):
                    free_solutions = _cpu_numpy(free_world_result.solution).astype(
                        np.float32, copy=False
                    ).reshape(-1, len(planner.joint_names))
                    if len(free_solutions) != len(free_success):
                        raise RuntimeError(
                            "free-world IK solutions do not match success entries"
                        )
                    solution = free_solutions[int(np.flatnonzero(free_success)[0])]
                    diagnostic_state = JointState.from_position(
                        torch.from_numpy(solution).to(device_cfg.device).unsqueeze(0),
                        joint_names=planner.joint_names,
                    )
                    diagnostic_kinematics = planner.compute_kinematics(
                        diagnostic_state
                    )
                    diagnostic_spheres = diagnostic_kinematics.robot_spheres
                    if diagnostic_spheres is None:
                        raise RuntimeError(
                            "cuRobo returned no spheres for free-world IK diagnosis"
                        )
                    diagnostic_spheres_np = _cpu_numpy(diagnostic_spheres).astype(
                        np.float32, copy=False
                    )
                    if sphere_link_names is None:
                        sphere_link_names, _ = _collision_sphere_link_names(
                            planner.kinematics.config.kinematics_config,
                            int(diagnostic_spheres_np.shape[-2]),
                        )

                    def query_diagnostic_collision(
                        activation_distance_m: float,
                    ) -> np.ndarray:
                        diagnostic_buffer = CollisionBuffer.from_shape(
                            diagnostic_spheres.shape, device_cfg
                        )
                        diagnostic_buffer.zero_()
                        values = planner.scene_collision_checker.get_sphere_collision(
                            diagnostic_kinematics,
                            diagnostic_buffer,
                            torch.tensor(
                                [1.0],
                                device=device_cfg.device,
                                dtype=torch.float32,
                            ),
                            torch.tensor(
                                [activation_distance_m],
                                device=device_cfg.device,
                                dtype=torch.float32,
                            ),
                        )
                        torch.cuda.synchronize(device_cfg.device)
                        return _cpu_numpy(values).astype(np.float32, copy=False)

                    penetration_cost = query_diagnostic_collision(0.0)
                    optimizer_cost = query_diagnostic_collision(0.01)
                    free_world_solution_collision = {
                        "joint_positions": solution.astype(float).tolist(),
                        "actual_penetration_activation_distance_m": 0.0,
                        "optimizer_activation_distance_m": 0.01,
                        "actual_penetration": _phase_contact_diagnostics(
                            "exact_grasp_free_world_ik_penetration",
                            penetration_cost,
                            diagnostic_spheres_np,
                            sphere_link_names,
                            observed_scene.points_robot_base_m,
                            target_robot_base,
                        ),
                        "optimizer_proximity": _phase_contact_diagnostics(
                            "exact_grasp_free_world_ik_optimizer_proximity",
                            optimizer_cost,
                            diagnostic_spheres_np,
                            sphere_link_names,
                            observed_scene.points_robot_base_m,
                            target_robot_base,
                        ),
                    }
            candidate_diagnostics.append(
                {
                    "goalset_rank": rank,
                    "source_candidate_index": int(source_indices[rank]),
                    "graspgenx_score": float(candidate_scores[rank]),
                    "collision_aware_ik": world_ik_summary,
                    "ik_without_world_scene_control": free_world_summary,
                    "free_world_solution_vs_observed_scene": (
                        free_world_solution_collision
                    ),
                }
            )

        failure_stage = _classify_preflight_failure(candidate_diagnostics)
        support_reviews = []
        if args.allow_reviewed_support_contact_preflight:
            assert support_surface_z_m is not None
            for candidate in candidate_diagnostics:
                review = _review_exact_grasp_support_candidate(
                    candidate,
                    support_surface_z_m=support_surface_z_m,
                    support_height_tolerance_m=(
                        map_discretization_allowance_m + 1e-6
                    ),
                    map_discretization_allowance_m=(
                        map_discretization_allowance_m
                    ),
                )
                candidate["reviewed_support_contact"] = review
                if review["accepted"]:
                    support_reviews.append(candidate)

        if support_reviews:
            # Candidate arrays are score ordered; retain only the highest-scoring
            # candidate that passed every narrow support-contact check.
            selected_rank = int(support_reviews[0]["goalset_rank"])
            support_contact_links = sorted(
                SIMULATION_FINGER_CONTACT_LINKS.intersection(
                    contact_collision_links
                )
            )
            if support_contact_links != sorted(SIMULATION_FINGER_CONTACT_LINKS):
                raise RuntimeError(
                    "reviewed Franka config is missing finger contact links"
                )
            selected_goal = GoalToolPose(
                tool_frames=planner.tool_frames,
                position=torch.from_numpy(
                    grasp_transforms[selected_rank : selected_rank + 1, :3, 3]
                ).to(device_cfg.device)[None, None, None, :, :],
                quaternion=torch.from_numpy(
                    quaternions[selected_rank : selected_rank + 1]
                ).to(device_cfg.device)[None, None, None, :, :],
            )
            planner.disable_link_collision(support_contact_links)
            planner.reset_seed()
            support_preflight_result = planner.plan_pose(
                current_state=current_state,
                goal_tool_poses=selected_goal,
                max_attempts=args.max_attempts,
            )
            planner.enable_link_collision(support_contact_links)
            preflight_success = bool(
                support_preflight_result is not None
                and support_preflight_result.success is not None
                and support_preflight_result.success.any().item()
            )
            support_contact_preflight = {
                "attempted": True,
                "succeeded": preflight_success,
                "selected_original_goalset_rank": selected_rank,
                "selected_source_candidate_index": int(
                    source_indices[selected_rank]
                ),
                "temporarily_disabled_links": support_contact_links,
                "review": support_reviews[0]["reviewed_support_contact"],
            }
            if preflight_success:
                grasp_transforms = grasp_transforms[
                    selected_rank : selected_rank + 1
                ]
                source_indices = source_indices[selected_rank : selected_rank + 1]
                candidate_scores = candidate_scores[
                    selected_rank : selected_rank + 1
                ]
                quaternions = quaternions[selected_rank : selected_rank + 1]
                grasp_goals = selected_goal

        if free_world_solver is not None:
            free_world_solver.destroy()

        if not preflight_success:
            failure_report = {
                "status": "preflight_failed",
                "reference": {
                    "curobo_commit": CUROBO_COMMIT,
                    "planner": "MotionPlanner.plan_pose Issue #663 preflight",
                    "issue_663": "https://github.com/NVlabs/curobo/issues/663",
                },
                "inputs": {
                    "capture": str(args.capture),
                    "segmentation": str(args.segmentation),
                    "pregrasp_plan": str(args.pregrasp_plan),
                    "prepared_map": str(prepared_map),
                },
                "parameters": {
                    "planner_config_policy": "GraspGenX end2end official defaults",
                    "max_attempts": args.max_attempts,
                    "candidate_count": int(len(grasp_transforms)),
                    "planner_parameters_changed_for_diagnosis": False,
                    "reviewed_support_contact_opt_in": bool(
                        args.allow_reviewed_support_contact_preflight
                    ),
                },
                "result": {
                    "issue_663_preflight_succeeded": False,
                    "strict_preflight_succeeded": strict_preflight_success,
                    "support_contact_preflight": support_contact_preflight,
                    "failure_stage": failure_stage,
                    "plan_grasp_called": False,
                },
                "candidate_diagnostics": candidate_diagnostics,
                "safety": {
                    "diagnosis_only": True,
                    "trajectory_saved": False,
                    "trajectory_executed": False,
                    "simulation_only": True,
                },
                "next_gate": (
                    "Use failure_stage to decide whether to regenerate grasps, "
                    "inspect world collision, or investigate trajectory "
                    "optimization. Do not tune planner parameters from the "
                    "preflight status alone."
                ),
            }
            failure_path = output / "grasp_preflight_failure.json"
            failure_path.write_text(
                json.dumps(failure_report, indent=2) + "\n", encoding="utf-8"
            )
            planner.destroy()
            print(json.dumps(failure_report, indent=2))
            print(f"saved: {failure_path}")
            return 2

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
        # Use cuRobo's documented default grasp-contact handling.  The Franka
        # contact links are re-enabled and every returned waypoint is checked
        # against the same observed scene before this artifact can pass.
    )
    elapsed_s = time.monotonic() - started
    success = bool(
        result is not None
        and result.success is not None
        and result.success.any().item()
    )
    if not success:
        status_text = str(getattr(result, "status", "plan_grasp returned no result"))
        failure_report = {
            "status": "planning_failed",
            "reference": {
                "curobo_commit": CUROBO_COMMIT,
                "planner": "MotionPlanner.plan_grasp",
                "official_source": (
                    "curobo/_src/motion/motion_planner.py::MotionPlanner.plan_grasp"
                ),
                "graspgenx_end2end": (
                    "GraspGenX/end2end/e2e_grasp_demo.py::init_planner"
                ),
            },
            "planner_status": status_text,
            "selected_goalset_rank": _goalset_index(result),
            "stage_success": {
                "issue_663_preflight": preflight_success,
                "strict_issue_663_preflight": strict_preflight_success,
                "reviewed_support_contact_preflight": bool(
                    support_contact_preflight
                    and support_contact_preflight["succeeded"]
                ),
                "goalset": _success_any(getattr(result, "goalset_result", None)),
                "approach": _success_any(getattr(result, "approach_result", None)),
                "grasp": _success_any(getattr(result, "grasp_result", None)),
                "lift": _success_any(getattr(result, "lift_result", None)),
            },
            "parameters": {
                "planner_config_policy": "GraspGenX end2end official defaults",
                "official_grasp_contact_link_names": contact_collision_links,
                "approach_offset_m": float(
                    pregrasp_report["parameters"]["approach_offset_m"]
                ),
                "lift_offset_m": args.lift_offset,
                "candidate_count": int(len(grasp_transforms)),
                "support_contact_preflight": support_contact_preflight,
            },
            "next_gate": (
                "Do not tune solver parameters from this status alone; inspect the "
                "recorded failing stage first."
            ),
        }
        failure_path = output / "grasp_lift_failure.json"
        failure_path.write_text(
            json.dumps(failure_report, indent=2) + "\n", encoding="utf-8"
        )
        planner.destroy()
        print(json.dumps(failure_report, indent=2))
        print(f"saved: {failure_path}")
        return 2

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

    # cuRobo intentionally disables configured grasp-contact links while it
    # plans contact motion.  Re-enable them and fail closed if any full-robot
    # collision sphere penetrates the observed table or surrounding geometry
    # at any returned waypoint.  The SAM3 target is absent by construction.
    planner.enable_link_collision(contact_collision_links)
    phase_penetration_costs = {}
    phase_robot_spheres = {}
    sphere_link_names = None
    link_sphere_indices = None
    for phase_name, values in phase_reports.items():
        phase_state = JointState.from_position(
            torch.from_numpy(values["position"]).to(device_cfg.device),
            joint_names=planner.joint_names,
        )
        phase_kinematics = planner.compute_kinematics(phase_state)
        if phase_kinematics.robot_spheres is None:
            raise RuntimeError(f"cuRobo returned no {phase_name} collision spheres")
        phase_spheres_np = _cpu_numpy(phase_kinematics.robot_spheres).astype(
            np.float32, copy=False
        )
        np.save(output / f"{phase_name}_full_robot_spheres_world.npy", phase_spheres_np)
        phase_robot_spheres[phase_name] = phase_spheres_np
        if sphere_link_names is None:
            sphere_link_names, link_sphere_indices = _collision_sphere_link_names(
                planner.kinematics.config.kinematics_config,
                int(phase_spheres_np.shape[-2]),
            )
        phase_buffer = CollisionBuffer.from_shape(
            phase_kinematics.robot_spheres.shape, device_cfg
        )
        phase_buffer.zero_()
        phase_cost = planner.scene_collision_checker.get_sphere_collision(
            phase_kinematics,
            phase_buffer,
            torch.tensor([1.0], device=device_cfg.device, dtype=torch.float32),
            torch.tensor([0.0], device=device_cfg.device, dtype=torch.float32),
        )
        torch.cuda.synchronize(device_cfg.device)
        phase_cost_np = _cpu_numpy(phase_cost).astype(np.float32, copy=False)
        np.save(output / f"{phase_name}_full_robot_penetration_cost.npy", phase_cost_np)
        phase_penetration_costs[phase_name] = phase_cost_np

    if sphere_link_names is None or link_sphere_indices is None:
        raise RuntimeError("cuRobo returned no collision-sphere ownership mapping")
    sphere_map_report = {
        "reference": "cuRobo KinematicsParams.get_sphere_index_from_link_name",
        "sphere_count": len(sphere_link_names),
        "sphere_index_to_link_name": sphere_link_names,
        "link_to_sphere_indices": link_sphere_indices,
    }
    (output / "collision_sphere_link_map.json").write_text(
        json.dumps(sphere_map_report, indent=2) + "\n", encoding="utf-8"
    )
    phase_contact_reports = {
        phase_name: _phase_contact_diagnostics(
            phase_name,
            phase_penetration_costs[phase_name],
            phase_robot_spheres[phase_name],
            sphere_link_names,
            observed_scene.points_robot_base_m,
            target_robot_base,
        )
        for phase_name in phase_reports
    }
    strict_full_robot_clear = bool(
        all(not np.any(cost > 0.0) for cost in phase_penetration_costs.values())
    )
    support_preflight_used = bool(
        support_contact_preflight and support_contact_preflight["succeeded"]
    )
    postvalidation_support_surface_z_m = (
        float(support_surface_z_m)
        if support_preflight_used and support_surface_z_m is not None
        else float(np.min(target_robot_base[:, 2]))
    )
    transient_finger_support_contact = _review_transient_finger_support_contact(
        phase_contact_reports,
        support_surface_z_m=postvalidation_support_surface_z_m,
        support_height_tolerance_m=float(observed_scene.voxel_size_m * 0.5 + 1e-6),
        map_discretization_allowance_m=(
            map_discretization_allowance_m if support_preflight_used else 0.0
        ),
    )
    contact_report = {
        "reference": {
            "sphere_ownership": (
                "cuRobo KinematicsParams.get_sphere_index_from_link_name"
            ),
            "collision_geometry": "the exact observed point-cloud mesh used by planner",
            "nearest_point_diagnostic": (
                "nearest measured source points used to build the meshes; these are "
                "not triangle-level closest points"
            ),
        },
        "interpretation_gate": (
            "Distances are diagnostics only. The simulation acceptance policy does "
            "not alter planner collision tolerances."
        ),
        "strict_full_robot_clear": strict_full_robot_clear,
        "transient_finger_support_contact": transient_finger_support_contact,
        "phases": phase_contact_reports,
    }
    (output / "grasp_contact_diagnostics.json").write_text(
        json.dumps(contact_report, indent=2) + "\n", encoding="utf-8"
    )

    selected_rank = int(_cpu_numpy(result.goalset_index).reshape(-1)[0])
    if not 0 <= selected_rank < len(grasp_transforms):
        raise RuntimeError("cuRobo returned an invalid grasp goalset index")

    # Prepare the target attachment only after the official lift plan. During
    # the first tabletop lift, the object starts in intentional contact with
    # its support surface; cuRobo has no per-pair allowed-collision matrix here.
    target_mesh = Mesh.from_pointcloud(
        target_robot_base,
        pitch=observed_scene.voxel_size_m,
        name="sam3_target_for_attachment",
    )
    # Mesh.from_pointcloud stores triangle indices as a flat list for cuRobo's
    # Warp mesh loader. AttachmentManager converts the same object to trimesh,
    # whose faces contract is Nx3. Restore only that shape; vertices and face
    # membership remain unchanged.
    target_triangle_faces = _restore_triangle_faces(
        target_mesh.vertices, target_mesh.faces
    )
    target_mesh.faces = target_triangle_faces.tolist()
    grasp_end = JointState.from_position(
        torch.from_numpy(phase_reports["grasp"]["end"])
        .to(device_cfg.device)
        .unsqueeze(0),
        joint_names=planner.joint_names,
    )
    lift_end = JointState.from_position(
        torch.from_numpy(phase_reports["lift"]["end"])
        .to(device_cfg.device)
        .unsqueeze(0),
        joint_names=planner.joint_names,
    )
    identity_pose = Pose.from_list(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device_cfg=device_cfg
    )
    # The pinned cuRobo commit exposes a broken MotionPlanner accessor
    # (trajopt_solver.attachment_manager).  Current upstream fixes that
    # accessor to use trajopt_solver.core.attachment_manager.  Use the same
    # official path here without modifying the vendored checkout.
    attachment_manager = planner.trajopt_solver.core.attachment_manager
    attachment_manager.attach(
        # Official API contract: joint_states is the grasp configuration that
        # defines the fixed object-to-hand transform. Evaluation below uses
        # lift_end after that transform has been established.
        joint_states=grasp_end,
        obstacles=[target_mesh],
        link_name="attached_object",
        num_spheres=4,
        world_objects_pose_offset=identity_pose,
    )
    attached_indices = (
        attachment_manager.kinematics_params.get_sphere_index_from_link_name(
            "attached_object"
        )
    )
    attached_indices_np = _cpu_numpy(attached_indices).astype(np.int64).reshape(-1)
    if attached_indices_np.size != 4:
        raise RuntimeError(
            f"franka.yml must expose 4 attached-object spheres, got {attached_indices_np.size}"
        )
    np.save(output / "attached_object_sphere_indices.npy", attached_indices_np)
    attached_local_spheres = _cpu_numpy(
        attachment_manager.kinematics_params.link_spheres[
            0, attached_indices, :
        ]
    ).astype(np.float32, copy=False)
    np.save(output / "attached_object_spheres_panda_hand.npy", attached_local_spheres)

    lifted_kinematics = planner.compute_kinematics(lift_end)
    if lifted_kinematics.robot_spheres is None:
        raise RuntimeError("cuRobo returned no lifted robot collision spheres")
    lifted_robot_spheres_np = _cpu_numpy(lifted_kinematics.robot_spheres).astype(
        np.float32, copy=False
    ).reshape(-1, 4)
    if int(attached_indices_np.max()) >= len(lifted_robot_spheres_np):
        raise RuntimeError("attached sphere index exceeds lifted robot sphere array")
    np.save(output / "lift_end_robot_spheres_world.npy", lifted_robot_spheres_np)
    np.save(
        output / "lift_end_attached_object_spheres_world.npy",
        lifted_robot_spheres_np[attached_indices_np],
    )
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

    # Optional fourth phase: keep the fitted object spheres attached and use
    # cuRobo's official pose planner from the lift endpoint. The position must
    # be supplied by the experiment configuration; it is not guessed here.
    transport_report = None
    transport_checks: dict[str, bool] = {}
    transport_cost_np = None
    if handover_goal_position is not None:
        if args.handover_goal_quaternion_wxyz is None:
            handover_goal_quaternion = rotation_matrix_to_quaternion_wxyz(
                grasp_transforms[selected_rank, :3, :3]
            )
            handover_orientation_policy = "preserve_selected_grasp_orientation"
        else:
            handover_goal_quaternion = _normalized_quaternion(
                args.handover_goal_quaternion_wxyz
            )
            handover_orientation_policy = "explicit_quaternion_wxyz"
        handover_goal = GoalToolPose(
            tool_frames=planner.tool_frames,
            position=torch.from_numpy(handover_goal_position).to(device_cfg.device)[
                None, None, None, None, :
            ],
            quaternion=torch.from_numpy(handover_goal_quaternion).to(device_cfg.device)[
                None, None, None, None, :
            ],
        )
        planner.reset_seed()
        transport_started = time.monotonic()
        transport_result = planner.plan_pose(
            current_state=lift_end,
            goal_tool_poses=handover_goal,
            max_attempts=args.max_attempts,
        )
        transport_elapsed_s = time.monotonic() - transport_started
        transport_success = bool(
            transport_result is not None
            and transport_result.success is not None
            and transport_result.success.any().item()
        )
        if not transport_success:
            failure_report = {
                "status": "handover_transport_planning_failed",
                "reference": {
                    "curobo_commit": CUROBO_COMMIT,
                    "planner": "MotionPlanner.plan_pose after AttachmentManager.attach",
                },
                "goal": {
                    "frame": "panda_link0 robot base",
                    "tool_frame": "panda_hand",
                    "position_m": handover_goal_position.tolist(),
                    "quaternion_wxyz": handover_goal_quaternion.tolist(),
                    "orientation_policy": handover_orientation_policy,
                },
                "planner_status": str(
                    getattr(transport_result, "status", "plan_pose returned no result")
                ),
                "next_gate": (
                    "Review the fixed goal and diagnostics before changing solver "
                    "parameters."
                ),
            }
            failure_path = output / "handover_transport_failure.json"
            failure_path.write_text(
                json.dumps(failure_report, indent=2) + "\n", encoding="utf-8"
            )
            planner.destroy()
            print(json.dumps(failure_report, indent=2))
            print(f"saved: {failure_path}")
            return 2

        transport_values = _save_phase(
            output,
            "transport",
            planner,
            transport_result.get_interpolated_plan(),
        )
        transport_state = JointState.from_position(
            torch.from_numpy(transport_values["position"]).to(device_cfg.device),
            joint_names=planner.joint_names,
        )
        transport_kinematics = planner.compute_kinematics(transport_state)
        if transport_kinematics.robot_spheres is None:
            raise RuntimeError("cuRobo returned no transport collision spheres")
        transport_buffer = CollisionBuffer.from_shape(
            transport_kinematics.robot_spheres.shape, device_cfg
        )
        transport_buffer.zero_()
        transport_cost = planner.scene_collision_checker.get_sphere_collision(
            transport_kinematics,
            transport_buffer,
            torch.tensor([1.0], device=device_cfg.device, dtype=torch.float32),
            torch.tensor([0.0], device=device_cfg.device, dtype=torch.float32),
        )
        torch.cuda.synchronize(device_cfg.device)
        transport_cost_np = _cpu_numpy(transport_cost).astype(np.float32, copy=False)
        transport_spheres_np = _cpu_numpy(
            transport_kinematics.robot_spheres
        ).astype(np.float32, copy=False)
        transport_attached_spheres = transport_spheres_np[..., attached_indices_np, :]
        np.save(output / "transport_full_robot_penetration_cost.npy", transport_cost_np)
        np.save(
            output / "transport_attached_object_spheres_world.npy",
            transport_attached_spheres,
        )
        transport_checks = {
            "transport_starts_at_lift_end": bool(
                np.allclose(
                    transport_values["start"],
                    phase_reports["lift"]["end"],
                    atol=2e-3,
                    rtol=0.0,
                )
            ),
            "transport_attached_spheres_are_positive_and_finite": bool(
                np.isfinite(transport_attached_spheres).all()
                and np.all(transport_attached_spheres[..., 3] > 0.0)
            ),
            "transport_all_waypoints_clear_of_observed_scene": bool(
                not np.any(transport_cost_np > 0.0)
            ),
        }
        transport_report = {
            "planner": "MotionPlanner.plan_pose",
            "result_type": type(transport_result).__name__,
            "planner_reported_success": bool(transport_result.success.any().item()),
            "curobo_total_time_s": float(transport_result.total_time),
            "curobo_solve_time_s": float(transport_result.solve_time),
            "wall_time_s": transport_elapsed_s,
            "waypoints": transport_values["waypoints"],
            "full_joint_names": transport_values["full_joint_names"],
            "goal": {
                "frame": "panda_link0 robot base",
                "tool_frame": "panda_hand",
                "position_m": handover_goal_position.tolist(),
                "quaternion_wxyz": handover_goal_quaternion.tolist(),
                "orientation_policy": handover_orientation_policy,
            },
        }

    report = {
        "status": "success",
        "reference": {
            "curobo_commit": CUROBO_COMMIT,
            "planner": "MotionPlanner.plan_grasp with optional attached plan_pose",
            "planner_source": (
                "curobo/_src/motion/motion_planner.py::MotionPlanner.plan_grasp"
            ),
            "graspgenx_end2end_planner_config": (
                "GraspGenX/end2end/e2e_grasp_demo.py::init_planner"
            ),
            "official_grasp_contact_handling": (
                "robot franka.yml grasp_contact_link_names"
            ),
            "issue_663_preflight": "https://github.com/NVlabs/curobo/issues/663",
            "issue_692_padding": "https://github.com/NVlabs/curobo/issues/692",
            "padding_trim": (
                "official trim_joint_state_trajectory using interpolated_last_tstep"
            ),
            "attachment": "AttachmentManager.attach with Mesh.from_pointcloud",
            "attached_transport": (
                "MotionPlanner.plan_pose after AttachmentManager.attach"
            ),
            "attachment_face_contract": (
                "cuRobo Mesh.from_pointcloud flat triangle indices restored to the "
                "Nx3 contract required by trimesh"
            ),
            "isaac_execution_precedent": "Isaac Sim 5.1 Franka Pick and Place",
        },
        "inputs": {
            "capture": str(args.capture),
            "segmentation": str(args.segmentation),
            "pregrasp_plan": str(args.pregrasp_plan),
            "prepared_map": str(prepared_map),
            "observed_surface_point_count": int(
                observed_scene.points_robot_base_m.shape[0]
            ),
            "target_point_count": int(len(target_robot_base)),
        },
        "parameters": {
            "robot": args.robot,
            "device": args.device,
            "planner_config_policy": "GraspGenX end2end official defaults",
            "approach_axis": "panda_hand +Z with negative offset",
            "approach_offset_m": float(
                pregrasp_report["parameters"]["approach_offset_m"]
            ),
            "lift_axis": "robot-base/world +Z",
            "lift_offset_m": args.lift_offset,
            "temporarily_disabled_grasp_contact_links": contact_collision_links,
            "target_absent_from_observed_scene": True,
            "attachment_sphere_count": 4,
            "attachment_mesh_triangle_count": int(len(target_triangle_faces)),
            "attachment_transform_configuration": "grasp phase end",
            "strict_issue_663_preflight_succeeded": strict_preflight_success,
            "support_contact_preflight": support_contact_preflight,
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
            "transport": transport_report,
        },
        "diagnostics": {
            "collision_sphere_link_map": str(
                output / "collision_sphere_link_map.json"
            ),
            "grasp_contact_diagnostics": str(
                output / "grasp_contact_diagnostics.json"
            ),
            "positive_full_robot_contacts_by_phase": {
                name: values["positive_count"]
                for name, values in phase_contact_reports.items()
            },
            "strict_all_returned_waypoints_clear_with_all_robot_links_enabled": (
                strict_full_robot_clear
            ),
            "transient_finger_support_contact": transient_finger_support_contact,
            "support_contact_preflight": support_contact_preflight,
        },
        "automatic_checks": {
            "issue_663_preflight_succeeded": preflight_success,
            "planner_reported_success": True,
            **continuity_checks,
            "returned_waypoints_pass_simulation_contact_policy": bool(
                strict_full_robot_clear
                or transient_finger_support_contact["accepted"]
            ),
            "lift_end_attached_spheres_are_finite": bool(
                np.isfinite(attached_local_spheres).all()
            ),
            "lift_end_attached_object_not_penetrating_observed_scene": bool(
                not np.any(lifted_cost_np > 0.0)
            ),
            **transport_checks,
        },
        "safety": {
            "simulation_only": True,
            "unknown_space_assumed_free": True,
            "non_contact_links_world_collision_enabled_during_planning": True,
            "grasp_contact_links_temporarily_disabled_by_official_planner": True,
            "all_robot_links_postvalidated_at_every_returned_waypoint": True,
            "strict_full_robot_clear": strict_full_robot_clear,
            "reviewed_transient_finger_support_contact_accepted": bool(
                transient_finger_support_contact["accepted"]
            ),
            "support_contact_acceptance_is_simulation_only": True,
            "reviewed_support_contact_preflight_used": support_preflight_used,
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
            "attachment_transform_defined_at_grasp_end": True,
            "handover_transport_planned": transport_report is not None,
            "held_object_collision_checked_during_transport": (
                transport_report is not None and transport_cost_np is not None
            ),
            "human_or_receiver_collision_model_present": False,
            "handover_release_planned": False,
            "trajectory_executed": False,
            "manual_review_required": True,
            "safe_for_real_robot_execution": False,
        },
        "next_gate": (
            "Replay approach, grasp, lift, and attached transport in Isaac Sim while "
            "holding the physical gripper closed."
            if transport_report is not None
            else (
                "Replay all three phases in Isaac Sim with a DynamicCuboid target, "
                "close the physical Franka gripper at the grasp boundary, and measure "
                "target lift."
            )
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
