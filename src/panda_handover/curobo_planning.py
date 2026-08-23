"""Fail-closed preparation helpers for cuRobo pre-grasp planning.

The GPU planner stays in NVIDIA's pinned cuRobo checkout.  This module only
validates persisted experiment artifacts and performs explicit frame/pose
conversions that can be unit tested without CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


CUROBO_COMMIT = "057a96ffb1088531535f9915154f9d0dabd62428"
CUROBO_VOXEL_PATCH_SHA256 = (
    "bab10d99e555fe722f2c3d893425ea9126978238ee43b9f6c0e250875c10e004"
)


@dataclass(frozen=True)
class ConservativeEsdf:
    """Validated dense ESDF and its serialized grid contract."""

    features_m: np.ndarray
    known_free: np.ndarray
    shape_xyz: tuple[int, int, int]
    voxel_size_m: float
    extent_m: tuple[float, float, float]
    center_robot_base_m: tuple[float, float, float]
    unknown_policy: str
    report: dict[str, Any]


@dataclass(frozen=True)
class PregraspGoalset:
    """Score-ordered Panda hand goals in the robot-base frame."""

    grasp_robot_base: np.ndarray
    pregrasp_robot_base: np.ndarray
    scores: np.ndarray
    candidate_indices: np.ndarray


@dataclass(frozen=True)
class ObservedPointcloudScene:
    """Validated robot/target-removed surface points in the robot-base frame."""

    points_robot_base_m: np.ndarray
    voxel_size_m: float
    report: dict[str, Any]
    view: dict[str, Any]


def _resolved_path_matches(recorded: str, expected: Path) -> bool:
    """Compare persisted experiment paths without accepting basename-only matches."""
    return Path(recorded).resolve() == expected.resolve()


def load_singleview_observed_pointcloud(
    prepared_map: str | Path,
    capture: str | Path,
) -> ObservedPointcloudScene:
    """Load cuRobo Mapper surface points only after checking their provenance.

    The input is the artifact written by ``curobo_map_capture.py``.  Requiring
    one view keeps this first comparison equivalent to the current single-view
    experiment and ensures that the target was absent from a fresh map for the
    entire integration.
    """
    root = Path(prepared_map)
    capture_path = Path(capture)
    report_path = root / "esdf_check.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"missing cuRobo Mapper report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "success":
        raise ValueError("cuRobo Mapper report did not finish successfully")
    checks = report.get("automatic_checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise ValueError("cuRobo Mapper automatic checks are incomplete or failed")
    if report.get("safe_to_plan") is not False:
        raise ValueError("expected the reviewed inspection-only Mapper gate")

    reference = report.get("reference", {})
    required_apis = {"RobotSegmenter", "FilterDepth", "Mapper.compute_esdf"}
    if not required_apis.issubset(set(reference.get("apis", ()))):
        raise ValueError("prepared map did not use the reviewed cuRobo perception APIs")
    if report.get("frames", {}).get("map") != "franka robot base":
        raise ValueError("prepared pointcloud is not expressed in the Franka base frame")

    parameters = report.get("parameters", {})
    if parameters.get("input_frames") != 1:
        raise ValueError("observed-mesh comparison currently requires exactly one view")
    voxel_size = float(parameters.get("voxel_size_m", float("nan")))
    extent = np.asarray(parameters.get("extent_m"), dtype=np.float64)
    center = np.asarray(parameters.get("grid_center_robot_base_m"), dtype=np.float64)
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("prepared map has an invalid voxel size")
    if extent.shape != (3,) or center.shape != (3,):
        raise ValueError("prepared map has invalid workspace bounds")
    if not np.all(np.isfinite(extent)) or not np.all(np.isfinite(center)):
        raise ValueError("prepared map workspace bounds contain non-finite values")
    if np.any(extent <= 0.0):
        raise ValueError("prepared map workspace extent must be positive")

    contract = report.get("unknown_environment_contract", {})
    required_contract = {
        "isaac_semantic_labels_used": False,
        "isaac_ground_truth_obstacle_geometry_used": False,
        "target_removed_with_sam3_mask": True,
        "robot_removed_with_curobo_kinematics": True,
        "unobserved_space_proven_occupied": False,
    }
    if any(contract.get(key) is not value for key, value in required_contract.items()):
        raise ValueError("prepared map provenance contract is missing")

    views = report.get("views")
    if not isinstance(views, list) or len(views) != 1 or not isinstance(views[0], dict):
        raise ValueError("prepared map must contain exactly one reported view")
    view = views[0]
    if not isinstance(view.get("capture"), str) or not _resolved_path_matches(
        view["capture"], capture_path
    ):
        raise ValueError("--capture does not match the prepared map view")
    if int(view.get("robot_mask_pixels", 0)) <= 0:
        raise ValueError("prepared map did not remove any robot pixels")
    if int(view.get("target_mask_pixels", 0)) <= 0:
        raise ValueError("prepared map did not remove any target pixels")

    points_path = root / "occupied_points_robot_base.npy"
    points = np.load(points_path, allow_pickle=False)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"occupied surface points must be non-empty Nx3, got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("occupied surface points contain non-finite values")
    reported_count = int(report.get("counts", {}).get("occupied_surface_voxels", -1))
    if reported_count != points.shape[0]:
        raise ValueError("occupied surface point count does not match the Mapper report")

    minimum = center - 0.5 * extent - voxel_size
    maximum = center + 0.5 * extent + voxel_size
    if np.any(points < minimum) or np.any(points > maximum):
        raise ValueError("occupied surface points fall outside the reported Mapper extent")

    return ObservedPointcloudScene(
        points_robot_base_m=points.astype(np.float32, copy=False),
        voxel_size_m=voxel_size,
        report=report,
        view=view,
    )


def summarize_ik_result_arrays(
    success: np.ndarray,
    *,
    feasible: np.ndarray | None = None,
    position_error: np.ndarray | None = None,
    rotation_error: np.ndarray | None = None,
    goalset_index: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build a small, JSON-safe diagnostic from official cuRobo IK fields."""
    success_array = np.asarray(success, dtype=bool).reshape(-1)
    summary: dict[str, Any] = {
        "returned_seed_count": int(success_array.size),
        "success_count": int(np.count_nonzero(success_array)),
    }

    def finite_minimum(value: np.ndarray | None) -> float | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        finite = array[np.isfinite(array)]
        return float(np.min(finite)) if finite.size else None

    if feasible is None:
        summary["feasible_count"] = None
    else:
        feasible_array = np.asarray(feasible, dtype=bool).reshape(-1)
        summary["feasible_count"] = int(np.count_nonzero(feasible_array))
    summary["minimum_position_error_m"] = finite_minimum(position_error)
    summary["minimum_rotation_error_rad"] = finite_minimum(rotation_error)

    if goalset_index is None:
        summary["successful_goalset_indices"] = []
    else:
        indices = np.asarray(goalset_index).reshape(-1)
        if indices.size != success_array.size:
            raise ValueError("goalset_index and success must have the same size")
        summary["successful_goalset_indices"] = [
            int(value) for value in indices[success_array]
        ]
    return summary


def classify_pregrasp_failure(
    *,
    planner_success: bool,
    world_ik_success_count: int,
    free_world_ik_success_count: int | None,
    start_penetrating_sphere_count: int,
    planner_returned_result: bool,
) -> str:
    """Classify a failed planning stage without changing any planner parameter."""
    if planner_success:
        return "plan_succeeded"
    if world_ik_success_count == 0:
        if free_world_ik_success_count is None:
            return "collision_aware_ik_failed_without_control_measurement"
        if free_world_ik_success_count > 0:
            if start_penetrating_sphere_count > 0:
                return "world_collision_rejects_ik_and_start_state_penetrates_map"
            return "world_collision_rejects_all_pregrasp_ik"
        return "pregrasp_ik_fails_even_without_world_collision"
    if not planner_returned_result:
        return "planner_ik_reproduction_mismatch"
    return "trajectory_optimization_failed_after_collision_aware_ik"


def _require_rigid_transform(transform: np.ndarray, *, label: str) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError(f"{label} must be 4x4, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{label} contains non-finite values")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{label} has an invalid homogeneous row")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise ValueError(f"{label} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-3):
        raise ValueError(f"{label} rotation determinant is not +1")
    return value


def load_backend_a_esdf(
    directory: str | Path,
    *,
    expected_unknown_policy: str,
) -> ConservativeEsdf:
    """Load Backend A only when its selected unknown-space contract is exact."""
    if expected_unknown_policy not in ("blocked", "free"):
        raise ValueError("expected_unknown_policy must be 'blocked' or 'free'")
    root = Path(directory)
    report_path = root / "esdf_check.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"missing conservative ESDF report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "success":
        raise ValueError("conservative ESDF report did not finish successfully")
    if report.get("backend") != "A_nvblox_torch_dense_edt":
        raise ValueError("pre-grasp A planner requires Backend A output")
    checks = report.get("automatic_checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise ValueError("conservative ESDF automatic checks are incomplete or failed")
    if report.get("safe_to_plan") is not False:
        raise ValueError("expected the reviewed pre-target-clear Backend A gate")

    parameters = report.get("parameters", {})
    unknown = report.get("unknown_environment_contract", {})
    report_policy = parameters.get("unknown_policy", "blocked")
    if report_policy != expected_unknown_policy:
        raise ValueError(
            f"Backend A unknown policy is {report_policy}, expected "
            f"{expected_unknown_policy}"
        )
    required_contract = (
        {
            "only_sensor_observed_space_can_be_free": True,
            "unobserved_space_is_blocked": True,
            "distance_to_unknown_boundary_recomputed": True,
            "target_is_currently_blocked": True,
        }
        if report_policy == "blocked"
        else {
            "only_sensor_observed_space_can_be_free": False,
            "unobserved_space_is_blocked": False,
            "unobserved_space_is_free": True,
            "distance_to_unknown_boundary_recomputed": True,
            "target_is_currently_blocked": False,
        }
    )
    if parameters.get("target_clear_applied") is not False:
        raise ValueError("unexpected target-clear mutation in Backend A map")
    if any(unknown.get(key) is not expected for key, expected in required_contract.items()):
        raise ValueError("Backend A unknown-space safety contract is missing")
    if report_policy == "free":
        scope = report.get("experiment_scope", {})
        required_scope = {
            "simulation_only": True,
            "safe_for_unknown_real_environment": False,
            "trajectory_execution_authorized": False,
        }
        if any(scope.get(key) is not value for key, value in required_scope.items()):
            raise ValueError("optimistic Backend A simulation scope is missing")

    grid = report.get("grid", {})
    if grid.get("index_order") != "x_slowest_z_fastest":
        raise ValueError("unsupported ESDF index order")
    if grid.get("sdf_sign") != "positive_free_negative_blocked":
        raise ValueError("unsupported ESDF sign convention")
    shape = tuple(int(value) for value in grid.get("shape_xyz", ()))
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("invalid ESDF shape metadata")
    voxel_size = float(grid.get("voxel_size_m", float("nan")))
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("invalid ESDF voxel size")
    extent = np.asarray(grid.get("extent_m"), dtype=np.float64)
    center = np.asarray(grid.get("center_robot_base_m"), dtype=np.float64)
    minimum = np.asarray(grid.get("min_corner_robot_base_m"), dtype=np.float64)
    if extent.shape != (3,) or center.shape != (3,) or minimum.shape != (3,):
        raise ValueError("invalid ESDF grid vectors")
    expected_extent = np.asarray(shape, dtype=np.float64) * voxel_size
    if not np.allclose(extent, expected_extent, atol=1e-7, rtol=0.0):
        raise ValueError("ESDF extent does not equal shape times voxel size")
    if not np.allclose(minimum, center - 0.5 * extent, atol=1e-7, rtol=0.0):
        raise ValueError("ESDF minimum corner is inconsistent with center and extent")

    features = np.load(root / "esdf_features.npy", allow_pickle=False)
    known_free = np.load(root / "known_free_mask.npy", allow_pickle=False).astype(
        bool, copy=False
    )
    if features.shape != shape or known_free.shape != shape:
        raise ValueError("ESDF arrays do not match reported shape")
    if not np.all(np.isfinite(features)):
        raise ValueError("ESDF contains non-finite values")
    if not np.all(features[known_free] > 0.0):
        raise ValueError("known-free voxels must have positive distance")
    if not np.all(features[~known_free] <= 0.0):
        raise ValueError("blocked voxels must have non-positive distance")
    if report_policy == "free":
        observed = np.load(root / "observed_mask.npy", allow_pickle=False).astype(
            bool, copy=False
        )
        sensor_known_free = np.load(
            root / "sensor_known_free_mask.npy", allow_pickle=False
        ).astype(bool, copy=False)
        if observed.shape != shape or sensor_known_free.shape != shape:
            raise ValueError("optimistic Backend A evidence masks do not match grid")
        expected_planning_free = sensor_known_free | ~observed
        if not np.array_equal(known_free, expected_planning_free):
            raise ValueError("optimistic planning-free mask changed observed obstacles")

    return ConservativeEsdf(
        features_m=features.astype(np.float32, copy=False),
        known_free=known_free,
        shape_xyz=shape,
        voxel_size_m=voxel_size,
        extent_m=tuple(float(value) for value in extent),
        center_robot_base_m=tuple(float(value) for value in center),
        unknown_policy=report_policy,
        report=report,
    )


def load_conservative_esdf(directory: str | Path) -> ConservativeEsdf:
    """Backward-compatible strict loader for unknown-as-blocked Backend A."""
    return load_backend_a_esdf(directory, expected_unknown_policy="blocked")


def validate_voxel_fix_report(path: str | Path) -> dict[str, Any]:
    """Require the GPU regression for the exact reviewed cuRobo source."""
    report_path = Path(path)
    if not report_path.is_file():
        raise FileNotFoundError(f"missing cuRobo voxel-fix regression: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    reference = report.get("reference", {})
    checks = report.get("automatic_checks")
    if report.get("status") != "success":
        raise ValueError("cuRobo voxel-fix regression did not succeed")
    if reference.get("curobo_commit") != CUROBO_COMMIT:
        raise ValueError("cuRobo voxel-fix regression used a different commit")
    if reference.get("patched_source_sha256") != CUROBO_VOXEL_PATCH_SHA256:
        raise ValueError("cuRobo voxel-fix regression used an unreviewed source file")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise ValueError("cuRobo voxel-fix regression checks are incomplete or failed")
    if report.get("safe_to_load_real_esdf") is not True:
        raise ValueError("cuRobo voxel-fix regression did not open its safety gate")
    return report


def prepare_pregrasp_goalset(
    panda_hand_world: np.ndarray,
    scores: np.ndarray,
    T_world_robot_base: np.ndarray,
    *,
    approach_offset_m: float = 0.15,
    max_candidates: int = 10,
    candidate_indices: np.ndarray | None = None,
) -> PregraspGoalset:
    """Transform and score-order grasps, then offset along negative tool Z."""
    poses = np.asarray(panda_hand_world, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"panda_hand_world must have shape (N,4,4), got {poses.shape}")
    if poses.shape[0] == 0 or values.shape != (poses.shape[0],):
        raise ValueError("candidate poses and scores must have one non-empty shared length")
    if not np.all(np.isfinite(values)):
        raise ValueError("candidate scores contain non-finite values")
    if not np.isfinite(approach_offset_m) or approach_offset_m <= 0.0:
        raise ValueError("approach_offset_m must be positive and finite")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    for index, pose in enumerate(poses):
        _require_rigid_transform(pose, label=f"panda_hand_world[{index}]")
    world_from_base = _require_rigid_transform(
        T_world_robot_base, label="T_world_robot_base"
    )

    if candidate_indices is None:
        source_indices = np.arange(poses.shape[0], dtype=np.int64)
    else:
        source_indices = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
        if source_indices.shape != values.shape:
            raise ValueError("candidate_indices length does not match candidates")
        if np.any(source_indices < 0) or len(np.unique(source_indices)) != len(source_indices):
            raise ValueError("candidate_indices must be unique non-negative integers")

    order = np.argsort(-values, kind="stable")[: min(max_candidates, len(values))]
    base_from_world = np.linalg.inv(world_from_base)
    grasp_base = np.einsum("ij,njk->nik", base_from_world, poses[order])
    pregrasp_base = grasp_base.copy()
    offset_tool = np.array([0.0, 0.0, -approach_offset_m], dtype=np.float64)
    pregrasp_base[:, :3, 3] += np.einsum(
        "nij,j->ni", grasp_base[:, :3, :3], offset_tool
    )
    return PregraspGoalset(
        grasp_robot_base=grasp_base.astype(np.float32),
        pregrasp_robot_base=pregrasp_base.astype(np.float32),
        scores=values[order].astype(np.float32),
        candidate_indices=source_indices[order].astype(np.int32),
    )


def rotation_matrix_to_quaternion_wxyz(rotations: np.ndarray) -> np.ndarray:
    """Convert one or more proper rotation matrices to normalized wxyz quaternions."""
    values = np.asarray(rotations, dtype=np.float64)
    if values.shape[-2:] != (3, 3):
        raise ValueError("rotations must end in shape (3,3)")
    flat = values.reshape(-1, 3, 3)
    output = np.empty((flat.shape[0], 4), dtype=np.float64)
    for index, matrix in enumerate(flat):
        transform = np.eye(4)
        transform[:3, :3] = matrix
        _require_rigid_transform(transform, label=f"rotation[{index}]")
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            quaternion = np.array(
                [0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
                 (matrix[0, 2] - matrix[2, 0]) / scale,
                 (matrix[1, 0] - matrix[0, 1]) / scale]
            )
        else:
            axis = int(np.argmax(np.diag(matrix)))
            if axis == 0:
                scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
                quaternion = np.array(
                    [(matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                     (matrix[0, 1] + matrix[1, 0]) / scale,
                     (matrix[0, 2] + matrix[2, 0]) / scale]
                )
            elif axis == 1:
                scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
                quaternion = np.array(
                    [(matrix[0, 2] - matrix[2, 0]) / scale,
                     (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                     (matrix[1, 2] + matrix[2, 1]) / scale]
                )
            else:
                scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
                quaternion = np.array(
                    [(matrix[1, 0] - matrix[0, 1]) / scale,
                     (matrix[0, 2] + matrix[2, 0]) / scale,
                     (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale]
                )
        quaternion /= np.linalg.norm(quaternion)
        if quaternion[0] < 0.0:
            quaternion *= -1.0
        output[index] = quaternion
    return output.reshape(*values.shape[:-2], 4).astype(np.float32)
