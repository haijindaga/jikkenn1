"""Validation and timing helpers for simulation-only trajectory replay."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PregraspReplay:
    """A validated, simulation-only cuRobo pre-grasp trajectory."""

    joint_names: tuple[str, ...]
    positions: np.ndarray
    segment_dt_s: np.ndarray
    capture_joint_names: tuple[str, ...]
    capture_joint_positions: np.ndarray
    capture_indices: np.ndarray
    plan_report: dict


@dataclass(frozen=True)
class GraspLiftReplay:
    """Validated cuRobo approach, grasp, and lift trajectories for Isaac Sim."""

    joint_names: tuple[str, ...]
    phase_positions: dict[str, np.ndarray]
    phase_segment_dt_s: dict[str, np.ndarray]
    capture_joint_names: tuple[str, ...]
    capture_joint_positions: np.ndarray
    plan_report: dict


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def normalize_segment_dt(dt: np.ndarray, waypoint_count: int) -> np.ndarray:
    """Return one positive duration for every pair of adjacent waypoints."""

    if waypoint_count < 2:
        raise ValueError("trajectory must contain at least two waypoints")
    value = np.asarray(dt, dtype=np.float64).reshape(-1)
    if value.size == 1:
        result = np.full(waypoint_count - 1, value.item(), dtype=np.float64)
    elif value.size == waypoint_count - 1:
        result = value.copy()
    elif value.size == waypoint_count:
        # JointState trajectories may serialize a dt value with each waypoint;
        # the final value has no following segment.
        result = value[:-1].copy()
    else:
        raise ValueError(
            "trajectory dt must be scalar, H-1, or H; "
            f"got {value.size} values for H={waypoint_count}"
        )
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("trajectory dt must contain only positive finite values")
    return result


def load_pregrasp_replay(
    capture_directory: str | Path,
    plan_directory: str | Path,
    *,
    start_tolerance: float = 2e-3,
) -> PregraspReplay:
    """Load a planner result without trusting its human-readable next-step text."""

    capture = Path(capture_directory)
    plan = Path(plan_directory)
    report = _load_json(plan / "pregrasp_plan_check.json")
    robot_report = _load_json(capture / "robot_state.json")

    result = report.get("result", {})
    checks = report.get("automatic_checks", {})
    safety = report.get("safety", {})
    if report.get("status") != "success" or not result.get("planner_reported_success"):
        raise ValueError("plan report does not contain a successful cuRobo plan")
    if not checks or not all(bool(value) for value in checks.values()):
        raise ValueError("plan report automatic checks did not all pass")
    if not safety.get("simulation_only"):
        raise ValueError("only an explicitly simulation-only plan may be replayed")
    if not safety.get("pregrasp_only_scope_gate_passed"):
        raise ValueError("plan is not approved for the pre-grasp-only scope")
    if safety.get("final_approach_planned") or safety.get("gripper_close_planned"):
        raise ValueError("this replay gate accepts pre-grasp motion only")

    joint_names = tuple(str(name) for name in result.get("trajectory_active_joint_names", ()))
    if not joint_names or len(set(joint_names)) != len(joint_names):
        raise ValueError("trajectory active joint names must be non-empty and unique")
    positions = np.load(plan / "trajectory_position.npy").astype(np.float64, copy=False)
    if positions.ndim != 2 or positions.shape[1] != len(joint_names):
        raise ValueError(
            "trajectory position must be HxD and match active joint names; "
            f"got {positions.shape} and {len(joint_names)} names"
        )
    if positions.shape[0] < 2 or not np.all(np.isfinite(positions)):
        raise ValueError("trajectory positions must have at least two finite waypoints")
    segment_dt = normalize_segment_dt(
        np.load(plan / "trajectory_dt_s.npy"), positions.shape[0]
    )

    capture_names = tuple(str(name) for name in robot_report.get("joint_names", ()))
    if not capture_names or len(set(capture_names)) != len(capture_names):
        raise ValueError("capture joint names must be non-empty and unique")
    capture_positions = np.load(capture / "panda_joint_positions.npy").astype(
        np.float64, copy=False
    )
    if capture_positions.shape != (len(capture_names),):
        raise ValueError("capture positions do not match capture joint names")
    name_to_capture_index = {name: index for index, name in enumerate(capture_names)}
    missing = [name for name in joint_names if name not in name_to_capture_index]
    if missing:
        raise ValueError(f"trajectory joints are missing from capture: {missing}")
    capture_indices = np.asarray(
        [name_to_capture_index[name] for name in joint_names], dtype=np.int64
    )
    if not np.allclose(
        positions[0], capture_positions[capture_indices], atol=start_tolerance, rtol=0.0
    ):
        maximum_error = float(
            np.max(np.abs(positions[0] - capture_positions[capture_indices]))
        )
        raise ValueError(
            "trajectory does not start at the captured Panda state; "
            f"maximum error={maximum_error:.6g}"
        )

    return PregraspReplay(
        joint_names=joint_names,
        positions=positions,
        segment_dt_s=segment_dt,
        capture_joint_names=capture_names,
        capture_joint_positions=capture_positions,
        capture_indices=capture_indices,
        plan_report=report,
    )


def load_grasp_lift_replay(
    capture_directory: str | Path,
    plan_directory: str | Path,
    *,
    continuity_tolerance: float = 2e-3,
) -> GraspLiftReplay:
    """Load three planned phases without authorizing non-simulation execution."""

    capture = Path(capture_directory)
    plan = Path(plan_directory)
    report = _load_json(plan / "grasp_lift_plan_check.json")
    robot_report = _load_json(capture / "robot_state.json")
    result = report.get("result", {})
    checks = report.get("automatic_checks", {})
    safety = report.get("safety", {})
    if report.get("status") != "success" or not result.get("planner_reported_success"):
        raise ValueError("grasp/lift report does not contain a successful cuRobo plan")
    if not checks or not all(bool(value) for value in checks.values()):
        raise ValueError("grasp/lift plan automatic checks did not all pass")
    if safety.get("simulation_only") is not True:
        raise ValueError("only an explicitly simulation-only grasp/lift plan is accepted")
    if safety.get("final_approach_planned") is not True or safety.get("lift_planned") is not True:
        raise ValueError("grasp/lift plan is missing required phases")
    if safety.get("trajectory_executed") is not False:
        raise ValueError("source plan unexpectedly claims prior execution")

    joint_names = tuple(str(name) for name in result.get("trajectory_active_joint_names", ()))
    if not joint_names or len(set(joint_names)) != len(joint_names):
        raise ValueError("trajectory active joint names must be non-empty and unique")
    phase_positions: dict[str, np.ndarray] = {}
    phase_dt: dict[str, np.ndarray] = {}
    for phase in ("approach", "grasp", "lift"):
        positions = np.load(plan / f"{phase}_trajectory_position.npy").astype(
            np.float64, copy=False
        )
        if positions.ndim != 2 or positions.shape[1] != len(joint_names):
            raise ValueError(f"{phase} trajectory shape does not match joint names")
        if positions.shape[0] < 2 or not np.isfinite(positions).all():
            raise ValueError(f"{phase} trajectory must contain finite waypoints")
        phase_positions[phase] = positions
        phase_dt[phase] = normalize_segment_dt(
            np.load(plan / f"{phase}_trajectory_dt_s.npy"), positions.shape[0]
        )

    capture_names = tuple(str(name) for name in robot_report.get("joint_names", ()))
    capture_positions = np.load(capture / "panda_joint_positions.npy").astype(
        np.float64, copy=False
    )
    if not capture_names or capture_positions.shape != (len(capture_names),):
        raise ValueError("capture joint names and positions do not match")
    name_to_capture = {name: index for index, name in enumerate(capture_names)}
    missing = [name for name in joint_names if name not in name_to_capture]
    if missing:
        raise ValueError(f"planned joints are missing from capture: {missing}")
    capture_active = capture_positions[[name_to_capture[name] for name in joint_names]]
    if not np.allclose(
        phase_positions["approach"][0],
        capture_active,
        atol=continuity_tolerance,
        rtol=0.0,
    ):
        raise ValueError("approach trajectory does not start at captured state")
    if not np.allclose(
        phase_positions["grasp"][0],
        phase_positions["approach"][-1],
        atol=continuity_tolerance,
        rtol=0.0,
    ):
        raise ValueError("grasp trajectory is discontinuous from approach")
    if not np.allclose(
        phase_positions["lift"][0],
        phase_positions["grasp"][-1],
        atol=continuity_tolerance,
        rtol=0.0,
    ):
        raise ValueError("lift trajectory is discontinuous from grasp")

    return GraspLiftReplay(
        joint_names=joint_names,
        phase_positions=phase_positions,
        phase_segment_dt_s=phase_dt,
        capture_joint_names=capture_names,
        capture_joint_positions=capture_positions,
        plan_report=report,
    )


def sample_positions_at_physics_rate(
    positions: np.ndarray,
    segment_dt_s: np.ndarray,
    physics_dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the saved piecewise-linear joint path at Isaac physics ticks."""

    q = np.asarray(positions, dtype=np.float64)
    dt = normalize_segment_dt(segment_dt_s, q.shape[0])
    if not np.isfinite(physics_dt_s) or physics_dt_s <= 0.0:
        raise ValueError("physics_dt_s must be positive and finite")
    waypoint_time = np.concatenate(([0.0], np.cumsum(dt)))
    duration = float(waypoint_time[-1])
    sample_time = np.arange(0.0, duration, physics_dt_s, dtype=np.float64)
    if sample_time.size == 0 or sample_time[-1] < duration:
        sample_time = np.append(sample_time, duration)
    sampled = np.column_stack(
        [np.interp(sample_time, waypoint_time, q[:, index]) for index in range(q.shape[1])]
    )
    sampled[0] = q[0]
    sampled[-1] = q[-1]
    return sample_time, sampled
