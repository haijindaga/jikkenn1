"""Classification and validation for saved grasp-candidate visualization."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np


STATIC_COLLISION_REJECTED = "static_collision_rejected"
STATIC_COLLISION_FREE = "static_collision_free"
CUROBO_PLAN_REJECTED = "curobo_plan_rejected"
CUROBO_PLAN_SUCCESS = "curobo_plan_success"
PLANNING_ERROR = "planning_error"

STATE_COLORS_RGB = {
    STATIC_COLLISION_REJECTED: (220, 45, 45),
    STATIC_COLLISION_FREE: (40, 190, 80),
    CUROBO_PLAN_REJECTED: (245, 180, 35),
    CUROBO_PLAN_SUCCESS: (35, 105, 235),
    PLANNING_ERROR: (220, 55, 180),
}


def resolve_saved_gripper_identity(
    candidate_report: Mapping[str, object],
    filter_report: Mapping[str, object],
) -> str:
    """Resolve matching gripper provenance across current and legacy reports.

    Before selectable grippers were introduced, both stages were hard-coded to
    ``franka_panda``. Legacy reports can therefore omit the identity. Explicit
    conflicting identities still fail closed.
    """

    def recorded(report: Mapping[str, object]) -> object:
        parameters = report.get("parameters", {})
        nested = parameters if isinstance(parameters, Mapping) else {}
        return report.get("gripper", nested.get("gripper"))

    candidate = recorded(candidate_report) or "franka_panda"
    filtered = recorded(filter_report) or "franka_panda"
    supported = {"franka_panda", "robotiq_2f_85"}
    if candidate not in supported:
        raise ValueError(f"unsupported candidate gripper identity: {candidate!r}")
    if filtered not in supported:
        raise ValueError(f"unsupported collision-filter gripper identity: {filtered!r}")
    if candidate != filtered:
        raise ValueError(
            "candidate and collision-filter grippers disagree: "
            f"candidate={candidate!r}, filtered={filtered!r}"
        )
    return str(candidate)


def classify_candidate_states(
    collision_free_mask: np.ndarray,
    plan_attempts: Iterable[Mapping[str, object]] = (),
) -> np.ndarray:
    """Return an exclusive, evidence-backed state for every raw candidate.

    Static collision filtering supplies the base red/green classification.
    Optional cuRobo attempts only refine candidates that passed that filter.
    Infrastructure failures remain a separate magenta state rather than being
    misreported as geometric or planning rejection.
    """

    collision_free = np.asarray(collision_free_mask, dtype=bool).reshape(-1)
    if collision_free.size == 0:
        raise ValueError("collision_free_mask is empty")
    states = np.full(
        collision_free.shape,
        STATIC_COLLISION_REJECTED,
        dtype="U32",
    )
    states[collision_free] = STATIC_COLLISION_FREE

    seen: set[int] = set()
    for attempt in plan_attempts:
        if "source_candidate_index" not in attempt:
            raise ValueError("planning attempt has no source_candidate_index")
        index = int(attempt["source_candidate_index"])
        if not 0 <= index < collision_free.size:
            raise ValueError(f"planning candidate index is outside raw pool: {index}")
        if index in seen:
            raise ValueError(f"planning candidate index is duplicated: {index}")
        seen.add(index)
        if not collision_free[index]:
            raise ValueError(
                f"cuRobo attempted statically rejected candidate {index}"
            )

        plan_status = str(attempt.get("plan_status", "missing_report"))
        return_code = int(attempt.get("return_code", 1))
        available = attempt.get("available_for_physical_trial") is True
        if available:
            if plan_status != "success" or return_code != 0:
                raise ValueError(
                    f"candidate {index} is marked available without a successful plan"
                )
            states[index] = CUROBO_PLAN_SUCCESS
        elif plan_status == "missing_report" or return_code not in (0, 2):
            states[index] = PLANNING_ERROR
        else:
            states[index] = CUROBO_PLAN_REJECTED
    return states


def verify_saved_world_grasps(
    grasps_camera: np.ndarray,
    grasps_world: np.ndarray,
    T_world_camera: np.ndarray,
    *,
    atol: float = 1e-5,
) -> None:
    """Fail if saved camera/world candidate frames are inconsistent."""

    camera = np.asarray(grasps_camera, dtype=np.float64)
    world = np.asarray(grasps_world, dtype=np.float64)
    transform = np.asarray(T_world_camera, dtype=np.float64)
    if camera.ndim != 3 or camera.shape[1:] != (4, 4):
        raise ValueError(f"grasps_camera must have shape (N,4,4), got {camera.shape}")
    if world.shape != camera.shape:
        raise ValueError("camera and world grasp arrays have different shapes")
    if transform.shape != (4, 4):
        raise ValueError(f"T_world_camera must have shape (4,4), got {transform.shape}")
    expected = np.einsum("ij,njk->nik", transform, camera)
    if not np.allclose(world, expected, atol=atol, rtol=0.0):
        maximum_error = float(np.max(np.abs(world - expected)))
        raise ValueError(
            "saved world grasps disagree with T_world_camera @ grasps_camera; "
            f"maximum element error={maximum_error:.6g}"
        )


def state_counts(states: np.ndarray) -> dict[str, int]:
    values = np.asarray(states).reshape(-1)
    return {
        state: int(np.count_nonzero(values == state))
        for state in STATE_COLORS_RGB
    }
