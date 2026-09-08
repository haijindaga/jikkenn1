"""Geometry helpers for affordance-aware robot-to-human handover goals."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


DEFAULT_HANDOVER_ROLL_DEGREES = (0.0, 45.0, -45.0, 90.0, -90.0, 180.0)


def receive_clear_score_order(
    scores: np.ndarray, receive_clear_mask: np.ndarray
) -> np.ndarray:
    """Return a stable lexicographic order without inventing a weighted score."""

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    clear = np.asarray(receive_clear_mask, dtype=bool).reshape(-1)
    if len(values) != len(clear):
        raise ValueError("scores and receive_clear_mask must have equal lengths")
    if not np.isfinite(values).all():
        raise ValueError("scores must contain only finite values")
    clear_indices = np.flatnonzero(clear)
    return clear_indices[np.argsort(-values[clear_indices], kind="stable")]


def _unit(vector: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain three finite values")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-8:
        raise ValueError(f"{name} must have non-zero length")
    return value / norm


def _points(value: np.ndarray, *, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"{name} must have non-empty shape (N,3)")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain only finite values")
    return points


def _fallback_perpendicular(axis: np.ndarray) -> np.ndarray:
    basis = np.eye(3)[int(np.argmin(np.abs(axis)))]
    return _unit(basis - axis * float(np.dot(basis, axis)), name="fallback axis")


def _projected_unit(vector: np.ndarray, axis: np.ndarray) -> np.ndarray:
    projected = vector - axis * float(np.dot(vector, axis))
    if np.linalg.norm(projected) <= 1e-8:
        return _fallback_perpendicular(axis)
    return _unit(projected, name="projected up axis")


def _axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _unit(axis, name="rotation axis")
    x, y, z = axis
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    identity = np.eye(3)
    return (
        identity * np.cos(angle_rad)
        + (1.0 - np.cos(angle_rad)) * np.outer(axis, axis)
        + np.sin(angle_rad) * cross
    )


def generate_affordance_handover_goals(
    grasp_transform_robot_base: np.ndarray,
    grasp_part_points_robot_base: np.ndarray,
    receive_part_points_robot_base: np.ndarray,
    receiver_position_robot_base_m: Iterable[float],
    human_direction_robot_base: Iterable[float],
    *,
    roll_degrees: Iterable[float] = DEFAULT_HANDOVER_ROLL_DEGREES,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create hand goals that present the receive part toward the human.

    The segmented part centroids are expressed in the grasped hand frame.  Each
    returned pose maps the grasp-to-receive axis onto ``human_direction`` and
    places the receive-part centroid at ``receiver_position``.  Roll variants
    are left for cuRobo to choose by reachability and attached-object collision.
    """

    transform = np.asarray(grasp_transform_robot_base, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("grasp_transform_robot_base must be a finite 4x4 matrix")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise ValueError("grasp_transform_robot_base rotation must be rigid")

    grasp_points = _points(grasp_part_points_robot_base, name="grasp part points")
    receive_points = _points(receive_part_points_robot_base, name="receive part points")
    receiver_position = np.asarray(
        tuple(receiver_position_robot_base_m), dtype=np.float64
    ).reshape(3)
    if not np.isfinite(receiver_position).all():
        raise ValueError("receiver position must contain three finite values")
    human_direction = _unit(
        np.asarray(tuple(human_direction_robot_base), dtype=np.float64),
        name="human direction",
    )

    grasp_center_base = np.median(grasp_points, axis=0)
    receive_center_base = np.median(receive_points, axis=0)
    inverse_rotation = rotation.T
    grasp_center_hand = inverse_rotation @ (grasp_center_base - transform[:3, 3])
    receive_center_hand = inverse_rotation @ (
        receive_center_base - transform[:3, 3]
    )
    presentation_axis_hand = _unit(
        receive_center_hand - grasp_center_hand,
        name="grasp-to-receive part axis",
    )

    # Choose the zero-roll solution that best preserves world-up.  This only
    # resolves the free roll around the required presentation direction.
    world_up = np.array([0.0, 0.0, 1.0])
    up_hand = rotation.T @ world_up
    source_up = _projected_unit(up_hand, presentation_axis_hand)
    source_side = np.cross(presentation_axis_hand, source_up)
    destination_up = _projected_unit(world_up, human_direction)
    destination_side = np.cross(human_direction, destination_up)
    source_frame = np.column_stack(
        (presentation_axis_hand, source_up, source_side)
    )
    destination_frame = np.column_stack(
        (human_direction, destination_up, destination_side)
    )
    zero_roll_rotation = destination_frame @ source_frame.T

    rolls = tuple(float(value) for value in roll_degrees)
    if not rolls or not np.isfinite(rolls).all():
        raise ValueError("roll_degrees must contain at least one finite value")
    goals = []
    for roll_degree in rolls:
        goal_rotation = (
            _axis_angle_rotation(human_direction, np.deg2rad(roll_degree))
            @ zero_roll_rotation
        )
        goal_translation = receiver_position - goal_rotation @ receive_center_hand
        goal = np.eye(4)
        goal[:3, :3] = goal_rotation
        goal[:3, 3] = goal_translation
        goals.append(goal)
    goal_transforms = np.asarray(goals, dtype=np.float32)

    diagnostics: dict[str, Any] = {
        "policy": (
            "align grasp-part-to-receive-part axis with the human direction; "
            "place receive-part median at the requested receiver position; "
            "let cuRobo choose among fixed roll variants"
        ),
        "representative_point": "coordinate-wise median of segmented 3-D points",
        "grasp_part_center_robot_base_m": grasp_center_base.tolist(),
        "receive_part_center_robot_base_m": receive_center_base.tolist(),
        "grasp_part_center_panda_hand_m": grasp_center_hand.tolist(),
        "receive_part_center_panda_hand_m": receive_center_hand.tolist(),
        "presentation_axis_panda_hand": presentation_axis_hand.tolist(),
        "human_direction_robot_base": human_direction.tolist(),
        "receiver_position_robot_base_m": receiver_position.tolist(),
        "roll_degrees": list(rolls),
        "goal_count": len(goals),
    }
    return goal_transforms, diagnostics
