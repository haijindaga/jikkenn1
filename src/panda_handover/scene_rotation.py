"""Rigid scene-variant transforms with explicit world-frame semantics."""

from __future__ import annotations

import math

import numpy as np


def rotate_world_pose_about_own_origin(
    T_world_object: np.ndarray,
    *,
    yaw_degrees: float,
) -> np.ndarray:
    """Rotate an object's pose about world +Z without translating its origin.

    Transforms use the column-vector convention used by the capture and
    planning pipeline.  The complete linear block is left-multiplied so any
    authored scale is preserved together with the object's local geometry.
    """

    transform = np.asarray(T_world_object, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("T_world_object must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("T_world_object must be an affine homogeneous matrix")
    if not math.isfinite(yaw_degrees):
        raise ValueError("yaw_degrees must be finite")

    angle = math.radians(float(yaw_degrees))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    world_yaw = np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    result = transform.copy()
    result[:3, :3] = world_yaw @ transform[:3, :3]
    return result


def local_pose_from_world_poses(
    T_world_parent: np.ndarray,
    T_world_child: np.ndarray,
) -> np.ndarray:
    """Return ``T_parent_child`` for two column-vector world transforms."""

    parent = np.asarray(T_world_parent, dtype=np.float64)
    child = np.asarray(T_world_child, dtype=np.float64)
    for transform, name in ((parent, "T_world_parent"), (child, "T_world_child")):
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(f"{name} must be a finite 4x4 matrix")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise ValueError(f"{name} must be an affine homogeneous matrix")
    try:
        return np.linalg.solve(parent, child)
    except np.linalg.LinAlgError as error:
        raise ValueError("T_world_parent must be invertible") from error
