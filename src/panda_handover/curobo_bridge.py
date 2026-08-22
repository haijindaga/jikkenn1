"""Pure validation helpers for the cuRobo integration boundary."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def select_named_joint_positions(
    joint_names: Sequence[str],
    joint_positions: np.ndarray,
    requested_names: Sequence[str],
) -> np.ndarray:
    """Return positions in a consumer's requested joint-name order."""
    names = tuple(str(name) for name in joint_names)
    requested = tuple(str(name) for name in requested_names)
    positions = np.asarray(joint_positions)
    if positions.shape != (len(names),):
        raise ValueError(
            f"joint_positions shape {positions.shape} does not match {len(names)} names"
        )
    if len(set(names)) != len(names):
        raise ValueError("joint_names must be unique")
    index = {name: i for i, name in enumerate(names)}
    missing = [name for name in requested if name not in index]
    if missing:
        raise ValueError(f"capture is missing requested joints: {missing}")
    return positions[[index[name] for name in requested]].copy()


def validate_mapping_inputs(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    target_mask: np.ndarray,
    T_robot_base_camera: np.ndarray,
) -> None:
    """Validate the saved Isaac/SAM3 arrays before allocating GPU state."""
    depth = np.asarray(depth_m)
    image = np.asarray(rgb)
    camera_matrix = np.asarray(intrinsics)
    mask = np.asarray(target_mask)
    transform = np.asarray(T_robot_base_camera)
    if depth.ndim != 2:
        raise ValueError(f"depth_m must be HxW, got {depth.shape}")
    if image.shape != (*depth.shape, 3):
        raise ValueError(f"rgb shape {image.shape} does not match depth {depth.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"rgb must be uint8, got {image.dtype}")
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"intrinsics must be 3x3, got {camera_matrix.shape}")
    if mask.shape != depth.shape:
        raise ValueError(f"target mask shape {mask.shape} does not match depth {depth.shape}")
    if transform.shape != (4, 4):
        raise ValueError(f"T_robot_base_camera must be 4x4, got {transform.shape}")
    if not np.all(np.isfinite(camera_matrix)) or not np.all(np.isfinite(transform)):
        raise ValueError("camera calibration contains non-finite values")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError("T_robot_base_camera has an invalid homogeneous last row")
    valid_depth = np.isfinite(depth) & (depth > 0.0)
    if not valid_depth.any():
        raise ValueError("capture has no valid metric depth")
    if not np.asarray(mask, dtype=bool).any():
        raise ValueError("target mask is empty")
