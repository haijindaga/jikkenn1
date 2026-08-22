"""Small, convention-explicit RGB-D geometry utilities.

Point clouds use the optical/OpenCV camera convention:

* +x: image right
* +y: image down
* +z: camera forward

Transforms are homogeneous matrices named ``T_destination_source``.  For
example, ``T_world_camera`` maps camera-frame points into the Isaac world.
"""

from __future__ import annotations

import numpy as np


def normalize_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Return a normalized scalar-first quaternion."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {quaternion.shape}")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion norm is zero")
    return quaternion / norm


def rotation_matrix_from_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Convert a normalized or unnormalized wxyz quaternion to a 3x3 matrix."""
    w, x, y, z = normalize_quaternion_wxyz(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_wxyz_from_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to a scalar-first quaternion."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {rotation.shape}")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("rotation matrix is not orthonormal")
    if np.linalg.det(rotation) < 0.0:
        raise ValueError("rotation matrix must be right-handed")

    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [0.25 * s,
             (rotation[2, 1] - rotation[1, 2]) / s,
             (rotation[0, 2] - rotation[2, 0]) / s,
             (rotation[1, 0] - rotation[0, 1]) / s]
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            q = np.array(
                [(rotation[2, 1] - rotation[1, 2]) / s,
                 0.25 * s,
                 (rotation[0, 1] + rotation[1, 0]) / s,
                 (rotation[0, 2] + rotation[2, 0]) / s]
            )
        elif axis == 1:
            s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            q = np.array(
                [(rotation[0, 2] - rotation[2, 0]) / s,
                 (rotation[0, 1] + rotation[1, 0]) / s,
                 0.25 * s,
                 (rotation[1, 2] + rotation[2, 1]) / s]
            )
        else:
            s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            q = np.array(
                [(rotation[1, 0] - rotation[0, 1]) / s,
                 (rotation[0, 2] + rotation[2, 0]) / s,
                 (rotation[1, 2] + rotation[2, 1]) / s,
                 0.25 * s]
            )
    return normalize_quaternion_wxyz(q)


def matrix_from_pose(position: np.ndarray, orientation_wxyz: np.ndarray) -> np.ndarray:
    """Build ``T_world_local`` from position and a wxyz quaternion."""
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError(f"position must have shape (3,), got {position.shape}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix_from_quaternion_wxyz(orientation_wxyz)
    transform[:3, 3] = position
    return transform


def look_at_quaternion_world(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Orient Isaac's world camera axes (+X forward, +Z up) at ``target``."""
    position = np.asarray(position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - position
    distance = float(np.linalg.norm(forward))
    if distance <= 1e-9:
        raise ValueError("camera position and target must differ")
    forward /= distance

    up_reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, up_reference))) > 0.98:
        up_reference = np.array([0.0, 1.0, 0.0])
    left = np.cross(up_reference, forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    rotation = np.column_stack((forward, left, up))
    return quaternion_wxyz_from_rotation_matrix(rotation)


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a rigid homogeneous transform to an ``(N, 3)`` point cloud."""
    transform = np.asarray(transform, dtype=np.float64)
    points = np.asarray(points)
    if transform.shape != (4, 4):
        raise ValueError(f"transform must be 4x4, got {transform.shape}")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    result = points @ transform[:3, :3].T + transform[:3, 3]
    return result.astype(points.dtype, copy=False)
