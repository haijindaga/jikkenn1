"""Geometry-only diagnostics for filtered RGB-D maps and robot spheres."""

from __future__ import annotations

import numpy as np


def backproject_depth_to_frame(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    T_frame_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project valid OpenCV-optical depth pixels into ``frame``.

    Pixel centres follow the Isaac capture contract: ``(u + 0.5, v + 0.5)``.
    Returns the 3-D points and their integer ``(v, u)`` source pixels.
    """
    depth = np.asarray(depth_m)
    camera_matrix = np.asarray(intrinsics)
    transform = np.asarray(T_frame_camera)
    if depth.ndim != 2:
        raise ValueError(f"depth_m must be HxW, got {depth.shape}")
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"intrinsics must be 3x3, got {camera_matrix.shape}")
    if transform.shape != (4, 4):
        raise ValueError(f"T_frame_camera must be 4x4, got {transform.shape}")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError("T_frame_camera has an invalid homogeneous last row")

    valid = np.isfinite(depth) & (depth > 0.0)
    pixels_vu = np.column_stack(np.nonzero(valid))
    if pixels_vu.size == 0:
        raise ValueError("depth_m contains no valid depth pixels")
    z = depth[valid].astype(np.float64, copy=False)
    v = pixels_vu[:, 0].astype(np.float64)
    u = pixels_vu[:, 1].astype(np.float64)
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")
    points_camera = np.column_stack(
        (
            (u + 0.5 - cx) * z / fx,
            (v + 0.5 - cy) * z / fy,
            z,
        )
    )
    points_frame = (
        points_camera @ transform[:3, :3].astype(np.float64).T
        + transform[:3, 3].astype(np.float64)
    )
    return points_frame.astype(np.float32), pixels_vu.astype(np.int32)


def nearest_point_to_spheres(
    points_xyz: np.ndarray,
    spheres_xyzr: np.ndarray,
    *,
    point_batch_size: int = 32768,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the nearest measured point to every sphere centre.

    Returns centre-to-point distance, surface clearance ``distance - radius``,
    and the source point index for each sphere.
    """
    points = np.asarray(points_xyz, dtype=np.float64)
    spheres = np.asarray(spheres_xyzr, dtype=np.float64).reshape(-1, 4)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("points_xyz must be a non-empty Nx3 array")
    if spheres.ndim != 2 or spheres.shape[1] != 4 or len(spheres) == 0:
        raise ValueError("spheres_xyzr must be a non-empty Nx4 array")
    if np.any(spheres[:, 3] < 0.0):
        raise ValueError("sphere radii must be non-negative")
    if point_batch_size <= 0:
        raise ValueError("point_batch_size must be positive")

    best_squared = np.full(len(spheres), np.inf, dtype=np.float64)
    best_indices = np.full(len(spheres), -1, dtype=np.int64)
    centres = spheres[:, :3]
    for start in range(0, len(points), point_batch_size):
        stop = min(start + point_batch_size, len(points))
        delta = centres[:, None, :] - points[None, start:stop, :]
        squared = np.einsum("ijk,ijk->ij", delta, delta)
        local_indices = np.argmin(squared, axis=1)
        local_squared = squared[np.arange(len(spheres)), local_indices]
        improved = local_squared < best_squared
        best_squared[improved] = local_squared[improved]
        best_indices[improved] = start + local_indices[improved]

    distances = np.sqrt(best_squared)
    clearance = distances - spheres[:, 3]
    return (
        distances.astype(np.float32),
        clearance.astype(np.float32),
        best_indices,
    )
