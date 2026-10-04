"""Geometry for a controlled handle-supported, head-clear diagnostic scene."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ElevatedHeadSupportPlan:
    """A support location derived from previously segmented 3-D parts."""

    support_center_xy_world_m: tuple[float, float]
    head_center_xy_world_m: tuple[float, float]
    handle_center_xy_world_m: tuple[float, float]
    support_width_m: float
    head_projection_margin_m: float
    nearest_head_projection_distance_m: float
    target_lift_m: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def masked_points_world(
    points_camera: np.ndarray,
    mask: np.ndarray,
    T_world_camera: np.ndarray,
) -> np.ndarray:
    """Return valid masked RGB-D points in the saved Isaac world frame."""

    points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    selected = np.asarray(mask, dtype=bool).reshape(-1)
    transform = np.asarray(T_world_camera, dtype=np.float64)
    if selected.shape != (len(points),):
        raise ValueError("mask and camera point cloud have different lengths")
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("T_world_camera must be a finite 4x4 matrix")
    valid = selected & np.isfinite(points).all(axis=1) & (points[:, 2] > 0.0)
    if np.count_nonzero(valid) < 3:
        raise ValueError("segmented part has fewer than three valid 3-D points")
    homogeneous = np.concatenate(
        (points[valid], np.ones((np.count_nonzero(valid), 1))), axis=1
    )
    world = (transform @ homogeneous.T).T[:, :3]
    if not np.all(np.isfinite(world)):
        raise ValueError("transformed segmented points are not finite")
    return world


def plan_handle_support(
    head_points_world: np.ndarray,
    handle_points_world: np.ndarray,
    *,
    support_width_m: float,
    head_projection_margin_m: float,
    target_lift_m: float,
) -> ElevatedHeadSupportPlan:
    """Place a square support on the handle without covering the head below.

    Candidate support centres are observed handle points.  The selected point
    is the closest one to the robust head centre whose square footprint, plus
    the requested margin, contains no observed head point.
    """

    head = np.asarray(head_points_world, dtype=np.float64)
    handle = np.asarray(handle_points_world, dtype=np.float64)
    for label, points in (("head", head), ("handle", handle)):
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
            raise ValueError(f"{label} points must have shape (N,3), N>=3")
        if not np.all(np.isfinite(points)):
            raise ValueError(f"{label} points contain non-finite values")
    if not np.isfinite(support_width_m) or support_width_m <= 0.0:
        raise ValueError("support_width_m must be positive and finite")
    if not np.isfinite(head_projection_margin_m) or head_projection_margin_m < 0.0:
        raise ValueError("head_projection_margin_m must be finite and non-negative")
    if not np.isfinite(target_lift_m) or target_lift_m <= 0.0:
        raise ValueError("target_lift_m must be positive and finite")

    head_xy = head[:, :2]
    handle_xy = handle[:, :2]
    head_center = np.median(head_xy, axis=0)
    handle_center = np.median(handle_xy, axis=0)
    required_linf_distance = 0.5 * support_width_m + head_projection_margin_m

    # L-infinity distance is the exact point-to-centre test for an axis-aligned
    # square footprint.  Chunking is unnecessary for the small SAM3 point sets.
    delta = np.abs(handle_xy[:, None, :] - head_xy[None, :, :])
    nearest_head_distance = np.min(np.max(delta, axis=2), axis=1)
    eligible = np.flatnonzero(nearest_head_distance > required_linf_distance)
    if not eligible.size:
        raise ValueError(
            "no observed handle point can support the target without covering "
            "the segmented head projection"
        )
    distances_to_head_center = np.linalg.norm(
        handle_xy[eligible] - head_center[None, :], axis=1
    )
    selected_index = int(eligible[np.argmin(distances_to_head_center)])
    support_xy = handle_xy[selected_index]

    return ElevatedHeadSupportPlan(
        support_center_xy_world_m=(float(support_xy[0]), float(support_xy[1])),
        head_center_xy_world_m=(float(head_center[0]), float(head_center[1])),
        handle_center_xy_world_m=(float(handle_center[0]), float(handle_center[1])),
        support_width_m=float(support_width_m),
        head_projection_margin_m=float(head_projection_margin_m),
        nearest_head_projection_distance_m=float(
            nearest_head_distance[selected_index]
        ),
        target_lift_m=float(target_lift_m),
    )
