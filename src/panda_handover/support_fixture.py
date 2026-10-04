"""Geometry for a controlled handle-supported, head-clear diagnostic scene."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ElevatedHeadSupportPlan:
    """A support location derived from previously segmented 3-D parts."""

    support_center_xy_world_m: tuple[float, float]
    support_axis_xy_world: tuple[float, float]
    support_yaw_deg: float
    support_length_m: float
    head_center_xy_world_m: tuple[float, float]
    handle_center_xy_world_m: tuple[float, float]
    support_width_m: float
    head_projection_margin_m: float
    nearest_head_projection_distance_m: float
    minimum_head_to_support_footprint_distance_m: float
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

    The handle's principal horizontal axis defines a rail extending from just
    outside the segmented head projection toward the observed handle end.  Its
    footprint, plus the requested margin, must contain no observed head point.
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
    centered_handle = handle_xy - handle_center[None, :]
    covariance = centered_handle.T @ centered_handle / float(len(handle_xy))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    if float(np.linalg.norm(axis)) <= 1e-8:
        raise ValueError("handle projection has no measurable principal axis")
    axis = axis / np.linalg.norm(axis)
    if float(np.dot(axis, handle_center - head_center)) < 0.0:
        axis = -axis
    lateral = np.array((-axis[1], axis[0]), dtype=np.float64)

    handle_axial = (handle_xy - head_center[None, :]) @ axis
    handle_lateral = (handle_xy - head_center[None, :]) @ lateral
    head_axial = (head_xy - head_center[None, :]) @ axis
    head_lateral = (head_xy - head_center[None, :]) @ lateral
    axial_start = max(
        float(np.quantile(handle_axial, 0.05)),
        float(np.max(head_axial) + head_projection_margin_m),
    )
    axial_end = float(np.quantile(handle_axial, 0.95))
    support_length_m = axial_end - axial_start
    if support_length_m <= max(0.01, support_width_m * 0.5):
        raise ValueError(
            "observed handle is too short to create a head-clear support rail"
        )
    lateral_center = float(np.median(handle_lateral))
    axial_center = 0.5 * (axial_start + axial_end)
    support_xy = (
        head_center + axial_center * axis + lateral_center * lateral
    )

    head_relative = head_xy - support_xy[None, :]
    head_support_axial = np.abs(head_relative @ axis)
    head_support_lateral = np.abs(head_relative @ lateral)
    head_overlaps_expanded_rail = (
        head_support_axial <= 0.5 * support_length_m + head_projection_margin_m
    ) & (
        head_support_lateral <= 0.5 * support_width_m + head_projection_margin_m
    )
    if np.any(head_overlaps_expanded_rail):
        raise ValueError("derived support rail overlaps the segmented head projection")

    # Report the nearest Euclidean XY distance for inspection.  The exact
    # non-overlap decision above is made in the rail's oriented local frame.
    nearest_head_distance = float(
        np.min(np.linalg.norm(head_xy - support_xy[None, :], axis=1))
    )
    outside_axial = np.maximum(
        head_support_axial - 0.5 * support_length_m, 0.0
    )
    outside_lateral = np.maximum(
        head_support_lateral - 0.5 * support_width_m, 0.0
    )
    minimum_footprint_distance = float(
        np.min(np.hypot(outside_axial, outside_lateral))
    )
    yaw_deg = float(np.degrees(np.arctan2(axis[1], axis[0])))

    return ElevatedHeadSupportPlan(
        support_center_xy_world_m=(float(support_xy[0]), float(support_xy[1])),
        support_axis_xy_world=(float(axis[0]), float(axis[1])),
        support_yaw_deg=yaw_deg,
        support_length_m=float(support_length_m),
        head_center_xy_world_m=(float(head_center[0]), float(head_center[1])),
        handle_center_xy_world_m=(float(handle_center[0]), float(handle_center[1])),
        support_width_m=float(support_width_m),
        head_projection_margin_m=float(head_projection_margin_m),
        nearest_head_projection_distance_m=nearest_head_distance,
        minimum_head_to_support_footprint_distance_m=minimum_footprint_distance,
        target_lift_m=float(target_lift_m),
    )
