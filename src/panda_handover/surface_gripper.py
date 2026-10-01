"""Geometry helpers for the Isaac Sim Surface Gripper retention mode.

The runtime replay copies NVIDIA's bundled 3 x 3 attachment-point template.
This module only relocates that rigid grid; it does not invent attachment
points or tune candidate-specific physics parameters.
"""

from __future__ import annotations

import numpy as np


EXPECTED_SURFACE_GRIPPER_ATTACHMENT_POINT_COUNT = 9


def rebase_attachment_point_grid(
    source_positions: np.ndarray,
    source_rotations: np.ndarray,
    destination_center: np.ndarray,
    destination_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rigidly move the official attachment-point grid to a new pose.

    Positions and rotations are expressed in their respective Body0 frames.
    The first source point supplies the template reference orientation, while
    the arithmetic mean supplies the template center. Relative geometry is
    preserved exactly apart from floating-point roundoff.
    """

    source_positions = np.asarray(source_positions, dtype=np.float64)
    source_rotations = np.asarray(source_rotations, dtype=np.float64)
    destination_center = np.asarray(destination_center, dtype=np.float64)
    destination_rotation = np.asarray(destination_rotation, dtype=np.float64)

    expected_count = EXPECTED_SURFACE_GRIPPER_ATTACHMENT_POINT_COUNT
    if source_positions.shape != (expected_count, 3):
        raise ValueError(
            "official Surface Gripper template must contain exactly "
            f"{expected_count} positions, got {source_positions.shape}"
        )
    if source_rotations.shape != (expected_count, 3, 3):
        raise ValueError(
            "official Surface Gripper template must contain exactly "
            f"{expected_count} rotations, got {source_rotations.shape}"
        )
    if destination_center.shape != (3,):
        raise ValueError(
            f"destination_center must have shape (3,), got {destination_center.shape}"
        )
    if destination_rotation.shape != (3, 3):
        raise ValueError(
            "destination_rotation must have shape (3, 3), got "
            f"{destination_rotation.shape}"
        )
    if not all(
        np.isfinite(value).all()
        for value in (
            source_positions,
            source_rotations,
            destination_center,
            destination_rotation,
        )
    ):
        raise ValueError("Surface Gripper grid geometry must be finite")

    rotations_to_check = [*source_rotations, destination_rotation]
    for rotation in rotations_to_check:
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("Surface Gripper grid contains a non-orthonormal rotation")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError("Surface Gripper grid contains a non-rigid rotation")

    source_center = source_positions.mean(axis=0)
    source_reference_rotation = source_rotations[0]
    offsets_in_reference = (
        source_positions - source_center
    ) @ source_reference_rotation
    relative_rotations = np.einsum(
        "ij,njk->nik", source_reference_rotation.T, source_rotations
    )

    destination_positions = (
        destination_center + offsets_in_reference @ destination_rotation.T
    )
    destination_rotations = np.einsum(
        "ij,njk->nik", destination_rotation, relative_rotations
    )
    return destination_positions, destination_rotations
