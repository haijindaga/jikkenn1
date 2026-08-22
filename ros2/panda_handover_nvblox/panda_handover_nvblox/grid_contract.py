"""Pure geometry contract for Isaac ROS nvblox dense AABB responses."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
from pathlib import Path

import numpy as np


def fingerprint_files(files: Sequence[tuple[str, str | Path]]) -> str:
    """Hash ordered labelled files without depending on container mount paths."""
    digest = hashlib.sha256()
    for label, path_like in files:
        path = Path(path_like)
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def inclusive_voxel_center_aabb(
    center_m: Sequence[float], extent_m: Sequence[float], voxel_size_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return physical grid origin, first centre, and inclusive centre span.

    The official dense-grid conversion floors both AABB endpoints and includes
    both resulting indices.  Requesting first-to-last voxel centres therefore
    returns exactly ``extent / voxel_size`` cells without post-hoc cropping.
    """
    center = np.asarray(center_m, dtype=np.float64)
    extent = np.asarray(extent_m, dtype=np.float64)
    if center.shape != (3,) or extent.shape != (3,):
        raise ValueError("center_m and extent_m must contain 3 values")
    if voxel_size_m <= 0.0 or not np.isfinite(voxel_size_m):
        raise ValueError("voxel_size_m must be positive and finite")
    cells = np.rint(extent / voxel_size_m).astype(np.int64)
    if np.any(cells < 1) or not np.allclose(
        cells * voxel_size_m, extent, atol=1e-8, rtol=0.0
    ):
        raise ValueError("extent must be a positive integer number of voxels")
    origin = center - 0.5 * extent
    minimum_center = origin + 0.5 * voxel_size_m
    center_span = (cells - 1) * voxel_size_m
    return origin, minimum_center, center_span.astype(np.float64)
