"""Dense ESDF helpers shared by mapping backends.

The default conservative contract only frees voxels proven free by sensor
integration. An explicit simulation-only policy can instead assume unobserved
voxels are free while preserving observed obstacles. Distances follow the
cuRobo sign convention: positive free and negative blocked.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DenseGridSpec:
    """Axis-aligned voxel-centre grid in x-slowest, z-fastest order."""

    shape: tuple[int, int, int]
    voxel_size_m: float
    center_m: tuple[float, float, float]
    extent_m: tuple[float, float, float]
    min_corner_m: tuple[float, float, float]


def fingerprint_files(files: Sequence[tuple[str, str | Path]]) -> str:
    """Hash ordered, semantically labelled inputs independent of mount paths."""
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


def validate_prepared_view_order(
    prepared_map: str | Path, captures: Sequence[str | Path]
) -> list[Path]:
    """Validate that prepared filtered depths match captures in exact order.

    ``curobo_map_capture.py`` records its source paths in ``esdf_check.json``.
    Both conservative backends consume those prepared views; checking the
    recorded paths prevents a plausible-looking but invalid A/B comparison.
    """
    prepared = Path(prepared_map)
    capture_paths = [Path(path) for path in captures]
    if not capture_paths:
        raise ValueError("at least one capture is required")
    report_path = prepared / "esdf_check.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"missing prepared-map report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_views = report.get("views")
    if not isinstance(report_views, list) or len(report_views) != len(capture_paths):
        raise ValueError(
            "prepared-map view count does not match --capture count; "
            f"got {len(report_views) if isinstance(report_views, list) else 'invalid'} "
            f"and {len(capture_paths)}"
        )

    depth_paths = []
    for index, (capture, view) in enumerate(zip(capture_paths, report_views)):
        recorded = view.get("capture") if isinstance(view, dict) else None
        if not isinstance(recorded, str):
            raise ValueError(f"prepared-map view {index} has no recorded capture")
        if Path(recorded).resolve() != capture.resolve():
            raise ValueError(
                f"prepared-map view {index} was built from {recorded}, not {capture}"
            )
        depth_path = prepared / "views" / f"camera_{index}" / "mapping_depth_m.npy"
        if not depth_path.is_file():
            raise FileNotFoundError(f"missing prepared mapping depth: {depth_path}")
        depth_paths.append(depth_path)
    return depth_paths


def make_dense_grid_spec(
    extent_m: Sequence[float],
    center_m: Sequence[float],
    voxel_size_m: float,
) -> DenseGridSpec:
    """Create a grid compatible with cuRobo ``VoxelGrid`` indexing."""
    extent = np.asarray(extent_m, dtype=np.float64)
    center = np.asarray(center_m, dtype=np.float64)
    if extent.shape != (3,) or center.shape != (3,):
        raise ValueError("extent_m and center_m must each contain 3 values")
    if not np.all(np.isfinite(extent)) or not np.all(np.isfinite(center)):
        raise ValueError("grid values must be finite")
    if voxel_size_m <= 0.0 or not np.isfinite(voxel_size_m):
        raise ValueError("voxel_size_m must be positive and finite")
    if np.any(extent <= 0.0):
        raise ValueError("extent_m must be positive")

    shape_array = np.rint(extent / voxel_size_m).astype(np.int64)
    actual_extent = shape_array.astype(np.float64) * voxel_size_m
    if not np.allclose(actual_extent, extent, atol=1e-8, rtol=0.0):
        raise ValueError("each extent must be an integer multiple of voxel_size_m")
    min_corner = center - 0.5 * actual_extent
    return DenseGridSpec(
        shape=tuple(int(value) for value in shape_array),
        voxel_size_m=float(voxel_size_m),
        center_m=tuple(float(value) for value in center),
        extent_m=tuple(float(value) for value in actual_extent),
        min_corner_m=tuple(float(value) for value in min_corner),
    )


def iter_voxel_centers(
    spec: DenseGridSpec, *, batch_size: int
) -> Iterator[tuple[slice, np.ndarray]]:
    """Yield flattened voxel centres without allocating the full Nx3 grid."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    nx, ny, nz = spec.shape
    count = nx * ny * nz
    minimum = np.asarray(spec.min_corner_m, dtype=np.float64)
    voxel = spec.voxel_size_m
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        flat = np.arange(start, stop, dtype=np.int64)
        ix = flat // (ny * nz)
        remainder = flat % (ny * nz)
        iy = remainder // nz
        iz = remainder % nz
        indices = np.column_stack((ix, iy, iz))
        centers = minimum + (indices.astype(np.float64) + 0.5) * voxel
        yield slice(start, stop), centers.astype(np.float32)


def classify_known_free(
    tsdf_distance_m: np.ndarray,
    tsdf_weight: np.ndarray,
    *,
    minimum_weight: float,
    free_distance_threshold_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(observed, known_free)`` from official TSDF query channels."""
    distance = np.asarray(tsdf_distance_m)
    weight = np.asarray(tsdf_weight)
    if distance.shape != weight.shape:
        raise ValueError("TSDF distance and weight shapes must match")
    if minimum_weight <= 0.0 or not np.isfinite(minimum_weight):
        raise ValueError("minimum_weight must be positive and finite")
    if not np.isfinite(free_distance_threshold_m):
        raise ValueError("free_distance_threshold_m must be finite")
    observed = np.isfinite(distance) & np.isfinite(weight) & (weight >= minimum_weight)
    known_free = observed & (distance > free_distance_threshold_m)
    return observed, known_free


def planning_free_from_unknown_policy(
    observed: np.ndarray,
    sensor_known_free: np.ndarray,
    *,
    unknown_policy: str,
) -> np.ndarray:
    """Create the planner free mask without changing observed obstacles.

    ``blocked`` is the conservative contract used for real-world safety work.
    ``free`` mirrors nvblox's optimistic unobserved-space policy and is allowed
    only for explicitly labelled simulation experiments.
    """
    observed_array = np.asarray(observed, dtype=bool)
    sensor_free_array = np.asarray(sensor_known_free, dtype=bool)
    if observed_array.shape != sensor_free_array.shape:
        raise ValueError("observed and sensor_known_free shapes must match")
    if np.any(sensor_free_array & ~observed_array):
        raise ValueError("sensor_known_free contains unobserved voxels")
    if unknown_policy == "blocked":
        return sensor_free_array.copy()
    if unknown_policy == "free":
        return sensor_free_array | ~observed_array
    raise ValueError("unknown_policy must be 'blocked' or 'free'")


def signed_distance_from_known_free(
    known_free: np.ndarray,
    *,
    voxel_size_m: float,
    max_distance_m: float | None = None,
) -> np.ndarray:
    """Compute an exact Euclidean SDF with unknown space treated as blocked.

    SciPy's EDT measures centre-to-centre distances.  Subtracting half a voxel
    places the zero crossing at the shared face between adjacent free and
    blocked voxel centres, matching the grid geometry consumed by cuRobo.
    """
    free = np.asarray(known_free, dtype=bool)
    if free.ndim != 3 or 0 in free.shape:
        raise ValueError("known_free must be a non-empty 3-D array")
    if not free.any():
        raise ValueError("known_free contains no proven-free voxels")
    if free.all():
        raise ValueError("known_free contains no blocked boundary or obstacle")
    if voxel_size_m <= 0.0 or not np.isfinite(voxel_size_m):
        raise ValueError("voxel_size_m must be positive and finite")
    if max_distance_m is not None and (
        max_distance_m <= 0.0 or not np.isfinite(max_distance_m)
    ):
        raise ValueError("max_distance_m must be positive and finite")

    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as error:
        raise RuntimeError(
            "scipy is required for the exact Euclidean distance transform"
        ) from error

    half_voxel = 0.5 * voxel_size_m
    distance_in_free = distance_transform_edt(free, sampling=voxel_size_m)
    distance_in_blocked = distance_transform_edt(~free, sampling=voxel_size_m)
    sdf = np.where(
        free,
        distance_in_free - half_voxel,
        -(distance_in_blocked - half_voxel),
    )
    if max_distance_m is not None:
        sdf = np.clip(sdf, -max_distance_m, max_distance_m)
    return sdf.astype(np.float32)


def conservative_esdf_checks(
    observed: np.ndarray, known_free: np.ndarray, esdf_m: np.ndarray
) -> dict[str, bool]:
    """Check invariants that prevent unknown space from becoming free."""
    observed_array = np.asarray(observed, dtype=bool)
    free_array = np.asarray(known_free, dtype=bool)
    esdf = np.asarray(esdf_m)
    if observed_array.shape != free_array.shape or free_array.shape != esdf.shape:
        raise ValueError("observed, known_free and esdf_m shapes must match")
    unknown = ~observed_array
    return {
        "esdf_is_finite": bool(np.isfinite(esdf).all()),
        "known_free_is_observed": bool(np.all(~free_array | observed_array)),
        "known_free_has_positive_distance": bool(np.all(esdf[free_array] > 0.0)),
        "unknown_has_nonpositive_distance": bool(np.all(esdf[unknown] <= 0.0)),
        "grid_has_known_free": bool(free_array.any()),
        "grid_has_unknown": bool(unknown.any()),
    }


def optimistic_sim_esdf_checks(
    observed: np.ndarray,
    sensor_known_free: np.ndarray,
    planning_free: np.ndarray,
    esdf_m: np.ndarray,
) -> dict[str, bool]:
    """Check the explicit simulation-only unknown-as-free contract."""
    observed_array = np.asarray(observed, dtype=bool)
    sensor_free_array = np.asarray(sensor_known_free, dtype=bool)
    planning_free_array = np.asarray(planning_free, dtype=bool)
    esdf = np.asarray(esdf_m)
    if not (
        observed_array.shape
        == sensor_free_array.shape
        == planning_free_array.shape
        == esdf.shape
    ):
        raise ValueError("all optimistic ESDF arrays must have the same shape")
    unknown = ~observed_array
    observed_blocked = observed_array & ~sensor_free_array
    return {
        "esdf_is_finite": bool(np.isfinite(esdf).all()),
        "sensor_known_free_is_observed": bool(
            np.all(~sensor_free_array | observed_array)
        ),
        "all_unknown_is_planning_free": bool(np.all(planning_free_array[unknown])),
        "observed_blocked_remains_blocked": bool(
            np.all(~planning_free_array[observed_blocked])
        ),
        "planning_free_has_positive_distance": bool(
            np.all(esdf[planning_free_array] > 0.0)
        ),
        "observed_blocked_has_nonpositive_distance": bool(
            np.all(esdf[observed_blocked] <= 0.0)
        ),
        "grid_has_unknown": bool(unknown.any()),
        "grid_has_observed_obstacle": bool(observed_blocked.any()),
    }
