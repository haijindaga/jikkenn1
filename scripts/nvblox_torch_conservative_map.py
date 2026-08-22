#!/usr/bin/env python3
"""Backend A: conservative dense ESDF from official nvblox_torch TSDF fusion.

Inputs are the robot/target-filtered depth frames already written by
``curobo_map_capture.py``.  This script only builds and inspects a map; it does
not create a motion planner or execute a robot trajectory.
"""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import sys

import numpy as np


NVBLOX_VERSION_SERIES = "0.0.10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--prepared-map",
        type=Path,
        required=True,
        help="Output directory from curobo_map_capture.py",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--extent", type=float, nargs=3, default=(1.6, 1.6, 1.6))
    parser.add_argument("--grid-center", type=float, nargs=3, default=(0.5, 0.0, 0.75))
    parser.add_argument("--minimum-tsdf-weight", type=float, default=0.1)
    parser.add_argument("--max-distance", type=float, default=1.0)
    parser.add_argument("--query-batch-size", type=int, default=262144)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from panda_handover.conservative_esdf import (
        classify_known_free,
        conservative_esdf_checks,
        fingerprint_files,
        iter_voxel_centers,
        make_dense_grid_spec,
        signed_distance_from_known_free,
        validate_prepared_view_order,
    )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("nvblox_torch mapping requires CUDA")
    try:
        import nvblox_torch
        from nvblox_torch.mapper import Mapper, QueryType
        from nvblox_torch.mapper_params import (
            MapperParams,
            ProjectiveIntegratorParams,
            ViewCalculatorParams,
        )
        from nvblox_torch.sensor import Sensor
    except ImportError as error:
        raise RuntimeError("install the official nvblox_torch 0.0.10 wheel") from error

    installed_version = metadata.version("nvblox-torch")
    if not installed_version.startswith(NVBLOX_VERSION_SERIES):
        raise RuntimeError(
            "backend A is pinned to nvblox_torch 0.0.10; "
            f"found {installed_version}"
        )

    depth_paths = validate_prepared_view_order(args.prepared_map, args.capture)
    fingerprint_inputs = [("prepared_map_report", args.prepared_map / "esdf_check.json")]
    for index, (capture, depth_path) in enumerate(zip(args.capture, depth_paths)):
        fingerprint_inputs.extend(
            [
                (f"camera_{index}_mapping_depth", depth_path),
                (f"camera_{index}_intrinsics", capture / "intrinsics.npy"),
                (
                    f"camera_{index}_robot_base_camera",
                    capture / "T_robot_base_camera.npy",
                ),
            ]
        )
    input_fingerprint = fingerprint_files(fingerprint_inputs)

    spec = make_dense_grid_spec(args.extent, args.grid_center, args.voxel_size)
    projective_params = ProjectiveIntegratorParams()
    projective_params.projective_integrator_max_integration_distance_m = 3.0
    mapper_params = MapperParams()
    mapper_params.set_projective_integrator_params(projective_params)
    view_params = ViewCalculatorParams()
    view_params.raycast_subsampling_factor = 1
    view_params.workspace_bounds_type = "kBoundingBox"
    view_params.workspace_bounds_min_corner_x_m = spec.min_corner_m[0]
    view_params.workspace_bounds_max_corner_x_m = (
        spec.min_corner_m[0] + spec.extent_m[0]
    )
    view_params.workspace_bounds_min_corner_y_m = spec.min_corner_m[1]
    view_params.workspace_bounds_max_corner_y_m = (
        spec.min_corner_m[1] + spec.extent_m[1]
    )
    view_params.workspace_bounds_min_height_m = spec.min_corner_m[2]
    view_params.workspace_bounds_max_height_m = (
        spec.min_corner_m[2] + spec.extent_m[2]
    )
    mapper_params.set_view_calculator_params(view_params)
    mapper = Mapper(voxel_sizes_m=float(args.voxel_size), mapper_parameters=mapper_params)

    view_reports = []
    device = torch.device(args.device)
    if device.type != "cuda" or (device.index not in (None, 0)):
        raise ValueError("the official 0.0.10 wheel is validated here only on cuda:0")
    for index, (capture, depth_path) in enumerate(zip(args.capture, depth_paths)):
        depth = np.load(depth_path).astype(np.float32, copy=False)
        intrinsics = np.load(capture / "intrinsics.npy").astype(np.float32, copy=False)
        transform = np.load(capture / "T_robot_base_camera.npy").astype(
            np.float32, copy=False
        )
        if depth.ndim != 2 or intrinsics.shape != (3, 3) or transform.shape != (4, 4):
            raise ValueError(f"camera_{index}: malformed prepared RGB-D geometry")
        sensor = Sensor.from_camera_matrix(
            torch.from_numpy(intrinsics), width=depth.shape[1], height=depth.shape[0]
        )
        mapper.add_depth_frame(
            torch.from_numpy(depth).to(device=device, dtype=torch.float32).contiguous(),
            torch.from_numpy(transform).to(device="cpu", dtype=torch.float32).contiguous(),
            sensor,
        )
        view_reports.append(
            {
                "index": index,
                "capture": str(capture),
                "prepared_depth": str(depth_path),
                "valid_depth_pixels": int((np.isfinite(depth) & (depth > 0.0)).sum()),
            }
        )

    voxel_count = int(np.prod(spec.shape))
    tsdf_distance = np.empty(voxel_count, dtype=np.float32)
    tsdf_weight = np.empty(voxel_count, dtype=np.float32)
    for destination, centers in iter_voxel_centers(spec, batch_size=args.query_batch_size):
        query = torch.from_numpy(centers).to(device=device, dtype=torch.float32)
        query_output = torch.empty(
            (query.shape[0], 2), device=device, dtype=torch.float32
        )
        result = mapper.query_layer(
            QueryType.TSDF, query, output=query_output, mapper_id=0
        )
        result_np = result.detach().cpu().numpy()
        tsdf_distance[destination] = result_np[:, 0]
        tsdf_weight[destination] = result_np[:, 1]

    tsdf_distance = tsdf_distance.reshape(spec.shape)
    tsdf_weight = tsdf_weight.reshape(spec.shape)
    observed, known_free = classify_known_free(
        tsdf_distance,
        tsdf_weight,
        minimum_weight=args.minimum_tsdf_weight,
    )
    esdf = signed_distance_from_known_free(
        known_free,
        voxel_size_m=args.voxel_size,
        max_distance_m=args.max_distance,
    )
    checks = conservative_esdf_checks(observed, known_free, esdf)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "tsdf_distance_m.npy", tsdf_distance)
    np.save(output / "tsdf_weight.npy", tsdf_weight)
    np.save(output / "observed_mask.npy", observed)
    np.save(output / "known_free_mask.npy", known_free)
    np.save(output / "esdf_features.npy", esdf)

    unknown = ~observed
    report = {
        "status": "success" if all(checks.values()) else "failed_checks",
        "backend": "A_nvblox_torch_dense_edt",
        "reference": {
            "nvblox_torch_required_version_series": NVBLOX_VERSION_SERIES,
            "nvblox_torch_runtime_version": installed_version,
            "tsdf_api": "Mapper.add_depth_frame + Mapper.query_layer(QueryType.TSDF)",
            "distance_transform": "scipy.ndimage.distance_transform_edt",
        },
        "grid": {
            "shape_xyz": list(spec.shape),
            "voxel_size_m": spec.voxel_size_m,
            "extent_m": list(spec.extent_m),
            "center_robot_base_m": list(spec.center_m),
            "min_corner_robot_base_m": list(spec.min_corner_m),
            "index_order": "x_slowest_z_fastest",
            "sdf_sign": "positive_free_negative_blocked",
        },
        "parameters": {
            "maximum_integration_distance_m": 3.0,
            "raycast_subsampling_factor": 1,
            "workspace_bounds_type": "kBoundingBox",
            "minimum_tsdf_weight": args.minimum_tsdf_weight,
            "max_distance_m": args.max_distance,
            "input_frames": len(args.capture),
            "target_clear_applied": False,
        },
        "input_fingerprint_sha256": input_fingerprint,
        "counts": {
            "total_voxels": voxel_count,
            "observed_voxels": int(observed.sum()),
            "known_free_voxels": int(known_free.sum()),
            "unknown_voxels": int(unknown.sum()),
            "blocked_voxels": int((~known_free).sum()),
        },
        "fractions": {
            "observed": float(observed.mean()),
            "known_free": float(known_free.mean()),
            "unknown": float(unknown.mean()),
        },
        "views": view_reports,
        "automatic_checks": checks,
        "unknown_environment_contract": {
            "only_sensor_observed_space_can_be_free": True,
            "unobserved_space_is_blocked": True,
            "distance_to_unknown_boundary_recomputed": True,
            "target_is_currently_blocked": True,
        },
        "safe_to_plan": False,
        "next_gate": "Compare with backend B, then choose a target-clear proxy before planning",
    }
    report_path = output / "esdf_check.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"integrated views: {len(args.capture)}")
    print(f"grid voxels: {voxel_count}")
    print(f"observed/free/unknown: {observed.sum()}/{known_free.sum()}/{unknown.sum()}")
    print(f"saved: {report_path}")
    if not all(checks.values()):
        raise RuntimeError(f"conservative ESDF checks failed; inspect {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
