#!/usr/bin/env python3
"""GPU regression test for the cuRobo Issue #699 rounding fix.

The synthetic grid deliberately reproduces a 120 * 0.01 float32 dimension,
whose stored ratio is 119.99999. The test then queries a blocked region in a
far x-slab and a known-free region through cuRobo's official voxel collision
kernel. A failed check aborts before any real map or trajectory is loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/curobo_voxel_fix_check.json"),
    )
    return parser.parse_args()


def verify_reviewed_source(curobo_source: Path) -> None:
    checker = Path(__file__).with_name("apply_curobo_voxel_round_fix.py")
    subprocess.run(
        [
            sys.executable,
            str(checker),
            "--source",
            str(curobo_source),
            "--check-only",
        ],
        check=True,
    )


def make_trigger_grid(torch, VoxelGrid, *, device: str):
    voxel_size = 0.01
    shape = (120, 120, 100)
    # Match the Mapper code path from Issue #699 instead of authoring ideal
    # Python float64 dimensions by hand.
    dims = [float(np.float32(n) * np.float32(voxel_size)) for n in shape]
    center = (0.0, 0.0, 0.0)
    box_center = (0.45, 0.0, 0.0)
    box_half = (0.03, 0.03, 0.03)

    axes = [
        torch.arange(n, device=device, dtype=torch.float32) for n in shape
    ]
    gx, gy, gz = torch.meshgrid(*axes, indexing="ij")
    coordinates = [
        center[axis]
        + (grid - (shape[axis] - 1) / 2.0) * voxel_size
        for axis, grid in enumerate((gx, gy, gz))
    ]
    distances = [
        (coordinates[axis] - box_center[axis]).abs() - box_half[axis]
        for axis in range(3)
    ]
    outside = torch.sqrt(sum(value.clamp(min=0) ** 2 for value in distances))
    inside = torch.stack(distances, dim=-1).max(dim=-1).values.clamp(max=0)
    sdf = (outside + inside).to(torch.float16)
    return VoxelGrid(
        name="issue_699_trigger",
        pose=[*center, 1.0, 0.0, 0.0, 0.0],
        dims=dims,
        voxel_size=voxel_size,
        feature_tensor=sdf,
        feature_dtype=torch.float16,
    ), shape


def launch_collision(torch, wp, voxel_data, spheres):
    from curobo._src.geom.collision.buffer_collision import CollisionBuffer
    from curobo._src.geom.collision.wp_collision_kernel import (
        sphere_obstacle_collision_kernel,
    )
    from curobo._src.types.device_cfg import DeviceCfg
    from curobo._src.util.warp import get_warp_device_stream

    device_cfg = DeviceCfg(device=spheres.device, dtype=torch.float32)
    batch, horizon, number, _ = spheres.shape
    buffer = CollisionBuffer.from_shape(spheres.shape, device_cfg)
    buffer.zero_()
    device, stream = get_warp_device_stream(spheres)
    data_wp = voxel_data.to_warp()
    spheres_wp = wp.from_torch(spheres.detach().view(-1, 4), dtype=wp.vec4)
    environment = torch.zeros(batch, dtype=torch.int32, device=spheres.device)
    weight = torch.tensor([1.0], dtype=torch.float32, device=spheres.device)
    activation = torch.tensor([0.02], dtype=torch.float32, device=spheres.device)
    wp.launch(
        kernel=sphere_obstacle_collision_kernel,
        dim=batch * horizon * number * voxel_data.max_n,
        inputs=[
            data_wp,
            spheres_wp,
            wp.from_torch(weight),
            wp.from_torch(activation),
            wp.from_torch(environment, dtype=wp.int32),
            wp.from_torch(buffer.distance.detach().view(-1)),
            wp.from_torch(buffer.gradient.detach().view(-1), dtype=wp.float32),
            batch,
            horizon,
            number,
            voxel_data.max_n,
            wp.uint8(0),
        ],
        stream=stream,
        device=device,
    )
    wp.synchronize_device(device)
    return buffer.distance.detach().cpu().numpy()


def main() -> int:
    args = parse_args()
    import curobo
    import torch
    import warp as wp
    from curobo._src.geom.data.data_voxel import VoxelData
    from curobo._src.geom.types import VoxelGrid
    from curobo._src.types.device_cfg import DeviceCfg

    if not torch.cuda.is_available():
        raise RuntimeError("cuRobo voxel regression requires CUDA")
    curobo_source = Path(curobo.__file__).resolve().parent.parent
    verify_reviewed_source(curobo_source)

    device = "cuda:0"
    grid, expected_shape = make_trigger_grid(torch, VoxelGrid, device=device)
    device_cfg = DeviceCfg(device=torch.device(device), dtype=torch.float32)
    voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)
    raw_params = voxel_data.params[0, 0, :3].detach().cpu().numpy()
    rounded_shape = np.rint(raw_params).astype(np.int64)
    truncated_shape = raw_params.astype(np.int64)

    spheres = torch.tensor(
        [[[[0.45, 0.0, 0.0, 0.005], [-0.45, 0.0, 0.0, 0.005]]]],
        dtype=torch.float32,
        device=device,
    )
    costs = launch_collision(torch, wp, voxel_data, spheres).reshape(-1)
    checks = {
        "trigger_contains_fractional_underflow": bool(
            np.any(truncated_shape != np.asarray(expected_shape))
        ),
        "rounded_shape_matches_feature_tensor": bool(
            np.array_equal(rounded_shape, expected_shape)
        ),
        "blocked_far_slab_has_collision_cost": bool(costs[0] > 0.0),
        "known_free_probe_has_zero_collision_cost": bool(abs(costs[1]) <= 1e-6),
    }
    report = {
        "status": "success" if all(checks.values()) else "failed_checks",
        "reference": {
            "issue": "https://github.com/NVlabs/curobo/issues/699",
            "curobo_commit": "057a96ffb1088531535f9915154f9d0dabd62428",
            "patched_source_sha256": (
                "bab10d99e555fe722f2c3d893425ea912"
                "6978238ee43b9f6c0e250875c10e004"
            ),
            "kernel": "sphere_obstacle_collision_kernel",
        },
        "trigger": {
            "expected_shape_xyz": list(expected_shape),
            "raw_float32_grid_params": raw_params.tolist(),
            "truncated_shape_xyz": truncated_shape.tolist(),
            "rounded_shape_xyz": rounded_shape.tolist(),
        },
        "probes": {
            "blocked_far_slab_cost": float(costs[0]),
            "known_free_cost": float(costs[1]),
        },
        "automatic_checks": checks,
        "safe_to_load_real_esdf": bool(all(checks.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not all(checks.values()):
        raise RuntimeError(f"cuRobo voxel regression failed; inspect {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
