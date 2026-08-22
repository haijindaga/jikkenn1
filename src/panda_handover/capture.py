"""Validated, portable RGB-D capture artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RgbdCapture:
    """One registered RGB-D frame and its calibrated world transform."""

    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    T_world_camera: np.ndarray
    camera_name: str = "camera_0"
    frame_convention: str = "opencv_optical_x_right_y_down_z_forward"

    def validate(self) -> None:
        rgb = np.asarray(self.rgb)
        depth = np.asarray(self.depth_m)
        intrinsics = np.asarray(self.intrinsics)
        transform = np.asarray(self.T_world_camera)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"rgb must be HxWx3, got {rgb.shape}")
        if rgb.dtype != np.uint8:
            raise ValueError(f"rgb must be uint8, got {rgb.dtype}")
        if depth.shape != rgb.shape[:2]:
            raise ValueError(f"depth shape {depth.shape} does not match rgb {rgb.shape[:2]}")
        if not np.issubdtype(depth.dtype, np.floating):
            raise ValueError(f"depth must be floating-point metres, got {depth.dtype}")
        if intrinsics.shape != (3, 3):
            raise ValueError(f"intrinsics must be 3x3, got {intrinsics.shape}")
        if transform.shape != (4, 4):
            raise ValueError(f"T_world_camera must be 4x4, got {transform.shape}")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
            raise ValueError("T_world_camera has an invalid homogeneous last row")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("T_world_camera rotation is not orthonormal")
        if not self.camera_name:
            raise ValueError("camera_name must not be empty")

    def save(self, root: str | Path, *, write_previews: bool = True) -> Path:
        """Save lossless arrays and optional human-readable PNG previews."""
        self.validate()
        output = Path(root) / self.camera_name
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / "rgb.npy", self.rgb)
        np.save(output / "depth_m.npy", self.depth_m.astype(np.float32, copy=False))
        np.save(output / "intrinsics.npy", self.intrinsics.astype(np.float64, copy=False))
        np.save(output / "T_world_camera.npy", self.T_world_camera.astype(np.float64, copy=False))

        valid_depth = np.asarray(self.depth_m)[np.isfinite(self.depth_m) & (self.depth_m > 0)]
        metadata = {
            "schema_version": 1,
            "camera_name": self.camera_name,
            "frame_convention": self.frame_convention,
            "rgb_shape": list(self.rgb.shape),
            "depth_shape": list(self.depth_m.shape),
            "depth_unit": "metre",
            "valid_depth_pixels": int(valid_depth.size),
            "depth_min_m": float(valid_depth.min()) if valid_depth.size else None,
            "depth_max_m": float(valid_depth.max()) if valid_depth.size else None,
        }
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        if write_previews:
            self._write_previews(output)
        return output

    def _write_previews(self, output: Path) -> None:
        try:
            import cv2
        except ImportError:
            return
        cv2.imwrite(str(output / "rgb.png"), self.rgb[:, :, ::-1])
        depth = np.asarray(self.depth_m, dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0.0)
        preview = np.zeros(depth.shape, dtype=np.uint8)
        if valid.any():
            lo, hi = np.percentile(depth[valid], [2.0, 98.0])
            if hi <= lo:
                hi = lo + 1e-3
            preview[valid] = np.clip((depth[valid] - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(255 - preview, cv2.COLORMAP_TURBO)
        colored[~valid] = 0
        cv2.imwrite(str(output / "depth_preview.png"), colored)

