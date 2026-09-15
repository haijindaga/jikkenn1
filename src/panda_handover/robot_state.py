"""Validated robot state saved alongside an RGB-D capture."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RobotStateCapture:
    """Robot state needed by robot masking and motion planning.

    Joint positions stay in Isaac articulation order. Consumers must select
    joints by name because cuRobo may use a different ordering or omit fingers.
    """

    joint_names: tuple[str, ...]
    joint_positions: np.ndarray
    T_world_robot_base: np.ndarray
    prim_path: str = "/World/Panda"
    robot_profile: str = "franka_panda"
    tool_frame: str | None = None
    T_world_tool: np.ndarray | None = None

    def validate(self) -> None:
        positions = np.asarray(self.joint_positions)
        transform = np.asarray(self.T_world_robot_base)
        if not self.joint_names:
            raise ValueError("joint_names must not be empty")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be unique")
        if positions.shape != (len(self.joint_names),):
            raise ValueError(
                "joint_positions must have one value per joint name, got "
                f"{positions.shape} for {len(self.joint_names)} names"
            )
        if not np.issubdtype(positions.dtype, np.floating):
            raise ValueError(f"joint_positions must be floating-point, got {positions.dtype}")
        if not np.all(np.isfinite(positions)):
            raise ValueError("joint_positions contains non-finite values")
        if transform.shape != (4, 4):
            raise ValueError(f"T_world_robot_base must be 4x4, got {transform.shape}")
        if not np.all(np.isfinite(transform)):
            raise ValueError("T_world_robot_base contains non-finite values")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
            raise ValueError("T_world_robot_base has an invalid homogeneous last row")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("T_world_robot_base rotation is not orthonormal")
        if not self.prim_path:
            raise ValueError("prim_path must not be empty")
        if (self.tool_frame is None) != (self.T_world_tool is None):
            raise ValueError("tool_frame and T_world_tool must be supplied together")
        if self.T_world_tool is not None:
            tool_transform = np.asarray(self.T_world_tool)
            if tool_transform.shape != (4, 4):
                raise ValueError("T_world_tool must be 4x4")
            if not np.all(np.isfinite(tool_transform)):
                raise ValueError("T_world_tool contains non-finite values")
            if not np.allclose(tool_transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
                raise ValueError("T_world_tool has an invalid homogeneous last row")

    def save(self, capture_directory: str | Path, T_world_camera: np.ndarray) -> Path:
        """Save state and the camera pose expressed in the robot-base frame."""
        self.validate()
        output = Path(capture_directory)
        output.mkdir(parents=True, exist_ok=True)
        camera_transform = np.asarray(T_world_camera, dtype=np.float64)
        if camera_transform.shape != (4, 4):
            raise ValueError(f"T_world_camera must be 4x4, got {camera_transform.shape}")
        T_robot_base_camera = np.linalg.inv(self.T_world_robot_base) @ camera_transform

        positions = np.asarray(self.joint_positions, dtype=np.float64)
        np.save(output / "joint_positions.npy", positions)
        if self.robot_profile == "franka_panda":
            np.save(output / "panda_joint_positions.npy", positions)
        np.save(
            output / "T_world_robot_base.npy",
            np.asarray(self.T_world_robot_base, dtype=np.float64),
        )
        np.save(output / "T_robot_base_camera.npy", T_robot_base_camera)
        if self.T_world_tool is not None:
            T_robot_base_tool = (
                np.linalg.inv(self.T_world_robot_base)
                @ np.asarray(self.T_world_tool, dtype=np.float64)
            )
            np.save(output / "T_robot_base_tool.npy", T_robot_base_tool)
        report = {
            "schema_version": 2,
            "reference": "Isaac Sim Articulation joint state and world pose APIs",
            "robot": self.robot_profile,
            "prim_path": self.prim_path,
            "tool_frame": self.tool_frame,
            "joint_names": list(self.joint_names),
            "joint_position_unit": "radian_or_metre_by_joint_type",
            "transforms": {
                "T_world_robot_base.npy": "robot base to Isaac world",
                "T_robot_base_camera.npy": (
                    "OpenCV optical camera frame to robot base; required by cuRobo RobotSegmenter"
                ),
                "T_robot_base_tool.npy": (
                    "observed tool frame in robot base; cross-checks Isaac and cuRobo kinematics"
                    if self.T_world_tool is not None
                    else None
                ),
            },
            "safety": {
                "joint_mapping_must_use_names": True,
                "simulator_semantic_labels_used_for_robot_masking": False,
            },
        }
        path = output / "robot_state.json"
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return path


def load_robot_joint_positions(capture_directory: str | Path) -> np.ndarray:
    """Load the generic artifact, with a read-only legacy Panda fallback."""
    capture = Path(capture_directory)
    generic = capture / "joint_positions.npy"
    legacy = capture / "panda_joint_positions.npy"
    path = generic if generic.is_file() else legacy
    if not path.is_file():
        raise FileNotFoundError(
            f"robot joint positions do not exist in {capture}; expected {generic.name}"
        )
    return np.load(path, allow_pickle=False)
