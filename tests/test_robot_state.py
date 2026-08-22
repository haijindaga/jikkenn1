import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from panda_handover.robot_state import RobotStateCapture


def make_robot_state() -> RobotStateCapture:
    transform = np.eye(4)
    transform[:3, 3] = [0.1, -0.2, 0.3]
    return RobotStateCapture(
        joint_names=("panda_joint1", "panda_joint2", "panda_finger_joint1"),
        joint_positions=np.array([0.1, -0.2, 0.04], dtype=np.float64),
        T_world_robot_base=transform,
    )


class RobotStateCaptureTests(unittest.TestCase):
    def test_save_preserves_names_and_computes_robot_camera_transform(self):
        state = make_robot_state()
        T_world_camera = np.eye(4)
        T_world_camera[:3, 3] = [1.0, 2.0, 3.0]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            report_path = state.save(output, T_world_camera)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["joint_names"], list(state.joint_names))
            np.testing.assert_allclose(
                np.load(output / "T_robot_base_camera.npy"),
                np.linalg.inv(state.T_world_robot_base) @ T_world_camera,
            )
            np.testing.assert_allclose(
                np.load(output / "panda_joint_positions.npy"), state.joint_positions
            )

    def test_rejects_joint_count_mismatch(self):
        state = RobotStateCapture(
            joint_names=("panda_joint1",),
            joint_positions=np.array([0.0, 1.0]),
            T_world_robot_base=np.eye(4),
        )
        with self.assertRaisesRegex(ValueError, "one value per joint name"):
            state.validate()

    def test_rejects_duplicate_joint_names(self):
        state = RobotStateCapture(
            joint_names=("panda_joint1", "panda_joint1"),
            joint_positions=np.zeros(2),
            T_world_robot_base=np.eye(4),
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            state.validate()
