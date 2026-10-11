import unittest
from pathlib import Path

import numpy as np

from panda_handover.scene_rotation import (
    local_pose_from_world_poses,
    rotate_world_pose_about_own_origin,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class SceneRotationTests(unittest.TestCase):
    def test_world_yaw_rotates_orientation_without_moving_origin(self) -> None:
        transform = np.eye(4)
        transform[:3, :3] = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -2.0],
                [0.0, 0.5, 0.0],
            ]
        )
        transform[:3, 3] = [0.4, -0.2, 0.08]

        result = rotate_world_pose_about_own_origin(
            transform,
            yaw_degrees=180.0,
        )

        expected = transform.copy()
        expected[:3, :3] = np.diag([-1.0, -1.0, 1.0]) @ transform[:3, :3]
        np.testing.assert_allclose(result, expected, atol=1e-12)
        np.testing.assert_array_equal(result[:3, 3], transform[:3, 3])

    def test_local_pose_reconstructs_requested_world_pose(self) -> None:
        parent = np.eye(4)
        parent[:3, 3] = [1.0, 2.0, 3.0]
        child = np.eye(4)
        child[:3, 3] = [1.5, 2.5, 3.5]
        local = local_pose_from_world_poses(parent, child)
        np.testing.assert_allclose(parent @ local, child)

    def test_invalid_transform_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite 4x4"):
            rotate_world_pose_about_own_origin(
                np.zeros((3, 3)),
                yaw_degrees=180.0,
            )
        singular = np.zeros((4, 4))
        singular[3, 3] = 1.0
        with self.assertRaisesRegex(ValueError, "invertible"):
            local_pose_from_world_poses(singular, np.eye(4))

    def test_isaac_script_is_non_destructive_and_does_not_step_physics(self) -> None:
        source = (
            REPO_ROOT / "scripts" / "isaac_create_rotated_target_scene.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"--base-scene"', source)
        self.assertIn('default="/World/Objects/Target"', source)
        self.assertIn('default=180.0', source)
        self.assertIn('"--overwrite"', source)
        self.assertIn("from isaacsim import SimulationApp", source)
        self.assertLess(source.index("SimulationApp("), source.index("from pxr import"))
        self.assertIn('"disable_viewport_updates": True', source)
        self.assertIn("MakeMatrixXform", source)
        self.assertIn("GetRootLayer().Export", source)
        self.assertIn('"used_to_choose_rotation_axis": False', source)
        self.assertNotIn("requires a Z-up stage", source)
        self.assertNotIn("requires metre stage units", source)
        self.assertIn('"source_scene_modified": False', source)
        self.assertIn('"candidate_or_planner_parameters_changed": False', source)
        self.assertIn('"physics_or_robot_parameters_changed": False', source)
        self.assertIn("source_hash_after != source_hash_before", source)
        self.assertNotIn("simulation_app.update()", source)
        self.assertIn("simulation_app.close()", source)


if __name__ == "__main__":
    unittest.main()
