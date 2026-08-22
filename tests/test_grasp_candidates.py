import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from panda_handover.grasp_candidates import (
    T_GRASP_PANDA_HAND,
    pose_quality,
    prepare_scene_point_cloud,
    save_grasp_candidates,
    transform_grasp_poses,
)


class GraspCandidateTests(unittest.TestCase):
    def test_prepare_scene_preserves_organized_points_and_zeros_invalid_labels(self):
        points = np.array(
            [[[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan]],
             [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]],
            dtype=np.float32,
        )
        mask = np.array([[True, True], [False, True]])

        prepared, labels, count = prepare_scene_point_cloud(points, mask)

        self.assertTrue(np.array_equal(prepared, points, equal_nan=True))
        np.testing.assert_array_equal(labels, np.array([[1, 0], [0, 1]], dtype=np.int32))
        self.assertEqual(count, 2)

    def test_prepare_scene_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            prepare_scene_point_cloud(np.zeros((2, 3, 3)), np.zeros((2, 2)))

    def test_transform_chain_uses_official_panda_axis_offset(self):
        grasp_camera = np.eye(4, dtype=np.float32)[None]
        T_world_camera = np.eye(4)
        T_world_camera[:3, 3] = [1.0, 2.0, 3.0]

        world, hand = transform_grasp_poses(grasp_camera, T_world_camera)

        np.testing.assert_allclose(world[0], T_world_camera)
        np.testing.assert_allclose(hand[0], T_world_camera @ T_GRASP_PANDA_HAND)
        np.testing.assert_allclose(
            T_GRASP_PANDA_HAND[:3, :3],
            np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        )

    def test_pose_quality_reports_exact_rigid_pose(self):
        quality = pose_quality(np.eye(4)[None])
        self.assertTrue(quality["finite"])
        self.assertLessEqual(quality["max_rotation_orthogonality_error"], 1e-12)
        self.assertLessEqual(quality["max_rotation_determinant_error"], 1e-12)

    def test_save_marks_candidates_as_not_safe_to_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            report = save_grasp_candidates(
                Path(directory),
                grasps_camera=np.eye(4, dtype=np.float32)[None],
                scores=np.array([0.8], dtype=np.float32),
                branch_tags=["diff"],
                T_world_camera=np.eye(4),
                input_point_count=123,
                parameters={"planner": "graspmoe"},
                server_health={"status": "ok"},
                server_metadata={"precision": {"weights": "fp32"}},
            )

            self.assertEqual(report["status"], "success")
            self.assertFalse(report["safety"]["safe_to_execute"])
            saved = json.loads((Path(directory) / "graspgenx_check.json").read_text())
            self.assertEqual(saved["candidates"]["count"], 1)
            self.assertTrue((Path(directory) / "panda_hand_world.npy").is_file())


if __name__ == "__main__":
    unittest.main()
