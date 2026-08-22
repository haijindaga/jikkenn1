import numpy as np
import unittest

from panda_handover.geometry import (
    depth_mask_to_points,
    look_at_quaternion_world,
    matrix_from_pose,
    rotation_matrix_from_quaternion_wxyz,
    transform_points,
)


class GeometryTests(unittest.TestCase):
    def test_depth_back_projection_center_pixel_is_forward(self):
        depth = np.zeros((3, 3), dtype=np.float32)
        depth[1, 1] = 2.0
        intrinsics = np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]])
        points = depth_mask_to_points(depth, intrinsics)
        self.assertEqual(points.shape, (1, 3))
        self.assertTrue(np.allclose(points[0], [0.0, 0.0, 2.0]))

    def test_depth_back_projection_respects_mask_and_stride(self):
        depth = np.full((4, 4), 1.0, dtype=np.float32)
        mask = np.ones((4, 4), dtype=bool)
        intrinsics = np.array([[2.0, 0.0, 1.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]])
        points = depth_mask_to_points(depth, intrinsics, mask, stride=2)
        self.assertEqual(points.shape, (4, 3))

    def test_transform_points_applies_rotation_and_translation(self):
        transform = matrix_from_pose(
            np.array([1.0, 2.0, 3.0]),
            np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]),
        )
        result = transform_points(transform, np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
        self.assertTrue(np.allclose(result, [[1.0, 3.0, 3.0]], atol=1e-6))

    def test_look_at_world_axes_points_local_x_at_target(self):
        position = np.array([1.0, 0.0, 1.0])
        target = np.array([0.0, 0.0, 0.0])
        quaternion = look_at_quaternion_world(position, target)
        rotation = rotation_matrix_from_quaternion_wxyz(quaternion)
        expected_forward = (target - position) / np.linalg.norm(target - position)
        self.assertTrue(np.allclose(rotation[:, 0], expected_forward, atol=1e-6))
        self.assertTrue(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6))
