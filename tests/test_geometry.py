import numpy as np
import unittest

from panda_handover.geometry import (
    look_at_quaternion_world,
    matrix_from_pose,
    rotation_matrix_from_quaternion_wxyz,
    transform_points,
)


class GeometryTests(unittest.TestCase):
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
