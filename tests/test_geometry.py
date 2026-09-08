import numpy as np
import unittest

from panda_handover.geometry import (
    look_at_quaternion_world,
    matrix_from_pose,
    relative_pose,
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

    def test_relative_pose_reconstructs_child_world_pose(self):
        parent_position = np.array([0.4, -0.2, 0.7])
        parent_orientation = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
        child_position = np.array([0.55, -0.1, 0.75])
        child_orientation = np.array([1.0, 0.0, 0.0, 0.0])

        relative_position, relative_orientation, T_parent_child = relative_pose(
            parent_position,
            parent_orientation,
            child_position,
            child_orientation,
        )
        reconstructed = matrix_from_pose(
            parent_position, parent_orientation
        ) @ matrix_from_pose(relative_position, relative_orientation)

        np.testing.assert_allclose(
            reconstructed,
            matrix_from_pose(child_position, child_orientation),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            T_parent_child,
            matrix_from_pose(relative_position, relative_orientation),
            atol=1e-7,
        )
