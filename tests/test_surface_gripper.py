import unittest

import numpy as np

from panda_handover.surface_gripper import (
    EXPECTED_SURFACE_GRIPPER_ATTACHMENT_POINT_COUNT,
    rebase_attachment_point_grid,
)


class SurfaceGripperGridTests(unittest.TestCase):
    def test_rebase_preserves_official_grid_pairwise_distances(self) -> None:
        positions = np.array(
            [[x, y, 0.0] for x in (-0.01, 0.0, 0.01) for y in (-0.01, 0.0, 0.01)]
        )
        rotations = np.repeat(
            np.eye(3)[None, :, :],
            EXPECTED_SURFACE_GRIPPER_ATTACHMENT_POINT_COUNT,
            axis=0,
        )
        angle = np.deg2rad(37.0)
        destination_rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        center = np.array([0.12, -0.04, 0.3])

        moved_positions, moved_rotations = rebase_attachment_point_grid(
            positions, rotations, center, destination_rotation
        )

        np.testing.assert_allclose(moved_positions.mean(axis=0), center, atol=1e-12)
        np.testing.assert_allclose(moved_rotations[0], destination_rotation, atol=1e-12)
        source_distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
        moved_distances = np.linalg.norm(
            moved_positions[:, None] - moved_positions[None, :], axis=2
        )
        np.testing.assert_allclose(moved_distances, source_distances, atol=1e-12)

    def test_rebase_rejects_an_incomplete_template(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 9 positions"):
            rebase_attachment_point_grid(
                np.zeros((1, 3)),
                np.eye(3)[None, :, :],
                np.zeros(3),
                np.eye(3),
            )


if __name__ == "__main__":
    unittest.main()
