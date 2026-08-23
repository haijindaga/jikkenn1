import unittest

import numpy as np

from panda_handover.map_diagnostics import (
    backproject_depth_to_frame,
    nearest_point_to_spheres,
)


class MapDiagnosticsTests(unittest.TestCase):
    def test_backprojects_pixel_centres_and_transform(self):
        depth = np.array([[2.0, 0.0], [4.0, np.nan]], dtype=np.float32)
        intrinsics = np.array(
            [[2.0, 0.0, 0.5], [0.0, 4.0, 0.5], [0.0, 0.0, 1.0]]
        )
        transform = np.eye(4)
        transform[:3, 3] = [1.0, 2.0, 3.0]
        points, pixels = backproject_depth_to_frame(depth, intrinsics, transform)
        np.testing.assert_allclose(
            points,
            [[1.0, 2.0, 5.0], [1.0, 3.0, 7.0]],
        )
        np.testing.assert_array_equal(pixels, [[0, 0], [1, 0]])

    def test_nearest_point_reports_surface_clearance(self):
        points = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        spheres = np.array([[1.0, 0.0, 0.0, 0.25], [2.8, 0.0, 0.0, 0.5]])
        distance, clearance, indices = nearest_point_to_spheres(
            points, spheres, point_batch_size=1
        )
        np.testing.assert_allclose(distance, [1.0, 0.2], atol=1e-6)
        np.testing.assert_allclose(clearance, [0.75, -0.3], atol=1e-6)
        np.testing.assert_array_equal(indices, [0, 1])

    def test_negative_radius_is_rejected_by_geometry_helper(self):
        with self.assertRaisesRegex(ValueError, "radii must be non-negative"):
            nearest_point_to_spheres(
                np.array([[0.0, 0.0, 0.0]]),
                np.array([[0.0, 0.0, 0.0, -1.0]]),
            )


if __name__ == "__main__":
    unittest.main()
