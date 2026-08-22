import tempfile
import unittest
from pathlib import Path

import numpy as np

from panda_handover.segmentation import (
    InstanceSegmentation,
    point_statistics,
    save_segmentation_artifacts,
    select_masked_points,
    write_ascii_ply,
)


class SegmentationTests(unittest.TestCase):
    def test_instance_prediction_requires_original_resolution_masks(self):
        prediction = InstanceSegmentation(
            masks=np.zeros((1, 2, 3), dtype=bool),
            boxes_xyxy=np.zeros((1, 4), dtype=np.float32),
            scores=np.ones(1, dtype=np.float32),
        )
        prediction.validate((2, 3))
        with self.assertRaisesRegex(ValueError, "masks must have shape"):
            prediction.validate((3, 2))

    def test_mask_selects_pixel_aligned_camera_and_world_points(self):
        depth = np.ones((2, 2), dtype=np.float32)
        camera = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        world = camera + 100.0
        rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        mask = np.array([[False, True], [True, False]])
        valid, camera_selected, world_selected, colors = select_masked_points(
            mask, depth, camera, world, rgb
        )
        self.assertTrue(np.array_equal(valid, mask))
        self.assertTrue(np.array_equal(camera_selected, camera[mask]))
        self.assertTrue(np.array_equal(world_selected, world[mask]))
        self.assertTrue(np.array_equal(colors, rgb[mask]))

    def test_invalid_depth_is_removed_from_selected_points(self):
        depth = np.array([[1.0, np.nan]], dtype=np.float32)
        points = np.ones((1, 2, 3), dtype=np.float32)
        rgb = np.zeros((1, 2, 3), dtype=np.uint8)
        valid, camera_points, _, _ = select_masked_points(
            np.ones((1, 2), dtype=bool), depth, points, points, rgb
        )
        self.assertTrue(np.array_equal(valid, [[True, False]]))
        self.assertEqual(camera_points.shape, (1, 3))

    def test_point_statistics_and_ply_vertex_count(self):
        points = np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]], dtype=np.float32)
        colors = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        statistics = point_statistics(points)
        self.assertEqual(statistics["count"], 2)
        self.assertEqual(statistics["centroid_m"], [1.0, 2.0, 3.0])
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "cloud.ply"
            write_ascii_ply(path, points, colors)
            text = path.read_text(encoding="ascii")
            self.assertIn("element vertex 2\n", text)
            self.assertTrue(text.endswith("2 3 4 4 5 6\n"))

    def test_segmentation_artifacts_preserve_instances_and_masked_points(self):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb[0, 1] = [10, 20, 30]
        depth = np.ones((2, 2), dtype=np.float32)
        camera = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        world = camera + 10.0
        prediction = InstanceSegmentation(
            masks=np.array([[[False, True], [False, False]]]),
            boxes_xyxy=np.array([[1.0, 0.0, 2.0, 1.0]], dtype=np.float32),
            scores=np.array([0.9], dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            report = save_segmentation_artifacts(
                output,
                rgb=rgb,
                depth_m=depth,
                points_camera=camera,
                points_world=world,
                prediction=prediction,
                prompt="blue block",
                model_id="facebook/sam3",
                score_threshold=0.5,
                mask_threshold=0.5,
            )
            self.assertEqual(report["instance_count"], 1)
            self.assertEqual(report["valid_3d_pixels"], 1)
            self.assertTrue(report["automatic_checks_passed"])
            self.assertTrue(np.array_equal(np.load(output / "points_camera.npy"), camera[0, 1][None]))
            self.assertTrue((output / "overlay.png").is_file())
            self.assertTrue((output / "points_world.ply").is_file())


if __name__ == "__main__":
    unittest.main()
