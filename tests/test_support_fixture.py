import unittest
from pathlib import Path

import numpy as np

from panda_handover.support_fixture import masked_points_world, plan_handle_support


class SupportFixtureTests(unittest.TestCase):
    def test_scene_builder_preserves_source_and_authors_static_support(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "isaac_create_elevated_head_scene.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"--base-scene"', source)
        self.assertIn('"--reference-segmentation"', source)
        self.assertIn('"parts" / "grasp_part" / "union_mask.npy"', source)
        self.assertIn('"parts" / "receive_part" / "union_mask.npy"', source)
        self.assertIn("UsdPhysics.CollisionAPI.Apply", source)
        self.assertIn('"source_scene_modified": False', source)
        self.assertIn('"candidate_or_planner_parameters_changed": False', source)

    def test_masked_points_are_transformed_to_world(self) -> None:
        points = np.array(
            [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0], [0.0, 0.0, 0.0]]
        )
        transform = np.eye(4)
        transform[:3, 3] = [1.0, 2.0, 3.0]
        result = masked_points_world(
            points,
            np.array([True, True, True, True]),
            transform,
        )
        np.testing.assert_allclose(
            result,
            [[1.0, 2.0, 4.0], [1.1, 2.0, 4.0], [1.2, 2.0, 4.0]],
        )

    def test_support_is_on_handle_and_clear_of_head_projection(self) -> None:
        head = np.array(
            [[0.00, -0.01, 0.03], [0.00, 0.00, 0.03], [0.00, 0.01, 0.03]]
        )
        handle = np.array(
            [[0.02, 0.00, 0.02], [0.04, 0.00, 0.02], [0.08, 0.00, 0.02]]
        )
        plan = plan_handle_support(
            head,
            handle,
            support_width_m=0.02,
            head_projection_margin_m=0.005,
            target_lift_m=0.03,
        )
        self.assertGreater(plan.support_center_xy_world_m[0], 0.02)
        self.assertAlmostEqual(plan.support_center_xy_world_m[1], 0.0)
        self.assertGreater(plan.support_length_m, 0.04)
        self.assertAlmostEqual(abs(plan.support_axis_xy_world[0]), 1.0)
        self.assertAlmostEqual(plan.support_axis_xy_world[1], 0.0)
        self.assertGreater(
            plan.minimum_head_to_support_footprint_distance_m, 0.005
        )
        self.assertEqual(plan.target_lift_m, 0.03)

    def test_support_fails_when_every_handle_point_overlaps_head(self) -> None:
        head = np.array(
            [[0.00, -0.01, 0.03], [0.00, 0.00, 0.03], [0.00, 0.01, 0.03]]
        )
        handle = np.array(
            [[0.001, 0.000, 0.02], [0.002, 0.000, 0.02], [0.003, 0.000, 0.02]]
        )
        with self.assertRaisesRegex(ValueError, "too short"):
            plan_handle_support(
                head,
                handle,
                support_width_m=0.02,
                head_projection_margin_m=0.005,
                target_lift_m=0.03,
            )


if __name__ == "__main__":
    unittest.main()
