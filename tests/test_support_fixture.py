import unittest
from pathlib import Path

import numpy as np

from panda_handover.support_fixture import (
    masked_points_world,
    plan_handle_support,
    plan_part_support_patch,
)


class SupportFixtureTests(unittest.TestCase):
    def test_scene_builder_preserves_source_and_authors_static_support(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "isaac_create_elevated_head_scene.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"--base-scene"', source)
        self.assertIn('"--reference-segmentation"', source)
        self.assertIn('"--target-clearance-m"', source)
        self.assertIn('"--head-clearance-m"', source)
        self.assertIn('"--object-label"', source)
        self.assertIn('"--grasp-part-label"', source)
        self.assertIn('"--support-part-label"', source)
        self.assertIn('"--grasp-mask-role"', source)
        self.assertIn('"--support-mask-role"', source)
        self.assertIn('"--support-geometry-policy"', source)
        self.assertIn('/ args.grasp_mask_role', source)
        self.assertIn('/ args.support_mask_role', source)
        self.assertIn('"grasp_mask_role": args.grasp_mask_role', source)
        self.assertIn('"support_mask_role": args.support_mask_role', source)
        self.assertIn(
            '"support_geometry_policy": args.support_geometry_policy', source
        )
        self.assertIn("traceback.print_exc()", source)
        self.assertIn("UsdPhysics.CollisionAPI.Apply", source)
        self.assertIn('"panda_handover:diagnostic_support_prim"', source)
        self.assertIn('"support-part-supported-grasp-clear"', source)
        self.assertIn('"source_scene_modified": False', source)
        self.assertIn('"candidate_or_planner_parameters_changed": False', source)

        capture_source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "isaac_capture_smoke.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"handle-supported-head-clear"', capture_source)
        self.assertIn('"support-part-supported-grasp-clear"', capture_source)
        self.assertIn(
            '"target_remains_clear_of_table_after_settling"', capture_source
        )
        self.assertIn('"target_is_near_support_top_after_settling"', capture_source)

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

    def test_part_patch_uses_dense_support_region_clear_of_grasp(self) -> None:
        grasp = np.array(
            [
                [0.00, -0.01, 0.03],
                [0.00, 0.00, 0.03],
                [0.00, 0.01, 0.03],
            ]
        )
        support = np.array(
            [
                [0.05, -0.01, 0.02],
                [0.05, 0.00, 0.02],
                [0.05, 0.01, 0.02],
                [0.06, -0.01, 0.02],
                [0.06, 0.00, 0.02],
                [0.06, 0.01, 0.02],
            ]
        )
        plan = plan_part_support_patch(
            grasp,
            support,
            support_width_m=0.02,
            grasp_projection_margin_m=0.005,
            target_lift_m=0.03,
        )
        self.assertGreater(plan.support_center_xy_world_m[0], 0.04)
        self.assertEqual(plan.support_length_m, 0.02)
        self.assertGreater(
            plan.minimum_head_to_support_footprint_distance_m, 0.005
        )

    def test_part_patch_fails_without_grasp_clear_support_region(self) -> None:
        grasp = np.array(
            [[0.00, 0.00, 0.03], [0.01, 0.00, 0.03], [0.00, 0.01, 0.03]]
        )
        support = np.array(
            [[0.00, 0.00, 0.02], [0.01, 0.00, 0.02], [0.00, 0.01, 0.02]]
        )
        with self.assertRaisesRegex(ValueError, "no dense patch"):
            plan_part_support_patch(
                grasp,
                support,
                support_width_m=0.02,
                grasp_projection_margin_m=0.005,
                target_lift_m=0.03,
            )

    def test_part_patch_shrinks_to_mask_derived_safe_width(self) -> None:
        grasp = np.array(
            [[0.00, -0.001, 0.03], [0.00, 0.00, 0.03], [0.00, 0.001, 0.03]]
        )
        support = np.array(
            [
                [0.015, -0.004, 0.02],
                [0.015, 0.000, 0.02],
                [0.015, 0.004, 0.02],
                [0.017, -0.004, 0.02],
                [0.017, 0.000, 0.02],
                [0.017, 0.004, 0.02],
            ]
        )
        plan = plan_part_support_patch(
            grasp,
            support,
            support_width_m=0.03,
            grasp_projection_margin_m=0.005,
            target_lift_m=0.03,
        )
        self.assertLess(plan.support_width_m, 0.03)
        self.assertGreater(plan.support_width_m, 0.015)
        self.assertGreater(
            plan.minimum_head_to_support_footprint_distance_m, 0.005
        )


if __name__ == "__main__":
    unittest.main()
