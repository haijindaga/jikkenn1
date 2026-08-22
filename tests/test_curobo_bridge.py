import unittest

import numpy as np

from panda_handover.curobo_bridge import (
    extent_covers_requested,
    select_named_joint_positions,
    validate_mapping_inputs,
)


class CuroboBridgeTests(unittest.TestCase):
    def test_actual_esdf_extent_must_cover_requested_extent(self):
        self.assertTrue(extent_covers_requested([1.6, 1.61, 1.6], [1.6, 1.6, 1.6]))
        self.assertFalse(extent_covers_requested([1.28, 1.28, 1.28], [1.6, 1.6, 1.6]))

    def test_extent_validation_rejects_non_xyz_input(self):
        with self.assertRaisesRegex(ValueError, "contain 3 values"):
            extent_covers_requested([1.6, 1.6], [1.6, 1.6, 1.6])

    def test_joint_positions_are_selected_by_name_not_position(self):
        result = select_named_joint_positions(
            ["finger", "joint_2", "joint_1"],
            np.array([0.04, 2.0, 1.0]),
            ["joint_1", "joint_2"],
        )
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_missing_requested_joint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing requested joints"):
            select_named_joint_positions(["joint_1"], np.array([1.0]), ["joint_2"])

    def test_mapping_contract_accepts_pixel_aligned_capture(self):
        validate_mapping_inputs(
            depth_m=np.ones((2, 3), dtype=np.float32),
            rgb=np.zeros((2, 3, 3), dtype=np.uint8),
            intrinsics=np.eye(3),
            target_mask=np.array([[False, True, False], [False, False, False]]),
            T_robot_base_camera=np.eye(4),
        )

    def test_mapping_contract_rejects_empty_target(self):
        with self.assertRaisesRegex(ValueError, "target mask is empty"):
            validate_mapping_inputs(
                depth_m=np.ones((2, 3), dtype=np.float32),
                rgb=np.zeros((2, 3, 3), dtype=np.uint8),
                intrinsics=np.eye(3),
                target_mask=np.zeros((2, 3), dtype=bool),
                T_robot_base_camera=np.eye(4),
            )
