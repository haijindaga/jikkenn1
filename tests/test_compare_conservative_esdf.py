import importlib.util
from pathlib import Path
import unittest

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "compare_conservative_esdf.py"
SPEC = importlib.util.spec_from_file_location("compare_conservative_esdf_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CompareConservativeEsdfTests(unittest.TestCase):
    def test_comparison_separates_directional_free_disagreement(self):
        a = np.array([[[0.2, -0.1, 0.3, -0.2]]], dtype=np.float32)
        b = np.array([[[0.1, 0.1, -0.1, -0.3]]], dtype=np.float32)
        metrics, masks = MODULE.compare_fields(a, b, surface_band_m=0.02)
        self.assertAlmostEqual(metrics["sign_agreement_fraction"], 0.5)
        np.testing.assert_array_equal(
            masks["backend_a_only_free_mask"], [[[False, False, True, False]]]
        )
        np.testing.assert_array_equal(
            masks["backend_b_only_free_mask"], [[[False, True, False, False]]]
        )

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "shapes must match"):
            MODULE.compare_fields(
                np.zeros((2, 2, 2)), np.zeros((2, 2, 3)), surface_band_m=0.0
            )

    def test_grid_match_tolerates_ros_float32_metadata(self):
        grid = {
            "shape_xyz": [160, 160, 160],
            "voxel_size_m": 0.01,
            "extent_m": [1.6, 1.6, 1.6],
            "center_robot_base_m": [0.5, 0.0, 0.75],
            "min_corner_robot_base_m": [-0.3, -0.8, -0.05],
        }
        ros_grid = dict(grid)
        ros_grid["voxel_size_m"] = np.float32(0.01).item()
        self.assertTrue(MODULE.grids_match(grid, ros_grid))
        ros_grid["center_robot_base_m"] = [0.51, 0.0, 0.75]
        self.assertFalse(MODULE.grids_match(grid, ros_grid))

    def test_input_fingerprint_must_exist_and_match(self):
        fingerprint = "a" * 64
        self.assertTrue(
            MODULE.input_fingerprints_match(
                {"input_fingerprint_sha256": fingerprint},
                {"input_fingerprint_sha256": fingerprint},
            )
        )
        self.assertFalse(MODULE.input_fingerprints_match({}, {}))
        self.assertFalse(
            MODULE.input_fingerprints_match(
                {"input_fingerprint_sha256": fingerprint},
                {"input_fingerprint_sha256": "b" * 64},
            )
        )


if __name__ == "__main__":
    unittest.main()
