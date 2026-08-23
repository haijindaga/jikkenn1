import unittest
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np

from panda_handover.conservative_esdf import (
    classify_known_free,
    conservative_esdf_checks,
    fingerprint_files,
    iter_voxel_centers,
    make_dense_grid_spec,
    optimistic_sim_esdf_checks,
    planning_free_from_unknown_policy,
    signed_distance_from_known_free,
    validate_prepared_view_order,
)


class ConservativeEsdfTests(unittest.TestCase):
    def test_optimistic_policy_frees_only_unknown(self):
        observed = np.array([[[False, True, True]]])
        sensor_free = np.array([[[False, True, False]]])
        planning_free = planning_free_from_unknown_policy(
            observed, sensor_free, unknown_policy="free"
        )
        np.testing.assert_array_equal(planning_free, [[[True, True, False]]])

    def test_optimistic_checks_reject_observed_obstacle_removal(self):
        observed = np.array([[[False, True, True]]])
        sensor_free = np.array([[[False, True, False]]])
        planning_free = np.array([[[True, True, False]]])
        esdf = np.array([[[0.05, 0.05, -0.05]]])
        self.assertTrue(
            all(
                optimistic_sim_esdf_checks(
                    observed, sensor_free, planning_free, esdf
                ).values()
            )
        )
        planning_free[0, 0, 2] = True
        esdf[0, 0, 2] = 0.05
        self.assertFalse(
            optimistic_sim_esdf_checks(
                observed, sensor_free, planning_free, esdf
            )["observed_blocked_remains_blocked"]
        )

    def test_grid_order_matches_x_slowest_z_fastest(self):
        spec = make_dense_grid_spec((0.2, 0.2, 0.2), (0.0, 0.0, 0.0), 0.1)
        centers = np.concatenate(
            [batch for _, batch in iter_voxel_centers(spec, batch_size=3)], axis=0
        )
        np.testing.assert_allclose(
            centers[:4],
            [
                [-0.05, -0.05, -0.05],
                [-0.05, -0.05, 0.05],
                [-0.05, 0.05, -0.05],
                [-0.05, 0.05, 0.05],
            ],
            atol=1e-7,
        )

    def test_non_integral_extent_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            make_dense_grid_spec((0.25, 0.2, 0.2), (0.0, 0.0, 0.0), 0.1)

    def test_unknown_never_becomes_known_free(self):
        distance = np.array([1.0, 1.0, -0.1, np.nan])
        weight = np.array([0.0, 0.2, 0.2, 0.2])
        observed, free = classify_known_free(distance, weight, minimum_weight=0.1)
        np.testing.assert_array_equal(observed, [False, True, True, False])
        np.testing.assert_array_equal(free, [False, True, False, False])

    def test_edt_places_zero_crossing_at_voxel_face(self):
        if importlib.util.find_spec("scipy") is None:
            self.skipTest("scipy is not installed in this test interpreter")
        free = np.zeros((5, 3, 3), dtype=bool)
        free[2:, :, :] = True
        sdf = signed_distance_from_known_free(free, voxel_size_m=0.1)
        self.assertAlmostEqual(float(sdf[1, 1, 1]), -0.05, places=6)
        self.assertAlmostEqual(float(sdf[2, 1, 1]), 0.05, places=6)
        self.assertAlmostEqual(float(sdf[3, 1, 1]), 0.15, places=6)

    def test_contract_detects_unknown_marked_free(self):
        observed = np.array([[[False, True]]])
        free = np.array([[[False, True]]])
        esdf = np.array([[[-0.05, 0.05]]])
        self.assertTrue(all(conservative_esdf_checks(observed, free, esdf).values()))
        esdf[0, 0, 0] = 0.05
        self.assertFalse(
            conservative_esdf_checks(observed, free, esdf)[
                "unknown_has_nonpositive_distance"
            ]
        )

    def test_prepared_view_order_is_enforced(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            captures = [root / "capture_0", root / "capture_1"]
            for capture in captures:
                capture.mkdir()
            prepared = root / "prepared"
            for index in range(2):
                view = prepared / "views" / f"camera_{index}"
                view.mkdir(parents=True)
                np.save(view / "mapping_depth_m.npy", np.ones((2, 2)))
            (prepared / "esdf_check.json").write_text(
                json.dumps({"views": [{"capture": str(path)} for path in captures]}),
                encoding="utf-8",
            )
            self.assertEqual(
                len(validate_prepared_view_order(prepared, captures)), 2
            )
            with self.assertRaisesRegex(ValueError, "was built from"):
                validate_prepared_view_order(prepared, list(reversed(captures)))

    def test_file_fingerprint_is_ordered_and_labelled(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first"
            second = root / "second"
            first.write_bytes(b"one")
            second.write_bytes(b"two")
            forward = fingerprint_files([("a", first), ("b", second)])
            self.assertEqual(forward, fingerprint_files([("a", first), ("b", second)]))
            self.assertNotEqual(
                forward, fingerprint_files([("b", second), ("a", first)])
            )


if __name__ == "__main__":
    unittest.main()
