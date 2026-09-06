import unittest

from panda_handover.physics_baselines import (
    FINGER_DRIVE_PRESETS,
    drive_value_matches_float_storage,
    resolve_finger_drive_values,
)


class PhysicsBaselineTests(unittest.TestCase):
    def test_authored_usd_control_does_not_request_overrides(self):
        self.assertEqual(
            resolve_finger_drive_values("authored-usd"),
            {"max_force": None, "stiffness": None, "damping": None},
        )

    def test_isaaclab_franka_values_are_named_and_source_backed(self):
        self.assertEqual(
            resolve_finger_drive_values("isaaclab-franka"),
            {"max_force": 200.0, "stiffness": 2000.0, "damping": 100.0},
        )
        preset = FINGER_DRIVE_PRESETS["isaaclab-franka"]
        self.assertIn("NVlabs/RoboLab", preset.source)
        self.assertIn(preset.source_revision, preset.source)
        self.assertIn("not a calibration", preset.notes)

    def test_explicit_max_force_only_overrides_that_one_field(self):
        self.assertEqual(
            resolve_finger_drive_values(
                "isaaclab-franka", explicit_max_force=70.0
            ),
            {"max_force": 70.0, "stiffness": 2000.0, "damping": 100.0},
        )

    def test_diagnostic_scale_preserves_approximate_damping_ratio(self):
        values = resolve_finger_drive_values(
            "isaaclab-franka", diagnostic_scale=5.0
        )
        self.assertEqual(values["max_force"], 1000.0)
        self.assertEqual(values["stiffness"], 10000.0)
        self.assertAlmostEqual(values["damping"], 223.60679774997897)

    def test_cannot_scale_unresolved_authored_usd_values(self):
        with self.assertRaisesRegex(ValueError, "cannot be scaled"):
            resolve_finger_drive_values("authored-usd", diagnostic_scale=5.0)

    def test_invalid_diagnostic_scale_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            resolve_finger_drive_values("isaaclab-franka", diagnostic_scale=0.0)

    def test_drive_readback_accepts_openusd_float_rounding(self):
        requested = 223.60679774997897
        stored_float = 223.60679626464844
        self.assertTrue(
            drive_value_matches_float_storage(stored_float, requested)
        )
        self.assertFalse(
            drive_value_matches_float_storage(223.5, requested)
        )

    def test_unknown_preset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown finger-drive preset"):
            resolve_finger_drive_values("hammer-special")


if __name__ == "__main__":
    unittest.main()
