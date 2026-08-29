import unittest

from panda_handover.physics_baselines import (
    FINGER_DRIVE_PRESETS,
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

    def test_unknown_preset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown finger-drive preset"):
            resolve_finger_drive_values("hammer-special")


if __name__ == "__main__":
    unittest.main()
