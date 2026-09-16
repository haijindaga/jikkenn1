import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from panda_handover.gripper_swap import (
    ARM_JOINTS, GRIPPER_VARIANT, bounded_gripper_velocity,
    captured_arm_positions, gripper_control_spec, select_official_gripper,
)


class VariantSet:
    selected = "Default"

    def GetVariantNames(self):
        return ["Default", GRIPPER_VARIANT]

    def SetVariantSelection(self, selection):
        self.selected = selection
        return True

    def GetVariantSelection(self):
        return self.selected


class GripperSwapTests(unittest.TestCase):
    def properties(self, stiffness=0, damping=5000):
        dtype = [(name, float) for name in (
            "type", "lower", "upper", "stiffness", "damping", "maxVelocity", "maxEffort"
        )]
        return np.array([(1, 0, 0.8, stiffness, damping, 2, 5)], dtype=dtype)

    def test_official_variant_selection(self):
        variants = VariantSet()
        class Prim:
            def GetVariantSet(self, name):
                self.name = name
                return variants
        prim = Prim()
        select_official_gripper(prim)
        self.assertEqual(prim.name, "Gripper")
        self.assertEqual(variants.selected, GRIPPER_VARIANT)

    def test_missing_official_variant_is_not_silently_substituted(self):
        variants = VariantSet()
        variants.GetVariantNames = lambda: ["Default"]
        class Prim:
            def GetVariantSet(self, name):
                return variants
        with self.assertRaisesRegex(RuntimeError, "lacks"):
            select_official_gripper(Prim())

    def test_arm_capture_maps_by_name_and_ignores_original_fingers(self):
        names = ["panda_finger_joint1"] + list(reversed(ARM_JOINTS))
        q = [0.04] + list(range(7, 0, -1))
        np.testing.assert_array_equal(captured_arm_positions(names, q), range(1, 8))

    def test_invalid_capture_rejected(self):
        for names, q in ((["panda_joint1"], [0]), (list(ARM_JOINTS), [math.nan] * 7),
                         (list(ARM_JOINTS), [0] * 6), (["x", "x"], [0, 0])):
            with self.assertRaises(ValueError):
                captured_arm_positions(names, q)

    def test_velocity_drive_keeps_authored_gains(self):
        properties = self.properties()
        before = properties.copy()
        result = gripper_control_spec(["finger_joint"], properties, 3)
        self.assertEqual(result["mode"], "velocity")
        self.assertEqual(result["authored_damping"], 5000)
        self.assertLessEqual(result["diagnostic_speed_rad_s"], 2)
        np.testing.assert_array_equal(properties, before)

    def test_position_drive_uses_position_action(self):
        result = gripper_control_spec(["finger_joint"], self.properties(100, 10), 3)
        self.assertEqual(result["mode"], "position")

    def test_unknown_master_or_invalid_drive_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "master"):
            gripper_control_spec(["panda_finger_joint1"], self.properties(), 3)
        with self.assertRaisesRegex(RuntimeError, "gain"):
            gripper_control_spec(["finger_joint"], self.properties(0, 0), 3)
        properties = self.properties()
        properties["lower"] = -0.8
        with self.assertRaisesRegex(RuntimeError, "limits"):
            gripper_control_spec(["finger_joint"], properties, 3)

    def test_velocity_is_bounded_and_stops_at_target(self):
        self.assertEqual(bounded_gripper_velocity(0, 0.8, 0.5, 1 / 60), 0.5)
        self.assertEqual(bounded_gripper_velocity(0.8, 0, 0.5, 1 / 60), -0.5)
        self.assertEqual(bounded_gripper_velocity(0.8, 0.8, 0.5, 1 / 60), 0)

    def test_cli_preserves_source_and_refuses_existing_output(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "isaac_try_panda_robotiq.py"
        spec = importlib.util.spec_from_file_location("gripper_swap_script", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # Must not launch/import Isaac at import time.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene = root / "source.usda"
            scene.write_text("#usda 1.0\n")
            capture = root / "capture"
            capture.mkdir()
            (capture / "robot_state.json").write_text("{}")
            np.save(capture / "panda_joint_positions.npy", np.zeros(9))
            output = root / "output"
            argv = [str(script), "--scene-usd", str(scene), "--capture", str(capture),
                    "--output", str(output), "--simulation-only"]
            with patch("sys.argv", argv):
                self.assertEqual(module.parse_args().phase_frames, 180)
                output.mkdir()
                with self.assertRaises(SystemExit):
                    module.parse_args()
            self.assertEqual(scene.read_text(), "#usda 1.0\n")
