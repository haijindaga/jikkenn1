import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from panda_handover.gripper_swap import (
    ARM_JOINTS, GRIPPER_VARIANT, bounded_gripper_velocity,
    captured_arm_positions, create_variant_scene, gripper_control_spec,
    scene_invariants, select_official_gripper,
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
        return np.array([(0, 0, 0.8, stiffness, damping, 2, 5)], dtype=dtype)

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

    def test_isaac_51_translation_joint_is_not_treated_as_rotation(self):
        properties = self.properties()
        properties["type"] = 1
        with self.assertRaisesRegex(RuntimeError, "revolute"):
            gripper_control_spec(["finger_joint"], properties, 3)

    def test_invalid_effort_is_rejected_without_changing_properties(self):
        for value in (0, -1, math.nan):
            properties = self.properties()
            properties["maxEffort"] = value
            with self.assertRaisesRegex(RuntimeError, "effort"):
                gripper_control_spec(["finger_joint"], properties, 3)

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
        self.assertEqual(bounded_gripper_velocity(0.8 - 0.001, 0.8, 0.5, 1 / 60), 0)
        self.assertEqual(bounded_gripper_velocity(0.5, 0.8, 0.5, 1 / 60), 0.5)

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
                args = module.parse_args()
                self.assertEqual(args.phase_frames, 180)
                self.assertEqual(args.replay_physics, "default")
                self.assertFalse(args.inspect_only)
                with patch("sys.argv", argv + ["--inspect-only", "--replay-physics", "cpu"]):
                    args = module.parse_args()
                    self.assertTrue(args.inspect_only)
                    self.assertEqual(args.replay_physics, "cpu")
                output.mkdir()
                with self.assertRaises(SystemExit):
                    module.parse_args()
            self.assertEqual(scene.read_text(), "#usda 1.0\n")


@unittest.skipUnless(importlib.util.find_spec("pxr"), "USD runtime is not installed locally")
class VariantSceneUSDTests(unittest.TestCase):
    def test_variant_only_preserves_metadata_paths_and_environment(self):
        from pxr import Gf, Sdf, Usd, UsdGeom

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.usda"
            asset = Usd.Stage.CreateNew(str(root / "object.usda"))
            cube = UsdGeom.Cube.Define(asset, "/Cube")
            asset.SetDefaultPrim(cube.GetPrim())
            asset.GetRootLayer().Save()
            stage = Usd.Stage.CreateNew(str(source))
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            stage.SetTimeCodesPerSecond(60)
            stage.SetStartTimeCode(1)
            stage.SetEndTimeCode(180)
            stage.SetMetadata("customLayerData", {"experiment": "unchanged"})
            robot = UsdGeom.Xform.Define(stage, "/World/Panda")
            robot.AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))
            variants = robot.GetPrim().GetVariantSets().AddVariantSet("Gripper")
            for name in ("Default", GRIPPER_VARIANT):
                variants.AddVariant(name)
                variants.SetVariantSelection(name)
                with variants.GetVariantEditContext():
                    stage.DefinePrim("/World/Panda/" + name, "Xform")
            variants.SetVariantSelection("Default")
            UsdGeom.Camera.Define(stage, "/World/camera_0").CreateFocalLengthAttr(35)
            target = stage.DefinePrim("/World/Objects/Target")
            target.GetReferences().AddReference("object.usda")
            stage.GetRootLayer().Save()
            before_bytes = source.read_bytes()
            before = scene_invariants(stage)
            out = root / "outputs"
            out.mkdir()
            changed, robot_path, _ = create_variant_scene(stage, out / "scene.usda")
            reopened = Usd.Stage.Open(str(out / "scene.usda"))
            self.assertEqual(robot_path, "/World/Panda")
            self.assertEqual(scene_invariants(reopened), before)
            self.assertTrue(reopened.GetPrimAtPath("/World/Panda/" + GRIPPER_VARIANT))
            self.assertFalse(reopened.GetPrimAtPath("/World/Panda/Default"))
            self.assertEqual(reopened.GetPrimAtPath("/World/Objects/Target").GetTypeName(), "Cube")
            self.assertEqual(source.read_bytes(), before_bytes)
            self.assertEqual(stage.GetPrimAtPath(robot_path).GetVariantSet("Gripper").GetVariantSelection(), "Default")
            # The experiment root does not rebuild, rename, or deactivate a robot.
            self.assertEqual(changed.GetRootLayer().GetPrimAtPath(robot_path).specifier, Sdf.SpecifierOver)

    def test_y_up_source_is_rejected_without_modifying_it(self):
        from pxr import Usd, UsdGeom

        with tempfile.TemporaryDirectory() as temporary:
            stage = Usd.Stage.CreateNew(str(Path(temporary) / "source.usda"))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
            UsdGeom.SetStageMetersPerUnit(stage, 1)
            UsdGeom.Xform.Define(stage, "/World/Panda")
            with self.assertRaisesRegex(RuntimeError, "Z-up"):
                create_variant_scene(stage, Path(temporary) / "unused.usda")
            self.assertFalse((Path(temporary) / "unused.usda").exists())
