import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "create_receiver_overlay.py"


def load_module():
    spec = importlib.util.spec_from_file_location("create_receiver_overlay", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReceiverOverlayTests(unittest.TestCase):
    def test_overlay_uses_measured_neo_bounds_without_modifying_sources(self):
        module = load_module()
        evidence = (
            REPO_ROOT
            / "config"
            / "receiver_assets"
            / "isaac_sim_5_1_1x_neo.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.usda"
            base.write_text(
                '#usda 1.0\n(defaultPrim = "World")\ndef Xform "World" {}\n',
                encoding="utf-8",
            )
            output = root / "with_receiver.usda"
            report = module.create_overlay(
                base_scene=base,
                output=output,
                evidence_path=evidence,
                receiver_root_xy=(-1.15, 0.0),
                receiver_yaw_deg=0.0,
                overwrite=False,
            )

            self.assertEqual(report["status"], "success")
            expected_z = -1.05 - (-0.8406724618970056)
            actual_z = report["placement"]["receiver_root_translation_world_m"][2]
            self.assertTrue(math.isclose(actual_z, expected_z, abs_tol=1e-12))
            self.assertTrue(
                report["automatic_checks"]["computed_lowest_z_matches_floor"]
            )
            self.assertFalse(report["composition"]["source_asset_modified"])
            self.assertFalse(report["composition"]["base_scene_modified"])
            self.assertFalse(report["composition"]["asset_rescaled"])
            self.assertFalse(report["composition"]["physics_apis_modified"])

            text = output.read_text(encoding="utf-8")
            self.assertIn("subLayers", text)
            self.assertIn("@base.usda@", text)
            self.assertIn("Isaac/Robots/1X/Neo/Neo.usd", text)
            self.assertIn('over "World"', text)
            self.assertIn('def Xform "Receiver"', text)
            self.assertIn('bool sourceAssetModified = false', text)

            saved_report = json.loads(
                output.with_suffix(".usda.check.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_report["status"], "success")

    def test_existing_output_is_not_overwritten_implicitly(self):
        module = load_module()
        evidence = (
            REPO_ROOT
            / "config"
            / "receiver_assets"
            / "isaac_sim_5_1_1x_neo.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.usda"
            output = root / "with_receiver.usda"
            base.write_text("#usda 1.0\n", encoding="utf-8")
            output.write_text("existing\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                module.create_overlay(
                    base_scene=base,
                    output=output,
                    evidence_path=evidence,
                    receiver_root_xy=(-1.15, 0.0),
                    receiver_yaw_deg=0.0,
                    overwrite=False,
                )


if __name__ == "__main__":
    unittest.main()
