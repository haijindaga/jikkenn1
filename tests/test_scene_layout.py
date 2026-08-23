import unittest
from pathlib import Path

from panda_handover.scene_layout import DEFAULT_TABLETOP_LAYOUT, TabletopSceneLayout


class SceneLayoutTests(unittest.TestCase):
    def test_reviewed_layout_passes_all_checks(self):
        report = DEFAULT_TABLETOP_LAYOUT.validation_report()
        self.assertEqual(report["status"], "success")
        self.assertTrue(all(report["automatic_checks"].values()))
        self.assertEqual(DEFAULT_TABLETOP_LAYOUT.table_top_z_m, 0.0)

    def test_old_floor_mounted_layout_is_rejected(self):
        old_layout = TabletopSceneLayout(
            ground_z_m=0.0,
            table_center_m=(0.50, 0.0, 0.35),
            table_size_m=(0.80, 1.00, 0.70),
            target_center_m=(0.55, 0.20, 0.725),
            obstacle_center_m=(0.55, -0.20, 0.75),
        )
        report = old_layout.validation_report()
        self.assertEqual(report["status"], "failure")
        self.assertFalse(report["automatic_checks"]["robot_mount_matches_tabletop"])
        self.assertFalse(report["automatic_checks"]["floor_is_below_table"])

    def test_capture_script_creates_only_the_lowered_ground_plane(self):
        script = (
            Path(__file__).resolve().parents[1] / "scripts" / "isaac_capture_smoke.py"
        ).read_text(encoding="utf-8")
        self.assertEqual(script.count("add_default_ground_plane("), 1)
        self.assertIn(
            "add_default_ground_plane(z_position=LAYOUT.ground_z_m)", script
        )

    def test_capture_script_registers_src_before_package_import(self):
        script = (
            Path(__file__).resolve().parents[1] / "scripts" / "isaac_capture_smoke.py"
        ).read_text(encoding="utf-8")
        path_registration = 'sys.path.insert(0, str(repo_root / "src"))'
        layout_import = (
            "from panda_handover.scene_layout import DEFAULT_TABLETOP_LAYOUT"
        )
        self.assertLess(script.index(path_registration), script.index(layout_import))


if __name__ == "__main__":
    unittest.main()
