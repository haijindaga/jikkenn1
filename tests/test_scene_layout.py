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

    def test_capture_script_loads_authored_physics_ready_target(self):
        script = (
            Path(__file__).resolve().parents[1] / "scripts" / "isaac_capture_smoke.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"--scene-usd"', script)
        self.assertIn("stage_utils.open_stage(str(scene_usd))", script)
        self.assertIn("UsdPhysics.RigidBodyAPI", script)
        self.assertIn("UsdPhysics.CollisionAPI", script)
        self.assertIn("PhysxSchema.PhysxCollisionAPI", script)
        self.assertIn("UsdPhysics.MassAPI", script)
        self.assertIn("include_children=True", script)
        self.assertNotIn("032_knife.usd", script)

    def test_scene_editor_follows_separate_usd_authoring_stage_contract(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "isaac_edit_tabletop_scene.py"
        ).read_text(encoding="utf-8")
        self.assertIn('create_prim("/World/Objects", prim_type="Scope")', script)
        self.assertIn('"/World/Objects/Target"', script)
        self.assertIn("stage_utils.save_stage(str(output))", script)
        self.assertIn("get_timeline_interface().stop()", script)
        self.assertIn('"--overwrite"', script)
        self.assertIn("output already exists", script)
        self.assertNotIn("Save Flattened", script)

    def test_scene_editor_can_reference_and_place_target_noninteractively(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "isaac_edit_tabletop_scene.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"--target-usd"', script)
        self.assertIn('"--target-center-xy"', script)
        self.assertIn('"--target-clearance-m"', script)
        self.assertIn(
            'stage.DefinePrim("/World/Objects/Target/Asset", "Xform")', script
        )
        self.assertIn("GetReferences().AddReference(str(target_usd))", script)
        self.assertIn("os.path.relpath(target_usd, start=output.parent)", script)
        self.assertIn("save_stage resolves authored", script)
        self.assertIn('"/World/Objects/Target"', script)
        self.assertIn("compute_aabb(", script)
        self.assertIn("UsdPhysics.RigidBodyAPI", script)
        self.assertIn("PhysxSchema.PhysxCollisionAPI", script)
        self.assertIn("UsdPhysics.MassAPI", script)
        self.assertIn('"source_assets_are_not_modified": True', script)


if __name__ == "__main__":
    unittest.main()
