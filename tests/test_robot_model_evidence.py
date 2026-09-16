import unittest
from unittest.mock import Mock

import numpy as np

from panda_handover.robot_model_evidence import arm_poses_in_base, collect_model_evidence, compare_arm_poses


class ModelEvidenceTests(unittest.TestCase):
    def evidence(self):
        bodies = []
        for i in range(8):
            value = np.eye(4)
            value[:3, 3] = [5, 4, i]
            bodies.append({"name": f"panda_link{i}", "T_world_body": value.tolist()})
        return {"rigid_bodies": bodies}

    def test_base_frame_conversion_and_exact_fk(self):
        observed = arm_poses_in_base(self.evidence())
        np.testing.assert_allclose(observed["panda_link7"][:3, 3], [0, 0, 7])
        self.assertTrue(all(r["passed"] for r in compare_arm_poses(observed, observed).values()))

    def test_rotation_mismatch_is_not_hidden_by_matching_positions(self):
        observed = arm_poses_in_base(self.evidence())
        predicted = {k: v.copy() for k, v in observed.items()}
        predicted["panda_link7"][:3, :3] = np.diag([-1, -1, 1])
        result = compare_arm_poses(observed, predicted)["panda_link7"]
        self.assertFalse(result["passed"])
        self.assertAlmostEqual(result["rotation_error_rad"], np.pi)

    def test_missing_duplicate_and_nonfinite_frames_rejected(self):
        for change in ("missing", "duplicate", "nonfinite"):
            evidence = self.evidence()
            if change == "missing":
                evidence["rigid_bodies"].pop()
            elif change == "duplicate":
                evidence["rigid_bodies"].append(evidence["rigid_bodies"][0])
            else:
                evidence["rigid_bodies"][0]["T_world_body"][0][0] = float("nan")
            with self.assertRaises(ValueError):
                arm_poses_in_base(evidence)

    def test_usd_inventory_does_not_reset_body_xforms_or_include_visual_names(self):
        try:
            from pxr import Usd, UsdGeom, UsdPhysics
        except ImportError:
            self.skipTest("USD bindings unavailable")
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/World/Panda")
        body = UsdGeom.Xform.Define(stage, "/World/Panda/panda_link0").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        mesh = UsdGeom.Cube.Define(stage, "/World/Panda/panda_link0/visuals/panda_link0").GetPrim()
        UsdPhysics.CollisionAPI.Apply(mesh)
        joint = UsdPhysics.FixedJoint.Define(stage, "/World/Panda/mount")
        joint.GetBody0Rel().SetTargets([body.GetPath()])
        wrapper = Mock()
        wrapper.get_world_pose.return_value = (np.zeros(3), np.array([1, 0, 0, 0]))
        factory = Mock(return_value=wrapper)
        before = stage.GetRootLayer().ExportToString()
        result = collect_model_evidence(stage, "/World/Panda", ["test_joint"], [0], factory)
        self.assertEqual(stage.GetRootLayer().ExportToString(), before)
        factory.assert_called_once_with(prim_path="/World/Panda/panda_link0", name="model_evidence_0", reset_xform_properties=False)
        self.assertEqual(len(result["rigid_bodies"]), 1)
        self.assertEqual(result["joints"][0]["body0"], ["/World/Panda/panda_link0"])
        self.assertEqual(len(result["collisions"]), 1)
        self.assertFalse(result["safety"]["profile_ready"])
