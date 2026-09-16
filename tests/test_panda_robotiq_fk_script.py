import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from panda_handover.gripper_swap import ARM_JOINTS


class FKScriptTests(unittest.TestCase):
    def test_public_fk_contract_measured_named_joints_no_collision_or_motion(self):
        observed = {}
        class Tensor:
            def __init__(self, data):
                self.data = np.asarray(data)
            def detach(self):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return self.data
        def load_cfg(filename, **kwargs):
            observed.update(filename=filename, **kwargs)
            return object()
        class Kinematics:
            joint_names = list(ARM_JOINTS)
            def __init__(self, cfg):
                pass
            def compute_kinematics(self, js):
                observed["state"] = js
                def pose(name):
                    return types.SimpleNamespace(position=Tensor([[0, 0, int(name[-1])]]),
                                                 quaternion=Tensor([[1, 0, 0, 0]]))
                return types.SimpleNamespace(tool_poses=types.SimpleNamespace(get_link_pose=pose))
        fake_torch = types.ModuleType("torch")
        fake_torch.as_tensor = lambda data, **kwargs: Tensor(data)
        fake_kin = types.ModuleType("curobo.kinematics")
        fake_kin.Kinematics = Kinematics
        fake_kin.KinematicsCfg = types.SimpleNamespace(from_robot_yaml_file=load_cfg)
        fake_types = types.ModuleType("curobo.types")
        fake_types.JointState = types.SimpleNamespace(from_position=lambda position, joint_names: (position, joint_names))
        fake_types.DeviceCfg = lambda: types.SimpleNamespace(device="cuda:0", dtype="float32")
        modules = {"torch": fake_torch, "curobo": types.ModuleType("curobo"),
                   "curobo.kinematics": fake_kin, "curobo.types": fake_types}
        script = Path(__file__).resolve().parents[1] / "scripts" / "check_panda_robotiq_arm_fk.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = {"open_close_checks_passed": True, "joint_names": list(reversed(ARM_JOINTS)),
                        "joint_positions": list(range(7, 0, -1)), "rigid_bodies": []}
            for i in range(8):
                transform = np.eye(4)
                transform[:3, 3] = [5, 4, i]
                evidence["rigid_bodies"].append({"name": f"panda_link{i}", "T_world_body": transform.tolist()})
            path = root / "evidence.json"
            path.write_text(json.dumps(evidence))
            output = root / "check.json"
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", [str(script), "--evidence", str(path), "--output", str(output)]):
                spec = importlib.util.spec_from_file_location("test_robotiq_fk_script", script)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.assertEqual(module.main(), 0)
            self.assertFalse(observed["load_collision_spheres"])
            self.assertEqual(observed["filename"], "franka.yml")
            np.testing.assert_array_equal(observed["state"][0].numpy(), [list(range(1, 8))])
            self.assertEqual(observed["state"][1], list(ARM_JOINTS))
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "arm_alignment_passed")
            self.assertFalse(report["safety"]["profile_ready"])
            self.assertFalse(report["safety"]["robot_moved"])
            self.assertFalse(report["safety"]["grasp_to_tool_verified"])
