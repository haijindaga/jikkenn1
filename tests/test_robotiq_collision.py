from copy import deepcopy
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
from panda_handover.robotiq_collision import (
    open_snapshot_config, snapshot_meshes_in_hand, validated_spheres, vertex_coverage,
)
from panda_handover.robotiq_model import file_identity


def fixtures():
    names = list(ARM_JOINTS) + ["finger_joint"]
    root = "/World/Panda/Robotiq/"
    body = root + "base_link"
    path = body + "/geometry/mesh"
    t = np.eye(4)
    bodies = [{"path": "/World/Panda/panda_hand", "name": "panda_hand", "T_world_body": t.tolist()},
              {"path": body, "name": "base_link", "T_world_body": t.tolist()}]
    for i in range(8):
        bodies.append({"path": f"/World/Panda/panda_link{i}", "name": f"panda_link{i}", "T_world_body": t.tolist()})
    evidence = {"open_close_checks_passed": True, "rigid_bodies": bodies, "joint_names": names,
        "joint_positions": [0] * len(names), "collisions": [{"path": path, "enabled": True}],
        "joints": [{"path": body + "/AssemblerFixedJoint", "type": "PhysicsFixedJoint",
                    "body0": [bodies[0]["path"]], "body1": [body]}]}
    geometry = {"status": "authored_collision_geometry_exported", "frame": "installed Robotiq base_link",
        "joint_names": names, "joint_positions": [0] * len(names), "meshes": [{
            "path": path, "body": body, "physics_approximation": "convexHull",
            "vertices_gripper_base_m": [[0, 0, 0], [.01, 0, 0], [0, .01, 0], [0, 0, .01]],
            "face_vertex_counts": [3, 3, 3, 3], "face_vertex_indices": [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]}]}
    return evidence, geometry


def stock_config():
    links = [f"panda_link{i}" for i in range(8)] + ["panda_hand", "panda_leftfinger", "panda_rightfinger", "attached_object"]
    names = list(ARM_JOINTS) + ["panda_finger_joint1", "panda_finger_joint2"]
    return {"robot_cfg": {"kinematics": {
        "collision_spheres": {name: [{"center": [0, 0, 0], "radius": .01}] for name in links},
        "collision_link_names": links, "self_collision_buffer": {name: .02 for name in links},
        "self_collision_ignore": {"panda_link0": ["panda_link1", "panda_link2"],
            "panda_link5": ["panda_link6", "panda_link7", "panda_hand", "panda_leftfinger"],
            "panda_hand": ["panda_leftfinger", "panda_rightfinger", "attached_object"]},
        "cspace": {"joint_names": names, "cspace_distance_weight": list(range(9)),
            "null_space_weight": list(range(9)), "default_joint_position": list(range(9)),
            "max_acceleration": 15.0, "max_jerk": 500.0}}}}


class CollisionSnapshotTests(unittest.TestCase):
    def test_preparation_calls_official_voxel_fit_per_collider_and_keeps_sources(self):
        evidence, geometry = fixtures()
        # A second separate body/collider verifies that no hull fills the jaw gap.
        second = deepcopy(geometry["meshes"][0])
        second["path"] = second["path"].replace("base_link", "finger")
        second["body"] = second["body"].replace("base_link", "finger")
        geometry["meshes"].append(second)
        evidence["collisions"].append({"path": second["path"], "enabled": True})
        evidence["rigid_bodies"].append({"path": second["body"], "name": "finger", "T_world_body": np.eye(4).tolist()})
        calls = {"fit": [], "hulls": 0, "exports": []}
        class Tensor:
            def __init__(self, data): self.data = np.asarray(data)
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self.data
        class Mesh:
            def __init__(self, vertices, faces, process=False):
                self.vertices, self.faces = np.asarray(vertices), np.asarray(faces)
            @property
            def convex_hull(self):
                calls["hulls"] += 1
                return self
            def export(self, filename): calls["exports"].append(filename)
            def copy(self): return Mesh(self.vertices.copy(), self.faces.copy())
            def apply_transform(self, value): calls["canonical_transform"] = value
        def fit(mesh, **kwargs):
            calls["fit"].append(kwargs)
            return types.SimpleNamespace(centers=Tensor([[0, 0, 0]]), radii=Tensor([.02]))
        names = ("trimesh", "yaml", "curobo", "curobo._src", "curobo._src.geom",
                 "curobo._src.geom.sphere_fit", "curobo._src.geom.sphere_fit.types",
                 "curobo._src.geom.sphere_fit.fit_spheres")
        modules = {name: types.ModuleType(name) for name in names}
        modules["trimesh"].Trimesh = Mesh
        modules["trimesh"].util = types.SimpleNamespace(concatenate=lambda hulls: Mesh(
            np.concatenate([h.vertices for h in hulls]), np.empty((0, 3), dtype=int)))
        modules["yaml"].safe_load = json.loads
        modules["yaml"].safe_dump = lambda cfg, **kwargs: json.dumps(cfg)
        modules["curobo._src.geom.sphere_fit.types"].SphereFitType = types.SimpleNamespace(VOXEL="official-voxel")
        modules["curobo._src.geom.sphere_fit.fit_spheres"].fit_spheres_to_mesh = fit
        script = Path(__file__).resolve().parents[1] / "scripts/prepare_panda_robotiq_collision.py"
        spec = importlib.util.spec_from_file_location("test_prepare_collision", script)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ep, gp, up, canonical = (root / name for name in ("evidence.json", "geometry.json", "robot.urdf", "canonical.obj"))
            ep.write_text(json.dumps(evidence)); gp.write_text(json.dumps(geometry))
            up.write_text("URDF fixture"); canonical.write_text("canonical fixture")
            prepared = root / "prepared.json"
            prepared.write_text(json.dumps({"status": "fk_model_prepared", "sources": {"evidence": file_identity(ep)},
                "urdf": file_identity(up), "canonical_collision_mesh": file_identity(canonical),
                "T_grasp_panda_hand_proposed": np.eye(4).tolist()}))
            check = root / "tool_fk.json"
            check.write_text(json.dumps({"status": "tool_mount_alignment_passed",
                "inputs": {"prepared_model": str(prepared), "evidence": str(ep), "urdf": file_identity(up)},
                "frames": {name: {"passed": True} for name in [f"panda_link{i}" for i in range(8)] +
                           ["panda_hand", "robotiq_arg2f_base_link"]}}))
            stock = root / "ext/curobo/curobo/content/configs/robot/franka.yml"
            stock.parent.mkdir(parents=True); stock.write_text(json.dumps(stock_config()))
            identities = [file_identity(p) for p in (ep, gp, up, canonical, prepared, check, stock)]
            output = root / "collision"
            args = [str(script), "--graspgenx-root", str(root), "--geometry", str(gp),
                    "--prepared-model", str(prepared), "--tool-fk-check", str(check), "--output", str(output)]
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", args):
                self.assertEqual(module.main(), 0)
            report = json.loads((output / "collision_model_check.json").read_text())
            self.assertEqual(report["gripper_sphere_count"], 2)
            self.assertEqual(calls["hulls"], 2)
            self.assertEqual(calls["fit"], [{"sphere_density": 1.0, "fit_type": "official-voxel"}] * 2)
            self.assertFalse(report["safety"]["profile_ready"])
            self.assertFalse(report["safety"]["lift_supported"])
            self.assertFalse(report["safety"]["safe_to_plan"])
            self.assertEqual(identities, [file_identity(p) for p in (ep, gp, up, canonical, prepared, check, stock)])
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", args), self.assertRaises(SystemExit):
                module.main()

    def test_measured_open_snapshot_in_hand_and_complete_inventory(self):
        evidence, geometry = fixtures()
        pieces = snapshot_meshes_in_hand(geometry, evidence)
        np.testing.assert_allclose(pieces[0]["vertices"], geometry["meshes"][0]["vertices_gripper_base_m"])
        self.assertEqual(pieces[0]["faces"].shape, (4, 3))
        for error in ("closed", "state_mismatch", "missing", "duplicate", "approximation", "indices", "nan", "follower"):
            e, g = fixtures()
            if error == "closed":
                e["joint_positions"][-1] = g["joint_positions"][-1] = .8
            elif error == "state_mismatch":
                g["joint_positions"][0] = .01
            elif error == "missing":
                e["collisions"].append({"path": "/World/Panda/Robotiq/other/mesh", "enabled": True})
            elif error == "duplicate":
                g["meshes"].append(deepcopy(g["meshes"][0]))
            elif error == "approximation":
                g["meshes"][0]["physics_approximation"] = "convexDecomposition"
            elif error == "indices":
                g["meshes"][0]["face_vertex_indices"][0] = 50
            elif error == "nan":
                g["meshes"][0]["vertices_gripper_base_m"][0][0] = float("nan")
            else:
                e["joint_names"].append("follower")
                g["joint_names"] = e["joint_names"]
                e["joint_positions"].append(.8)
                g["joint_positions"] = e["joint_positions"]
            with self.subTest(error=error), self.assertRaises(ValueError):
                snapshot_meshes_in_hand(g, e)

    def test_coverage_is_metric_not_geometry_safety_claim(self):
        spheres = [{"center": [0, 0, 0], "radius": .02}]
        result = vertex_coverage(np.array([[0, 0, .01], [0, 0, .03]]), spheres)
        self.assertEqual(result["outside_vertex_count"], 1)
        self.assertAlmostEqual(result["maximum_vertex_outside_distance_m"], .01)
        self.assertFalse(result["volume_coverage_proven"])
        for invalid in ([], [{"center": [0, 0, 0], "radius": 0}], [{"center": [0, 0, float("nan")], "radius": 1}]):
            with self.assertRaises(ValueError):
                validated_spheres(invalid)

    def test_only_old_hand_spheres_replaced_and_no_blanket_arm_ignore(self):
        stock = stock_config()
        original = deepcopy(stock)
        spheres = [{"center": [0, .03, .1], "radius": .01}]
        result = open_snapshot_config(stock, Path("/tmp/diagnostic.urdf"), spheres)
        self.assertEqual(stock, original)
        kin = result["robot_cfg"]["kinematics"]
        for i in range(8):
            name = f"panda_link{i}"
            self.assertEqual(kin["collision_spheres"][name], stock["robot_cfg"]["kinematics"]["collision_spheres"][name])
        self.assertEqual(kin["collision_spheres"]["panda_hand"], spheres)
        self.assertEqual(kin["self_collision_ignore"]["panda_hand"], [])
        self.assertNotIn("panda_hand", kin["self_collision_ignore"]["panda_link0"])
        self.assertIn("panda_hand", kin["self_collision_ignore"]["panda_link5"])
        self.assertEqual(kin["self_collision_buffer"]["panda_hand"], 0)
        self.assertIsNone(kin["lock_joints"])
        self.assertEqual(kin["extra_links"], {})
        self.assertEqual(kin["cspace"]["joint_names"], list(ARM_JOINTS))
        self.assertEqual(kin["cspace"]["max_acceleration"], 15.0)
        self.assertEqual(kin["cspace"]["default_joint_position"], list(range(7)))
        self.assertEqual(kin["grasp_contact_link_names"], [])

    def test_runtime_check_uses_collision_enabled_loader_and_official_cost(self):
        evidence, _ = fixtures()
        spheres = [{"center": [0, .03, .1], "radius": .01}]
        configuration = open_snapshot_config(stock_config(), Path("/tmp/diagnostic.urdf"), spheres)
        calls = {}
        class Tensor:
            def __init__(self, data): self.data = np.asarray(data)
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self.data
        def load_cfg(filename, **kwargs):
            calls["cfg"] = kwargs
            self.assertNotIn("load_collision_spheres", kwargs)
            return object()
        class Kin:
            joint_names = list(ARM_JOINTS)
            def __init__(self, cfg): pass
            def compute_kinematics(self, js):
                calls["joint_state"] = js
                pose = types.SimpleNamespace(position=Tensor([[0, 0, 0]]), quaternion=Tensor([[1, 0, 0, 0]]))
                return types.SimpleNamespace(tool_poses=types.SimpleNamespace(get_link_pose=lambda name: pose),
                    robot_spheres=Tensor(np.array([[[[0, 0, 0, .01]] * 9]])))
            def get_self_collision_config(self):
                return types.SimpleNamespace(collision_pairs=Tensor([[0, 3], [1, 4]]))
        class Cost:
            def __init__(self, cfg): calls["cost_cfg"] = cfg
            def setup_batch_tensors(self, batch, horizon): calls["shape"] = (batch, horizon)
            def forward(self, spheres): return Tensor([[[calls.get("penetration", 0.0)]]])
        modules = {name: types.ModuleType(name) for name in (
            "torch", "yaml", "curobo", "curobo.kinematics", "curobo.types",
            "curobo._src.cost.cost_self_collision", "curobo._src.cost.cost_self_collision_cfg")}
        modules["torch"].as_tensor = lambda data, **kwargs: Tensor(data)
        modules["yaml"].safe_load = lambda text: configuration
        modules["curobo.kinematics"].Kinematics = Kin
        modules["curobo.kinematics"].KinematicsCfg = types.SimpleNamespace(from_robot_yaml_file=load_cfg)
        modules["curobo.types"].DeviceCfg = lambda: types.SimpleNamespace(device="cuda:0", dtype="float32")
        modules["curobo.types"].JointState = types.SimpleNamespace(from_position=lambda q, joint_names: (q, joint_names))
        modules["curobo._src.cost.cost_self_collision"].SelfCollisionCost = Cost
        modules["curobo._src.cost.cost_self_collision_cfg"].SelfCollisionCostCfg = lambda **kwargs: kwargs
        script = Path(__file__).resolve().parents[1] / "scripts/check_panda_robotiq_collision.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ep, yp, up = root / "evidence.json", root / "robot.yml", root / "robot.urdf"
            ep.write_text(json.dumps(evidence)); yp.write_text("fixture yaml"); up.write_text("fixture urdf")
            manifest = root / "collision_model_check.json"
            manifest.write_text(json.dumps({"status": "open_snapshot_collision_draft_prepared",
                "sources": {"evidence": file_identity(ep)}, "yaml": file_identity(yp), "urdf": file_identity(up),
                "colliders": [{"coverage": {"outside_vertex_count": 0}}]}))
            spec = importlib.util.spec_from_file_location("test_collision_preflight", script)
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            for name, penetration, status, code in (("passed", 0, "open_snapshot_preflight_passed", 0),
                ("rejected", .005, "open_snapshot_preflight_rejected", 2)):
                calls["penetration"] = penetration
                output = root / name
                with patch.dict(sys.modules, modules), patch.object(sys, "argv", [str(script),
                    "--collision-model", str(manifest), "--output", str(output)]):
                    self.assertEqual(module.main(), code)
                report = json.loads((output / "collision_preflight_check.json").read_text())
                self.assertEqual(report["status"], status)
                self.assertFalse(report["safety"]["profile_ready"])
                self.assertFalse(report["safety"]["safe_to_execute"])
            self.assertEqual(calls["shape"], (1, 1))
            self.assertEqual(calls["cost_cfg"]["weight"], 1.0)
            self.assertEqual(calls["joint_state"][0].numpy().shape, (1, 1, 7))
