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
from panda_handover.robotiq_model import file_identity
from panda_handover.robotiq_trial import (
    candidate_in_world, capture_arm_matches, contact_waypoints, pick_result,
    rigid_transform, solve_waypoints, verified_trial_model,
)


def load_script(name):
    filename = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(filename.parent), *sys.path]):
        spec.loader.exec_module(module)
    return module


class TrialHelpersTests(unittest.TestCase):
    def test_waypoints_use_grasp_axis_before_hand_transform_and_world_z_lift(self):
        grasp = np.eye(4)
        grasp[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        grasp[:3, 3] = [.4, .2, .1]
        hand = np.eye(4)
        hand[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        goals = contact_waypoints(grasp, hand)
        np.testing.assert_allclose(goals["contact"][0, :3, 3], [.3, .2, .1])
        np.testing.assert_allclose(goals["contact"][-1], grasp @ hand)
        np.testing.assert_allclose(goals["lift"][-1, :3, 3], [.4, .2, .25])
        np.testing.assert_allclose(goals["lift"][0], goals["contact"][-1])
        for key in goals:
            self.assertLessEqual(np.max(np.linalg.norm(np.diff(goals[key][:, :3, 3], axis=0), axis=1)), .005 + 1e-10)
        with self.assertRaises(ValueError): contact_waypoints(grasp, hand, lift_m=-1)
        bad = hand.copy(); bad[0, 0] = 3
        with self.assertRaises(ValueError): rigid_transform(bad)

    def test_ik_uses_prior_warm_start_rejects_failures_limits_and_branch_jumps(self):
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        calls = []
        def solve(frame, position, rotation, warm_start):
            calls.append((frame, warm_start.copy()))
            return warm_start + .01, True
        solver = types.SimpleNamespace(compute_inverse_kinematics=solve)
        q, reason = solve_waypoints(solver, poses, np.zeros(7), -np.ones(7), np.ones(7))
        self.assertIsNone(reason)
        np.testing.assert_allclose(q[-1], .03)
        np.testing.assert_allclose(calls[1][1], .01)
        solver.compute_inverse_kinematics = lambda *a, **kw: (np.zeros(7), False)
        self.assertEqual(solve_waypoints(solver, poses, np.zeros(7), -np.ones(7), np.ones(7))[1], "ik_failed")
        solver.compute_inverse_kinematics = lambda *a, **kw: (np.ones(7)*2, True)
        self.assertEqual(solve_waypoints(solver, poses, np.zeros(7), -np.ones(7), np.ones(7))[1], "joint_limit_rejected")
        solver.compute_inverse_kinematics = lambda *a, **kw: (kw["warm_start"] + .4, True)
        self.assertEqual(solve_waypoints(solver, poses, np.zeros(7), -np.ones(7)*5, np.ones(7)*5)[1], "ik_branch_jump_rejected")
        solver.compute_inverse_kinematics = lambda *a, **kw: (np.full(7, np.nan), True)
        with self.assertRaises(RuntimeError): solve_waypoints(solver, poses, np.zeros(7), -np.ones(7), np.ones(7))

    def test_pick_requires_entire_hold_not_a_bounce(self):
        phase = ["lift", "lift", "hold", "hold"]
        points = np.zeros((4, 3)); points[:, 2] = [.1, .2, .2, .2]
        self.assertTrue(pick_result(phase, points, .05)["physical_pick_observed"])
        points[2, 2] = 0
        self.assertFalse(pick_result(phase, points, .05)["physical_pick_observed"])
        with self.assertRaises(ValueError): pick_result(["lift"], [[0, 0, .2]], 0)

    def test_candidate_gripper_camera_provenance_and_score_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera = np.repeat(np.eye(4)[None], 2, axis=0)
            camera[:, 0, 3] = [.1, .2]
            transform = np.eye(4); transform[1, 3] = .3
            np.save(root / "T_world_camera.npy", transform)
            np.save(root / "grasps_camera.npy", camera)
            np.save(root / "grasps_world.npy", transform @ camera)
            np.save(root / "scores.npy", [.5, .9])
            report = {"status": "success", "gripper": "robotiq_2f_85", "candidates": {"count": 2}}
            path = root / "graspgenx_check.json"
            path.write_text(json.dumps(report))
            pose, index, score = candidate_in_world(root, root, 0)
            self.assertEqual(index, 1)
            self.assertEqual(score, .9)
            np.testing.assert_allclose(pose[:3, 3], [.2, .3, 0])
            with self.assertRaises(ValueError): candidate_in_world(root, root, 2)
            report["gripper"] = "franka_panda"; path.write_text(json.dumps(report))
            with self.assertRaises(ValueError): candidate_in_world(root, root, 0)
            report["gripper"] = "robotiq_2f_85"; path.write_text(json.dumps(report))
            np.save(root / "T_world_camera.npy", np.eye(4))
            with self.assertRaises(ValueError): candidate_in_world(root, root, 0)

    def test_capture_preserves_measured_named_arm_posture(self):
        evidence = {"joint_names": list(ARM_JOINTS), "joint_positions": np.arange(7)*.01}
        names = list(reversed(ARM_JOINTS)) + ["panda_finger_joint1"]
        result = capture_arm_matches(evidence, names, [*.01*np.arange(6, -1, -1), 0])
        np.testing.assert_allclose(result, evidence["joint_positions"])
        with self.assertRaises(ValueError): capture_arm_matches(evidence, list(ARM_JOINTS), np.ones(7))

    def test_verified_sources_scene_and_transform_cannot_be_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene, urdf, mesh = (root / name for name in ("scene", "urdf", "mesh"))
            for path in (scene, urdf, mesh): path.write_text("unchanged")
            evidence_path = root / "evidence.json"
            evidence_path.write_text(json.dumps({"source_scene_sha256": file_identity(scene)["sha256"]}))
            transform = np.eye(4); transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
            prepared_path = root / "prepared.json"
            prepared = {"status": "fk_model_prepared", "sources": {"evidence": file_identity(evidence_path)},
                        "urdf": file_identity(urdf), "canonical_collision_mesh": file_identity(mesh),
                        "mesh_files": [], "T_grasp_panda_hand_proposed": transform.tolist()}
            prepared_path.write_text(json.dumps(prepared))
            fk_path = root / "fk.json"
            frames = [*(f"panda_link{i}" for i in range(8)), "panda_hand", "robotiq_arg2f_base_link"]
            fk = {"status": "tool_mount_alignment_passed", "frames": {name: {"passed": True} for name in frames},
                  "inputs": {"prepared_model": str(prepared_path), "evidence": str(evidence_path), "urdf": file_identity(urdf)}}
            fk_path.write_text(json.dumps(fk))
            with patch("panda_handover.robotiq_trial.installed_mount"):
                np.testing.assert_allclose(verified_trial_model(prepared_path, fk_path, scene)[1], transform)
                scene.write_text("changed")
                with self.assertRaises(ValueError): verified_trial_model(prepared_path, fk_path, scene)


class IsaacTrialContractTests(unittest.TestCase):
    def test_trial_uses_official_apis_only_on_arm_and_master_and_measures_hold(self):
        script = load_script("isaac_try_robotiq_pick")
        dtype = [(name, float) for name in ("lower", "upper", "stiffness", "damping", "maxEffort", "maxVelocity", "type")]
        properties = np.zeros(8, dtype=dtype)
        properties["lower"] = -3; properties["upper"] = 3
        properties["stiffness"] = 2; properties["damping"] = 1
        properties["maxEffort"] = 26; properties["maxVelocity"] = 1
        properties[7]["lower"] = 0; properties[7]["upper"] = np.deg2rad(47)
        calls = []
        class Action:
            def __init__(self, **kwargs): self.__dict__.update(kwargs)
        class Robot:
            def __init__(self, **kw): self.q = np.zeros(8); self.dof_names = list(ARM_JOINTS) + ["finger_joint"]; self.dof_properties = properties
            def set_joint_positions(self, q, joint_indices): self.q[joint_indices] = q
            def set_joint_velocities(self, *a, **kw): pass
            def get_joint_positions(self): return self.q.copy()
            def apply_action(self, action):
                calls.append(action)
                if getattr(action, "joint_positions", None) is not None: self.q[action.joint_indices] = action.joint_positions
        robot = Robot()
        class Body:
            def __init__(self, prim_path, **kw): self.path = prim_path
            def initialize(self): pass
            def get_world_pose(self):
                if "panda_link0" in self.path: return np.zeros(3), np.array([1, 0, 0, 0])
                if "Target" in self.path:
                    z = robot.q[2] if robot.q[7] > .5 else 0
                    return np.array([0, 0, z]), np.array([1, 0, 0, 0])
                return robot.q[:3].copy(), np.array([1, 0, 0, 0])
        class World:
            def __init__(self, **kw): self.scene = types.SimpleNamespace(add=lambda value: robot)
            def reset(self): pass
            def get_physics_context(self): return None
            def step(self, **kw): pass
            def is_playing(self): return True
        class Solver:
            def __init__(self, **kw): pass
            def get_joint_names(self): return list(ARM_JOINTS)
            def get_all_frame_names(self): return ["panda_hand"]
            def set_robot_base_pose(self, *args): calls.append("base_pose")
            def compute_forward_kinematics(self, name, q): return q[:3], np.eye(3)
            def compute_inverse_kinematics(self, name, position, orientation, warm_start):
                q = np.zeros(7); q[:3] = position
                return q, True
        class Trajectory:
            start_time = 0; end_time = .1
            def __init__(self, q): self.q = q
            def get_joint_targets(self, time):
                fraction = time / .1
                return self.q[0]*(1-fraction) + self.q[-1]*fraction, (self.q[-1]-self.q[0])/.1
        class Generator:
            def __init__(self, **kw): pass
            def get_active_joints(self): return list(ARM_JOINTS)
            def compute_c_space_trajectory(self, q): return Trajectory(q)
        app = types.SimpleNamespace(update=lambda: None, close=lambda: None, is_running=lambda: True)
        prim = types.SimpleNamespace(HasAPI=lambda api: True)
        active = types.SimpleNamespace(GetPrimAtPath=lambda path: prim, TraverseAll=lambda: [])
        context = types.SimpleNamespace(open_stage=lambda path: True, get_stage=lambda: active)
        api = lambda prim: types.SimpleNamespace(GetKinematicEnabledAttr=lambda: types.SimpleNamespace(Get=lambda: False),
                                                 GetRigidBodyEnabledAttr=lambda: types.SimpleNamespace(Get=lambda: True))
        names = ["isaacsim", "omni", "omni.usd", "isaacsim.core", "isaacsim.core.api", "isaacsim.core.prims",
                 "isaacsim.core.simulation_manager", "isaacsim.core.utils", "isaacsim.core.utils.types",
                 "isaacsim.robot_motion", "isaacsim.robot_motion.motion_generation",
                 "isaacsim.robot_motion.motion_generation.interface_config_loader", "pxr"]
        modules = {name: types.ModuleType(name) for name in names}
        modules["isaacsim"].SimulationApp = lambda settings: app
        modules["omni"].usd = modules["omni.usd"]
        modules["omni.usd"].get_context = lambda: context
        modules["isaacsim.core.api"].World = World
        modules["isaacsim.core.prims"].SingleArticulation = Robot
        modules["isaacsim.core.prims"].SingleRigidPrim = Body
        modules["isaacsim.core.simulation_manager"].SimulationManager = object
        modules["isaacsim.core.utils.types"].ArticulationAction = Action
        modules["isaacsim.robot_motion.motion_generation"].LulaKinematicsSolver = Solver
        modules["isaacsim.robot_motion.motion_generation"].LulaCSpaceTrajectoryGenerator = Generator
        modules["isaacsim.robot_motion.motion_generation.interface_config_loader"].load_supported_lula_kinematics_solver_config = lambda name: {}
        modules["pxr"].Usd = types.SimpleNamespace(Stage=types.SimpleNamespace(Open=lambda name: active))
        modules["pxr"].UsdPhysics = types.SimpleNamespace(RigidBodyAPI=api)
        evidence = {"rigid_bodies": [{"name": name, "path": "/World/Panda/" + name}
                                    for name in ("panda_link0", "panda_hand", "base_link")]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "scene"; scene.write_text("untouched")
            (root / "robot_state.json").write_text(json.dumps({"joint_names": list(ARM_JOINTS)}))
            np.save(root / "panda_joint_positions.npy", np.zeros(7))
            args = types.SimpleNamespace(output=root / "trial", scene_usd=scene, capture=root, candidates=root,
                prepared_model=root, tool_fk_check=root, candidate_rank=0, target_prim="/World/Objects/Target",
                pregrasp_distance_m=.1, lift_distance_m=.15, trajectory_time_scale=3,
                headless=True, keep_open=False, replay_physics="default")
            goal = np.eye(4); goal[2, 3] = .1
            with patch.dict(sys.modules, modules), patch.object(script, "parse_args", return_value=args), \
                 patch.object(script, "verified_trial_model", return_value=(evidence, np.eye(4))), \
                 patch.object(script, "capture_arm_matches", return_value=np.zeros(7)), \
                 patch.object(script, "candidate_in_world", return_value=(goal, 3, .9)), \
                 patch.object(script, "create_variant_scene", return_value=(active, "/World/Panda", [])), \
                 patch.object(script, "scene_invariants", return_value={}), \
                 patch.object(script, "configure_replay_physics", return_value={}), \
                 patch.object(script, "replay_physics_state", return_value={}), \
                 patch.object(script, "validate_replay_physics"), patch("builtins.print"):
                code = script.main()
            result = json.loads((args.output / "robotiq_pick_check.json").read_text())
            self.assertEqual(code, 0, result.get("failure"))
            self.assertEqual(result["status"], "physical_pick_observed")
            self.assertTrue(result["source_scene_unchanged"])
            self.assertFalse(result["safety"]["attachment_created"])
            self.assertFalse(result["safety"]["drive_parameters_modified"])
            self.assertFalse(result["safety"]["world_collision_checked"])
            self.assertFalse(result["safety"]["profile_ready"])
            phase = np.load(args.output / "measured_phase.npy")
            self.assertEqual(np.sum(phase == "close"), 180)
            self.assertEqual(np.sum(phase == "hold"), 180)
            self.assertIn("base_pose", calls)
            for action in calls:
                if isinstance(action, Action):
                    self.assertIn(tuple(action.joint_indices), (tuple(range(7)), (7,)))


class BatchTrialContractTests(unittest.TestCase):
    def test_batch_owns_server_retries_only_expected_rejections_and_stops_at_success(self):
        script = load_script("run_robotiq_pick_diagnostic")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts_dir = Path(script.__file__).parent
            collision_dir = root / "collision"
            collision_dir.mkdir()
            (collision_dir / "runtime_preflight").mkdir()
            source = root / "source"; source.write_text("unchanged")
            geometry = root / "geometry.json"; geometry.write_text("unchanged")
            collision_path = collision_dir / "collision_model_check.json"
            collision = {"status": "open_snapshot_collision_draft_prepared", "sources": {
                "prepared_model": file_identity(source), "tool_fk_check": file_identity(source),
                "evidence": file_identity(source), "geometry": file_identity(geometry)}}
            collision_path.write_text(json.dumps(collision))
            runtime = {"status": "open_snapshot_preflight_passed", "inputs": {"collision_model": file_identity(collision_path)}}
            (collision_dir / "runtime_preflight/collision_preflight_check.json").write_text(json.dumps(runtime))
            (root / "gripper_swap_check.json").write_text(json.dumps({"status": "success", "inputs": {"scene_usd": str(source)}}))
            for name in ("points_camera.npy", "T_world_camera.npy", "panda_joint_positions.npy", "union_mask.npy"):
                np.save(root / name, np.zeros(1))
            (root / "robot_state.json").write_text("{}")
            reference = root / "reference.json"
            reference.write_text(json.dumps({"inputs": {"capture": str(root), "segmentation": str(root)}}))
            grasp_root = root / "GraspGenX"
            (grasp_root / ".venv/bin").mkdir(parents=True)
            (grasp_root / ".venv/bin/python").write_text("fake")
            stages = []; stopped = []; owned = object()

            def run(command, **kwargs):
                args = list(map(str, command))
                stages.append(args)
                output = Path(args[args.index("--output")+1])
                output.mkdir()
                if "graspgenx_infer_capture.py" in args[1]:
                    self.assertEqual(args[args.index("--gripper-name")+1], "robotiq_2f_85")
                    np.save(output / "scores.npy", [.9, .8, .7])
                    return types.SimpleNamespace(returncode=0)
                rank = int(args[args.index("--candidate-rank")+1])
                self.assertIn(owned, stopped)  # GPU model stopped before any Isaac process.
                status = ["candidate_rejected", "physical_pick_not_observed", "physical_pick_observed"][rank]
                (output / "robotiq_pick_check.json").write_text(json.dumps({"status": status, "candidate": {"source_candidate_index": rank}}))
                return types.SimpleNamespace(returncode=0 if rank == 2 else 2)

            common = ["run", "--collision-model", str(collision_path), "--reference-plan", str(reference),
                      "--graspgenx-root", str(grasp_root), "--isaac-python", str(source),
                      "--simulation-only", "--allow-collision-unchecked-simulation"]
            with patch.object(sys, "argv", [*common, "--output", str(root / "batch")]), \
                 patch.object(script, "verified_trial_model"), \
                 patch.object(script, "_port_accepts_connections", return_value=False), \
                 patch.object(script, "_wait_for_server"), \
                 patch.object(script, "_stop_server", side_effect=lambda process: stopped.append(process)), \
                 patch.object(script.subprocess, "Popen", return_value=owned) as popen, \
                 patch.object(script.subprocess, "run", side_effect=run), patch("builtins.print"):
                self.assertEqual(script.main(), 0)
            report = json.loads((root / "batch/robotiq_pick_trials.json").read_text())
            self.assertEqual(len(report["attempts"]), 3)
            self.assertEqual(report["status"], "physical_pick_observed")
            self.assertEqual(report["selected_success"]["score_rank"], 2)
            self.assertEqual(popen.call_args.args[0][-1], "robotiq_2f_85")
            self.assertEqual(stopped, [owned])
            self.assertFalse(report["safety"]["normal_pipeline_changed"])

            def broken_trial(command, **kwargs):
                if "graspgenx_infer_capture.py" in str(command[1]): return run(command, **kwargs)
                output = Path(command[command.index("--output")+1]); output.mkdir()
                (output / "robotiq_pick_check.json").write_text('{"status":"failure"}')
                return types.SimpleNamespace(returncode=1)
            with patch.object(sys, "argv", [*common, "--output", str(root / "broken")]), \
                 patch.object(script, "verified_trial_model"), \
                 patch.object(script, "_port_accepts_connections", return_value=False), \
                 patch.object(script, "_wait_for_server"), patch.object(script, "_stop_server"), \
                 patch.object(script.subprocess, "Popen", return_value=owned), \
                 patch.object(script.subprocess, "run", side_effect=broken_trial), patch("builtins.print"):
                self.assertEqual(script.main(), 1)
            broken = json.loads((root / "broken/robotiq_pick_trials.json").read_text())
            self.assertEqual(len(broken["attempts"]), 1)
            self.assertEqual(broken["status"], "failure")


if __name__ == "__main__":
    unittest.main()
