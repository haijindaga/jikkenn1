#!/usr/bin/env python3
"""Isolated Robotiq pick diagnostic: official Lula IK/trajectory + Isaac control.

NO collision-avoidance guarantee, NO attachment, NO hardware execution.
The normal Panda/cuRobo pipeline is untouched.
"""

import argparse
import json
from pathlib import Path
import sys
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from panda_handover.geometry import matrix_from_pose
from panda_handover.gripper_swap import (
    ARM_JOINTS, bounded_gripper_velocity, create_variant_scene, gripper_control_spec, scene_invariants,
)
from panda_handover.replay_physics import configure_replay_physics, replay_physics_state, validate_replay_physics
from panda_handover.robot_model_evidence import compare_arm_poses
from panda_handover.robotiq_model import file_identity
from panda_handover.robotiq_trial import (
    candidate_in_world, capture_arm_matches, contact_waypoints, pick_result, solve_waypoints, verified_trial_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scene-usd", "capture", "candidates", "prepared-model", "tool-fk-check", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--candidate-rank", type=int, default=0)
    parser.add_argument("--target-prim", default="/World/Objects/Target")
    parser.add_argument("--pregrasp-distance-m", type=float, default=0.1)
    parser.add_argument("--lift-distance-m", type=float, default=0.15)
    parser.add_argument("--trajectory-time-scale", type=float, default=3.0,
                        help="Slow official time-optimal trajectories; >=1, not drive scaling")
    parser.add_argument("--replay-physics", choices=("default", "cpu"), default="default")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--simulation-only", action="store_true", required=True)
    parser.add_argument("--allow-collision-unchecked-simulation", action="store_true", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; use a new directory")
    if args.candidate_rank < 0 or not np.isfinite(args.trajectory_time_scale) or args.trajectory_time_scale < 1:
        parser.error("rank must be nonnegative and trajectory-time-scale >=1")
    if args.headless and args.keep_open:
        parser.error("--keep-open is only for GUI trials")
    return args


def main():
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True)
    path = output / "robotiq_pick_check.json"
    report = {"status": "in_progress", "purpose": "collision-unchecked isolated simulation pick diagnostic",
              "inputs": {k: str(getattr(args, k).resolve()) for k in
                         ("scene_usd", "capture", "candidates", "prepared_model", "tool_fk_check")},
              "safety": {"simulation_only": True, "safe_to_execute_on_hardware": False,
                         "world_collision_checked": False, "self_collision_checked": False,
                         "profile_ready": False, "attachment_created": False,
                         "drive_parameters_modified": False, "friction_modified": False,
                         "old_panda_grasp_trajectory_reused": False,
                         "canonical_installed_mesh_equivalence_verified": False,
                         "capture_reused_without_robotiq_recapture": True},
              "parameters": {"candidate_rank": args.candidate_rank,
                             "pregrasp_distance_m": args.pregrasp_distance_m,
                             "lift_distance_m": args.lift_distance_m,
                             "trajectory_time_scale": args.trajectory_time_scale,
                             "ik_waypoint_spacing_m": 0.005, "ik_branch_jump_limit_rad": 0.35,
                             "tracking_abort_limit_rad": 0.35,
                             "settle_frames": 120, "close_frames": 180, "hold_frames": 180}}
    app = None
    history = {"phase": [], "hand": [], "target": [], "q": [], "command_q": []}

    def save():
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        evidence, grasp_hand = verified_trial_model(args.prepared_model, args.tool_fk_check, args.scene_usd)
        state = json.loads((args.capture / "robot_state.json").read_text(encoding="utf-8"))
        initial_q = capture_arm_matches(evidence, state["joint_names"], np.load(args.capture / "panda_joint_positions.npy"))
        grasp, index, score = candidate_in_world(args.candidates, args.capture, args.candidate_rank)
        poses = contact_waypoints(grasp, grasp_hand, args.pregrasp_distance_m, args.lift_distance_m)
        report["candidate"] = {"source_candidate_index": index, "graspgenx_score": score,
                               "T_world_grasp": grasp.tolist(), "T_grasp_panda_hand": grasp_hand.tolist()}
        report["source_scene_before"] = file_identity(args.scene_usd)
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": args.headless})
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, LulaCSpaceTrajectoryGenerator
        from isaacsim.robot_motion.motion_generation.interface_config_loader import load_supported_lula_kinematics_solver_config
        from pxr import Usd, UsdPhysics

        original = Usd.Stage.Open(str(args.scene_usd.resolve()))
        stage, robot_path, _ = create_variant_scene(original, output / "scene.usda")
        if scene_invariants(original) != scene_invariants(stage):
            raise RuntimeError("Scene invariants changed during the official variant swap")
        if not omni.usd.get_context().open_stage(str(output / "scene.usda")):
            raise RuntimeError("Could not open isolated variant scene")
        for _ in range(10):
            app.update()
        world = World(stage_units_in_meters=1.0, physics_dt=1/60, rendering_dt=1/30)
        context = world.get_physics_context()
        report["physics"] = configure_replay_physics(context, SimulationManager, args.replay_physics)
        robot = world.scene.add(SingleArticulation(prim_path=robot_path, name="robotiq_pick_robot"))
        world.reset()
        report["physics"]["after_reset"] = replay_physics_state(context)
        validate_replay_physics(args.replay_physics, report["physics"]["after_reset"])
        active = omni.usd.get_context().get_stage()
        target_prim = active.GetPrimAtPath(args.target_prim)
        if not target_prim or not target_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError("Target must be an existing dynamic rigid body; specify its actual --target-prim")
        if UsdPhysics.RigidBodyAPI(target_prim).GetKinematicEnabledAttr().Get():
            raise RuntimeError("Kinematic target is not a physical pick test")
        if UsdPhysics.RigidBodyAPI(target_prim).GetRigidBodyEnabledAttr().Get() is False:
            raise RuntimeError("Target rigid body is disabled")
        for prim in active.TraverseAll():
            if prim.IsA(UsdPhysics.Joint) or str(prim.GetTypeName()) == "PhysxPhysicsAttachment":
                endpoints = [str(value) for relation in prim.GetRelationships() for value in relation.GetTargets()]
                if any(value == args.target_prim or value.startswith(args.target_prim + "/") for value in endpoints):
                    raise RuntimeError("Pre-existing target joint/attachment would invalidate the physical retention test")
        bodies = {}
        for name in ("panda_link0", "panda_hand", "base_link"):
            prim_path = next(b["path"] for b in evidence["rigid_bodies"] if b["name"] == name)
            bodies[name] = SingleRigidPrim(prim_path=prim_path, name="trial_" + name, reset_xform_properties=False)
            bodies[name].initialize()
        target = SingleRigidPrim(prim_path=args.target_prim, name="trial_target", reset_xform_properties=False)
        target.initialize()
        names, properties = list(robot.dof_names), robot.dof_properties
        arm_indices = np.asarray([names.index(name) for name in ARM_JOINTS])
        lower, upper = properties[arm_indices]["lower"], properties[arm_indices]["upper"]
        if any(name in names for name in ("panda_finger_joint1", "panda_finger_joint2")):
            raise RuntimeError("Old Panda finger DOFs remain")
        extra_driven = [name for i, name in enumerate(names) if name not in ARM_JOINTS and name != "finger_joint"
                        and (properties[i]["stiffness"] > 0 or properties[i]["damping"] > 0)]
        if extra_driven:
            raise RuntimeError(f"Unreviewed independently driven gripper joints: {extra_driven}")
        spec = gripper_control_spec(names, properties, 3.0)
        report["gripper_control"] = spec
        if np.any(initial_q < lower) or np.any(initial_q > upper):
            raise RuntimeError("Captured initial arm pose violates installed joint limits")
        robot.set_joint_positions(initial_q, joint_indices=arm_indices)
        robot.set_joint_velocities(np.zeros(7), joint_indices=arm_indices)

        def step(q, opened, phase, velocity=None):
            if not app.is_running() or not world.is_playing():
                raise RuntimeError("Simulation stopped before trial completed")
            robot.apply_action(ArticulationAction(joint_positions=q, joint_velocities=velocity, joint_indices=arm_indices))
            finger_target = spec["open_rad"] if opened else spec["closed_rad"]
            if spec["mode"] == "position":
                action = ArticulationAction(joint_positions=np.asarray([finger_target]), joint_indices=np.asarray([spec["index"]]))
            else:
                finger_q = float(robot.get_joint_positions()[spec["index"]])
                speed = bounded_gripper_velocity(finger_q, finger_target, spec["diagnostic_speed_rad_s"], 1/60)
                action = ArticulationAction(joint_velocities=np.asarray([speed]), joint_indices=np.asarray([spec["index"]]))
            robot.apply_action(action)
            world.step(render=True)
            measured = np.asarray(robot.get_joint_positions())
            if not np.all(np.isfinite(measured)):
                raise RuntimeError("Non-finite measured joint state")
            error = float(np.max(np.abs(measured[arm_indices] - q)))
            errors = report.setdefault("phase_maximum_tracking_error_rad", {})
            errors[phase] = max(errors.get(phase, 0), error)
            history["phase"].append(phase)
            history["hand"].append(matrix_from_pose(*bodies["panda_hand"].get_world_pose()))
            history["target"].append(matrix_from_pose(*target.get_world_pose()))
            history["q"].append(measured.copy())
            history["command_q"].append(np.asarray(q).copy())
            if error > 0.35:
                raise RuntimeError(f"Arm tracking error {error:.4f} rad exceeds diagnostic limit in {phase}")

        for _ in range(120):
            step(initial_q, True, "settle")
        baseline_z = float(target.get_world_pose()[0][2])
        base_position, base_quaternion = bodies["panda_link0"].get_world_pose()
        config = load_supported_lula_kinematics_solver_config("Franka")
        solver = LulaKinematicsSolver(**config)
        if list(solver.get_joint_names()) != list(ARM_JOINTS) or "panda_hand" not in solver.get_all_frame_names():
            raise RuntimeError("Official Lula arm joint/frame conventions differ")
        solver.set_robot_base_pose(base_position, base_quaternion)
        current_q = np.asarray(robot.get_joint_positions())[arm_indices]
        predicted_position, predicted_rotation = solver.compute_forward_kinematics("panda_hand", current_q)
        predicted = np.eye(4)
        predicted[:3, :3], predicted[:3, 3] = predicted_rotation, np.asarray(predicted_position).reshape(3)
        hand_pose = matrix_from_pose(*bodies["panda_hand"].get_world_pose())
        mount_pose = matrix_from_pose(*bodies["base_link"].get_world_pose())
        report["runtime_model_alignment"] = compare_arm_poses(
            {"lula_hand": hand_pose, "native_mount": hand_pose},
            {"lula_hand": predicted, "native_mount": mount_pose})
        if not all(f["passed"] for f in report["runtime_model_alignment"].values()):
            raise RuntimeError("Runtime Lula FK or native mount disagrees with measured hand")
        generator = LulaCSpaceTrajectoryGenerator(**config)
        if list(generator.get_active_joints()) != list(ARM_JOINTS):
            raise RuntimeError("Trajectory generator arm order differs")
        report["reference"] = {"ik": "Isaac Sim 5.1 official LulaKinematicsSolver Franka/panda_hand",
                               "trajectory": "official LulaCSpaceTrajectoryGenerator; default limits, slowed in time",
                               "control": "ArticulationAction; installed Robotiq master/mimics and authored gains"}
        contact_q, rejection = solve_waypoints(solver, poses["contact"], current_q, lower, upper)
        lift_q = None
        if rejection is None:
            lift_q, rejection = solve_waypoints(solver, poses["lift"], contact_q[-1], lower, upper)
        if rejection is not None:
            report.update(status="candidate_rejected", rejection=rejection)
            return 2
        waypoints = {"approach": contact_q[:2], "contact": contact_q[1:], "lift": lift_q}
        trajectories = {}
        for phase, q in waypoints.items():
            q = q[np.r_[True, np.any(np.abs(np.diff(q, axis=0)) > 1e-8, axis=1)]]
            trajectory = generator.compute_c_space_trajectory(q) if len(q) >= 2 else None
            if trajectory is None and len(q) >= 2:
                report.update(status="candidate_rejected", rejection="trajectory_generation_failed", rejection_phase=phase)
                return 2
            trajectories[phase] = (trajectory, q[-1].copy())
            np.save(output / f"{phase}_ik_joint_waypoints_rad.npy", q)
        np.save(output / "contact_hand_world_goals.npy", poses["contact"])
        np.save(output / "lift_hand_world_goals.npy", poses["lift"])
        report["baseline_target_z_m"] = baseline_z
        save()

        def move(phase, opened):
            trajectory, q = trajectories[phase]
            if trajectory is not None:
                duration = (trajectory.end_time - trajectory.start_time) * args.trajectory_time_scale
                frames = int(np.ceil(duration * 60))
                for frame in range(frames + 1):
                    time = min(trajectory.end_time, trajectory.start_time + frame / 60 / args.trajectory_time_scale)
                    q, velocity = trajectory.get_joint_targets(time)
                    q, velocity = np.asarray(q), np.asarray(velocity) / args.trajectory_time_scale
                    if q.shape != (7,) or velocity.shape != (7,) or not np.all(np.isfinite([q, velocity])):
                        raise RuntimeError("Invalid official trajectory sample")
                    if np.any(q < lower) or np.any(q > upper):
                        raise RuntimeError("Interpolated trajectory violates installed joint limits")
                    step(q, opened, phase, velocity)
            for _ in range(60):
                step(q, opened, phase)
            goal = poses["contact"][0 if phase == "approach" else -1] if phase != "lift" else poses["lift"][-1]
            reached = matrix_from_pose(*bodies["panda_hand"].get_world_pose())
            endpoint = compare_arm_poses({phase: goal}, {phase: reached},
                                        translation_tolerance=0.01, rotation_tolerance=np.deg2rad(5))[phase]
            report.setdefault("phase_endpoint_pose_error", {})[phase] = endpoint
            if not endpoint["passed"]:
                raise RuntimeError(f"Hand did not reach {phase} endpoint; refusing next phase")
            return q

        print(f"Robotiq candidate {index}: approach -> contact -> close -> lift (collision-unchecked simulation)", flush=True)
        move("approach", True)
        closed_q = move("contact", True)
        for _ in range(180):
            step(closed_q, False, "close")
        lifted_q = move("lift", False)
        for _ in range(180):
            step(lifted_q, False, "hold")
        result = pick_result(history["phase"], np.asarray(history["target"])[:, :3, 3], baseline_z)
        report["object"] = result
        report["status"] = "physical_pick_observed" if result["physical_pick_observed"] else "physical_pick_not_observed"
        report["next_gate"] = "Inspect the measured motion; this diagnostic does not validate collision avoidance or handover."
        save()
        print(json.dumps({"status": report["status"], "candidate": index, "object": result}), flush=True)
        print(f"saved: {path}", flush=True)
        if args.keep_open:
            while app.is_running():
                if world.is_playing():
                    step(lifted_q, False, "post_report")
                else:
                    app.update()
        return 0 if result["physical_pick_observed"] else 2
    except Exception as error:
        report["status"] = "failure"
        report["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        print(traceback.format_exc(), flush=True)
        return 1
    finally:
        if history["phase"]:
            for key, value in history.items():
                np.save(output / f"measured_{key}.npy", np.asarray(value))
        if "source_scene_before" in report:
            report["source_scene_unchanged"] = file_identity(args.scene_usd) == report["source_scene_before"]
        save()
        if app is not None:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
