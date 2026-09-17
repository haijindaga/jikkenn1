#!/usr/bin/env python3
"""Swap the hand using the official Panda USD variant in an isolated scene.

This is an open/close smoke test, NOT a grasp/lift planner. It does not run
Panda-specific grasp candidates against a different gripper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from panda_handover.gripper_swap import (
    ARM_JOINTS, PANDA_ASSET, GRIPPER_VARIANT, bounded_gripper_velocity,
    captured_arm_positions, create_variant_scene, gripper_control_spec, scene_invariants,
)
from panda_handover.replay_physics import (
    configure_replay_physics, replay_physics_state, validate_replay_physics,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-usd", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True,
                        help="Existing Panda capture supplying the initial arm posture")
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory; refuses to replace an existing experiment")
    parser.add_argument("--replay-physics", choices=("default", "cpu"), default="default",
                        help="Keep scene settings unless CPU diagnostics are explicitly requested")
    parser.add_argument("--inspect-only", action="store_true",
                        help="Save replacement scene and joint inventory without commanding motion")
    parser.add_argument("--phase-frames", type=int, default=180)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--export-model-evidence", action="store_true",
                        help="After reopening, save measured rigid-body frames and USD joint/collision inventory")
    parser.add_argument("--export-collision-geometry", action="store_true",
                        help="With --export-model-evidence, also save authored gripper mesh topology and runtime placement")
    parser.add_argument("--simulation-only", action="store_true", required=True)
    args = parser.parse_args()
    if not args.scene_usd.is_file():
        parser.error(f"scene does not exist: {args.scene_usd}")
    for filename in ("robot_state.json", "panda_joint_positions.npy"):
        if not (args.capture / filename).is_file():
            parser.error(f"capture lacks {filename}: {args.capture}")
    if args.output.exists():
        parser.error("output already exists; choose a new output directory")
    if args.phase_frames <= 0:
        parser.error("--phase-frames must be positive")
    if args.inspect_only and args.export_model_evidence:
        parser.error("Model evidence requires the completed open/close pass, not --inspect-only")
    if args.export_collision_geometry and not args.export_model_evidence:
        parser.error("--export-collision-geometry requires --export-model-evidence")
    return args


def main():
    args = parse_args()
    import numpy as np

    state = json.loads((args.capture / "robot_state.json").read_text(encoding="utf-8"))
    arm_q = captured_arm_positions(
        state["joint_names"], np.load(args.capture / "panda_joint_positions.npy")
    )
    source = args.scene_usd.resolve()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    output = args.output.resolve()
    output.mkdir(parents=True)
    report_path = output / "gripper_swap_check.json"
    report = {
        "status": "in_progress", "purpose": "official gripper swap and open/close smoke test",
        "inputs": {"scene_usd": str(source), "capture": str(args.capture.resolve())},
        "reference": {
            "expected_official_asset": PANDA_ASSET, "variant": GRIPPER_VARIANT,
            "documentation": "https://docs.isaacsim.omniverse.nvidia.com/5.1.0/assets/usd_assets_robots.html",
        },
        "safety": {
            "simulation_only": True, "grasp_tested": False,
            "existing_grasp_plans_reused": False, "pipeline_robot_profile_ready": False,
            "source_assets_modified": False, "drive_parameters_modified": False,
            "arm_initialization_only_no_arm_trajectory": True,
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    app = None
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": args.headless})
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.utils.types import ArticulationAction
        from pxr import Usd, UsdPhysics

        original_stage = Usd.Stage.Open(str(source))
        if original_stage is None:
            raise RuntimeError("Input scene could not be opened")
        before_scene = scene_invariants(original_stage)
        scene_path = output / "scene.usda"
        stage, robot_path, available = create_variant_scene(original_stage, scene_path)
        after_scene = scene_invariants(stage)
        report["scene_preservation"] = {
            "passed": before_scene == after_scene,
            "up_axis": after_scene["up_axis"],
            "meters_per_unit": after_scene["meters_per_unit"],
            "same_robot_prim_path": robot_path == "/World/Panda",
            "scope": "stage metadata, robot root, and non-robot prims/attributes/relationships before simulation",
        }
        if before_scene != after_scene:
            raise RuntimeError("Variant change altered scene invariants; refusing motion")
        report["scene_usd"] = str(scene_path)
        report["robot_prim"] = robot_path
        report["available_gripper_variants"] = available
        if not omni.usd.get_context().open_stage(str(scene_path)):
            raise RuntimeError("Could not open the separate gripper test scene")
        for _ in range(10):
            app.update()

        world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 30)
        context = world.get_physics_context()
        report["physics"] = configure_replay_physics(context, SimulationManager, args.replay_physics)
        robot = world.scene.add(SingleArticulation(prim_path=robot_path, name="panda_robotiq_test"))
        world.reset()
        report["physics"]["after_reset"] = replay_physics_state(context)
        validate_replay_physics(args.replay_physics, report["physics"]["after_reset"])
        names = list(robot.dof_names)
        properties = robot.dof_properties
        # Persist installed-model evidence BEFORE interpreting control conventions.
        report["joint_names"] = names
        report["joint_properties"] = [
            {"name": name, **{
                field: properties[index][field].item() for field in properties.dtype.names
            }} for index, name in enumerate(names)
        ]
        active_stage = omni.usd.get_context().get_stage()
        report["usd_joints"] = [
            {"path": str(prim.GetPath()), "type": str(prim.GetTypeName()),
             "applied_schemas": list(map(str, prim.GetAppliedSchemas())),
             "attributes": {str(a.GetName()): str(a.Get()) for a in prim.GetAttributes()
                            if str(a.GetName()).startswith(("drive:", "physxMimicJoint:", "physics:"))}}
            for prim in Usd.PrimRange(active_stage.GetPrimAtPath(robot_path))
            if prim.IsA(UsdPhysics.Joint)
        ]
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        arm_indices = np.asarray([names.index(name) for name in ARM_JOINTS])
        if any(name in names for name in ("panda_finger_joint1", "panda_finger_joint2")):
            raise RuntimeError("Original Panda finger DOFs remain; variant not properly replaced")
        for index, value in zip(arm_indices, arm_q):
            if not properties[index]["lower"] <= value <= properties[index]["upper"]:
                raise RuntimeError("Captured arm position outside official joint limits")
        if args.inspect_only:
            report["status"] = "inspection_complete"
            report["safety"]["motion_commanded"] = False
            report["safety"]["gripper_open_close_tested"] = False
            report["next_gate"] = "Review installed drive/mimic inventory before the open/close smoke test."
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"saved: {report_path}; inspection_complete", flush=True)
            return 0
        extra_driven = [name for i, name in enumerate(names)
                        if name not in ARM_JOINTS and name != "finger_joint"
                        and (properties[i]["stiffness"] > 0 or properties[i]["damping"] > 0)]
        if extra_driven:
            raise RuntimeError(f"Additional driven gripper DOFs require review: {extra_driven}; see joint inventory")
        spec = gripper_control_spec(names, properties, args.phase_frames / 60)
        report.update({
            "scene_usd": str(scene_path), "robot_prim": robot_path,
            "available_gripper_variants": available, "joint_names": names,
            "gripper_control": spec, "captured_arm_positions_rad": arm_q.tolist(),
        })
        robot.set_joint_positions(arm_q, joint_indices=arm_indices)
        robot.set_joint_velocities(np.zeros(7), joint_indices=arm_indices)

        def command(opened):
            robot.apply_action(ArticulationAction(joint_positions=arm_q, joint_indices=arm_indices))
            target = spec["open_rad"] if opened else spec["closed_rad"]
            if spec["mode"] == "position":
                action = ArticulationAction(joint_positions=np.asarray([target]), joint_indices=np.asarray([spec["index"]]))
            else:
                q = float(robot.get_joint_positions()[spec["index"]])
                velocity = bounded_gripper_velocity(q, target, spec["diagnostic_speed_rad_s"], 1 / 60)
                action = ArticulationAction(joint_velocities=np.asarray([velocity]), joint_indices=np.asarray([spec["index"]]))
            robot.apply_action(action)

        report["safety"]["motion_commanded"] = True
        phases = []
        for phase, opened in (("open", True), ("close", False), ("reopen", True)):
            max_arm_error = 0.0
            for _ in range(args.phase_frames):
                if not app.is_running():
                    raise RuntimeError("Window closed before open/close test completed")
                command(opened)
                world.step(render=True)
                measured = np.asarray(robot.get_joint_positions())
                if not np.all(np.isfinite(measured)):
                    raise RuntimeError("Non-finite joint state during test")
                max_arm_error = max(max_arm_error, float(np.max(np.abs(measured[arm_indices] - arm_q))))
            result = {"phase": phase, "finger_joint_rad": float(measured[spec["index"]]),
                      "maximum_arm_tracking_error_rad": max_arm_error}
            phases.append(result)
            print(json.dumps(result), flush=True)
        report["phases"] = phases
        span = spec["closed_rad"] - spec["open_rad"]
        checks = {
            "close_motion_observed": phases[1]["finger_joint_rad"] - phases[0]["finger_joint_rad"] > 0.1 * span,
            "reopen_motion_observed": phases[1]["finger_joint_rad"] - phases[2]["finger_joint_rad"] > 0.1 * span,
            "arm_posture_retained": max(p["maximum_arm_tracking_error_rad"] for p in phases) < 0.1,
            "source_scene_unchanged": hashlib.sha256(source.read_bytes()).hexdigest() == source_hash,
        }
        report["automatic_checks"] = checks
        report["status"] = "success" if all(checks.values()) else "smoke_test_failed"
        if args.export_model_evidence:
            from isaacsim.core.prims import SingleRigidPrim
            from panda_handover.robot_model_evidence import collect_model_evidence
            evidence = collect_model_evidence(active_stage, robot_path, names,
                                              robot.get_joint_positions(), SingleRigidPrim)
            evidence["reference"] = report["reference"]
            evidence["source_scene_sha256"] = source_hash
            evidence["open_close_checks_passed"] = all(checks.values())
            evidence["joint_properties"] = report["joint_properties"]
            evidence_path = output / "robot_model_evidence.json"
            evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
            report["model_evidence"] = str(evidence_path)
            print(f"saved: {evidence_path}", flush=True)
            if args.export_collision_geometry:
                from panda_handover.robot_model_evidence import collect_gripper_collision_geometry
                geometry = collect_gripper_collision_geometry(active_stage, evidence)
                geometry["evidence_path"] = str(evidence_path)
                geometry_path = output / "gripper_collision_geometry.json"
                geometry_path.write_text(json.dumps(geometry, indent=2) + "\n", encoding="utf-8")
                report["collision_geometry"] = str(geometry_path)
                print(f"saved: {geometry_path}", flush=True)
        report["next_gate"] = "Review opening/closing visually, then prepare matching GraspGenX and cuRobo profiles before grasp trials."
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"saved: {report_path}; {report['status']}", flush=True)
        if not args.headless:
            import omni.ui as ui
            requested = {"opened": True}
            window = ui.Window("Panda / Robotiq test", width=340, height=150)
            with window.frame:
                with ui.VStack():
                    ui.Label("Standalone gripper test - no grasp planning")
                    ui.Button("Open", clicked_fn=lambda: requested.update(opened=True))
                    ui.Button("Close", clicked_fn=lambda: requested.update(opened=False))
            while app.is_running():
                if world.is_playing():
                    command(requested["opened"])
                world.step(render=True)
        return 0 if report["status"] == "success" else 2
    except Exception as error:
        report["status"] = "failure"
        report["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
