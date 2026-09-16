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
    captured_arm_positions, gripper_control_spec, select_official_gripper,
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
    parser.add_argument("--assets-root", help="Optional official Isaac asset root URL/path")
    parser.add_argument("--phase-frames", type=int, default=180)
    parser.add_argument("--headless", action="store_true")
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
            "asset": PANDA_ASSET, "variant": GRIPPER_VARIANT,
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
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Sdf, Usd, UsdGeom

        asset_root = args.assets_root or get_assets_root_path()
        if not asset_root:
            raise RuntimeError("Official Isaac asset root unavailable; use --assets-root")
        asset_url = asset_root.rstrip("/\\") + PANDA_ASSET
        original_stage = Usd.Stage.Open(str(source))
        if original_stage is None or UsdGeom.GetStageMetersPerUnit(original_stage) != 1.0:
            raise RuntimeError("Input scene must open successfully and use metres")
        original = original_stage.GetPrimAtPath("/World/Panda")
        if not original.IsValid() or not original.IsActive():
            raise RuntimeError("Input scene lacks active /World/Panda")
        original_xform = UsdGeom.Xformable(original)
        if not original_xform:
            raise RuntimeError("Original Panda root is not transformable")
        original_matrix = original_xform.GetLocalTransformation()

        # A new authoring layer inherits the table, objects, camera and lighting.
        # The old robot is only inactive IN THIS LAYER; source USD is untouched.
        scene_path = output / "scene.usda"
        layer = Sdf.Layer.CreateNew(str(scene_path))
        layer.subLayerPaths = [str(source)]
        stage = Usd.Stage.Open(layer)
        stage.OverridePrim("/World/Panda").SetActive(False)
        robot_path = "/World/PandaRobotiq"
        replacement = UsdGeom.Xform.Define(stage, robot_path)
        replacement.GetPrim().GetReferences().AddReference(asset_url)
        available = select_official_gripper(replacement.GetPrim())
        replacement.MakeMatrixXform().Set(original_matrix)
        replacement.SetResetXformStack(original_xform.GetResetXformStack())
        layer.Save()
        if not omni.usd.get_context().open_stage(str(scene_path)):
            raise RuntimeError("Could not open the separate gripper test scene")
        for _ in range(10):
            app.update()

        world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 30)
        context = world.get_physics_context()
        report["physics"] = configure_replay_physics(context, SimulationManager, "cpu")
        robot = world.scene.add(SingleArticulation(prim_path=robot_path, name="panda_robotiq_test"))
        world.reset()
        report["physics"]["after_reset"] = replay_physics_state(context)
        validate_replay_physics("cpu", report["physics"]["after_reset"])
        names = list(robot.dof_names)
        arm_indices = np.asarray([names.index(name) for name in ARM_JOINTS])
        if any(name in names for name in ("panda_finger_joint1", "panda_finger_joint2")):
            raise RuntimeError("Original Panda finger DOFs remain; variant not properly replaced")
        properties = robot.dof_properties
        for index, value in zip(arm_indices, arm_q):
            if not properties[index]["lower"] <= value <= properties[index]["upper"]:
                raise RuntimeError("Captured arm position outside official joint limits")
        spec = gripper_control_spec(names, properties, args.phase_frames / 60)
        report.update({
            "scene_usd": str(scene_path), "robot_prim": robot_path,
            "available_gripper_variants": available, "joint_names": names,
            "gripper_control": spec, "captured_arm_positions_rad": arm_q.tolist(),
        })
        report["joint_properties"] = [
            {"name": name, **{
                field: properties[index][field].item() for field in properties.dtype.names
            }} for index, name in enumerate(names)
        ]
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
