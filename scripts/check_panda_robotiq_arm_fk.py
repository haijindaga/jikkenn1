#!/usr/bin/env python3
"""Compare installed Panda arm frames with official cuRobo FK; no planning."""

import argparse
import json
from pathlib import Path
import sys
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from panda_handover.geometry import matrix_from_pose
from panda_handover.gripper_swap import ARM_JOINTS
from panda_handover.robot_model_evidence import arm_poses_in_base, compare_arm_poses
from panda_handover.robotiq_model import body_transform, file_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared-model", type=Path,
                        help="model_preparation_check.json; additionally verify hand and Robotiq base FK")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a new name")
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    if evidence.get("open_close_checks_passed") is not True:
        raise ValueError("Installed gripper open/close checks did not pass")
    report = {"status": "in_progress", "purpose": "arm-only model alignment preflight",
              "inputs": {"evidence": str(args.evidence.resolve()), "robot": "franka.yml"},
              "safety": {"robot_moved": False, "trajectory_saved": False, "profile_ready": False,
                         "grasp_to_tool_verified": False, "collision_geometry_equivalence_verified": False}}
    try:
        import torch
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.types import JointState, DeviceCfg

        observed = arm_poses_in_base(evidence)
        model_overrides = {}
        if args.prepared_model is not None:
            prepared = json.loads(args.prepared_model.read_text(encoding="utf-8"))
            if prepared.get("status") != "fk_model_prepared":
                raise ValueError("Model preparation did not pass")
            if file_identity(args.evidence) != prepared["sources"]["evidence"]:
                raise ValueError("Evidence differs from the prepared model's source")
            urdf = prepared["urdf"]
            if file_identity(urdf["path"]) != urdf:
                raise ValueError("Prepared URDF changed after preparation")
            inverse_base = np.linalg.inv(body_transform(evidence, "panda_link0"))
            observed["panda_hand"] = inverse_base @ body_transform(evidence, "panda_hand")
            observed["robotiq_arg2f_base_link"] = inverse_base @ body_transform(evidence, "base_link")
            model_overrides["urdf_path"] = urdf["path"]
            report["inputs"]["prepared_model"] = str(args.prepared_model.resolve())
            report["inputs"]["urdf"] = urdf
            report["purpose"] = "arm and native gripper mount FK; not collision or grasp-frame validation"
        # The loader builds only chains to requested frames. The stock finger
        # locks refer to joints outside these arm-only chains and cause KeyError
        # before FK. Remove those locks and the stock hand's attachment extension
        # only for this diagnostic; do not alter the YAML, URDF or simulator.
        report["diagnostic_model_overrides"] = {
            "load_collision_spheres": False, "lock_joints": None, "extra_links": {},
            "tool_frames": list(observed),
            **model_overrides,
        }
        cfg = KinematicsCfg.from_robot_yaml_file("franka.yml", tool_frames=list(observed),
                                                load_collision_spheres=False,
                                                lock_joints=None, extra_links={}, **model_overrides)
        kin = Kinematics(cfg)
        if set(kin.joint_names) != set(ARM_JOINTS):
            raise ValueError(f"Unexpected official arm joints: {kin.joint_names}")
        measured = dict(zip(evidence["joint_names"], evidence["joint_positions"]))
        q = np.asarray([measured[name] for name in kin.joint_names], dtype=float)
        if not np.all(np.isfinite(q)):
            raise ValueError("Non-finite measured joint state")
        device = DeviceCfg()
        state = kin.compute_kinematics(JointState.from_position(
            torch.as_tensor(q[None], device=device.device, dtype=device.dtype), joint_names=kin.joint_names))
        predicted = {}
        for name in observed:
            pose = state.tool_poses.get_link_pose(name)
            predicted[name] = matrix_from_pose(pose.position.detach().cpu().numpy().reshape(-1, 3)[0],
                                               pose.quaternion.detach().cpu().numpy().reshape(-1, 4)[0])
        results = compare_arm_poses(observed, predicted)
        report["frames"] = results
        report["status"] = "arm_alignment_passed" if all(r["passed"] for r in results.values()) else "arm_alignment_failed"
        if args.prepared_model is not None:
            report["status"] = "tool_mount_alignment_passed" if all(r["passed"] for r in results.values()) else "tool_mount_alignment_failed"
        report["tolerances"] = {"translation_m": 0.005, "rotation_rad": float(np.deg2rad(2))}
        report["next_gate"] = "Arm-only comparison at one state; not a verified Robotiq mount or collision model. Review installed joints/geometry and canonical grasp frame before preparing the full profile."
    except Exception as error:
        report["status"] = "failure"
        report["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] in ("arm_alignment_passed", "tool_mount_alignment_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
