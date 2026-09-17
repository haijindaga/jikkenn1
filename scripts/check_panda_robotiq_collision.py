#!/usr/bin/env python3
"""No-motion cuRobo FK/self-collision check for the open Robotiq snapshot draft."""

import argparse
import json
from pathlib import Path
import sys
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from panda_handover.geometry import matrix_from_pose
from panda_handover.gripper_swap import ARM_JOINTS, captured_arm_positions
from panda_handover.robotiq_model import body_transform, file_identity
from panda_handover.robot_model_evidence import arm_poses_in_base, compare_arm_poses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collision-model", type=Path, required=True,
                        help="collision_model_check.json from preparation")
    parser.add_argument("--output", type=Path, required=True, help="New directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a new directory")
    output = args.output.resolve()
    output.mkdir(parents=True)
    report = {"status": "in_progress", "scope": "one measured posture, open gripper only",
              "safety": {"robot_moved": False, "trajectory_saved": False, "profile_ready": False,
                         "safe_to_plan": False, "safe_to_execute": False, "gripper_close_supported": False}}
    try:
        import torch
        import yaml
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.types import DeviceCfg, JointState
        from curobo._src.cost.cost_self_collision import SelfCollisionCost
        from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg

        draft = json.loads(args.collision_model.read_text(encoding="utf-8"))
        if draft.get("status") != "open_snapshot_collision_draft_prepared":
            raise ValueError("Open snapshot preparation did not succeed")
        for source in list(draft["sources"].values()) + [draft["yaml"], draft["urdf"]]:
            if file_identity(source["path"]) != source:
                raise ValueError(f"Collision model source changed: {source['path']}")
        evidence = json.loads(Path(draft["sources"]["evidence"]["path"]).read_text(encoding="utf-8"))
        observed = arm_poses_in_base(evidence)
        inverse_base = np.linalg.inv(body_transform(evidence, "panda_link0"))
        observed["panda_hand"] = inverse_base @ body_transform(evidence, "panda_hand")
        observed["robotiq_arg2f_base_link"] = inverse_base @ body_transform(evidence, "base_link")
        cfg = KinematicsCfg.from_robot_yaml_file(draft["yaml"]["path"], tool_frames=list(observed))
        kin = Kinematics(cfg)
        if set(kin.joint_names) != set(ARM_JOINTS):
            raise ValueError(f"Unexpected collision-profile arm joints: {kin.joint_names}")
        measured = dict(zip(ARM_JOINTS, captured_arm_positions(evidence["joint_names"], evidence["joint_positions"])))
        q = np.asarray([measured[name] for name in kin.joint_names])
        device = DeviceCfg()
        state = kin.compute_kinematics(JointState.from_position(
            torch.as_tensor(q[None, None], device=device.device, dtype=device.dtype), joint_names=kin.joint_names))
        predicted = {}
        for name in observed:
            pose = state.tool_poses.get_link_pose(name)
            predicted[name] = matrix_from_pose(pose.position.detach().cpu().numpy().reshape(-1, 3)[0],
                                               pose.quaternion.detach().cpu().numpy().reshape(-1, 4)[0])
        report["frames"] = compare_arm_poses(observed, predicted)
        spheres = state.robot_spheres
        if spheres is None:
            raise ValueError("Collision-enabled model returned no spheres")
        array = spheres.detach().cpu().numpy()
        configuration = yaml.safe_load(Path(draft["yaml"]["path"]).read_text(encoding="utf-8"))
        expected = sum(len(value) for value in configuration["robot_cfg"]["kinematics"]["collision_spheres"].values())
        if array.shape != (1, 1, expected, 4) or not np.all(np.isfinite(array)) or np.any(array[..., 3] <= 0):
            raise ValueError("Collision sphere inventory is incomplete or invalid")
        self_cfg = kin.get_self_collision_config()
        if self_cfg is None:
            raise ValueError("Self collision configuration is absent")
        cost = SelfCollisionCost(SelfCollisionCostCfg(weight=1.0, device_cfg=device,
                                self_collision_kin_config=self_cfg, store_pair_distance=True))
        cost.setup_batch_tensors(1, 1)
        values = cost.forward(spheres).detach().cpu().numpy()
        if not np.all(np.isfinite(values)):
            raise ValueError("Non-finite official self-collision cost")
        np.save(output / "robot_spheres_robot_base_m.npy", array[0, 0])
        pairs = self_cfg.collision_pairs.detach().cpu().numpy()
        if len(pairs) == 0:
            raise ValueError("No self-collision sphere pairs are checked")
        np.save(output / "checked_self_collision_sphere_pairs.npy", pairs)
        report["self_collision"] = {"checked": True, "sphere_count": expected,
            "checked_pair_count": len(pairs), "maximum_penetration_cost_m": float(np.max(values)),
            "passed": bool(np.max(values) <= 0), "solver": "official cuRobo SelfCollisionCost"}
        report["inputs"] = {"collision_model": file_identity(args.collision_model), "robot": draft["yaml"]}
        report["coverage"] = [c["coverage"] for c in draft["colliders"]]
        checks = {"all_frames_match": all(value["passed"] for value in report["frames"].values()),
                  "self_collision_free_at_measured_posture": report["self_collision"]["passed"]}
        report["automatic_checks"] = checks
        report["status"] = "open_snapshot_preflight_passed" if all(checks.values()) else "open_snapshot_preflight_rejected"
        report["next_gate"] = "Review gripper sphere coverage and wrist adjacency exclusions, then adapt Robotiq capture/static filtering/pregrasp replay. No closing/lift/handover trajectory is authorized by this result."
    except Exception as error:
        report["status"] = "failure"
        report["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    path = output / "collision_preflight_check.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"saved: {path}", flush=True)
    return 0 if report["status"] == "open_snapshot_preflight_passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
