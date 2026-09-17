#!/usr/bin/env python3
"""Prepare an isolated FK-only model from official Panda and Robotiq URDFs."""

import argparse
import json
from pathlib import Path
import sys
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from panda_handover.robotiq_model import compose_fk_urdf, file_identity, installed_mount


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graspgenx-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--arm-fk-check", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    output = args.output.resolve()
    output.mkdir(parents=True)
    report = {"status": "in_progress", "purpose": "FK-only URDF composition, not a planning profile",
              "safety": {"profile_ready": False, "trajectory_saved": False, "robot_moved": False,
                         "collision_geometry_equivalence_verified": False,
                         "source_assets_modified": False}}
    try:
        from graspgenx.x_grippers import resolve_gripper_asset_dir
        root = args.graspgenx_root.resolve()
        arm_path = root / "ext/curobo/curobo/content/assets/robot/franka_description/franka_panda.urdf"
        gripper_dir = Path(resolve_gripper_asset_dir("robotiq_2f_85", assets_dir=str(root / "assets"))).resolve()
        evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
        check = json.loads(args.arm_fk_check.read_text(encoding="utf-8"))
        if (check.get("status") != "arm_alignment_passed"
                or Path(check["inputs"]["evidence"]).resolve() != args.evidence.resolve()
                or set(check.get("frames", {})) != {f"panda_link{i}" for i in range(8)}
                or not all(frame.get("passed") is True for frame in check["frames"].values())):
            raise ValueError("A passing arm FK check for this exact evidence path is required")
        report["mount"] = installed_mount(evidence)
        config = json.loads((gripper_dir / "config.json").read_text(encoding="utf-8"))
        canonical_mesh = file_identity(gripper_dir / "coll_mesh.obj")
        xml, transform, meshes = compose_fk_urdf(arm_path, gripper_dir / "gripper.urdf", config)
        report["sources"] = {name: file_identity(path) for name, path in (
            ("evidence", args.evidence), ("arm_fk_check", args.arm_fk_check),
            ("arm_urdf", arm_path), ("gripper_urdf", gripper_dir / "gripper.urdf"),
            ("gripper_config", gripper_dir / "config.json"))}
        report["canonical_collision_mesh"] = canonical_mesh
        report["mesh_files"] = meshes
        urdf = output / "panda_robotiq85_fk_only.urdf"
        urdf.write_text(xml + "\n", encoding="utf-8")
        report["urdf"] = file_identity(urdf)
        np.save(output / "T_grasp_panda_hand_proposed.npy", transform)
        report["T_grasp_panda_hand_proposed"] = transform.tolist()
        report["grasp_frame_status"] = "derived from source world_joint; canonical mesh equivalence still unverified"
        report["status"] = "fk_model_prepared"
        report["next_gate"] = "Run tool/mount FK and inspect exported installed collision meshes before building a collision-ready profile. Do not use this URDF for robot masking, planning or replay."
    except Exception as error:
        report["status"] = "failure"
        report["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    path = output / "model_preparation_check.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"saved: {path}", flush=True)
    return 0 if report["status"] == "fk_model_prepared" else 2


if __name__ == "__main__":
    raise SystemExit(main())
