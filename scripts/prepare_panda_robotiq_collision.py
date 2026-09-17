#!/usr/bin/env python3
"""Fit installed open Robotiq colliders using cuRobo's official sphere fitter."""

import argparse
import json
from pathlib import Path
import sys
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from panda_handover.robotiq_model import file_identity
from panda_handover.robotiq_collision import open_snapshot_config, snapshot_meshes_in_hand, validated_spheres, vertex_coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graspgenx-root", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--prepared-model", type=Path, required=True)
    parser.add_argument("--tool-fk-check", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    output = args.output.resolve()
    output.mkdir(parents=True)
    report = {"status": "in_progress", "scope": "open-gripper collision snapshot draft",
              "safety": {"simulation_only": True, "profile_ready": False, "safe_to_plan": False,
                         "safe_to_execute": False, "robot_moved": False, "source_assets_modified": False,
                         "gripper_close_supported": False, "lift_supported": False,
                         "physx_cooked_geometry_equivalence_verified": False}}
    try:
        import trimesh
        import yaml
        from curobo._src.geom.sphere_fit.types import SphereFitType
        from curobo._src.geom.sphere_fit.fit_spheres import fit_spheres_to_mesh

        prepared = json.loads(args.prepared_model.read_text(encoding="utf-8"))
        check = json.loads(args.tool_fk_check.read_text(encoding="utf-8"))
        if prepared.get("status") != "fk_model_prepared" or check.get("status") != "tool_mount_alignment_passed":
            raise ValueError("Passing model preparation and native tool/mount FK are required")
        if Path(check["inputs"]["prepared_model"]).resolve() != args.prepared_model.resolve():
            raise ValueError("Tool FK check is for another prepared model")
        expected_frames = {f"panda_link{i}" for i in range(8)} | {"panda_hand", "robotiq_arg2f_base_link"}
        if set(check["frames"]) != expected_frames or not all(v.get("passed") is True for v in check["frames"].values()):
            raise ValueError("Tool FK did not pass every required frame")
        for source in list(prepared["sources"].values()) + [prepared["urdf"], prepared["canonical_collision_mesh"]]:
            if file_identity(source["path"]) != source:
                raise ValueError(f"Prepared source changed: {source['path']}")
        if check["inputs"]["urdf"] != prepared["urdf"]:
            raise ValueError("Tool FK used a different URDF")
        evidence_path = Path(prepared["sources"]["evidence"]["path"])
        if Path(check["inputs"]["evidence"]).resolve() != evidence_path.resolve():
            raise ValueError("Tool FK used different measured evidence")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        geometry = json.loads(args.geometry.read_text(encoding="utf-8"))
        pieces = snapshot_meshes_in_hand(geometry, evidence)
        all_spheres, hulls, summaries = [], [], []
        for i, piece in enumerate(pieces):
            mesh = trimesh.Trimesh(vertices=piece["vertices"], faces=piece["faces"], process=False)
            # Each USD collider is convexHull. Do NOT hull the whole gripper:
            # that would fill the space between the fingers.
            hull = mesh.convex_hull
            fitted = fit_spheres_to_mesh(hull, sphere_density=1.0, fit_type=SphereFitType.VOXEL)
            centers = fitted.centers.detach().cpu().numpy()
            radii = fitted.radii.detach().cpu().numpy()
            if centers.ndim != 2 or centers.shape[1] != 3 or radii.shape != (len(centers),):
                raise ValueError("Unexpected official sphere-fit result shape")
            spheres = validated_spheres([{"center": center.tolist(), "radius": float(radius)}
                for center, radius in zip(centers, radii)])
            hull.export(str(output / f"collider_{i:02d}_hand_frame.ply"))
            summary = {"path": piece["path"], "sphere_count": len(spheres),
                       "coverage": vertex_coverage(hull.vertices, spheres)}
            summaries.append(summary)
            hulls.append(hull)
            all_spheres.extend(spheres)
            print(json.dumps(summary), flush=True)
        native_mesh = trimesh.util.concatenate(hulls)
        canonical_mesh = native_mesh.copy()
        canonical_mesh.apply_transform(np.asarray(prepared["T_grasp_panda_hand_proposed"]))
        canonical_mesh.export(str(output / "installed_gripper_open_grasp_frame.obj"))
        stock_path = args.graspgenx_root.resolve() / "ext/curobo/curobo/content/configs/robot/franka.yml"
        stock = yaml.safe_load(stock_path.read_text(encoding="utf-8"))
        cfg = open_snapshot_config(stock, Path(prepared["urdf"]["path"]), all_spheres)
        yaml_path = output / "panda_robotiq85_open_snapshot.yml"
        yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        report.update(status="open_snapshot_collision_draft_prepared", yaml=file_identity(yaml_path),
                      urdf=prepared["urdf"], colliders=summaries, gripper_sphere_count=len(all_spheres),
                      sources={"geometry": file_identity(args.geometry), "evidence": file_identity(evidence_path),
                               "prepared_model": file_identity(args.prepared_model),
                               "tool_fk_check": file_identity(args.tool_fk_check), "stock_config": file_identity(stock_path)})
        report["parameters"] = {"sphere_fit": "cuRobo VOXEL", "sphere_density": 1.0,
            "convex_hull_scope": "each installed collider separately; not exact PhysX cooking",
            "gripper_geometry_frame": "panda_hand, fixed measured open snapshot",
            "arm_spheres_and_arm_pairs": "official franka.yml unchanged",
            "hand_self_collision_padding_m": 0.0, "old_panda_hand_padding_removed_m": 0.02,
            "wrist_hand_adjacency_ignores": "inherited from stock; still require review for new hand"}
        report["next_gate"] = "Run collision-profile FK and official self-collision preflight, then inspect sphere coverage. Do not run grasp/lift or the normal end-to-end pipeline with this draft."
    except Exception as error:
        report["status"] = "failure"
        report["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    path = output / "collision_model_check.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"saved: {path}", flush=True)
    return 0 if report["status"] == "open_snapshot_collision_draft_prepared" else 2


if __name__ == "__main__":
    raise SystemExit(main())
