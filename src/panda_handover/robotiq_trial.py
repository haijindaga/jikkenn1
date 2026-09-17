"""Pure helpers for an isolated, collision-unchecked Robotiq simulation trial."""

import numpy as np

from .grasp_candidates import pose_quality
from .geometry import quaternion_wxyz_from_rotation_matrix
from .gripper_swap import captured_arm_positions
from .robotiq_model import file_identity, installed_mount


def rigid_transform(value):
    value = np.asarray(value, dtype=float)
    quality = pose_quality(value[None])
    if not quality["finite"] or any(v > 1e-5 for k, v in quality.items() if k != "finite"):
        raise ValueError("Invalid rigid transform; do not repair or guess offsets")
    return value


def verified_trial_model(prepared_path, fk_path, scene_path):
    import json
    from pathlib import Path

    prepared = json.loads(Path(prepared_path).read_text(encoding="utf-8"))
    fk = json.loads(Path(fk_path).read_text(encoding="utf-8"))
    if prepared.get("status") != "fk_model_prepared" or fk.get("status") != "tool_mount_alignment_passed":
        raise ValueError("Passing prepared model and tool/mount FK are required")
    if Path(fk["inputs"]["prepared_model"]).resolve() != Path(prepared_path).resolve():
        raise ValueError("Tool FK refers to a different model")
    expected = {f"panda_link{i}" for i in range(8)} | {"panda_hand", "robotiq_arg2f_base_link"}
    if set(fk["frames"]) != expected or not all(f["passed"] is True for f in fk["frames"].values()):
        raise ValueError("Incomplete tool/mount FK check")
    for source in [*prepared["sources"].values(), prepared["urdf"],
                   prepared["canonical_collision_mesh"], *prepared["mesh_files"]]:
        if file_identity(source["path"]) != source:
            raise ValueError(f"Prepared source changed: {source['path']}")
    if fk["inputs"]["urdf"] != prepared["urdf"]:
        raise ValueError("Tool FK used a different URDF")
    if Path(fk["inputs"]["evidence"]).resolve() != Path(prepared["sources"]["evidence"]["path"]).resolve():
        raise ValueError("Tool FK used different evidence")
    evidence = json.loads(Path(prepared["sources"]["evidence"]["path"]).read_text(encoding="utf-8"))
    installed_mount(evidence)
    if file_identity(scene_path)["sha256"] != evidence["source_scene_sha256"]:
        raise ValueError("Scene changed since the verified gripper run; collect evidence again")
    transform = rigid_transform(prepared["T_grasp_panda_hand_proposed"])
    # Source world_joint Rz(+90 deg), supported by the existing mount inspection.
    expected_transform = np.eye(4)
    expected_transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    if not np.allclose(transform, expected_transform, atol=1e-5):
        raise ValueError("Unreviewed canonical-to-hand transform")
    return evidence, transform


def candidate_in_world(candidates, capture, rank):
    import json
    from pathlib import Path

    candidates, capture = Path(candidates), Path(capture)
    report = json.loads((candidates / "graspgenx_check.json").read_text(encoding="utf-8"))
    if report.get("status") != "success" or report.get("gripper") != "robotiq_2f_85":
        raise ValueError("Fresh Robotiq 2F-85 candidates required; old Panda poses cannot be reused")
    scores = np.load(candidates / "scores.npy")
    poses = np.load(candidates / "grasps_world.npy")
    camera = np.load(candidates / "grasps_camera.npy")
    if (scores.ndim != 1 or poses.shape != (len(scores), 4, 4)
            or camera.shape != poses.shape or not np.all(np.isfinite(scores))):
        raise ValueError("Invalid candidate arrays")
    if report["candidates"]["count"] != len(scores) or not 0 <= rank < len(scores):
        raise ValueError("Candidate rank/count invalid")
    world_camera = rigid_transform(np.load(capture / "T_world_camera.npy"))
    if not np.allclose(poses, world_camera @ camera, atol=1e-5):
        raise ValueError("Candidates do not match this capture's camera transform")
    index = int(np.argsort(-scores, kind="stable")[rank])
    return rigid_transform(poses[index]), index, float(scores[index])


def contact_waypoints(world_grasp, grasp_hand, approach_m=0.1, lift_m=0.15, step_m=0.005):
    if not all(np.isfinite([approach_m, lift_m, step_m])) or min(approach_m, lift_m, step_m) <= 0:
        raise ValueError("Positive finite motion distances required")
    grasp, hand = rigid_transform(world_grasp), rigid_transform(grasp_hand)
    # GraspGenX canonical +Z is the approach axis; back off BEFORE tool conversion.
    pregrasp = grasp.copy()
    pregrasp[:3, 3] -= approach_m * grasp[:3, 2]
    lift = grasp.copy()
    lift[2, 3] += lift_m  # world +Z, scene is checked to be Z-up/metre

    def line(start, end, distance):
        values = []
        for fraction in np.linspace(0, 1, int(np.ceil(distance / step_m)) + 1):
            pose = start.copy()
            pose[:3, 3] = start[:3, 3] * (1 - fraction) + end[:3, 3] * fraction
            values.append(pose @ hand)
        return np.asarray(values)

    return {"contact": line(pregrasp, grasp, approach_m), "lift": line(grasp, lift, lift_m)}


def solve_waypoints(solver, poses, start, lower, upper, max_step_rad=0.35):
    """Call the official Lula IK with the preceding solution as warm start."""
    current = np.asarray(start, dtype=float)
    output = [current.copy()]
    for pose in poses:
        q, success = solver.compute_inverse_kinematics(
            "panda_hand", pose[:3, 3], quaternion_wxyz_from_rotation_matrix(pose[:3, :3]),
            warm_start=current)
        q = np.asarray(q, dtype=float)
        if not success:
            return None, "ik_failed"
        if q.shape != current.shape or not np.all(np.isfinite(q)):
            raise RuntimeError("Invalid Lula IK solution")
        if np.any(q < lower) or np.any(q > upper):
            return None, "joint_limit_rejected"
        if len(output) > 1 and np.max(np.abs(q - current)) > max_step_rad:
            return None, "ik_branch_jump_rejected"
        output.append(q.copy())
        current = q
    return np.asarray(output), None


def capture_arm_matches(evidence, names, q):
    reference = captured_arm_positions(evidence["joint_names"], evidence["joint_positions"])
    captured = captured_arm_positions(names, q)
    if np.max(np.abs(reference - captured)) > 0.1:
        raise ValueError("Capture and verified gripper run have different arm postures")
    return captured


def pick_result(phase, target_positions, baseline_z, threshold_m=0.05):
    phase = np.asarray(phase)
    positions = np.asarray(target_positions, dtype=float)
    if positions.shape != (len(phase), 3) or not np.all(np.isfinite(positions)):
        raise ValueError("Invalid measured target history")
    hold = positions[phase == "hold", 2] - baseline_z
    lift = positions[phase == "lift", 2] - baseline_z
    if not len(hold) or not len(lift):
        raise ValueError("Incomplete lift/hold observations")
    return {"physical_pick_observed": bool(np.min(hold) >= threshold_m and np.max(lift) >= threshold_m),
            "peak_object_lift_m": float(np.max(lift)), "lift_after_hold_m": float(hold[-1]),
            "minimum_lift_during_hold_m": float(np.min(hold)), "success_threshold_m": threshold_m}
