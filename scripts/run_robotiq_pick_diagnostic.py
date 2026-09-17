#!/usr/bin/env python3
"""Generate fresh Robotiq candidates, then run up to five isolated Isaac trials.

Reuse saved RGB-D/SAM3 only. This is NOT the collision-aware end-to-end pipeline.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from panda_handover.robotiq_model import file_identity
from panda_handover.robotiq_trial import verified_trial_model
from run_sim_grasp_pipeline import _port_accepts_connections, _stop_server, _wait_for_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collision-model", type=Path, required=True, help="Passing open-snapshot collision_model_check.json")
    parser.add_argument("--reference-plan", type=Path, required=True,
                        help="Old grasp_lift_plan_check.json: read capture/mask paths ONLY, never its trajectory or poses")
    parser.add_argument("--segmentation", type=Path, help="Optional replacement SAM3 mask directory")
    parser.add_argument("--graspgenx-root", type=Path, default=Path("/home/suzutaro/GraspGenX"))
    parser.add_argument("--isaac-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-trials", type=int, default=5)
    parser.add_argument("--num-grasps", type=int, default=500)
    parser.add_argument("--topk", type=int, default=300)
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--replay-physics", choices=("default", "cpu"), default="default")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--simulation-only", action="store_true", required=True)
    parser.add_argument("--allow-collision-unchecked-simulation", action="store_true", required=True)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.max_trials <= 5 or min(args.num_grasps, args.topk) <= 0:
        parser.error("Use a new output directory, positive candidate counts, and 1..5 trials")
    project = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True)
    report = {"status": "in_progress", "attempts": [],
              "policy": {"maximum_trials": args.max_trials, "stop_at_first_physical_pick": True,
                         "candidate_order": "original GraspGenX score descending",
                         "candidate_specific_parameter_tuning": False,
                         "world_and_self_collision_avoidance_guaranteed": False,
                         "grasp_retention": "physical contact only; no attachment"},
              "safety": {"simulation_only": True, "profile_ready": False,
                         "safe_to_execute_on_hardware": False, "normal_pipeline_changed": False,
                         "capture_and_segmentation_reused": True}, "selected_success": None}
    status_path = output / "robotiq_pick_trials.json"

    def save():
        status_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def run(command, log, cwd):
        print("Executing: " + " ".join(map(str, command)), flush=True)
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(list(map(str, command)), cwd=cwd, stdout=stream, stderr=subprocess.STDOUT)
        print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]), flush=True)
        return result.returncode

    save()
    try:
        collision = json.loads(args.collision_model.read_text(encoding="utf-8"))
        runtime_path = args.collision_model.parent / "runtime_preflight/collision_preflight_check.json"
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if collision.get("status") != "open_snapshot_collision_draft_prepared" or runtime.get("status") != "open_snapshot_preflight_passed":
            raise ValueError("Existing open-snapshot model preflight must pass first")
        if runtime["inputs"]["collision_model"] != file_identity(args.collision_model):
            raise ValueError("Runtime preflight refers to a changed collision draft")
        for key in ("prepared_model", "tool_fk_check", "evidence", "geometry"):
            source = collision["sources"][key]
            if file_identity(source["path"]) != source:
                raise ValueError(f"Verified source changed: {key}")
        prepared = Path(collision["sources"]["prepared_model"]["path"])
        tool_fk = Path(collision["sources"]["tool_fk_check"]["path"])
        connection = Path(collision["sources"]["geometry"]["path"]).parent
        swap = json.loads((connection / "gripper_swap_check.json").read_text(encoding="utf-8"))
        if swap.get("status") != "success":
            raise ValueError("Gripper open/close smoke test must have passed")
        scene = Path(swap["inputs"]["scene_usd"])
        verified_trial_model(prepared, tool_fk, scene)
        reference = json.loads(args.reference_plan.read_text(encoding="utf-8"))
        capture = Path(reference["inputs"]["capture"])
        segmentation = args.segmentation or Path(reference["inputs"]["segmentation"])
        for name in ("points_camera.npy", "T_world_camera.npy", "robot_state.json", "panda_joint_positions.npy"):
            if not (capture / name).is_file():
                raise FileNotFoundError(capture / name)
        if not (segmentation / "union_mask.npy").is_file():
            raise FileNotFoundError(segmentation / "union_mask.npy")
        root = args.graspgenx_root.resolve()
        grasp_python = root / ".venv/bin/python"
        if not grasp_python.is_file() or not args.isaac_python.is_file():
            raise FileNotFoundError("Existing GraspGenX and Isaac Python executables are required")
        report["inputs"] = {"scene_usd": str(scene), "capture": str(capture), "segmentation": str(segmentation),
                            "reference_plan_used_for_paths_only": str(args.reference_plan.resolve()),
                            "prepared_model": str(prepared), "tool_fk_check": str(tool_fk)}
        save()
        if _port_accepts_connections("127.0.0.1", args.port):
            raise RuntimeError(f"Port {args.port} is busy: stop the existing server or choose --port; it will not be killed")
        candidates = output / "graspgenx_candidates"
        with (output / "graspgenx_server.log").open("w", encoding="utf-8") as server_log:
            process = subprocess.Popen([
                str(grasp_python), str(root / "client-server/graspgenx_server.py"),
                "--config", str(root / "ext/graspgenx_checkpoints/release"),
                "--assets_dir", str(root / "assets"), "--port", str(args.port),
                "--default_gripper", "robotiq_2f_85"], cwd=root, stdout=server_log, stderr=subprocess.STDOUT)
            try:
                print("Starting managed Robotiq GraspGenX server; see graspgenx_server.log", flush=True)
                _wait_for_server(process, "127.0.0.1", args.port, 180)
                command = [grasp_python, project / "scripts/graspgenx_infer_capture.py",
                           "--capture", capture, "--segmentation", segmentation,
                           "--segmentation-role", "grasp_part" if "grasp_part" in segmentation.parts else "whole_object",
                           "--output", candidates, "--gripper-name", "robotiq_2f_85",
                           "--num-grasps", args.num_grasps, "--topk", args.topk, "--port", args.port]
                if run(command, output / "inference.log", root) != 0:
                    raise RuntimeError("Robotiq inference failed; inspect inference.log")
            finally:
                _stop_server(process)  # Only the process started above; free its GPU memory before Isaac.
        count = len(np.load(candidates / "scores.npy"))
        for rank in range(min(args.max_trials, count)):
            trial = output / f"trial_{rank + 1:02d}"
            command = [args.isaac_python, project / "scripts/isaac_try_robotiq_pick.py",
                       "--scene-usd", scene, "--capture", capture, "--candidates", candidates,
                       "--prepared-model", prepared, "--tool-fk-check", tool_fk,
                       "--candidate-rank", rank, "--output", trial, "--replay-physics", args.replay_physics,
                       "--simulation-only", "--allow-collision-unchecked-simulation"]
            if args.headless:
                command.append("--headless")
            code = run(command, output / f"trial_{rank + 1:02d}.log", project)
            trial_path = trial / "robotiq_pick_check.json"
            result = json.loads(trial_path.read_text(encoding="utf-8")) if trial_path.exists() else {}
            attempt = {"score_rank": rank, "return_code": code, "status": result.get("status", "missing_report"),
                       "candidate": result.get("candidate"), "rejection": result.get("rejection"),
                       "object": result.get("object"), "report": str(trial_path)}
            report["attempts"].append(attempt)
            save()
            if code == 0 and result.get("status") == "physical_pick_observed":
                report.update(status="physical_pick_observed", selected_success=attempt)
                return 0
            if code != 2 or result.get("status") not in ("candidate_rejected", "physical_pick_not_observed"):
                raise RuntimeError(f"Trial {rank + 1} had a non-physical error; stop instead of hiding it with retries")
        report["status"] = "physical_pick_not_observed"
        return 2
    except Exception as error:
        report["status"] = "failure"
        report["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        print(traceback.format_exc(), flush=True)
        return 1
    finally:
        save()
        print(f"{report['status']}; saved: {status_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
