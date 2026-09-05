#!/usr/bin/env python3
"""Run the reviewed RGB-D-to-physical-pick simulation pipeline end to end.

Launch this script with the Isaac Lab environment's Python.  Child processes
that need GraspGenX/cuRobo are launched with GraspGenX's own virtualenv, so the
incompatible Torch environments remain isolated.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))


@dataclass(frozen=True)
class PipelinePaths:
    root: Path
    capture_root: Path
    capture: Path
    vlm: Path
    segmentation: Path
    prepared_map: Path
    raw_candidates: Path
    filtered_candidates: Path
    pregrasp: Path
    plan_trials: Path
    replay_trials: Path
    manifest: Path
    server_log: Path
    result_report: Path


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]
    report: Path
    accepted_statuses: tuple[str, ...] = ("success",)


class PipelineStageError(RuntimeError):
    pass


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture RGB-D, segment the target, generate/filter grasps, plan "
            "with cuRobo, and replay up to five candidates in Isaac Sim."
        )
    )
    parser.add_argument("--scene-usd", type=Path, required=True)
    prompt_source = parser.add_mutually_exclusive_group(required=True)
    prompt_source.add_argument(
        "--prompt", help="Manual SAM3 whole-object prompt; bypasses the VLM"
    )
    prompt_source.add_argument(
        "--target-object",
        help="Target object phrase supplied to the Ollama handover-parts prompt",
    )
    parser.add_argument("--grasp-part-prompt")
    parser.add_argument("--receive-part-prompt")
    parser.add_argument(
        "--ollama-model",
        help="Explicit Ollama vision model tag; required with --target-object",
    )
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-timeout-s", type=float, default=180.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--graspgenx-root",
        type=Path,
        default=Path("/home/suzutaro/GraspGenX"),
    )
    parser.add_argument("--graspgenx-python", type=Path)
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--server-start-timeout-s", type=float, default=180.0)
    parser.add_argument("--num-grasps", type=int, default=500)
    parser.add_argument("--topk", type=int, default=300)
    parser.add_argument("--collision-threshold", type=float, default=0.005)
    parser.add_argument("--max-pregrasp-candidates", type=int, default=100)
    parser.add_argument("--max-physical-trials", type=int, default=5)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--sam3-allow-download", action="store_true")
    parser.add_argument(
        "--no-show-results",
        action="store_true",
        help="Generate results.html but do not open it in a browser",
    )
    parser.add_argument(
        "--allow-reviewed-support-contact-preflight",
        action="store_true",
        help=(
            "Use the existing simulation-only reviewed finger/support contact "
            "policy during exact-grasp preflight"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only stages whose saved JSON report already says success",
    )
    args = parser.parse_args(argv)
    if args.prompt is not None and not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if args.target_object is not None and not args.target_object.strip():
        parser.error("--target-object must not be empty")
    if args.target_object is not None and not (args.ollama_model or "").strip():
        parser.error("--ollama-model is required with --target-object")
    if args.prompt is not None and args.ollama_model is not None:
        parser.error("--ollama-model is only valid with --target-object")
    if bool(args.grasp_part_prompt) != bool(args.receive_part_prompt):
        parser.error(
            "--grasp-part-prompt and --receive-part-prompt must be supplied together"
        )
    if args.target_object is not None and (
        args.grasp_part_prompt or args.receive_part_prompt
    ):
        parser.error("manual part prompts cannot be combined with --target-object")
    if args.ollama_timeout_s <= 0:
        parser.error("--ollama-timeout-s must be positive")
    if args.port <= 0 or args.port > 65535:
        parser.error("--port must be in 1..65535")
    if args.server_start_timeout_s <= 0:
        parser.error("--server-start-timeout-s must be positive")
    if min(args.num_grasps, args.topk, args.max_pregrasp_candidates) <= 0:
        parser.error("candidate counts must be positive")
    if args.topk > args.num_grasps:
        parser.error("--topk cannot exceed --num-grasps")
    if args.max_physical_trials <= 0:
        parser.error("--max-physical-trials must be positive")
    if args.collision_threshold <= 0:
        parser.error("--collision-threshold must be positive")
    return args


def pipeline_paths(root: Path) -> PipelinePaths:
    root = root.expanduser().resolve()
    capture_root = root / "capture"
    return PipelinePaths(
        root=root,
        capture_root=capture_root,
        capture=capture_root / "camera_0",
        vlm=root / "vlm",
        segmentation=capture_root / "sam3",
        prepared_map=root / "curobo_map",
        raw_candidates=root / "graspgenx_candidates",
        filtered_candidates=root / "graspgenx_candidates_filtered",
        pregrasp=root / "curobo_pregrasp",
        plan_trials=root / "curobo_grasp_lift_trials",
        replay_trials=root / "isaac_grasp_lift_trials",
        manifest=root / "pipeline_status.json",
        server_log=root / "graspgenx_server.log",
        result_report=root / "results.html",
    )


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def report_succeeded(path: Path, accepted_statuses: Iterable[str]) -> bool:
    report = _load_json(path)
    return report is not None and report.get("status") in set(accepted_statuses)


def _write_manifest(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _display_command(command: Iterable[str]) -> str:
    return shlex.join(str(item) for item in command)


def _archive_partial_stage_output(stage: Stage, manifest_path: Path) -> Path | None:
    output = stage.report.parent
    if not output.is_dir() or not any(output.iterdir()):
        return None
    archive_root = manifest_path.parent / "failed_stage_outputs"
    archive_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    archive = archive_root / f"{stage.name}_{timestamp}"
    suffix = 1
    while archive.exists():
        archive = archive_root / f"{stage.name}_{timestamp}_{suffix}"
        suffix += 1
    shutil.move(str(output), str(archive))
    return archive


def _run_stage(
    stage: Stage,
    *,
    project_root: Path,
    resume: bool,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> None:
    if resume and report_succeeded(stage.report, stage.accepted_statuses):
        print(f"=== {stage.name}: already successful; skipping ===", flush=True)
        manifest["stages"].append(
            {"name": stage.name, "status": "skipped_success", "report": str(stage.report)}
        )
        _write_manifest(manifest_path, manifest)
        return

    archived = _archive_partial_stage_output(stage, manifest_path) if resume else None
    if archived is not None:
        print(f"archived incomplete output to {archived}", flush=True)

    print(f"=== {stage.name} ===", flush=True)
    print(_display_command(stage.command), flush=True)
    started = time.monotonic()
    completed = subprocess.run(stage.command, cwd=project_root, check=False)
    elapsed = time.monotonic() - started
    saved_ok = report_succeeded(stage.report, stage.accepted_statuses)
    stage_status = "success" if completed.returncode == 0 and saved_ok else "failed"
    manifest["stages"].append(
        {
            "name": stage.name,
            "status": stage_status,
            "return_code": completed.returncode,
            "elapsed_s": elapsed,
            "report": str(stage.report),
            "previous_incomplete_output": str(archived) if archived else None,
        }
    )
    _write_manifest(manifest_path, manifest)
    if stage_status != "success":
        raise PipelineStageError(
            f"{stage.name} failed (return code {completed.returncode}); "
            f"inspect {stage.report}"
        )


def _port_accepts_connections(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def _wait_for_server(
    process: subprocess.Popen[Any], host: str, port: int, timeout_s: float
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise PipelineStageError(
                f"GraspGenX server exited during startup with code {return_code}"
            )
        if _port_accepts_connections(host, port):
            return
        time.sleep(0.25)
    raise PipelineStageError(
        f"GraspGenX server did not listen on {host}:{port} within {timeout_s:g}s"
    )


def _stop_server(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10.0)


def _resolve_graspgenx_python(args: argparse.Namespace) -> Path:
    if args.graspgenx_python is not None:
        return args.graspgenx_python.expanduser().resolve()
    root = args.graspgenx_root.expanduser().resolve()
    candidates = (root / ".venv" / "bin" / "python", root / ".venv" / "Scripts" / "python.exe")
    return next((path for path in candidates if path.is_file()), candidates[0])


def build_stages(
    args: argparse.Namespace,
    *,
    project_root: Path,
    paths: PipelinePaths,
    isaac_python: Path,
    graspgenx_python: Path,
) -> dict[str, Stage]:
    script = project_root / "scripts"
    scene_usd = args.scene_usd.expanduser().resolve()
    use_grasp_part = bool(args.target_object is not None or args.grasp_part_prompt)
    grasp_candidate_segmentation = (
        paths.segmentation / "parts" / "grasp_part"
        if use_grasp_part
        else paths.segmentation
    )
    grasp_candidate_segmentation_role = (
        "grasp_part" if use_grasp_part else "whole_object"
    )

    capture_command = [
        str(isaac_python),
        str(script / "isaac_capture_smoke.py"),
        "--scene-usd",
        str(scene_usd),
        "--output",
        str(paths.capture_root),
    ]
    if args.headless:
        capture_command.append("--headless")

    vlm_command = None
    if args.target_object is not None:
        vlm_command = [
            str(isaac_python),
            str(script / "ollama_discover_handover_parts.py"),
            "--capture",
            str(paths.capture),
            "--target-object",
            args.target_object,
            "--model",
            args.ollama_model,
            "--ollama-url",
            args.ollama_url,
            "--timeout-s",
            str(args.ollama_timeout_s),
            "--output",
            str(paths.vlm),
        ]

    sam3_command = [
        str(isaac_python),
        str(script / "sam3_segment_capture.py"),
        "--capture",
        str(paths.capture),
        "--output",
        str(paths.segmentation),
    ]
    if args.target_object is not None:
        sam3_command.extend(
            ["--vlm-result", str(paths.vlm / "vlm_part_discovery.json")]
        )
    else:
        sam3_command.extend(["--prompt", args.prompt])
        if args.grasp_part_prompt:
            sam3_command.extend(["--grasp-part-prompt", args.grasp_part_prompt])
            sam3_command.extend(["--receive-part-prompt", args.receive_part_prompt])
    if args.sam3_allow_download:
        sam3_command.append("--sam3-allow-download")

    map_command = [
        str(graspgenx_python),
        str(script / "curobo_map_capture.py"),
        "--capture",
        str(paths.capture),
        "--segmentation",
        str(paths.segmentation),
        "--output",
        str(paths.prepared_map),
    ]
    infer_command = [
        str(graspgenx_python),
        str(script / "graspgenx_infer_capture.py"),
        "--capture",
        str(paths.capture),
        "--segmentation",
        str(grasp_candidate_segmentation),
        "--segmentation-role",
        grasp_candidate_segmentation_role,
        "--output",
        str(paths.raw_candidates),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--num-grasps",
        str(args.num_grasps),
        "--topk",
        str(args.topk),
    ]
    filter_command = [
        str(graspgenx_python),
        str(script / "graspgenx_filter_capture.py"),
        "--capture",
        str(paths.capture),
        "--segmentation",
        str(paths.segmentation),
        "--candidates",
        str(paths.raw_candidates),
        "--output",
        str(paths.filtered_candidates),
        "--collision-threshold",
        str(args.collision_threshold),
    ]
    pregrasp_command = [
        str(graspgenx_python),
        str(script / "curobo_plan_pregrasp_a.py"),
        "--scene-backend",
        "observed_pointcloud_mesh",
        "--prepared-map",
        str(paths.prepared_map),
        "--capture",
        str(paths.capture),
        "--candidates",
        str(paths.filtered_candidates),
        "--output",
        str(paths.pregrasp),
        "--max-candidates",
        str(args.max_pregrasp_candidates),
    ]
    plan_trials_command = [
        str(graspgenx_python),
        str(script / "curobo_plan_grasp_lift_trials.py"),
        "--capture",
        str(paths.capture),
        "--segmentation",
        str(paths.segmentation),
        "--pregrasp-plan",
        str(paths.pregrasp),
        "--output",
        str(paths.plan_trials),
        "--max-physical-trials",
        str(args.max_physical_trials),
    ]
    if args.allow_reviewed_support_contact_preflight:
        plan_trials_command.append("--allow-reviewed-support-contact-preflight")

    replay_command = [
        str(isaac_python),
        str(script / "isaac_replay_grasp_lift_trials.py"),
        "--capture",
        str(paths.capture),
        "--plan-trials",
        str(paths.plan_trials),
        "--scene-usd",
        str(scene_usd),
        "--output",
        str(paths.replay_trials),
        "--max-physical-trials",
        str(args.max_physical_trials),
        "--finger-drive-preset",
        "isaaclab-franka",
        "--simulation-only",
    ]
    if args.headless:
        replay_command.append("--headless")

    stages = {
        "capture_rgbd": Stage(
            "capture_rgbd",
            tuple(capture_command),
            paths.capture_root / "run_status.json",
        ),
        "sam3_segmentation": Stage(
            "sam3_segmentation",
            tuple(sam3_command),
            paths.segmentation / "run_status.json",
        ),
        "curobo_map": Stage(
            "curobo_map", tuple(map_command), paths.prepared_map / "esdf_check.json"
        ),
        "graspgenx_inference": Stage(
            "graspgenx_inference",
            tuple(infer_command),
            paths.raw_candidates / "graspgenx_check.json",
        ),
        "static_collision_filter": Stage(
            "static_collision_filter",
            tuple(filter_command),
            paths.filtered_candidates / "collision_filter_check.json",
        ),
        "curobo_pregrasp": Stage(
            "curobo_pregrasp",
            tuple(pregrasp_command),
            paths.pregrasp / "pregrasp_plan_check.json",
        ),
        "curobo_grasp_lift_trials": Stage(
            "curobo_grasp_lift_trials",
            tuple(plan_trials_command),
            paths.plan_trials / "grasp_lift_trial_plans.json",
            ("plans_ready",),
        ),
        "isaac_physical_trials": Stage(
            "isaac_physical_trials",
            tuple(replay_command),
            paths.replay_trials / "grasp_lift_physical_trials.json",
        ),
    }
    if vlm_command is not None:
        stages = {
            "capture_rgbd": stages["capture_rgbd"],
            "ollama_vlm": Stage(
                "ollama_vlm",
                tuple(vlm_command),
                paths.vlm / "vlm_part_discovery.json",
            ),
            **{name: stage for name, stage in stages.items() if name != "capture_rgbd"},
        }
    return stages


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    paths = pipeline_paths(args.output)
    scene_usd = args.scene_usd.expanduser().resolve()
    graspgenx_root = args.graspgenx_root.expanduser().resolve()
    graspgenx_python = _resolve_graspgenx_python(args)
    isaac_python = Path(sys.executable).resolve()

    if not scene_usd.is_file():
        raise FileNotFoundError(f"scene USD does not exist: {scene_usd}")
    if not graspgenx_root.is_dir():
        raise FileNotFoundError(f"GraspGenX root does not exist: {graspgenx_root}")
    if not graspgenx_python.is_file():
        raise FileNotFoundError(f"GraspGenX Python does not exist: {graspgenx_python}")
    if paths.root.exists() and not paths.root.is_dir():
        raise NotADirectoryError(f"pipeline output is not a directory: {paths.root}")
    if paths.root.exists() and any(paths.root.iterdir()) and not args.resume:
        raise FileExistsError(
            f"pipeline output already contains files: {paths.root}; "
            "use a new --output or pass --resume"
        )
    paths.root.mkdir(parents=True, exist_ok=True)

    stages = build_stages(
        args,
        project_root=project_root,
        paths=paths,
        isaac_python=isaac_python,
        graspgenx_python=graspgenx_python,
    )
    manifest: dict[str, Any] = {
        "status": "running",
        "simulation_only": True,
        "inputs": {
            "scene_usd": str(scene_usd),
            "prompt": args.prompt,
            "target_object": args.target_object,
            "ollama_model": args.ollama_model,
            "ollama_url": args.ollama_url if args.target_object else None,
            "isaac_python": str(isaac_python),
            "graspgenx_python": str(graspgenx_python),
        },
        "policy": {
            "num_grasps": args.num_grasps,
            "topk": args.topk,
            "collision_threshold_m": args.collision_threshold,
            "max_pregrasp_candidates": args.max_pregrasp_candidates,
            "max_physical_trials": args.max_physical_trials,
            "finger_drive_preset": "isaaclab-franka",
            "grasp_candidate_segmentation": (
                "grasp_part" if args.target_object is not None or args.grasp_part_prompt
                else "whole_object"
            ),
            "grasp_candidate_segmentation_path": str(
                paths.segmentation / "parts" / "grasp_part"
                if args.target_object is not None or args.grasp_part_prompt
                else paths.segmentation
            ),
            "whole_object_segmentation_path": str(paths.segmentation),
            "whole_object_uses": [
                "target_removal_from_observed_collision_map",
                "static_surrounding-scene_collision_filter",
                "attached_object_geometry",
            ],
            "grasp_part_fallback_to_whole_object": False,
            "allow_reviewed_support_contact_preflight": bool(
                args.allow_reviewed_support_contact_preflight
            ),
            "stop_on_first_failed_stage": True,
            "stop_on_first_physical_pick": True,
        },
        "stages": [],
    }
    _write_manifest(paths.manifest, manifest)

    exit_code = 0
    try:
        for name in ("capture_rgbd", "ollama_vlm", "sam3_segmentation", "curobo_map"):
            if name in stages:
                _run_stage(
                    stages[name],
                    project_root=project_root,
                    resume=args.resume,
                    manifest=manifest,
                    manifest_path=paths.manifest,
                )

        infer_stage = stages["graspgenx_inference"]
        inference_already_done = args.resume and report_succeeded(
            infer_stage.report, infer_stage.accepted_statuses
        )
        if inference_already_done:
            _run_stage(
                infer_stage,
                project_root=project_root,
                resume=True,
                manifest=manifest,
                manifest_path=paths.manifest,
            )
        else:
            if _port_accepts_connections("127.0.0.1", args.port):
                raise PipelineStageError(
                    f"port {args.port} is already in use. Stop the existing "
                    "GraspGenX server with Ctrl+C, then rerun with --resume; "
                    "this runner only terminates a server that it started itself."
                )
            server_command = [
                str(graspgenx_python),
                str(graspgenx_root / "client-server" / "graspgenx_server.py"),
                "--config",
                str(graspgenx_root / "ext" / "graspgenx_checkpoints" / "release"),
                "--assets_dir",
                str(graspgenx_root / "assets"),
                "--port",
                str(args.port),
                "--default_gripper",
                "franka_panda",
            ]
            print("=== starting managed GraspGenX server ===", flush=True)
            print(_display_command(server_command), flush=True)
            with paths.server_log.open("w", encoding="utf-8") as server_log:
                server_process = subprocess.Popen(
                    server_command,
                    cwd=graspgenx_root,
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                )
                try:
                    _wait_for_server(
                        server_process,
                        "127.0.0.1",
                        args.port,
                        args.server_start_timeout_s,
                    )
                    _run_stage(
                        infer_stage,
                        project_root=project_root,
                        resume=False,
                        manifest=manifest,
                        manifest_path=paths.manifest,
                    )
                finally:
                    print("=== stopping managed GraspGenX server ===", flush=True)
                    _stop_server(server_process)

        for name in (
            "static_collision_filter",
            "curobo_pregrasp",
            "curobo_grasp_lift_trials",
            "isaac_physical_trials",
        ):
            _run_stage(
                stages[name],
                project_root=project_root,
                resume=args.resume,
                manifest=manifest,
                manifest_path=paths.manifest,
            )
    except Exception as exc:
        exit_code = 2
        manifest["status"] = "failed"
        manifest["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        print(f"pipeline failed: {exc}", file=sys.stderr, flush=True)
    else:
        manifest["status"] = "success"
        manifest["result"] = {
            "physical_trial_report": str(
                paths.replay_trials / "grasp_lift_physical_trials.json"
            )
        }
        print("RGB-D-to-grasp simulation pipeline succeeded", flush=True)
    finally:
        manifest["result_report"] = str(paths.result_report)
        _write_manifest(paths.manifest, manifest)
        try:
            from panda_handover.result_report import (
                generate_result_report,
                open_result_report,
            )

            generate_result_report(
                paths.root,
                manifest_path=paths.manifest,
                destination=paths.result_report,
            )
            print(f"results: {paths.result_report}", flush=True)
            if not args.headless and not args.no_show_results:
                opened = open_result_report(paths.result_report)
                if not opened:
                    print(
                        "the browser did not confirm opening the report; use the path above",
                        flush=True,
                    )
        except Exception as report_error:
            manifest["result_report_error"] = {
                "type": type(report_error).__name__,
                "message": str(report_error),
            }
            _write_manifest(paths.manifest, manifest)
            print(
                f"warning: could not generate result report: {report_error}",
                file=sys.stderr,
                flush=True,
            )
    print(f"saved: {paths.manifest}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
