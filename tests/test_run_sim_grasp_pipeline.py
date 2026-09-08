from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "run_sim_grasp_pipeline.py"
SPEC = importlib.util.spec_from_file_location("run_sim_grasp_pipeline_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RunSimGraspPipelineTests(unittest.TestCase):
    def test_builds_complete_ordered_pipeline_with_isolated_pythons(self) -> None:
        args = MODULE.parse_args(
            [
                "--scene-usd",
                str(PROJECT / "scene.usda"),
                "--prompt",
                "scissors",
                "--output",
                str(PROJECT / "outputs" / "e2e"),
                "--allow-reviewed-support-contact-preflight",
            ]
        )
        paths = MODULE.pipeline_paths(args.output)
        stages = MODULE.build_stages(
            args,
            project_root=PROJECT,
            paths=paths,
            isaac_python=Path("/envs/isaac/bin/python"),
            graspgenx_python=Path("/graspgenx/.venv/bin/python"),
        )
        self.assertEqual(
            list(stages),
            [
                "capture_rgbd",
                "sam3_segmentation",
                "curobo_map",
                "graspgenx_inference",
                "static_collision_filter",
                "curobo_pregrasp",
                "curobo_grasp_lift_trials",
                "isaac_physical_trials",
            ],
        )
        self.assertEqual(
            stages["capture_rgbd"].command[0], str(Path("/envs/isaac/bin/python"))
        )
        self.assertEqual(
            stages["isaac_physical_trials"].command[0],
            str(Path("/envs/isaac/bin/python")),
        )
        self.assertEqual(
            stages["graspgenx_inference"].command[0],
            str(Path("/graspgenx/.venv/bin/python")),
        )
        self.assertIn("--simulation-only", stages["isaac_physical_trials"].command)
        replay = stages["isaac_physical_trials"].command
        self.assertEqual(replay[replay.index("--finger-drive-scale") + 1], "1.0")
        self.assertEqual(
            replay[replay.index("--grasp-retention-mode") + 1], "physics"
        )
        self.assertIn(
            "--allow-reviewed-support-contact-preflight",
            stages["curobo_grasp_lift_trials"].command,
        )
        self.assertIn("--prompt", stages["sam3_segmentation"].command)
        infer = stages["graspgenx_inference"].command
        self.assertEqual(
            infer[infer.index("--segmentation") + 1], str(paths.segmentation)
        )
        self.assertEqual(
            infer[infer.index("--segmentation-role") + 1], "whole_object"
        )

    def test_vlm_mode_inserts_ollama_before_sam3(self) -> None:
        args = MODULE.parse_args(
            [
                "--scene-usd",
                str(PROJECT / "scene.usda"),
                "--target-object",
                "scissors",
                "--ollama-model",
                "qwen3-vl:4b",
                "--output",
                str(PROJECT / "outputs" / "e2e"),
            ]
        )
        paths = MODULE.pipeline_paths(args.output)
        stages = MODULE.build_stages(
            args,
            project_root=PROJECT,
            paths=paths,
            isaac_python=Path("/envs/isaac/bin/python"),
            graspgenx_python=Path("/graspgenx/.venv/bin/python"),
        )
        self.assertEqual(
            list(stages)[:3],
            ["capture_rgbd", "ollama_vlm", "sam3_segmentation"],
        )
        self.assertIn("--vlm-result", stages["sam3_segmentation"].command)
        self.assertNotIn("--sam3-prompt", stages["capture_rgbd"].command)
        infer = stages["graspgenx_inference"].command
        self.assertEqual(
            infer[infer.index("--segmentation") + 1],
            str(paths.segmentation / "parts" / "grasp_part"),
        )
        self.assertEqual(
            infer[infer.index("--segmentation-role") + 1], "grasp_part"
        )
        for stage_name in (
            "curobo_map",
            "static_collision_filter",
            "curobo_grasp_lift_trials",
        ):
            command = stages[stage_name].command
            self.assertEqual(
                command[command.index("--segmentation") + 1], str(paths.segmentation)
            )

    def test_manual_part_prompt_also_drives_candidate_generation(self) -> None:
        args = MODULE.parse_args(
            [
                "--scene-usd",
                str(PROJECT / "scene.usda"),
                "--prompt",
                "hammer",
                "--grasp-part-prompt",
                "hammer handle",
                "--receive-part-prompt",
                "hammer head",
                "--output",
                str(PROJECT / "outputs" / "e2e"),
            ]
        )
        paths = MODULE.pipeline_paths(args.output)
        stages = MODULE.build_stages(
            args,
            project_root=PROJECT,
            paths=paths,
            isaac_python=Path("/envs/isaac/bin/python"),
            graspgenx_python=Path("/graspgenx/.venv/bin/python"),
        )
        infer = stages["graspgenx_inference"].command
        self.assertEqual(
            infer[infer.index("--segmentation") + 1],
            str(paths.segmentation / "parts" / "grasp_part"),
        )

    def test_vlm_mode_requires_explicit_model(self) -> None:
        with self.assertRaises(SystemExit):
            MODULE.parse_args(
                [
                    "--scene-usd",
                    "scene.usda",
                    "--target-object",
                    "hammer",
                    "--output",
                    "outputs/e2e",
                ]
            )

    def test_fivefold_drive_diagnostic_is_forwarded_to_every_replay(self) -> None:
        args = MODULE.parse_args(
            [
                "--scene-usd",
                str(PROJECT / "scene.usda"),
                "--prompt",
                "hammer",
                "--output",
                str(PROJECT / "outputs" / "e2e"),
                "--finger-drive-scale",
                "5",
            ]
        )
        paths = MODULE.pipeline_paths(args.output)
        stages = MODULE.build_stages(
            args,
            project_root=PROJECT,
            paths=paths,
            isaac_python=Path("/envs/isaac/bin/python"),
            graspgenx_python=Path("/graspgenx/.venv/bin/python"),
        )
        replay = stages["isaac_physical_trials"].command
        self.assertEqual(replay[replay.index("--finger-drive-scale") + 1], "5.0")

    def test_handover_goal_enables_attached_planning_and_rigid_replay(self) -> None:
        args = MODULE.parse_args(
            [
                "--scene-usd",
                str(PROJECT / "scene.usda"),
                "--prompt",
                "hammer",
                "--output",
                str(PROJECT / "outputs" / "e2e"),
                "--handover-goal-position-robot-base-m",
                "0.45",
                "-0.25",
                "0.65",
            ]
        )
        paths = MODULE.pipeline_paths(args.output)
        stages = MODULE.build_stages(
            args,
            project_root=PROJECT,
            paths=paths,
            isaac_python=Path("/envs/isaac/bin/python"),
            graspgenx_python=Path("/graspgenx/.venv/bin/python"),
        )
        planner = stages["curobo_grasp_lift_trials"].command
        position_at = planner.index("--handover-goal-position-robot-base-m")
        self.assertEqual(
            planner[position_at + 1 : position_at + 4],
            ("0.45", "-0.25", "0.65"),
        )
        replay = stages["isaac_physical_trials"].command
        self.assertEqual(
            replay[replay.index("--grasp-retention-mode") + 1],
            "rigid-attachment",
        )

    def test_handover_orientation_requires_position(self) -> None:
        with self.assertRaises(SystemExit):
            MODULE.parse_args(
                [
                    "--scene-usd",
                    "scene.usda",
                    "--prompt",
                    "hammer",
                    "--output",
                    "outputs/e2e",
                    "--handover-goal-quaternion-wxyz",
                    "1",
                    "0",
                    "0",
                    "0",
                ]
            )

    def test_affordance_handover_inserts_reranking_and_candidate_specific_goal(self):
        args = MODULE.parse_args(
            [
                "--scene-usd",
                str(PROJECT / "scene.usda"),
                "--target-object",
                "hammer",
                "--ollama-model",
                "gemma3:12b",
                "--output",
                str(PROJECT / "outputs" / "e2e"),
                "--handover-receiver-position-robot-base-m",
                "0.55",
                "-0.30",
                "0.65",
                "--handover-human-direction-robot-base",
                "0",
                "-1",
                "0",
            ]
        )
        paths = MODULE.pipeline_paths(args.output)
        stages = MODULE.build_stages(
            args,
            project_root=PROJECT,
            paths=paths,
            isaac_python=Path("/envs/isaac/bin/python"),
            graspgenx_python=Path("/graspgenx/.venv/bin/python"),
        )
        names = list(stages)
        self.assertEqual(
            names[names.index("static_collision_filter") + 1],
            "handover_aware_rerank",
        )
        rerank = stages["handover_aware_rerank"].command
        self.assertIn("--receive-segmentation", rerank)
        pregrasp = stages["curobo_pregrasp"].command
        self.assertEqual(
            pregrasp[pregrasp.index("--candidates") + 1],
            str(paths.handover_candidates),
        )
        planner = stages["curobo_grasp_lift_trials"].command
        self.assertIn("--handover-receiver-position-robot-base-m", planner)
        self.assertIn("--handover-human-direction-robot-base", planner)
        replay = stages["isaac_physical_trials"].command
        self.assertEqual(
            replay[replay.index("--grasp-retention-mode") + 1],
            "rigid-attachment",
        )

    def test_affordance_handover_requires_part_segmentation_and_complete_geometry(self):
        with self.assertRaises(SystemExit):
            MODULE.parse_args(
                [
                    "--scene-usd",
                    "scene.usda",
                    "--prompt",
                    "hammer",
                    "--output",
                    "outputs/e2e",
                    "--handover-receiver-position-robot-base-m",
                    "0.5",
                    "0",
                    "0.6",
                    "--handover-human-direction-robot-base",
                    "1",
                    "0",
                    "0",
                ]
            )
        with self.assertRaises(SystemExit):
            MODULE.parse_args(
                [
                    "--scene-usd",
                    "scene.usda",
                    "--target-object",
                    "hammer",
                    "--ollama-model",
                    "gemma3:12b",
                    "--output",
                    "outputs/e2e",
                    "--handover-receiver-position-robot-base-m",
                    "0.5",
                    "0",
                    "0.6",
                ]
            )

    def test_task_instruction_is_forwarded_only_to_vlm(self) -> None:
        args = MODULE.parse_args(
            [
                "--scene-usd",
                str(PROJECT / "scene.usda"),
                "--target-object",
                "hammer",
                "--task-instruction",
                "Grasp near the estimated center of mass.",
                "--ollama-model",
                "gemma3:12b",
                "--output",
                str(PROJECT / "outputs" / "e2e"),
            ]
        )
        paths = MODULE.pipeline_paths(args.output)
        stages = MODULE.build_stages(
            args,
            project_root=PROJECT,
            paths=paths,
            isaac_python=Path("/envs/isaac/bin/python"),
            graspgenx_python=Path("/graspgenx/.venv/bin/python"),
        )
        vlm = stages["ollama_vlm"].command
        self.assertEqual(
            vlm[vlm.index("--task-instruction") + 1],
            "Grasp near the estimated center of mass.",
        )
        self.assertNotIn("--task-instruction", stages["sam3_segmentation"].command)

    def test_resume_accepts_only_expected_report_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            report.write_text(json.dumps({"status": "success"}), encoding="utf-8")
            self.assertTrue(MODULE.report_succeeded(report, ("success",)))
            self.assertFalse(MODULE.report_succeeded(report, ("plans_ready",)))
            report.write_text(json.dumps({"status": "failed"}), encoding="utf-8")
            self.assertFalse(MODULE.report_succeeded(report, ("success",)))

    def test_resume_archives_incomplete_stage_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "stage"
            output.mkdir()
            report = output / "report.json"
            report.write_text(json.dumps({"status": "failed"}), encoding="utf-8")
            stage = MODULE.Stage("example", ("python",), report)
            archived = MODULE._archive_partial_stage_output(
                stage, root / "pipeline_status.json"
            )
            self.assertIsNotNone(archived)
            assert archived is not None
            self.assertTrue((archived / "report.json").is_file())
            self.assertFalse(output.exists())

    def test_rejects_topk_above_generated_count(self) -> None:
        with self.assertRaises(SystemExit):
            MODULE.parse_args(
                [
                    "--scene-usd",
                    "scene.usda",
                    "--prompt",
                    "hammer",
                    "--output",
                    "outputs/e2e",
                    "--num-grasps",
                    "100",
                    "--topk",
                    "101",
                ]
            )


if __name__ == "__main__":
    unittest.main()
