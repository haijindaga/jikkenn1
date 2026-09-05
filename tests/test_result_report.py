from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from panda_handover.result_report import generate_result_report


class ResultReportTests(unittest.TestCase):
    def test_includes_images_json_and_failed_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "capture" / "camera_0"
            artifact.mkdir(parents=True)
            (artifact / "rgb.png").write_bytes(b"not-a-real-png")
            stage_report = artifact / "stage.json"
            stage_report.write_text(
                json.dumps({"status": "failed", "reason": "test"}),
                encoding="utf-8",
            )
            manifest = root / "pipeline_status.json"
            manifest.write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "simulation_only": True,
                        "stages": [
                            {
                                "name": "capture_rgbd",
                                "status": "failed",
                                "report": str(stage_report),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = generate_result_report(root, manifest_path=manifest)
            text = output.read_text(encoding="utf-8")
            self.assertIn("capture_rgbd", text)
            self.assertIn("capture/camera_0/rgb.png", text)
            self.assertIn("capture/camera_0/stage.json", text)
            self.assertIn("FAILED", text)

    def test_shows_semantic_grasp_routing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "pipeline_status.json"
            manifest.write_text(
                json.dumps(
                    {
                        "status": "success",
                        "policy": {
                            "grasp_candidate_segmentation": "grasp_part",
                            "grasp_candidate_segmentation_path": "capture/sam3/parts/grasp_part",
                            "whole_object_segmentation_path": "capture/sam3",
                            "whole_object_uses": ["attached_object_geometry"],
                            "grasp_part_fallback_to_whole_object": False,
                        },
                        "stages": [],
                    }
                ),
                encoding="utf-8",
            )
            output = generate_result_report(root, manifest_path=manifest)
            text = output.read_text(encoding="utf-8")
            self.assertIn("Semantic grasp routing", text)
            self.assertIn("grasp_part", text)
            self.assertIn("attached_object_geometry", text)
            self.assertIn("False", text)

    def test_highlights_vlm_input_prompt_and_structured_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "capture" / "camera_0"
            capture.mkdir(parents=True)
            (capture / "rgb.png").write_bytes(b"image")
            vlm = root / "vlm"
            vlm.mkdir()
            (vlm / "vlm_part_discovery.json").write_text(
                json.dumps(
                    {
                        "status": "success",
                        "inputs": {"target_object": "hammer"},
                        "parameters": {"model": "gemma3:12b"},
                        "result": {
                            "object": "hammer",
                            "grasp_part": "hammer head",
                            "receive_part": "hammer handle",
                        },
                        "provenance": {
                            "model_digest": "abc123",
                            "request": {
                                "system_prompt": "system instructions",
                                "user_prompt": "Target object: hammer",
                                "schema": {"type": "object"},
                            },
                            "response": {"done": True},
                        },
                    }
                ),
                encoding="utf-8",
            )
            manifest = root / "pipeline_status.json"
            manifest.write_text(
                json.dumps({"status": "success", "stages": []}), encoding="utf-8"
            )
            output = generate_result_report(root, manifest_path=manifest)
            text = output.read_text(encoding="utf-8")
            self.assertIn("VLM input and output", text)
            self.assertIn("gemma3:12b", text)
            self.assertIn("Target object: hammer", text)
            self.assertIn("hammer head", text)
            self.assertIn("capture/camera_0/rgb.png", text)


if __name__ == "__main__":
    unittest.main()
