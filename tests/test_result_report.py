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


if __name__ == "__main__":
    unittest.main()
