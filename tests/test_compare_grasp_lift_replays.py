import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


def replay_report(*, held_lift: float, peak_lift: float, lift_loss: float) -> dict:
    return {
        "status": "success",
        "inputs": {
            "capture": "outputs/capture/camera_0",
            "plan": "outputs/plan",
            "scene_usd": "/tmp/scene.usda",
        },
        "physical_object": {
            "physical_pick_observed": held_lift > 0.04,
            "lift_after_hold_m": held_lift,
        },
        "physical_parameters": {
            "target_effective_mass_kg": 0.5,
            "finger_drive_preset": "authored-usd",
            "finger_joint_drives": [
                {
                    "found": True,
                    "max_force_after": 7.2,
                    "stiffness_after": 400.0,
                    "damping_after": 80.0,
                }
            ],
        },
        "retention_diagnostics": {
            "peak_object_lift_m": peak_lift,
            "lift_lost_from_peak_to_final_m": lift_loss,
            "finger_gap_at_peak_lift_m": 0.03,
            "finger_gap_at_final_hold_m": 0.02,
        },
    }


class CompareGraspLiftReplayTests(unittest.TestCase):
    def test_controlled_comparison_records_metric_deltas(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "compare_grasp_lift_replays.py"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            output = root / "comparison.json"
            baseline.write_text(
                json.dumps(
                    replay_report(held_lift=-0.0001, peak_lift=0.018, lift_loss=0.0181)
                ),
                encoding="utf-8",
            )
            candidate_report = replay_report(
                held_lift=0.06, peak_lift=0.07, lift_loss=0.01
            )
            candidate_report["physical_parameters"]["finger_drive_preset"] = (
                "isaaclab-franka"
            )
            candidate.write_text(json.dumps(candidate_report), encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--baseline",
                    str(baseline),
                    "--candidate",
                    str(candidate),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(output.read_text(encoding="utf-8"))

        self.assertTrue(result["comparison_is_controlled"])
        self.assertTrue(result["candidate"]["physical_pick_observed"])
        self.assertAlmostEqual(
            result["candidate_minus_baseline"]["lift_after_hold_m"], 0.0601
        )
        self.assertLess(
            result["candidate_minus_baseline"][
                "lift_lost_from_peak_to_final_m"
            ],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
