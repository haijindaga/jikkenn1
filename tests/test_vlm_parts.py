from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

from panda_handover.vlm_parts import (
    HandoverParts,
    build_user_prompt,
    discover_handover_parts,
)


class HandoverPartsTests(unittest.TestCase):
    def test_accepts_original_three_field_contract(self) -> None:
        parts = HandoverParts.from_mapping(
            {
                "object": "scissors",
                "grasp_part": "scissors blade near the pivot",
                "receive_part": "scissors handles",
            },
            target_object="scissors",
        )
        self.assertEqual(parts.object, "scissors")

    def test_rejects_extra_vlm_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly"):
            HandoverParts.from_mapping(
                {
                    "object": "hammer",
                    "grasp_part": "hammer head",
                    "receive_part": "hammer handle",
                    "confidence": 0.9,
                },
                target_object="hammer",
            )

    def test_rejects_ambiguous_part_without_object_phrase(self) -> None:
        with self.assertRaisesRegex(ValueError, "complete object phrase"):
            HandoverParts.from_mapping(
                {
                    "object": "hammer",
                    "grasp_part": "head",
                    "receive_part": "hammer handle",
                },
                target_object="hammer",
            )

    def test_rejects_object_substitution(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not exactly match"):
            HandoverParts.from_mapping(
                {
                    "object": "pliers",
                    "grasp_part": "pliers jaws",
                    "receive_part": "pliers handles",
                },
                target_object="scissors",
            )

    def test_user_prompt_keeps_task_fixed(self) -> None:
        prompt = build_user_prompt("knife")
        self.assertIn("Target object: knife", prompt)
        self.assertIn("hand", prompt)

    def test_ollama_request_uses_schema_and_unloads_model(self) -> None:
        calls = []

        def fake_request(url, *, method="GET", payload=None, timeout_s):
            calls.append((url, method, payload, timeout_s))
            return {
                "model": "qwen3-vl:4b",
                "message": {
                    "role": "assistant",
                    "content": (
                        '{"object":"hammer","grasp_part":"hammer head",'
                        '"receive_part":"hammer handle"}'
                    ),
                },
            }

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "rgb.png"
            image.write_bytes(b"image")
            with patch("panda_handover.vlm_parts._request_json", fake_request), patch(
                "panda_handover.vlm_parts._model_digest", return_value="digest"
            ):
                parts, metadata = discover_handover_parts(
                    image,
                    target_object="hammer",
                    model="qwen3-vl:4b",
                )
        self.assertEqual(parts.receive_part, "hammer handle")
        payload = calls[0][2]
        self.assertEqual(calls[0][1], "POST")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["keep_alive"], 0)
        self.assertEqual(payload["options"]["temperature"], 0)
        self.assertFalse(payload["format"]["additionalProperties"])
        self.assertEqual(metadata["model_digest"], "digest")


if __name__ == "__main__":
    unittest.main()
