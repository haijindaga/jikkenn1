"""Strict Ollama structured output for task-oriented handover parts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


HANDOVER_PARTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "object": {"type": "string"},
        "grasp_part": {"type": "string"},
        "receive_part": {"type": "string"},
    },
    "required": ["object", "grasp_part", "receive_part"],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """You identify semantic parts for a robot handover experiment.
Return exactly the JSON object required by the supplied schema.

Rules:
- object must exactly match the target object phrase supplied by the user.
- grasp_part is the part the robot should hold during handover.
- receive_part is the safe, natural part a human should receive or hold.
- Every value must be a short English noun phrase suitable as a SAM3 text prompt.
- Each part phrase must be self-contained and include the complete object phrase.
- grasp_part and receive_part must name different regions.
- Do not add explanations, markdown, confidence scores, or extra fields.
"""


@dataclass(frozen=True)
class HandoverParts:
    object: str
    grasp_part: str
    receive_part: str

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, target_object: str
    ) -> "HandoverParts":
        expected = {"object", "grasp_part", "receive_part"}
        if set(value) != expected:
            raise ValueError(
                f"VLM output fields must be exactly {sorted(expected)}, got {sorted(value)}"
            )
        if any(not isinstance(value[field], str) for field in expected):
            raise ValueError("all VLM output fields must be strings")
        result = cls(
            object=value["object"].strip(),
            grasp_part=value["grasp_part"].strip(),
            receive_part=value["receive_part"].strip(),
        )
        result.validate(target_object=target_object)
        return result

    def validate(self, *, target_object: str) -> None:
        target = target_object.strip()
        if not target:
            raise ValueError("target object must not be empty")
        for name, phrase in asdict(self).items():
            if not phrase:
                raise ValueError(f"{name} must not be empty")
            if len(phrase) > 160:
                raise ValueError(f"{name} is too long for a SAM3 noun phrase")
            if not any(character.isascii() and character.isalnum() for character in phrase):
                raise ValueError(f"{name} must be an English SAM3 noun phrase")
        if self.object.casefold() != target.casefold():
            raise ValueError(
                f"VLM object {self.object!r} does not exactly match target {target!r}"
            )
        target_folded = target.casefold()
        for name, phrase in (
            ("grasp_part", self.grasp_part),
            ("receive_part", self.receive_part),
        ):
            if target_folded not in phrase.casefold():
                raise ValueError(
                    f"{name} must include the complete object phrase {target!r}"
                )
        if self.grasp_part.casefold() == self.receive_part.casefold():
            raise ValueError("grasp_part and receive_part must be different")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def build_user_prompt(target_object: str) -> str:
    target = target_object.strip()
    if not target:
        raise ValueError("target object must not be empty")
    return (
        f"Target object: {target}\n"
        "Task: hand the object to a human. Identify the region the robot should "
        "grasp and the distinct region the human should receive."
    )


def _request_json(
    url: str,
    *,
    method: str = "GET",
    payload: Mapping[str, Any] | None = None,
    timeout_s: float,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            decoded = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code} from {url}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"cannot reach Ollama at {url}: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama returned invalid JSON from {url}") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("Ollama response must be a JSON object")
    return decoded


def _model_digest(
    base_url: str, model: str, *, timeout_s: float
) -> str | None:
    response = _request_json(
        urljoin(base_url.rstrip("/") + "/", "api/tags"), timeout_s=timeout_s
    )
    models = response.get("models", [])
    if not isinstance(models, list):
        return None
    requested = model.casefold()
    for item in models:
        if not isinstance(item, dict):
            continue
        names = (str(item.get("name", "")), str(item.get("model", "")))
        if any(name.casefold() == requested for name in names):
            digest = item.get("digest")
            return str(digest) if digest else None
    return None


def discover_handover_parts(
    image_path: str | Path,
    *,
    target_object: str,
    model: str,
    base_url: str = "http://127.0.0.1:11434",
    timeout_s: float = 180.0,
) -> tuple[HandoverParts, dict[str, Any]]:
    """Call Ollama's official chat API with a strict JSON schema."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"VLM input image does not exist: {image_path}")
    if not model.strip():
        raise ValueError("Ollama model must not be empty")
    if timeout_s <= 0:
        raise ValueError("timeout must be positive")

    image_bytes = image_path.read_bytes()
    user_prompt = build_user_prompt(target_object)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_prompt,
                "images": [base64.b64encode(image_bytes).decode("ascii")],
            },
        ],
        "stream": False,
        "format": HANDOVER_PARTS_SCHEMA,
        "options": {"temperature": 0},
        "keep_alive": 0,
    }
    response = _request_json(
        urljoin(base_url.rstrip("/") + "/", "api/chat"),
        method="POST",
        payload=payload,
        timeout_s=timeout_s,
    )
    message = response.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise RuntimeError("Ollama response is missing message.content")
    try:
        structured = json.loads(message["content"])
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama message.content is not valid JSON") from exc
    if not isinstance(structured, dict):
        raise RuntimeError("Ollama structured output must be a JSON object")
    parts = HandoverParts.from_mapping(structured, target_object=target_object)
    metadata = {
        "reference": "Ollama /api/chat vision and structured outputs",
        "request": {
            "base_url": base_url,
            "model": model,
            "target_object": target_object.strip(),
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": user_prompt,
            "schema": HANDOVER_PARTS_SCHEMA,
            "temperature": 0,
            "keep_alive": 0,
            "image_path": str(image_path.resolve()),
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        },
        "response": response,
        "model_digest": _model_digest(base_url, model, timeout_s=timeout_s),
    }
    return parts, metadata


def load_handover_parts_report(path: str | Path) -> HandoverParts:
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("status") != "success":
        raise ValueError(f"VLM report is not successful: {path}")
    result = report.get("result")
    target = report.get("inputs", {}).get("target_object")
    if not isinstance(result, dict) or not isinstance(target, str):
        raise ValueError(f"VLM report is missing result or target object: {path}")
    return HandoverParts.from_mapping(result, target_object=target)
