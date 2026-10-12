"""Validation for presentation-only receiver overlays used during replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


OVERLAY_PURPOSE = "visual-only 1X NEO receiver overlay"
OVERLAY_METHOD = "OpenUSD subLayer plus reference"
RECEIVER_PRIM = "/World/Receiver"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_visual_receiver_overlay(
    *,
    replay_scene: Path,
    recorded_scene: Path,
    receiver_prim: str,
) -> dict:
    """Fail closed unless replay_scene is a checked visual-only overlay.

    The overlay must have been produced from the exact scene recorded by the
    capture.  Both the recorded base scene and overlay bytes are checked
    against the adjacent generator report.  This permits adding a nonphysical
    presentation asset without weakening the normal scene-identity gate.
    """

    replay_scene = replay_scene.expanduser().resolve()
    recorded_scene = recorded_scene.expanduser().resolve()
    if receiver_prim != RECEIVER_PRIM:
        raise ValueError(
            "validated receiver overlays require "
            f"--visual-only-receiver-prim {RECEIVER_PRIM}"
        )
    if not replay_scene.is_file():
        raise FileNotFoundError(f"receiver overlay scene does not exist: {replay_scene}")
    if not recorded_scene.is_file():
        raise FileNotFoundError(f"recorded base scene does not exist: {recorded_scene}")

    report_path = replay_scene.with_suffix(replay_scene.suffix + ".check.json")
    if not report_path.is_file():
        raise FileNotFoundError(
            "receiver overlay validation report does not exist: "
            f"{report_path}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "success":
        raise ValueError("receiver overlay report does not have status=success")
    if report.get("purpose") != OVERLAY_PURPOSE:
        raise ValueError("receiver overlay report purpose is not recognized")
    if Path(report.get("output", "")).expanduser().resolve() != replay_scene:
        raise ValueError("receiver overlay report output does not match replay scene")

    composition = report.get("composition", {})
    if composition.get("method") != OVERLAY_METHOD:
        raise ValueError("receiver overlay composition method is not recognized")
    if Path(composition.get("base_scene", "")).expanduser().resolve() != recorded_scene:
        raise ValueError(
            "receiver overlay base scene does not match the scene recorded by capture"
        )

    recorded_hash = sha256_file(recorded_scene)
    if composition.get("base_scene_sha256") != recorded_hash:
        raise ValueError("recorded base scene changed after receiver overlay creation")
    overlay_hash = sha256_file(replay_scene)
    if composition.get("overlay_scene_sha256") != overlay_hash:
        raise ValueError("receiver overlay changed after its validation report was created")

    required_false = (
        "source_asset_modified",
        "base_scene_modified",
        "physics_apis_modified",
        "asset_rescaled",
    )
    if any(composition.get(key) is not False for key in required_false):
        raise ValueError("receiver overlay report permits a source or physics mutation")

    safety = report.get("safety", {})
    if safety.get("presentation_only") is not True:
        raise ValueError("receiver overlay is not marked presentation-only")
    if safety.get("human_collision_model_present") is not False:
        raise ValueError("receiver overlay unexpectedly declares a human collision model")
    if safety.get("safe_for_physical_replay") is not False:
        raise ValueError("receiver overlay safety scope is not fail-closed")
    if (
        safety.get("runtime_replay_requires_receiver_physics_and_collision_disabled")
        is not True
    ):
        raise ValueError("receiver overlay lacks the runtime visual-only guard requirement")

    checks = report.get("automatic_checks", {})
    if not checks or not all(value is True for value in checks.values()):
        raise ValueError("receiver overlay automatic checks did not all pass")

    return {
        "mode": "validated-visual-only-receiver-overlay",
        "report": str(report_path),
        "recorded_base_scene": str(recorded_scene),
        "recorded_base_scene_sha256": recorded_hash,
        "replay_overlay_scene": str(replay_scene),
        "replay_overlay_scene_sha256": overlay_hash,
        "receiver_prim": receiver_prim,
        "human_collision_model_present": False,
        "runtime_physics_and_collision_must_be_disabled": True,
    }
