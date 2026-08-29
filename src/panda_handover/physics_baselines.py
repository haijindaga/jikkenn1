"""Named, source-backed physics baselines for grasp replay experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FingerDrivePreset:
    """OpenUSD linear-drive values applied to the Panda finger actuator."""

    name: str
    max_force: float | None
    stiffness: float | None
    damping: float | None
    source: str
    source_revision: str | None
    notes: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


FINGER_DRIVE_PRESETS: dict[str, FingerDrivePreset] = {
    "authored-usd": FingerDrivePreset(
        name="authored-usd",
        max_force=None,
        stiffness=None,
        damping=None,
        source="Values already authored in the loaded Panda USD",
        source_revision=None,
        notes="Control condition; the replay script does not modify the finger drive.",
    ),
    "isaaclab-franka": FingerDrivePreset(
        name="isaaclab-franka",
        max_force=200.0,
        stiffness=2000.0,
        damping=100.0,
        source=(
            "https://github.com/NVlabs/RoboLab/blob/"
            "9db0aaf09d9fe5d4f37b168320788258c7012463/robolab/robots/franka.py"
        ),
        source_revision="9db0aaf09d9fe5d4f37b168320788258c7012463",
        notes=(
            "RoboLab FrankaCfg, adapted from Isaac Lab FRANKA_PANDA_CFG. "
            "These are simulator actuator/drive values, not a calibration of "
            "total hardware grasp force."
        ),
    ),
}


def resolve_finger_drive_values(
    preset_name: str,
    *,
    explicit_max_force: float | None = None,
) -> dict[str, float | None]:
    """Resolve a named baseline with the legacy explicit max-force override."""

    try:
        preset = FINGER_DRIVE_PRESETS[preset_name]
    except KeyError as exc:
        choices = ", ".join(sorted(FINGER_DRIVE_PRESETS))
        raise ValueError(f"unknown finger-drive preset {preset_name!r}; choose {choices}") from exc
    return {
        "max_force": (
            float(explicit_max_force)
            if explicit_max_force is not None
            else preset.max_force
        ),
        "stiffness": preset.stiffness,
        "damping": preset.damping,
    }
