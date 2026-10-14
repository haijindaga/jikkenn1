"""Validated presentation-only handover-region overlay specifications."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

from panda_handover.handover import handover_region_from_scene_layout


@dataclass(frozen=True)
class HandoverRegionOverlaySpec:
    """A visual box expressed in the robot-base frame."""

    source: Path
    center_robot_base_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    yaw_degrees: float
    color_rgb: tuple[float, float, float]
    opacity: float

    def to_report(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "presentation_only": True,
            "affects_physics": False,
            "authored_scene_modified": False,
            "source": str(self.source),
            "frame": "robot_base",
            "shape": "box",
            "center_robot_base_m": list(self.center_robot_base_m),
            "size_m": list(self.size_m),
            "yaw_degrees": self.yaw_degrees,
            "color_rgb": list(self.color_rgb),
            "opacity": self.opacity,
        }


def _parse_hex_rgb(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, str) or len(value) != 7 or not value.startswith("#"):
        raise ValueError("handover overlay color must use '#RRGGBB'")
    try:
        channels = tuple(int(value[index : index + 2], 16) / 255.0 for index in (1, 3, 5))
    except ValueError as exc:
        raise ValueError("handover overlay color must use '#RRGGBB'") from exc
    return channels


def handover_region_overlay_spec(
    layout: Mapping[str, Any],
    *,
    source: str | Path,
    opacity: float,
) -> HandoverRegionOverlaySpec:
    """Create a display specification from the same reviewed box used by planning."""

    if not math.isfinite(opacity) or not 0.0 < opacity <= 1.0:
        raise ValueError("handover overlay opacity must be in (0, 1]")
    center, size, _, report = handover_region_from_scene_layout(layout)
    if report["shape"] != "box":
        raise ValueError(
            "handover-region overlay requires a box; a point has no reviewed visual extent"
        )
    handover = layout["entities"]["handover"]
    yaw_degrees = float(handover.get("yaw", 0.0))
    if not math.isfinite(yaw_degrees):
        raise ValueError("handover overlay yaw must be finite")
    color_rgb = _parse_hex_rgb(handover.get("color", "#df57dc"))
    return HandoverRegionOverlaySpec(
        source=Path(source).expanduser().resolve(),
        center_robot_base_m=tuple(float(value) for value in center),
        size_m=tuple(float(value) for value in size),
        yaw_degrees=yaw_degrees,
        color_rgb=color_rgb,
        opacity=float(opacity),
    )
