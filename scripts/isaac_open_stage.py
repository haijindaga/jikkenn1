#!/usr/bin/env python3
"""Open an existing USD stage for visual inspection without playing physics."""

from __future__ import annotations

import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=Path, required=True)
parser.add_argument(
    "--eye",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=(2.2, -2.7, 1.6),
    help="Initial presentation viewport eye position in world metres",
)
parser.add_argument(
    "--target",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=(-0.2, 0.0, 0.35),
    help="Initial presentation viewport look-at target in world metres",
)
args = parser.parse_args()
stage_path = args.stage.expanduser().resolve()
if not stage_path.is_file():
    parser.error(f"stage does not exist: {stage_path}")

from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": False})
try:
    import numpy as np
    import omni.timeline
    import omni.usd
    from isaacsim.core.utils.viewports import (
        set_active_viewport_camera,
        set_camera_view,
    )

    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    context = omni.usd.get_context()
    result = context.open_stage(str(stage_path))
    print(f"open_stage returned: {result}", flush=True)
    for _ in range(120):
        simulation_app.update()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError(f"USD stage did not open: {stage_path}")
    # The active Kit perspective camera can retain an unrelated previous
    # viewport roll.  Reset only that transient display camera; do not edit or
    # save the inspected USD stage or its authored sensor camera.
    set_camera_view(
        eye=np.asarray(args.eye, dtype=np.float64),
        target=np.asarray(args.target, dtype=np.float64),
        camera_prim_path="/OmniverseKit_Persp",
    )
    set_active_viewport_camera("/OmniverseKit_Persp")
    for _ in range(2):
        simulation_app.update()
    print(f"opened stage: {stage.GetRootLayer().identifier}", flush=True)
    print(
        f"presentation view: eye={tuple(args.eye)} target={tuple(args.target)}",
        flush=True,
    )
    print("Timeline is stopped. Close the Isaac Sim window when done.", flush=True)
    while simulation_app.is_running():
        simulation_app.update()
finally:
    simulation_app.close()
