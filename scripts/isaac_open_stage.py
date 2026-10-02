#!/usr/bin/env python3
"""Open an existing USD stage for visual inspection without playing physics."""

from __future__ import annotations

import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=Path, required=True)
args = parser.parse_args()
stage_path = args.stage.expanduser().resolve()
if not stage_path.is_file():
    parser.error(f"stage does not exist: {stage_path}")

from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": False})
try:
    import omni.timeline
    import omni.usd

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
    print(f"opened stage: {stage.GetRootLayer().identifier}", flush=True)
    print("Timeline is stopped. Close the Isaac Sim window when done.", flush=True)
    while simulation_app.is_running():
        simulation_app.update()
finally:
    simulation_app.close()
