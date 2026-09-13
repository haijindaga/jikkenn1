#!/usr/bin/env python3
"""Inspect the installed Isaac Sim Surface Gripper implementation.

This is a read-only compatibility preflight. It does not create a stage,
attach an object, or move a robot. The report lets the experiment use the
exact Isaac Sim 5.1 API instead of guessing version-specific schema details.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable


REQUIRED_INTERFACE_METHODS = (
    "close_gripper",
    "open_gripper",
    "get_gripper_status",
    "get_gripped_objects",
    "set_write_to_usd",
)
EXAMPLE_FILENAMES = ("gripper_grasp.py", "SurfaceGripper_gantry.usda")
RELEVANT_SOURCE_TERMS = (
    "surfacegripper",
    "surface_gripper",
    "attachmentpoint",
    "attachment_point",
    "d6",
    "close_gripper",
    "open_gripper",
    "gripped_objects",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/isaac_surface_gripper_preflight.json"),
    )
    return parser.parse_args()


def callable_contract(value: Any, names: Iterable[str]) -> dict[str, bool]:
    return {name: callable(getattr(value, name, None)) for name in names}


def module_path(module: Any) -> Path | None:
    try:
        return Path(inspect.getfile(module)).resolve()
    except (OSError, TypeError):
        raw = getattr(module, "__file__", None)
        return Path(raw).resolve() if raw else None


def find_named_files(roots: Iterable[Path], names: Iterable[str]) -> list[Path]:
    wanted = set(names)
    matches: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.name in wanted:
                matches.add(path.resolve())
    return sorted(matches)


def extract_relevant_source_lines(path: Path, *, maximum: int = 120) -> list[dict[str, Any]]:
    if path.suffix != ".py":
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        lowered = line.lower()
        if any(term in lowered for term in RELEVANT_SOURCE_TERMS):
            records.append({"line": line_number, "text": line.strip()})
            if len(records) >= maximum:
                break
    return records


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in (
        "isaacsim",
        "isaacsim-robot-surface-gripper",
        "isaacsim-robot-schema",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def import_first(names: Iterable[str]) -> tuple[Any | None, list[dict[str, str]]]:
    failures = []
    for name in names:
        try:
            return importlib.import_module(name), failures
        except Exception as error:  # Isaac schema packaging differs by release.
            failures.append(
                {"module": name, "error": f"{type(error).__name__}: {error}"}
            )
    return None, failures


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Isaac extensions must be imported after SimulationApp construction.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})
    report: dict[str, Any]
    return_code = 1
    try:
        surface_module = importlib.import_module("isaacsim.robot.surface_gripper")
        binding_module = importlib.import_module(
            "isaacsim.robot.surface_gripper._surface_gripper"
        )
        interface = binding_module.acquire_surface_gripper_interface()
        interface_methods = callable_contract(interface, REQUIRED_INTERFACE_METHODS)

        schema_module, schema_import_failures = import_first(
            ("usd.schema.isaac.robot_schema", "isaacsim.robot.schema")
        )
        schema_symbols = []
        if schema_module is not None:
            schema_symbols = sorted(
                name
                for name in dir(schema_module)
                if any(term in name.lower() for term in ("gripper", "attach"))
            )

        discovered_module_paths = [
            path
            for path in (
                module_path(surface_module),
                module_path(binding_module),
                module_path(schema_module) if schema_module is not None else None,
            )
            if path is not None
        ]
        search_roots = {Path(sys.prefix).resolve()}
        for path in discovered_module_paths:
            search_roots.add(path.parent)
            for parent in path.parents:
                if parent.name in {"site-packages", "exts", "extscache"}:
                    search_roots.add(parent)
                    break

        examples = find_named_files(search_roots, EXAMPLE_FILENAMES)
        example_records = [
            {
                "path": str(path),
                "relevant_source_lines": extract_relevant_source_lines(path),
            }
            for path in examples
        ]

        required_api_present = all(interface_methods.values())
        report = {
            "status": "success" if required_api_present else "incompatible_api",
            "purpose": "read-only Isaac Sim Surface Gripper compatibility preflight",
            "python": sys.executable,
            "python_prefix": sys.prefix,
            "package_versions": package_versions(),
            "modules": {
                "surface_gripper": str(module_path(surface_module)),
                "binding": str(module_path(binding_module)),
                "schema": str(module_path(schema_module)) if schema_module else None,
                "schema_import_failures": schema_import_failures,
            },
            "interface_methods": interface_methods,
            "required_interface_api_present": required_api_present,
            "schema_symbols": schema_symbols,
            "search_roots": [str(path) for path in sorted(search_roots)],
            "bundled_examples": example_records,
            "bundled_python_example_found": any(
                path.name == "gripper_grasp.py" for path in examples
            ),
            "automatic_checks": {
                "surface_gripper_imported": True,
                "interface_acquired": interface is not None,
                "required_interface_api_present": required_api_present,
            },
            "safety": {
                "read_only": True,
                "stage_created": False,
                "attachment_created": False,
                "robot_moved": False,
            },
            "next_gate": (
                "Implement a separate surface-gripper retention mode from this exact "
                "installed API and bundled example; retain FixedJoint as the baseline."
            ),
        }
        return_code = 0 if required_api_present else 2
    except Exception as error:
        report = {
            "status": "preflight_failed",
            "purpose": "read-only Isaac Sim Surface Gripper compatibility preflight",
            "python": sys.executable,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "safety": {
                "read_only": True,
                "stage_created": False,
                "attachment_created": False,
                "robot_moved": False,
            },
        }
    finally:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"saved: {args.output.resolve()}")
        simulation_app.close()
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
