#!/usr/bin/env python3
"""Resolve the official Isaac ROS image tag using the pinned official resolver."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli-source", type=Path, required=True)
    parser.add_argument("--build-arg", action="append", default=[])
    parser.add_argument("--platform", default="amd64")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.cli_source.resolve()
    resolver = source / "scripts" / "run_dev"
    config_path = source / "config" / ".build_image_layers.yaml"
    if not resolver.is_dir() or not config_path.is_file():
        raise FileNotFoundError("--cli-source is not an Isaac ROS CLI checkout")
    sys.path.insert(0, str(resolver))

    # These are NVIDIA's resolver classes, loaded from the pinned v4.5-0
    # checkout.  This helper does not reproduce its hashing rules.
    from build_image_layers import (  # type: ignore[import-not-found]
        Config,
        ImageKey,
        parse_build_args,
        resolve_dockerfiles,
    )

    architecture = "x86_64" if args.platform == "amd64" else "aarch64"
    with redirect_stdout(io.StringIO()):
        config = Config(platform_=architecture)
        if not config.load_yaml(str(config_path)):
            raise RuntimeError(f"could not load {config_path}")
        # An installed CLI obtains this directory from
        # /etc/isaac-ros-cli/.isaac_ros_common-config.  A source checkout has
        # the identical Dockerfiles under docker/, so supply that installation
        # path explicitly without modifying NVIDIA's configuration.
        config.docker_search_dirs_.insert(0, str(source / "docker"))
        image_key = ImageKey.from_key_set(
            {"isaac_ros"}, key_order=config.image_key_order_
        )
        plan = resolve_dockerfiles(
            image_key,
            config.docker_search_dirs_,
            context_overrides=config.context_overrides_,
        )
    if plan is None:
        raise RuntimeError("official resolver could not resolve Dockerfile.isaac_ros")
    plan.build_variables_.update(parse_build_args(args.build_arg))
    print(
        "nvcr.io/nvidia/isaac/ros:"
        f"{plan.target_name()}-{args.platform}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
