#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/jazzy/setup.bash
source "${ISAAC_ROS_WS}/.backend-b/ws/install/setup.bash"

ros2 launch panda_handover_nvblox conservative_nvblox.launch.py &
nvblox_launch_pid=$!
cleanup() {
    kill -INT "${nvblox_launch_pid}" >/dev/null 2>&1 || true
    wait "${nvblox_launch_pid}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

ros2 run panda_handover_nvblox offline_esdf "$@"
