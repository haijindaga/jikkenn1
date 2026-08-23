#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/docker/isaac_ros_b_pins.env"

if ! docker image inspect "${NVBLOX_B_IMAGE}" >/dev/null 2>&1; then
    echo "Backend B image is missing. Run scripts/setup_nvblox_b_docker.sh first." >&2
    exit 1
fi
if [[ ! -f "${PROJECT_ROOT}/.backend-b/ws/install/setup.bash" ]]; then
    echo "Backend B ROS workspace is not built. Run setup first." >&2
    exit 1
fi

if (( $# == 0 )); then
    set -- \
        --capture \
            "${PROJECT_ROOT}/outputs/multiview_v1/camera_0" \
            "${PROJECT_ROOT}/outputs/multiview_v1/camera_1" \
            "${PROJECT_ROOT}/outputs/multiview_v1/camera_2" \
        --prepared-map "${PROJECT_ROOT}/outputs/curobo_map_multiview_v1" \
        --output "${PROJECT_ROOT}/outputs/conservative_esdf_b_v1"
fi

docker run --rm --gpus all --network host --ipc=host \
    -e "HOST_USER_UID=$(id -u)" \
    -e "HOST_USER_GID=$(id -g)" \
    -e "ISAAC_ROS_WS=${PROJECT_ROOT}" \
    -v "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    --workdir "${PROJECT_ROOT}" \
    --entrypoint /usr/local/bin/scripts/workspace-entrypoint.sh \
    "${NVBLOX_B_IMAGE}" \
    bash "${PROJECT_ROOT}/scripts/run_nvblox_b_inside.sh" "$@"
