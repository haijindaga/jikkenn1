#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PINS_FILE="${PROJECT_ROOT}/docker/isaac_ros_b_pins.env"
# shellcheck source=/dev/null
source "${PINS_FILE}"

BACKEND_ROOT="${PROJECT_ROOT}/.backend-b"
VENDOR_ROOT="${BACKEND_ROOT}/vendor"
CLI_SOURCE="${VENDOR_ROOT}/isaac-ros-cli"
CLI_VENV="${BACKEND_ROOT}/cli-venv"
NVBLOX_SOURCE="${BACKEND_ROOT}/ws/src/isaac_ros_nvblox"
BUILD_ROOT="${BACKEND_ROOT}/ws"
HOST_PYTHON=/usr/bin/python3

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "missing required command: $1" >&2
        exit 1
    fi
}

clone_pinned() {
    local repository="$1"
    local reference="$2"
    local expected_commit="$3"
    local destination="$4"
    if [[ ! -e "${destination}" ]]; then
        git clone --recursive --branch "${reference}" --depth 1 \
            "${repository}" "${destination}"
    fi
    if [[ ! -d "${destination}/.git" ]]; then
        echo "existing path is not a Git checkout: ${destination}" >&2
        exit 1
    fi
    local actual_commit
    actual_commit="$(git -C "${destination}" rev-parse HEAD)"
    if [[ "${actual_commit}" != "${expected_commit}" ]]; then
        echo "pin mismatch at ${destination}" >&2
        echo "expected: ${expected_commit}" >&2
        echo "actual:   ${actual_commit}" >&2
        echo "Refusing to reset an existing checkout automatically." >&2
        exit 1
    fi
    git -C "${destination}" lfs pull
    git -C "${destination}" submodule update --init --recursive --depth 1
}

require_command docker
require_command git
require_command nvidia-smi
git lfs version >/dev/null
if [[ ! -x "${HOST_PYTHON}" ]]; then
    echo "system Python is missing: ${HOST_PYTHON}" >&2
    exit 1
fi

if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "Backend B is pinned for the official Isaac ROS 4.5 x86_64 image." >&2
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    echo "Docker is not available to the current user." >&2
    echo "The official setup requires membership in the docker group." >&2
    exit 1
fi
if ! docker info --format '{{json .Runtimes}}' | grep -q 'nvidia'; then
    echo "Docker does not report the NVIDIA runtime." >&2
    exit 1
fi
driver_major="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | cut -d. -f1)"
if [[ ! "${driver_major}" =~ ^[0-9]+$ ]] || (( driver_major < 580 )); then
    echo "Isaac ROS 4.5 requires NVIDIA driver 580 or newer." >&2
    exit 1
fi

mkdir -p "${VENDOR_ROOT}" "${BUILD_ROOT}/src"
clone_pinned \
    "${ISAAC_ROS_CLI_REPOSITORY}" \
    "${ISAAC_ROS_CLI_REF}" \
    "${ISAAC_ROS_CLI_COMMIT}" \
    "${CLI_SOURCE}"
clone_pinned \
    "${ISAAC_ROS_NVBLOX_REPOSITORY}" \
    "${ISAAC_ROS_NVBLOX_REF}" \
    "${ISAAC_ROS_NVBLOX_COMMIT}" \
    "${NVBLOX_SOURCE}"

if [[ ! -f "${NVBLOX_SOURCE}/nvblox_ros/nvblox_core/CMakeLists.txt" ]]; then
    echo "nvblox_core submodule was not populated" >&2
    exit 1
fi
actual_core_commit="$(git -C "${NVBLOX_SOURCE}/nvblox_ros/nvblox_core" rev-parse HEAD)"
if [[ "${actual_core_commit}" != "${NVBLOX_CORE_COMMIT}" ]]; then
    echo "nvblox_core pin mismatch" >&2
    echo "expected: ${NVBLOX_CORE_COMMIT}" >&2
    echo "actual:   ${actual_core_commit}" >&2
    exit 1
fi

if [[ ! -x "${CLI_VENV}/bin/python" ]]; then
    "${HOST_PYTHON}" -m venv "${CLI_VENV}"
fi
"${CLI_VENV}/bin/python" -m pip install --disable-pip-version-check \
    'PyYAML==6.0.2' 'termcolor==2.4.0'

build_args=(
    "ISAAC_DEBIAN_KEY_URL=${ISAAC_DEBIAN_KEY_URL}"
    "ISAAC_DEBIAN_REPOSITORY=${ISAAC_DEBIAN_REPOSITORY}"
    "ISAAC_DEBIAN_DIST=${ISAAC_DEBIAN_DIST}"
    "ISAAC_DEBIAN_COMPONENTS=${ISAAC_DEBIAN_COMPONENTS}"
)
resolver_args=(
    --cli-source "${CLI_SOURCE}"
    --platform amd64
)
for argument in "${build_args[@]}"; do
    resolver_args+=(--build-arg "${argument}")
done
official_base_image="$(
    "${CLI_VENV}/bin/python" \
        "${PROJECT_ROOT}/scripts/resolve_isaac_ros_image.py" \
        "${resolver_args[@]}"
)"
echo "official Isaac ROS image: ${official_base_image}"

if ! docker pull "${official_base_image}"; then
    echo "Prebuilt image was unavailable; building the pinned official Dockerfile locally."
    docker buildx build --load --progress=plain \
        --file "${CLI_SOURCE}/docker/Dockerfile.isaac_ros" \
        --tag "${official_base_image}" \
        --build-arg PLATFORM=amd64 \
        --build-arg ISAAC_ROS_PLATFORM=amd64 \
        --build-arg "ISAAC_DEBIAN_KEY_URL=${ISAAC_DEBIAN_KEY_URL}" \
        --build-arg "ISAAC_DEBIAN_REPOSITORY=${ISAAC_DEBIAN_REPOSITORY}" \
        --build-arg "ISAAC_DEBIAN_DIST=${ISAAC_DEBIAN_DIST}" \
        --build-arg "ISAAC_DEBIAN_COMPONENTS=${ISAAC_DEBIAN_COMPONENTS}" \
        "${CLI_SOURCE}"
fi

docker buildx build --load --progress=plain \
    --file "${PROJECT_ROOT}/docker/Dockerfile.nvblox_b" \
    --tag "${NVBLOX_B_IMAGE}" \
    --build-arg "BASE_IMAGE=${official_base_image}" \
    --build-arg "ISAAC_ROS_NVBLOX_REPOSITORY=${ISAAC_ROS_NVBLOX_REPOSITORY}" \
    --build-arg "ISAAC_ROS_NVBLOX_REF=${ISAAC_ROS_NVBLOX_REF}" \
    --build-arg "ISAAC_ROS_NVBLOX_COMMIT=${ISAAC_ROS_NVBLOX_COMMIT}" \
    "${PROJECT_ROOT}"

docker run --rm --gpus all --network host --ipc=host \
    -e "HOST_USER_UID=$(id -u)" \
    -e "HOST_USER_GID=$(id -g)" \
    -e "ISAAC_ROS_WS=${PROJECT_ROOT}" \
    -v "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    --workdir "${PROJECT_ROOT}" \
    --entrypoint /usr/local/bin/scripts/workspace-entrypoint.sh \
    "${NVBLOX_B_IMAGE}" \
    bash -lc '
        set -euo pipefail
        source /opt/ros/jazzy/setup.bash
        colcon --log-base "${ISAAC_ROS_WS}/.backend-b/ws/log" build \
            --symlink-install \
            --base-paths \
                "${ISAAC_ROS_WS}/.backend-b/ws/src/isaac_ros_nvblox" \
                "${ISAAC_ROS_WS}/ros2/panda_handover_nvblox" \
            --build-base "${ISAAC_ROS_WS}/.backend-b/ws/build" \
            --install-base "${ISAAC_ROS_WS}/.backend-b/ws/install" \
            --packages-up-to panda_handover_nvblox \
            --cmake-args -DCMAKE_BUILD_TYPE=Release
    '

"${HOST_PYTHON}" "${PROJECT_ROOT}/scripts/check_conservative_backends.py" \
    --backend b \
    --isaac-ros-nvblox-source "${NVBLOX_SOURCE}"

image_commit="$(
    docker image inspect --format \
        '{{ index .Config.Labels "org.opencontainers.image.revision" }}' \
        "${NVBLOX_B_IMAGE}"
)"
if [[ "${image_commit}" != "${ISAAC_ROS_NVBLOX_COMMIT}" ]]; then
    echo "Backend B image provenance label mismatch: ${image_commit}" >&2
    exit 1
fi
test -f "${BUILD_ROOT}/install/setup.bash"
echo "Backend B setup complete: ${NVBLOX_B_IMAGE}"
