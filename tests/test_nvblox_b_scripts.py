from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_backend_b_pins_exact_official_release_commits():
    pins = read("docker/isaac_ros_b_pins.env")
    assert "ISAAC_ROS_CLI_REF=v4.5-0" in pins
    assert "680c8c5f85854bee8b0074a00a17c2ba2aeab906" in pins
    assert "ISAAC_ROS_NVBLOX_REF=v4.5-0" in pins
    assert "a0dbb2a06475dc8fa0dbdf5b919ec53973843d17" in pins
    assert "24eee4948768682fa1ffb969b881efee4fca29c2" in pins
    assert "release-4.5" in pins
    assert "ISAAC_DEBIAN_DIST=noble" in pins


def test_backend_b_uses_official_source_rosdep_and_isolated_docker():
    dockerfile = read("docker/Dockerfile.nvblox_b")
    setup = read("scripts/setup_nvblox_b_docker.sh")
    resolver = read("scripts/resolve_isaac_ros_image.py")
    assert "ARG BASE_IMAGE" in dockerfile
    assert "git clone --recursive" in dockerfile
    assert "rosdep install -i -r" in dockerfile
    assert "--rosdistro jazzy" in dockerfile
    assert "env_isaaclab" not in dockerfile
    assert "env_isaaclab" not in setup
    assert "HOST_PYTHON=/usr/bin/python3" in setup
    assert "--gpus all" in setup
    assert "'PyYAML==6.0.2' 'termcolor==2.4.0'" in setup
    assert "workspace-entrypoint.sh" in setup
    assert "--packages-up-to panda_handover_nvblox" in setup
    assert "Refusing to reset an existing checkout automatically" in setup
    assert "from build_image_layers import" in resolver
    assert "hashlib" not in resolver


def test_backend_b_runtime_preserves_same_project_paths_and_safety_policy():
    outer = read("scripts/run_nvblox_b.sh")
    inner = read("scripts/run_nvblox_b_inside.sh")
    config = read(
        "ros2/panda_handover_nvblox/config/conservative_nvblox.yaml"
    )
    assert '-v "${PROJECT_ROOT}:${PROJECT_ROOT}"' in outer
    assert "outputs/multiview_v1/camera_0" in outer
    assert "outputs/curobo_map_multiview_v1" in outer
    assert "outputs/conservative_esdf_b_v1" in outer
    assert "source /opt/ros/jazzy/setup.bash" in inner
    assert "trap cleanup EXIT INT TERM" in inner
    assert 'offline_esdf "$@"' in inner
    assert 'unobserved_esdf_policy: "occupied"' in config
    assert 'esdf_mode: "3d"' in config


def test_generated_backend_b_state_is_ignored_and_readme_stays_empty():
    assert ".backend-b/" in read(".gitignore")
    dockerignore = read(".dockerignore")
    assert ".backend-b" in dockerignore
    assert "outputs" in dockerignore
    assert (ROOT / "README.md").read_text(encoding="utf-8") == ""
