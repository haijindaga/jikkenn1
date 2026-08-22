"""Launch the official nvblox component with a conservative 3-D ESDF policy."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("panda_handover_nvblox"),
        "config",
        "conservative_nvblox.yaml",
    )
    node = ComposableNode(
        name="nvblox_node",
        package="nvblox_ros",
        plugin="nvblox::NvbloxNode",
        parameters=[config],
    )
    container = ComposableNodeContainer(
        name="nvblox_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=[node],
        output="screen",
    )
    return LaunchDescription([container])
