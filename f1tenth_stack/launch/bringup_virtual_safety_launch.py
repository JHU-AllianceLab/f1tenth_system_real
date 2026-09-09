"""Hardware bringup plus optional real-ego/virtual-opponent safety control."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    stack_share = get_package_share_directory("f1tenth_stack")
    deployment_share = get_package_share_directory("f1tenth_safety_deployment")
    deployment_config = os.path.join(
        deployment_share, "config", "deployment.yaml"
    )
    default_model_bundle = os.path.join(
        deployment_share,
        "model_bundles",
        "real_track_compact_seed17_hold25_qcert_20260909",
    )

    model_bundle = DeclareLaunchArgument(
        "model_bundle",
        default_value=default_model_bundle,
        description="Directory containing the exported P1/P2 TorchScript bundle",
    )
    safety_config = DeclareLaunchArgument(
        "safety_deployment_config",
        default_value=deployment_config,
        description="Parameters for real-ego/virtual-opponent deployment",
    )
    enable_controller = DeclareLaunchArgument(
        "enable_safety_controller",
        default_value="false",
        description="Start the learned controller (shadow mode is configured separately)",
    )
    enable_opponent = DeclareLaunchArgument(
        "enable_virtual_opponent",
        default_value="false",
        description="Start the centerline-following virtual opponent",
    )

    hardware = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(stack_share, "launch", "bringup_launch.py")
        )
    )
    controller = Node(
        package="f1tenth_safety_deployment",
        executable="composed_controller",
        name="composed_safety_controller",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_safety_controller")),
        parameters=[
            LaunchConfiguration("safety_deployment_config"),
            {"model_bundle": LaunchConfiguration("model_bundle")},
        ],
    )
    opponent = Node(
        package="f1tenth_safety_deployment",
        executable="virtual_opponent",
        name="virtual_opponent",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_virtual_opponent")),
        parameters=[
            LaunchConfiguration("safety_deployment_config"),
            {"model_bundle": LaunchConfiguration("model_bundle")},
        ],
    )

    return LaunchDescription(
        [
            model_bundle,
            safety_config,
            enable_controller,
            enable_opponent,
            hardware,
            controller,
            opponent,
        ]
    )
