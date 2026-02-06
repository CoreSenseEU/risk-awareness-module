import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Load Config From File
    config = os.path.join(
        get_package_share_directory("riskam_bringup"), "config", "riskam_config.yml"
    )

    # Create argument for launching bagger
    bagger_arg = DeclareLaunchArgument("run_bagger", default_value="false")

    # Bringup the RiskAM Node
    riskam_node = Node(
        package="riskam_ros", executable="riskam_node.py", name="riskam_node", parameters=[config]
    )

    # Bringup the Logger Node
    bagger_node = Node(
        package="riskam_ros",
        executable="riskam_bagger.py",
        name="riskam_bagger",
        condition=IfCondition(LaunchConfiguration("run_bagger")),
        parameters=[config],
    )

    return LaunchDescription([bagger_arg, riskam_node, bagger_node])
