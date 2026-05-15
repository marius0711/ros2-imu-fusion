from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="imu_fusion",
            executable="imu_fusion_node",
            name="imu_fusion_node",
            parameters=[{
                "sigma_gyro":  1.7e-4,
                "sigma_bias":  1.9e-5,
                "sigma_accel": 2.0e-3,
                "fs":          200.0,
                "g":           9.81,
                "imu_topic":   "/imu/data",
            }],
            output="screen",
        )
    ])
