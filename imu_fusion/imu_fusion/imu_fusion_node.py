"""
imu_fusion_node.py — ROS2 node wrapping the EKF for IMU orientation estimation.

Subscribes:
  /imu/data  (sensor_msgs/Imu)

Publishes:
  /imu/orientation  (geometry_msgs/PoseWithCovarianceStamped)
  /imu/bias         (geometry_msgs/Vector3Stamped)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseWithCovarianceStamped, Vector3Stamped

from imu_fusion.ekf import EKF


class ImuFusionNode(Node):
    """
    ROS2 node that fuses gyroscope and accelerometer data via an EKF.

    On each incoming Imu message:
      1. EKF predict step using gyroscope reading and dt.
      2. EKF update step using accelerometer reading.
      3. Publish orientation as PoseWithCovarianceStamped.
      4. Publish estimated gyro bias as Vector3Stamped.
    """

    def __init__(self) -> None:
        super().__init__("imu_fusion_node")

        # Parameters (can be overridden from launch file or CLI)
        self.declare_parameter("sigma_gyro",  1.7e-4)
        self.declare_parameter("sigma_bias",  1.9e-5)
        self.declare_parameter("sigma_accel", 2.0e-3)
        self.declare_parameter("fs",          200.0)
        self.declare_parameter("g",           9.81)
        self.declare_parameter("imu_topic",   "/imu/data")

        sigma_gyro  = self.get_parameter("sigma_gyro").value
        sigma_bias  = self.get_parameter("sigma_bias").value
        sigma_accel = self.get_parameter("sigma_accel").value
        fs          = self.get_parameter("fs").value
        g           = self.get_parameter("g").value
        imu_topic   = self.get_parameter("imu_topic").value

        self.ekf = EKF(
            sigma_gyro=sigma_gyro,
            sigma_bias=sigma_bias,
            sigma_accel=sigma_accel,
            fs=fs,
            g=g,
        )

        self._last_stamp = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._sub = self.create_subscription(
            Imu,
            imu_topic,
            self._imu_callback,
            qos,
        )

        self._pub_pose = self.create_publisher(
            PoseWithCovarianceStamped,
            "/imu/orientation",
            10,
        )

        self._pub_bias = self.create_publisher(
            Vector3Stamped,
            "/imu/bias",
            10,
        )

        self.get_logger().info(
            f"ImuFusionNode started — subscribing to {imu_topic}"
        )

    def _imu_callback(self, msg: Imu) -> None:
        # Compute dt from message timestamps
        stamp = msg.header.stamp
        t_now = stamp.sec + stamp.nanosec * 1e-9

        if self._last_stamp is None:
            self._last_stamp = t_now
            return

        dt = t_now - self._last_stamp
        self._last_stamp = t_now

        if dt <= 0.0 or dt > 1.0:
            self.get_logger().warn(f"Skipping step with dt={dt:.4f} s")
            return

        # Extract IMU readings
        gyro  = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ])
        accel = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ])

        # EKF predict + update
        self.ekf.predict(gyro, dt)
        self.ekf.update(accel)

        q    = self.ekf.get_quaternion()
        bias = self.ekf.get_bias()
        cov  = self.ekf.get_covariance_quaternion()

        # Publish orientation
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp    = msg.header.stamp
        pose_msg.header.frame_id = "imu_link"

        pose_msg.pose.pose.orientation.w = float(q[0])
        pose_msg.pose.pose.orientation.x = float(q[1])
        pose_msg.pose.pose.orientation.y = float(q[2])
        pose_msg.pose.pose.orientation.z = float(q[3])

        # Fill 6x6 covariance (orientation block only, rows/cols 3-5)
        flat_cov = [0.0] * 36
        for i in range(3):
            for j in range(3):
                flat_cov[(3 + i) * 6 + (3 + j)] = float(cov[i + 1, j + 1])
        pose_msg.pose.covariance = flat_cov

        self._pub_pose.publish(pose_msg)

        # Publish bias
        bias_msg = Vector3Stamped()
        bias_msg.header.stamp    = msg.header.stamp
        bias_msg.header.frame_id = "imu_link"
        bias_msg.vector.x = float(bias[0])
        bias_msg.vector.y = float(bias[1])
        bias_msg.vector.z = float(bias[2])

        self._pub_bias.publish(bias_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImuFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
