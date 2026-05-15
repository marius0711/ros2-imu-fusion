"""
Tests for ImuFusionNode — run without a live ROS2 stack.

Strategy: mock rclpy and all ROS2 message types so the node logic
can be tested as plain Python. The EKF integration, dt calculation,
covariance mapping, and topic wiring are all tested here.
"""

import sys
import types
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# ROS2 mock — must happen before any import of imu_fusion modules
# ---------------------------------------------------------------------------

def _make_ros2_mocks():
    """Inject minimal rclpy and sensor/geometry message stubs."""

    # --- rclpy ---
    rclpy_mod = types.ModuleType("rclpy")
    rclpy_mod.init = lambda args=None: None
    rclpy_mod.spin = lambda node: None
    rclpy_mod.shutdown = lambda: None

    node_mod = types.ModuleType("rclpy.node")

    class FakeNode:
        def __init__(self, name):
            self._name = name
            self._params = {}
            self._publishers = {}
            self._subscriptions = []
            self._log = _FakeLogger()

        def declare_parameter(self, name, default):
            self._params[name] = default

        def get_parameter(self, name):
            class _P:
                def __init__(self, v): self.value = v
            return _P(self._params.get(name))

        def create_publisher(self, msg_type, topic, qos):
            pub = _FakePublisher(topic)
            self._publishers[topic] = pub
            return pub

        def create_subscription(self, msg_type, topic, cb, qos):
            self._subscriptions.append((topic, cb))
            return (topic, cb)

        def get_logger(self):
            return self._log

        def destroy_node(self):
            pass

    class _FakeLogger:
        def info(self, msg): pass
        def warn(self, msg): pass
        def error(self, msg): pass

    class _FakePublisher:
        def __init__(self, topic):
            self.topic = topic
            self.messages = []
        def publish(self, msg):
            self.messages.append(msg)

    node_mod.Node = FakeNode
    rclpy_mod.node = node_mod

    qos_mod = types.ModuleType("rclpy.qos")

    class _QP:
        def __init__(self, **kw): pass

    class _RP:
        BEST_EFFORT = "BEST_EFFORT"

    class _HP:
        KEEP_LAST = "KEEP_LAST"

    qos_mod.QoSProfile = _QP
    qos_mod.ReliabilityPolicy = _RP
    qos_mod.HistoryPolicy = _HP

    # --- sensor_msgs ---
    sensor_mod = types.ModuleType("sensor_msgs")
    sensor_msg_mod = types.ModuleType("sensor_msgs.msg")

    class _Vec3:
        def __init__(self): self.x = self.y = self.z = 0.0

    class _Quat:
        def __init__(self): self.w = self.x = self.y = self.z = 0.0

    class _Header:
        def __init__(self):
            self.stamp = _Stamp()
            self.frame_id = ""

    class _Stamp:
        def __init__(self): self.sec = 0; self.nanosec = 0

    class _Imu:
        def __init__(self):
            self.header = _Header()
            self.angular_velocity = _Vec3()
            self.linear_acceleration = _Vec3()

    sensor_msg_mod.Imu = _Imu

    # --- geometry_msgs ---
    geo_mod = types.ModuleType("geometry_msgs")
    geo_msg_mod = types.ModuleType("geometry_msgs.msg")

    class _Pose:
        def __init__(self):
            self.orientation = _Quat()

    class _PoseWithCov:
        def __init__(self):
            self.pose = _Pose()
            self.covariance = [0.0] * 36

    class _PoseWithCovStamped:
        def __init__(self):
            self.header = _Header()
            self.pose = _PoseWithCov()

    class _V3Stamped:
        def __init__(self):
            self.header = _Header()
            self.vector = _Vec3()

    geo_msg_mod.PoseWithCovarianceStamped = _PoseWithCovStamped
    geo_msg_mod.Vector3Stamped = _V3Stamped

    # Register all mocks
    for name, mod in [
        ("rclpy", rclpy_mod),
        ("rclpy.node", node_mod),
        ("rclpy.qos", qos_mod),
        ("sensor_msgs", sensor_mod),
        ("sensor_msgs.msg", sensor_msg_mod),
        ("geometry_msgs", geo_mod),
        ("geometry_msgs.msg", geo_msg_mod),
    ]:
        sys.modules[name] = mod

    return _Imu, _FakePublisher


_Imu, _FakePublisher = _make_ros2_mocks()

# Now safe to import
sys.path.insert(0, "imu_fusion")
from imu_fusion.imu_fusion_node import ImuFusionNode  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FS = 200.0
G = 9.81


def make_node() -> ImuFusionNode:
    return ImuFusionNode()


def make_imu_msg(
    t_sec: float,
    gyro: tuple = (0.0, 0.0, 0.0),
    accel: tuple = (0.0, 0.0, G),
) -> _Imu:
    msg = _Imu()
    msg.header.stamp.sec = int(t_sec)
    msg.header.stamp.nanosec = int((t_sec % 1) * 1e9)
    msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = gyro
    msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = accel
    return msg


def feed_static(node: ImuFusionNode, n: int = 200, dt: float = 1.0 / FS):
    """Feed n static IMU messages (gravity aligned, zero gyro)."""
    for i in range(n):
        t = (i + 1) * dt
        msg = make_imu_msg(t, gyro=(0.0, 0.0, 0.0), accel=(0.0, 0.0, G))
        node._imu_callback(msg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNodeInit:
    def test_node_creates_without_error(self):
        node = make_node()
        assert node is not None

    def test_default_ekf_quaternion_is_identity(self):
        node = make_node()
        q = node.ekf.get_quaternion()
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-10)

    def test_publishers_registered(self):
        node = make_node()
        assert "/imu/orientation" in node._publishers
        assert "/imu/bias" in node._publishers

    def test_subscriber_registered(self):
        node = make_node()
        topics = [t for t, _ in node._subscriptions]
        assert "/imu/data" in topics


class TestFirstMessageSkipped:
    def test_first_message_does_not_publish(self):
        node = make_node()
        msg = make_imu_msg(1.0)
        node._imu_callback(msg)
        assert len(node._publishers["/imu/orientation"].messages) == 0

    def test_second_message_publishes(self):
        node = make_node()
        node._imu_callback(make_imu_msg(1.0))
        node._imu_callback(make_imu_msg(1.005))
        assert len(node._publishers["/imu/orientation"].messages) == 1


class TestStaticOrientation:
    def test_static_orientation_stays_near_identity(self):
        node = make_node()
        feed_static(node, n=400)

        pub = node._publishers["/imu/orientation"]
        assert len(pub.messages) > 0

        last = pub.messages[-1]
        q = np.array([
            last.pose.pose.orientation.w,
            last.pose.pose.orientation.x,
            last.pose.pose.orientation.y,
            last.pose.pose.orientation.z,
        ])
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=0.02)

    def test_static_euler_angles_near_zero(self):
        node = make_node()
        feed_static(node, n=400)
        roll, pitch, _ = node.ekf.get_euler_deg()
        assert abs(roll) < 1.0
        assert abs(pitch) < 1.0


class TestCovarianceOutput:
    def test_covariance_has_36_elements(self):
        node = make_node()
        node._imu_callback(make_imu_msg(0.0))
        node._imu_callback(make_imu_msg(0.005))
        msg = node._publishers["/imu/orientation"].messages[-1]
        assert len(msg.pose.covariance) == 36

    def test_covariance_orientation_block_is_positive(self):
        node = make_node()
        node._imu_callback(make_imu_msg(0.0))
        node._imu_callback(make_imu_msg(0.005))
        msg = node._publishers["/imu/orientation"].messages[-1]
        cov = msg.pose.covariance
        # Diagonal elements of orientation block (rows/cols 3-5)
        diag = [cov[3*6+3], cov[4*6+4], cov[5*6+5]]
        assert all(v > 0 for v in diag), f"Non-positive diagonal: {diag}"


class TestBiasOutput:
    def test_bias_published_after_second_message(self):
        node = make_node()
        node._imu_callback(make_imu_msg(0.0))
        node._imu_callback(make_imu_msg(0.005))
        assert len(node._publishers["/imu/bias"].messages) == 1

    def test_bias_near_zero_at_startup(self):
        node = make_node()
        feed_static(node, n=10)
        bias = node._publishers["/imu/bias"].messages[-1]
        assert abs(bias.vector.x) < 0.01
        assert abs(bias.vector.y) < 0.01
        assert abs(bias.vector.z) < 0.01


class TestDtHandling:
    def test_zero_dt_skipped(self):
        node = make_node()
        node._imu_callback(make_imu_msg(1.0))
        node._imu_callback(make_imu_msg(1.0))  # same timestamp
        assert len(node._publishers["/imu/orientation"].messages) == 0

    def test_large_dt_skipped(self):
        node = make_node()
        node._imu_callback(make_imu_msg(0.0))
        node._imu_callback(make_imu_msg(2.0))  # 2 s gap > 1 s threshold
        assert len(node._publishers["/imu/orientation"].messages) == 0

    def test_normal_dt_processed(self):
        node = make_node()
        node._imu_callback(make_imu_msg(0.0))
        node._imu_callback(make_imu_msg(0.005))
        assert len(node._publishers["/imu/orientation"].messages) == 1


class TestMessageTimestamps:
    def test_output_timestamp_matches_input(self):
        node = make_node()
        node._imu_callback(make_imu_msg(1.0))
        t = 1.005
        node._imu_callback(make_imu_msg(t))
        msg = node._publishers["/imu/orientation"].messages[-1]
        t_out = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        assert abs(t_out - t) < 1e-6

    def test_output_frame_id_is_imu_link(self):
        node = make_node()
        node._imu_callback(make_imu_msg(0.0))
        node._imu_callback(make_imu_msg(0.005))
        msg = node._publishers["/imu/orientation"].messages[-1]
        assert msg.header.frame_id == "imu_link"
