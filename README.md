# ros2-imu-fusion

A ROS2 node that fuses gyroscope and accelerometer data from a 6-axis IMU into a real-time
orientation estimate using an Extended Kalman Filter.

Part of a robotics sensing portfolio — see also
[imu-kalman-filter](https://github.com/marius0711/imu-kalman-filter) (EKF from scratch, noise
characterization) and [sensor-calibration](https://github.com/marius0711/sensor-calibration)
(accelerometer calibration pipeline).

---

## What this does

Subscribes to `/imu/data` (`sensor_msgs/Imu`) and publishes:

- `/imu/orientation` (`geometry_msgs/PoseWithCovarianceStamped`) — quaternion orientation with
  covariance from the EKF state
- `/imu/bias` (`geometry_msgs/Vector3Stamped`) — online gyroscope bias estimate

The EKF tracks a 7-dimensional state: quaternion `[q0, q1, q2, q3]` and gyro bias `[bx, by, bz]`.
Bias is estimated online — not pre-calibrated — which is what separates a 1-minute filter from a
10-minute filter in practice. Noise matrices Q and R are derived from Allan deviation results on
the ADIS16448 IMU (see imu-kalman-filter).

---

## Architecture

```
/imu/data  (sensor_msgs/Imu)
      │
      ▼
ImuFusionNode
  ├── EKF.predict(gyro, dt)     # quaternion kinematics + bias propagation
  └── EKF.update(accel)         # gravity reference in body frame
      │
      ├── /imu/orientation  (geometry_msgs/PoseWithCovarianceStamped)
      └── /imu/bias         (geometry_msgs/Vector3Stamped)
```

---

## Repository structure

```
ros2-imu-fusion/
├── .devcontainer/
│   └── devcontainer.json          # ROS2 Humble on Ubuntu 22.04
├── imu_fusion/
│   ├── imu_fusion/
│   │   ├── ekf.py                 # EKF core (7D state, quaternion + bias)
│   │   └── imu_fusion_node.py     # ROS2 node
│   ├── launch/
│   │   └── imu_fusion.launch.py   # parameterized launch file
│   ├── test/
│   │   └── test_imu_fusion_node.py  # 17 tests (no ROS2 runtime needed)
│   ├── package.xml
│   └── setup.py
└── .gitignore
```

---

## Quickstart in GitHub Codespaces

Open in Codespaces — the devcontainer pulls ROS2 Humble automatically.

```bash
# Build
source /opt/ros/humble/setup.bash
colcon build --packages-select imu_fusion
source install/setup.bash

# Run node
ros2 run imu_fusion imu_fusion_node

# Publish test IMU data (second terminal)
ros2 topic pub /imu/data sensor_msgs/msg/Imu \
  "{header: {stamp: 'now', frame_id: 'imu_link'}, \
    angular_velocity: {x: 0.0, y: 0.2, z: 0.0}, \
    linear_acceleration: {x: 0.0, y: 0.0, z: 9.81}}" \
  --rate 50

# Read orientation output (third terminal)
ros2 topic echo /imu/orientation --qos-reliability best_effort --once
```

### Expected output

```yaml
header:
  frame_id: imu_link
pose:
  pose:
    orientation:
      x: 0.0
      y: -0.983
      z: 0.0
      w: -0.186
  covariance:
    - ...  # non-zero EKF covariance block
```

---

## Run tests

Tests use a full ROS2 mock — no running ROS2 stack required. Works on any machine with Python 3.9+
and numpy.

```bash
PYTHONPATH=imu_fusion python3 -m pytest imu_fusion/test/test_imu_fusion_node.py -v \
  -p no:ament_flake8 -p no:ament_pep257 -p no:ament_copyright
```

```
17 passed in 0.57s
```

| Test class | Tests | What is verified |
|---|---|---|
| TestNodeInit | 4 | Node creation, EKF init, publisher/subscriber registration |
| TestFirstMessageSkipped | 2 | First message skipped (no dt), second message publishes |
| TestStaticOrientation | 2 | Static IMU stays near identity, Euler angles < 1° |
| TestCovarianceOutput | 2 | 36-element covariance, orientation block positive |
| TestBiasOutput | 2 | Bias published, near zero at startup |
| TestDtHandling | 3 | Zero dt skipped, large dt skipped, normal dt processed |
| TestMessageTimestamps | 2 | Output timestamp matches input, frame_id correct |

---

## Launch file

```bash
ros2 launch imu_fusion imu_fusion.launch.py
```

Parameters (all overridable):

| Parameter | Default | Description |
|---|---|---|
| `sigma_gyro` | 1.7e-4 | Gyro noise density [rad/s/√Hz] |
| `sigma_bias` | 1.9e-5 | Bias random walk [rad/s²/√Hz] |
| `sigma_accel` | 2.0e-3 | Accel noise density [m/s²/√Hz] |
| `fs` | 200.0 | Nominal sampling rate [Hz] |
| `g` | 9.81 | Gravitational acceleration [m/s²] |
| `imu_topic` | /imu/data | Input IMU topic |

---

## EKF design

The filter tracks orientation as a unit quaternion to avoid gimbal lock. The gyroscope bias is
included as a state (not pre-calibrated), which means the filter actively learns sensor drift
during operation.

Noise parameters Q and R are derived from Allan deviation analysis on the ADIS16448 IMU — not
manually tuned. The Jacobians F (7×7) and H (3×7) are derived analytically and verified against
numerical finite differences in the test suite of the companion project.

For the full derivation and benchmark results (EKF RMSE 0.15° vs. 175° for accel-only), see
[imu-kalman-filter](https://github.com/marius0711/imu-kalman-filter).

---

## Related projects

1. [imu-kalman-filter](https://github.com/marius0711/imu-kalman-filter) — EKF from scratch,
   Allan deviation noise characterization, filter comparison (36 tests)
2. **ros2-imu-fusion** (this repo) — EKF as a production ROS2 node
3. [sensor-calibration](https://github.com/marius0711/sensor-calibration) — six-position
   accelerometer calibration pipeline
