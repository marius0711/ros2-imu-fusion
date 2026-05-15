"""
Extended Kalman Filter for IMU orientation estimation.

State vector (7D): x = [q0, q1, q2, q3, bx, by, bz]
  - Quaternion (scalar-first convention)
  - Gyroscope bias (3D)

Process model: quaternion kinematics driven by gyro measurements.
Measurement model: gravity projection in body frame via rotation matrix.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
from typing import Tuple


class EKF:
    """
    Extended Kalman Filter for attitude estimation from a 6-DOF IMU.

    Tracks quaternion orientation and gyroscope bias. The measurement update
    uses accelerometer readings compared against the expected gravity vector
    in the body frame.

    Args:
        sigma_gyro:  Gyro noise density [rad/s/sqrt(Hz)], e.g. ADIS16448 ARW.
        sigma_bias:  Gyro bias walk density [rad/s^2/sqrt(Hz)].
        sigma_accel: Accel noise density [m/s^2/sqrt(Hz)].
        fs:          Sampling frequency [Hz].
        g:           Gravitational acceleration [m/s^2].
    """

    def __init__(
        self,
        sigma_gyro: float = 1.7e-4,
        sigma_bias: float = 1.9e-5,
        sigma_accel: float = 2.0e-3,
        fs: float = 200.0,
        g: float = 9.81,
    ) -> None:
        self.g = g
        self.fs = fs
        dt = 1.0 / fs

        # Process noise covariance Q (7x7)
        # Quaternion states: driven by gyro ARW discretised to variance/step
        q_gyro = (sigma_gyro ** 2) * dt
        # Bias states: driven by bias instability random walk
        q_bias = (sigma_bias ** 2) * dt
        self.Q = np.diag([q_gyro] * 4 + [q_bias] * 3)

        # Measurement noise covariance R (3x3)
        # Noise density [m/s^2/sqrt(Hz)] -> variance at fs: sigma^2 * fs
        sigma_accel_meas = sigma_accel * np.sqrt(fs)
        self.R = np.eye(3) * (sigma_accel_meas ** 2)

        # Initial state: identity quaternion, zero bias
        self.x: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Initial covariance: small uncertainty on quat, larger on bias
        self.P: np.ndarray = np.diag([1e-4] * 4 + [1e-6] * 3)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _omega_matrix(self, w: np.ndarray) -> np.ndarray:
        """
        4x4 skew-symmetric Omega matrix for quaternion kinematics.

        dq/dt = 0.5 * Omega(w) @ q
        """
        wx, wy, wz = w
        return np.array(
            [
                [0.0, -wx, -wy, -wz],
                [wx, 0.0, wz, -wy],
                [wy, -wz, 0.0, wx],
                [wz, wy, -wx, 0.0],
            ]
        )

    def _quat_to_rot(self, q: np.ndarray) -> np.ndarray:
        """
        Rotation matrix from quaternion [q0, q1, q2, q3] (scalar first).

        Convention: rotates vectors from body frame to world frame.
        R @ v_body = v_world
        """
        q0, q1, q2, q3 = q
        return np.array(
            [
                [1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
                [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
                [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)],
            ]
        )

    def _h(self, q: np.ndarray) -> np.ndarray:
        """
        Measurement model: expected accelerometer reading in body frame.

        h(q) = R(q)^T @ [0, 0, g]
        """
        R = self._quat_to_rot(q)
        gravity_world = np.array([0.0, 0.0, self.g])
        return R.T @ gravity_world

    def _H_jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        3x7 Jacobian of the measurement model h w.r.t. state x.

        dh/dx = [dh/dq (3x4) | dh/db (3x3)]

        h = [2g(q1*q3 - q0*q2),
             2g(q2*q3 + q0*q1),
             g*(1 - 2*q1^2 - 2*q2^2)]
        """
        q0, q1, q2, q3 = q
        g = self.g

        # dh/dq (3x4)
        H_q = 2.0 * g * np.array(
            [
                [-q2, q3, -q0, q1],
                [q1, q0, q3, q2],
                [0.0, -2.0 * q1, -2.0 * q2, 0.0],
            ]
        )
        # Third row factor: h3 = g*(1 - 2q1^2 - 2q2^2)
        # dh3/dqi = g*(-4q1, -4q2, 0) = 2g*(0, -2q1, -2q2, 0)  checked above

        # dh/db = 0 (measurement does not depend on bias directly)
        H_b = np.zeros((3, 3))

        return np.hstack([H_q, H_b])

    def _F_jacobian(self, q: np.ndarray, omega_c: np.ndarray, dt: float) -> np.ndarray:
        """
        7x7 Jacobian of the process model f w.r.t. state x.

        F = [[F_qq (4x4), F_qb (4x3)],
             [0    (3x4), I    (3x3)]]

        F_qq = I + 0.5*dt*Omega(omega_c)
        F_qb = -0.5*dt * d(Omega(omega_c)*q)/d(omega_c)   (chain rule: d(omega_c)/d(b) = -I)
        """
        Omega = self._omega_matrix(omega_c)
        F_qq = np.eye(4) + 0.5 * dt * Omega

        q0, q1, q2, q3 = q
        # d(Omega(w)*q)/d(w) is a 4x3 matrix (derivative w.r.t. [wx, wy, wz])
        # Negated because d(omega_c)/d(b) = -I
        F_qb = 0.5 * dt * np.array(
            [
                [q1, q2, q3],
                [-q0, q3, -q2],
                [-q3, -q0, q1],
                [q2, -q1, -q0],
            ]
        )

        F = np.zeros((7, 7))
        F[:4, :4] = F_qq
        F[:4, 4:] = F_qb
        F[4:, 4:] = np.eye(3)
        return F

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, q0: np.ndarray = None, P0: np.ndarray = None) -> None:
        """Reset state and covariance to initial values."""
        if q0 is not None:
            self.x = np.concatenate([q0, np.zeros(3)])
        else:
            self.x = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if P0 is not None:
            self.P = P0
        else:
            self.P = np.diag([1e-4] * 4 + [1e-6] * 3)

    def predict(self, omega_measured: np.ndarray, dt: float) -> None:
        """
        EKF prediction step.

        Args:
            omega_measured: Gyroscope reading [rad/s], shape (3,).
            dt:             Time step [s].
        """
        q = self.x[:4]
        bias = self.x[4:]
        omega_c = omega_measured - bias

        # Compute Jacobian before state update (linearise at current state)
        F = self._F_jacobian(q, omega_c, dt)

        # Propagate quaternion via first-order integration
        Omega = self._omega_matrix(omega_c)
        q_new = q + 0.5 * dt * Omega @ q
        q_new /= np.linalg.norm(q_new)

        self.x = np.concatenate([q_new, bias])
        self.P = F @ self.P @ F.T + self.Q

    def update(self, accel_measured: np.ndarray) -> None:
        """
        EKF measurement update step using accelerometer.

        Args:
            accel_measured: Accelerometer reading [m/s^2], shape (3,).
        """
        q = self.x[:4]

        h = self._h(q)
        H = self._H_jacobian(q)

        y = accel_measured - h                       # innovation
        S = H @ self.P @ H.T + self.R                # innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)         # Kalman gain

        self.x = self.x + K @ y
        self.x[:4] /= np.linalg.norm(self.x[:4])    # re-normalise quaternion
        self.P = (np.eye(7) - K @ H) @ self.P

    def get_quaternion(self) -> np.ndarray:
        """Return current quaternion estimate [q0, q1, q2, q3]."""
        return self.x[:4].copy()

    def get_bias(self) -> np.ndarray:
        """Return current gyroscope bias estimate [rad/s]."""
        return self.x[4:].copy()

    def get_covariance_quaternion(self) -> np.ndarray:
        """Return 4x4 covariance block for the quaternion states."""
        return self.P[:4, :4].copy()

    def get_euler_deg(self) -> Tuple[float, float, float]:
        """
        Return roll, pitch, yaw [degrees] from current quaternion.

        Angles follow ZYX convention (yaw-pitch-roll intrinsic).
        """
        q0, q1, q2, q3 = self.x[:4]
        roll = np.degrees(np.arctan2(2.0 * (q0 * q1 + q2 * q3), 1.0 - 2.0 * (q1 ** 2 + q2 ** 2)))
        pitch = np.degrees(np.arcsin(np.clip(2.0 * (q0 * q2 - q3 * q1), -1.0, 1.0)))
        yaw = np.degrees(np.arctan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 ** 2 + q3 ** 2)))
        return roll, pitch, yaw


def run_ekf_on_dataset(
    gyro: np.ndarray,
    accel: np.ndarray,
    timestamps: np.ndarray,
    sigma_gyro: float = 1.7e-4,
    sigma_bias: float = 1.9e-5,
    sigma_accel: float = 2.0e-3,
    fs: float = 200.0,
) -> np.ndarray:
    """
    Run EKF over a full IMU dataset.

    Args:
        gyro:       Gyroscope data, shape (N, 3) [rad/s].
        accel:      Accelerometer data, shape (N, 3) [m/s^2].
        timestamps: Time array, shape (N,) [s].
        sigma_gyro: Gyro noise density [rad/s/sqrt(Hz)].
        sigma_bias: Bias walk density [rad/s^2/sqrt(Hz)].
        sigma_accel: Accel noise density [m/s^2/sqrt(Hz)].
        fs:         Nominal sampling rate [Hz].

    Returns:
        quaternions: Shape (N, 4) estimated quaternions [q0, q1, q2, q3].
    """
    N = gyro.shape[0]
    quaternions = np.zeros((N, 4))

    ekf = EKF(sigma_gyro=sigma_gyro, sigma_bias=sigma_bias, sigma_accel=sigma_accel, fs=fs)
    quaternions[0] = ekf.get_quaternion()

    for i in range(1, N):
        dt = float(timestamps[i] - timestamps[i - 1])
        ekf.predict(gyro[i], dt)
        ekf.update(accel[i])
        quaternions[i] = ekf.get_quaternion()

    return quaternions


def quat_angle_error_deg(q_est: np.ndarray, q_true: np.ndarray) -> np.ndarray:
    """
    Compute angular error between estimated and true quaternions.

    Returns angle [degrees] of the relative rotation for each timestep.

    Args:
        q_est:  Estimated quaternions, shape (N, 4).
        q_true: Ground truth quaternions, shape (N, 4).

    Returns:
        errors: Angular errors in degrees, shape (N,).
    """
    # Dot product of unit quaternions gives cos(theta/2)
    dots = np.clip(np.abs(np.sum(q_est * q_true, axis=1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run EKF on a synthetic IMU dataset.")
    parser.add_argument(
        "--dataset",
        default="data/MH_01_easy",
        help="Path to EuRoC-format dataset directory.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Save quaternion output as .npy to this path.",
    )
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(__file__))
    from euroc_loader import load_imu

    imu = load_imu(args.dataset)

    timestamps = imu["timestamps"]
    gyro = imu["gyro"]
    accel = imu["accel"]

    print(f"Loaded {len(timestamps)} IMU samples from {args.dataset}")

    quats = run_ekf_on_dataset(gyro, accel, timestamps)
    print(f"EKF complete. Final quaternion: {quats[-1]}")

    if args.save:
        np.save(args.save, quats)
        print(f"Saved quaternions to {args.save}")
