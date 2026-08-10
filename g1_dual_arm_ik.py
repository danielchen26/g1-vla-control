"""Damped least-squares dual-arm IK for the MuJoCo Menagerie Unitree G1."""

from __future__ import annotations

import numpy as np
import mujoco


LEFT_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
]
RIGHT_JOINTS = [name.replace("left_", "right_") for name in LEFT_JOINTS]


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def orientation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """World-frame rotation vector from current to target (wxyz)."""
    error = _quat_multiply(target, _quat_conjugate(current))
    if error[0] < 0.0:
        error = -error
    vector_norm = np.linalg.norm(error[1:])
    if vector_norm < 1e-9:
        return 2.0 * error[1:]
    angle = 2.0 * np.arctan2(vector_norm, np.clip(error[0], -1.0, 1.0))
    return error[1:] / vector_norm * angle


class G1DualArmIK:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        damping: float = 0.08,
        max_joint_speed: float = 1.2,
    ):
        self.model = model
        self.data = data
        self.damping = damping
        self.max_joint_speed = max_joint_speed
        self.left_body = self._body_id("left_wrist_yaw_link")
        self.right_body = self._body_id("right_wrist_yaw_link")
        self.left = self._joint_info(LEFT_JOINTS)
        self.right = self._joint_info(RIGHT_JOINTS)
        self.q_target = data.qpos.copy()

    def _body_id(self, name: str) -> int:
        result = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if result < 0:
            raise ValueError(f"Missing body: {name}")
        return result

    def _joint_info(self, names: list[str]) -> dict[str, np.ndarray]:
        joint_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in names
        ])
        actuator_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in names
        ])
        if np.any(joint_ids < 0) or np.any(actuator_ids < 0):
            raise ValueError("G1 arm joint/actuator mapping is incomplete")
        return {
            "joint": joint_ids,
            "qpos": self.model.jnt_qposadr[joint_ids],
            "dof": self.model.jnt_dofadr[joint_ids],
            "actuator": actuator_ids,
        }

    def reset(self) -> None:
        self.q_target[:] = self.data.qpos

    def pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        body = self.left_body if side == "left" else self.right_body
        return self.data.xpos[body].copy(), self.data.xquat[body].copy()

    def _solve_side(
        self,
        info: dict[str, np.ndarray],
        body: int,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
        dt: float,
    ) -> tuple[float, float]:
        jac_position = np.zeros((3, self.model.nv))
        jac_rotation = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jac_position, jac_rotation, body)
        jacobian = np.vstack((jac_position[:, info["dof"]], jac_rotation[:, info["dof"]]))

        position_error = np.asarray(target_position) - self.data.xpos[body]
        rotation_error = orientation_error(
            np.asarray(target_quaternion), self.data.xquat[body]
        )
        error = np.concatenate((position_error, 0.45 * rotation_error))
        regularizer = self.damping ** 2 * np.eye(6)
        joint_delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + regularizer, error
        )
        joint_velocity = np.clip(
            5.0 * joint_delta, -self.max_joint_speed, self.max_joint_speed
        )
        qpos_indices = info["qpos"]
        self.q_target[qpos_indices] += joint_velocity * dt

        for local_index, joint_id in enumerate(info["joint"]):
            if self.model.jnt_limited[joint_id]:
                low, high = self.model.jnt_range[joint_id]
                q_index = qpos_indices[local_index]
                self.q_target[q_index] = np.clip(self.q_target[q_index], low, high)
        self.data.ctrl[info["actuator"]] = self.q_target[qpos_indices]
        return float(np.linalg.norm(position_error)), float(np.linalg.norm(rotation_error))

    def step(
        self,
        left_action: np.ndarray,
        right_action: np.ndarray,
        dt: float,
    ) -> dict[str, float]:
        left_errors = self._solve_side(
            self.left, self.left_body, left_action[:3], left_action[3:7], dt
        )
        right_errors = self._solve_side(
            self.right, self.right_body, right_action[:3], right_action[3:7], dt
        )
        return {
            "left_position_error": left_errors[0],
            "left_orientation_error": left_errors[1],
            "right_position_error": right_errors[0],
            "right_orientation_error": right_errors[1],
        }
