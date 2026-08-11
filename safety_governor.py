"""Simulation-only motion filter and conservative IK/collision preflight gate.

The defaults come from P99 finite differences over the public training targets.
They are useful engineering starting points, not Unitree hardware safety limits.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from action_schema import LEFT_POS, LEFT_QUAT, RIGHT_POS, RIGHT_QUAT, GRIPPERS
from g1_dual_arm_ik import G1DualArmIK, orientation_error


@dataclass(frozen=True)
class MotionEnvelope:
    max_eef_speed_m_s: float = 0.227
    max_eef_acceleration_m_s2: float = 2.760
    max_eef_jerk_m_s3: float = 125.67
    max_angular_speed_rad_s: float = 1.044
    max_gripper_speed_rad_s: float = 5.793
    source: str = "public training-target P99; simulation-only"


@dataclass(frozen=True)
class FilterTelemetry:
    max_translation_speed_m_s: float
    max_translation_acceleration_m_s2: float
    max_translation_jerk_m_s3: float
    max_angular_speed_rad_s: float
    max_gripper_speed_rad_s: float


@dataclass(frozen=True)
class TargetSafetyResult:
    accepted: bool
    reachable: bool
    collision_free: bool
    joint_limits_ok: bool
    position_error_m: float
    orientation_error_rad: float
    reason: str


def _clip_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= maximum or norm < 1e-12:
        return vector
    return vector * (maximum / norm)


def _quat_slerp_wxyz(q0: np.ndarray, q1: np.ndarray, max_angle: float) -> tuple[np.ndarray, float]:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    angle = 2.0 * np.arccos(dot)
    if angle < 1e-10:
        return q1, 0.0
    fraction = min(1.0, max_angle / angle)
    theta = np.arccos(dot)
    if theta < 1e-8:
        result = q0 + fraction * (q1 - q0)
    else:
        result = (
            np.sin((1.0 - fraction) * theta) / np.sin(theta) * q0
            + np.sin(fraction * theta) / np.sin(theta) * q1
        )
    result /= np.linalg.norm(result)
    return result, angle * fraction


class JerkLimitedActionFilter:
    """Track world-frame EEF targets while hard-limiting translational v/a/j.

    Input/output layout remains 16-D, but quaternion slices are MuJoCo wxyz.
    This safety layer may lag the requested path; it is not a pure retimer.
    """

    def __init__(
        self,
        envelope: MotionEnvelope | None = None,
        position_gain: float = 8.0,
        velocity_gain: float = 20.0,
    ):
        self.envelope = envelope or MotionEnvelope()
        self.position_gain = position_gain
        self.velocity_gain = velocity_gain
        self.command: np.ndarray | None = None
        self.velocity = np.zeros((2, 3))
        self.acceleration = np.zeros((2, 3))

    def reset(self, command: np.ndarray) -> None:
        command = np.asarray(command, dtype=np.float64)
        if command.shape != (16,):
            raise ValueError("command must have shape (16,)")
        self.command = command.copy()
        for quaternion_slice in (LEFT_QUAT, RIGHT_QUAT):
            norm = np.linalg.norm(self.command[quaternion_slice])
            if norm < 1e-9:
                raise ValueError("zero quaternion")
            self.command[quaternion_slice] /= norm
        self.velocity.fill(0.0)
        self.acceleration.fill(0.0)

    def step(self, desired: np.ndarray, dt: float) -> tuple[np.ndarray, FilterTelemetry]:
        if dt <= 0:
            raise ValueError("dt must be positive")
        desired = np.asarray(desired, dtype=np.float64)
        if desired.shape != (16,):
            raise ValueError("desired must have shape (16,)")
        if self.command is None:
            self.reset(desired)
        assert self.command is not None
        speed_values, acceleration_values, jerk_values = [], [], []
        angular_values = []
        for side_index, (position_slice, quaternion_slice) in enumerate((
            (LEFT_POS, LEFT_QUAT), (RIGHT_POS, RIGHT_QUAT)
        )):
            position = self.command[position_slice]
            error = desired[position_slice] - position
            desired_velocity = _clip_norm(
                self.position_gain * error,
                self.envelope.max_eef_speed_m_s,
            )
            desired_acceleration = _clip_norm(
                self.velocity_gain * (desired_velocity - self.velocity[side_index]),
                self.envelope.max_eef_acceleration_m_s2,
            )
            requested_jerk = (
                desired_acceleration - self.acceleration[side_index]
            ) / dt
            jerk = _clip_norm(requested_jerk, self.envelope.max_eef_jerk_m_s3)
            acceleration = self.acceleration[side_index] + jerk * dt
            acceleration = _clip_norm(
                acceleration, self.envelope.max_eef_acceleration_m_s2
            )
            velocity = self.velocity[side_index] + acceleration * dt
            velocity = _clip_norm(velocity, self.envelope.max_eef_speed_m_s)
            position = position + velocity * dt
            self.command[position_slice] = position
            self.velocity[side_index] = velocity
            self.acceleration[side_index] = acceleration
            quaternion, angular_step = _quat_slerp_wxyz(
                self.command[quaternion_slice],
                desired[quaternion_slice],
                self.envelope.max_angular_speed_rad_s * dt,
            )
            self.command[quaternion_slice] = quaternion
            speed_values.append(np.linalg.norm(velocity))
            acceleration_values.append(np.linalg.norm(acceleration))
            jerk_values.append(np.linalg.norm(jerk))
            angular_values.append(angular_step / dt)
        previous_gripper = self.command[GRIPPERS].copy()
        maximum_step = self.envelope.max_gripper_speed_rad_s * dt
        self.command[GRIPPERS] += np.clip(
            desired[GRIPPERS] - self.command[GRIPPERS],
            -maximum_step,
            maximum_step,
        )
        gripper_speed = float(np.max(np.abs(
            self.command[GRIPPERS] - previous_gripper
        )) / dt)
        telemetry = FilterTelemetry(
            max_translation_speed_m_s=float(max(speed_values)),
            max_translation_acceleration_m_s2=float(max(acceleration_values)),
            max_translation_jerk_m_s3=float(max(jerk_values)),
            max_angular_speed_rad_s=float(max(angular_values)),
            max_gripper_speed_rad_s=gripper_speed,
        )
        return self.command.copy(), telemetry


def forbidden_manipulator_contact(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """Conservative free-space collision rule used by preflight.

    Dex1-to-cube contact is allowed. Same-side adjacent-chain contacts are
    ignored. Table/floor, torso, and cross-arm contacts are rejected.
    """
    def name(object_type, object_id) -> str:
        return mujoco.mj_id2name(model, object_type, object_id) or "world"

    def is_manipulator(body: str) -> bool:
        return any(part in body for part in ("shoulder", "elbow", "wrist", "dex1"))

    def is_cube(body: str) -> bool:
        return body.endswith("_cube")

    for index in range(data.ncon):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1 = name(mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom1]))
        body2 = name(mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom2]))
        geom_name1 = name(mujoco.mjtObj.mjOBJ_GEOM, geom1)
        geom_name2 = name(mujoco.mjtObj.mjOBJ_GEOM, geom2)
        manip1, manip2 = is_manipulator(body1), is_manipulator(body2)
        if not (manip1 or manip2):
            continue
        if ("dex1" in body1 and is_cube(body2)) or ("dex1" in body2 and is_cube(body1)):
            continue
        same_chain = (
            (body1.startswith("left_") and body2.startswith("left_"))
            or (body1.startswith("right_") and body2.startswith("right_"))
        )
        if same_chain:
            continue
        if "work_table" in (geom_name1, geom_name2) or not same_chain:
            return True
    return False


class G1TargetPreflight:
    """Conservative kinematic feasibility and final-pose collision check."""

    def __init__(
        self,
        model: mujoco.MjModel,
        position_tolerance_m: float = 0.005,
        orientation_tolerance_rad: float = np.deg2rad(3.0),
        iterations: int = 250,
    ):
        self.model = model
        self.position_tolerance_m = position_tolerance_m
        self.orientation_tolerance_rad = orientation_tolerance_rad
        self.iterations = iterations

    def check(self, source: mujoco.MjData, target_world_wxyz: np.ndarray) -> TargetSafetyResult:
        target = np.asarray(target_world_wxyz, dtype=np.float64)
        if target.shape != (16,) or not np.all(np.isfinite(target)):
            return TargetSafetyResult(False, False, False, False, np.inf, np.inf, "invalid_target")
        data = mujoco.MjData(self.model)
        data.qpos[:] = source.qpos
        data.qvel[:] = source.qvel
        data.ctrl[:] = source.ctrl
        mujoco.mj_forward(self.model, data)
        solver = G1DualArmIK(
            self.model, data, damping=0.06, max_joint_speed=2.0
        )
        solver.reset()
        arm_qpos = np.concatenate((solver.left["qpos"], solver.right["qpos"]))
        for _ in range(self.iterations):
            solver.step(target[:7], target[7:14], 0.02)
            data.qpos[arm_qpos] = solver.q_target[arm_qpos]
            data.qvel[:] = 0.0
            mujoco.mj_fwdPosition(self.model, data)
        left_pose = solver.pose("left")
        right_pose = solver.pose("right")
        position_error = float(max(
            np.linalg.norm(left_pose[0] - target[:3]),
            np.linalg.norm(right_pose[0] - target[7:10]),
        ))
        rotation_error = float(max(
            np.linalg.norm(orientation_error(target[3:7], left_pose[1])),
            np.linalg.norm(orientation_error(target[10:14], right_pose[1])),
        ))
        reachable = bool(
            position_error <= self.position_tolerance_m
            and rotation_error <= self.orientation_tolerance_rad
        )
        joint_limits_ok = True
        saturated = False
        for info in (solver.left, solver.right):
            for joint_id, qpos_index in zip(info["joint"], info["qpos"], strict=True):
                if self.model.jnt_limited[joint_id]:
                    low, high = self.model.jnt_range[joint_id]
                    value = data.qpos[qpos_index]
                    joint_limits_ok &= bool(low - 1e-9 <= value <= high + 1e-9)
                    saturated |= bool(min(abs(value - low), abs(value - high)) < 1e-4)
        collision_free = not forbidden_manipulator_contact(self.model, data)
        accepted = bool(reachable and collision_free and joint_limits_ok and not saturated)
        if not reachable:
            reason = "unreachable"
        elif not joint_limits_ok or saturated:
            reason = "joint_limit"
        elif not collision_free:
            reason = "collision"
        else:
            reason = "accepted"
        return TargetSafetyResult(
            accepted, reachable, collision_free, joint_limits_ok,
            position_error, rotation_error, reason,
        )
