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
class JointMotionEnvelope:
    """Per-joint P99 limits in LEFT_JOINTS + RIGHT_JOINTS order."""

    max_speeds_rad_s: tuple[float, ...] = (
        0.533597, 0.252561, 0.474892, 1.145755, 0.328392, 0.697807, 0.365966,
        0.553127, 0.272480, 0.435653, 1.185016, 0.304985, 0.728678, 0.431426,
    )
    max_accelerations_rad_s2: tuple[float, ...] = (
        4.827004, 3.162481, 4.641246, 8.887587, 4.219851, 5.072449, 3.990434,
        5.032260, 3.042590, 4.189361, 8.400302, 3.793993, 5.347711, 4.485692,
    )
    max_jerks_rad_s3: tuple[float, ...] = (
        219.410593, 136.371927, 200.813817, 411.824612, 180.969172, 225.600934,
        168.006894, 228.110088, 134.318513, 186.512517, 390.869447, 169.460381,
        231.259257, 201.427131,
    )
    source: str = "public per-joint training-target P99; simulation-only"

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = tuple(np.asarray(item, dtype=np.float64) for item in (
            self.max_speeds_rad_s,
            self.max_accelerations_rad_s2,
            self.max_jerks_rad_s3,
        ))
        if any(value.shape != (14,) or np.any(value <= 0) for value in values):
            raise ValueError("joint limits must contain 14 positive values")
        return values


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
    collision_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class JointFilterTelemetry:
    max_speed_rad_s: float
    max_acceleration_rad_s2: float
    max_jerk_rad_s3: float
    max_speed_ratio: float
    max_acceleration_ratio: float
    max_jerk_ratio: float


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


class JerkLimitedJointFilter:
    """Track 14-D IK targets with per-joint simulation-only v/a/jerk caps."""

    def __init__(
        self,
        envelope: JointMotionEnvelope | None = None,
        position_gain: float = 10.0,
        velocity_gain: float = 30.0,
    ):
        self.envelope = envelope or JointMotionEnvelope()
        self.max_speed, self.max_acceleration, self.max_jerk = self.envelope.arrays()
        self.position_gain = position_gain
        self.velocity_gain = velocity_gain
        self.command: np.ndarray | None = None
        self.velocity = np.zeros(14)
        self.acceleration = np.zeros(14)

    def reset(
        self,
        command: np.ndarray,
        velocity: np.ndarray | None = None,
        acceleration: np.ndarray | None = None,
    ) -> None:
        command = np.asarray(command, dtype=np.float64)
        if command.shape != (14,) or not np.all(np.isfinite(command)):
            raise ValueError("joint command must contain 14 finite values")
        self.command = command.copy()
        self.velocity[:] = 0.0 if velocity is None else np.asarray(velocity)
        self.acceleration[:] = 0.0 if acceleration is None else np.asarray(acceleration)
        if self.velocity.shape != (14,) or self.acceleration.shape != (14,):
            raise ValueError("joint derivatives must have shape (14,)")
        self.velocity[:] = np.clip(self.velocity, -self.max_speed, self.max_speed)
        self.acceleration[:] = np.clip(
            self.acceleration, -self.max_acceleration, self.max_acceleration
        )

    def step(
        self,
        desired: np.ndarray,
        dt: float,
        lower_limits: np.ndarray | None = None,
        upper_limits: np.ndarray | None = None,
    ) -> tuple[np.ndarray, JointFilterTelemetry]:
        if dt <= 0:
            raise ValueError("dt must be positive")
        desired = np.asarray(desired, dtype=np.float64)
        if desired.shape != (14,) or not np.all(np.isfinite(desired)):
            raise ValueError("desired joint target must contain 14 finite values")
        if self.command is None:
            self.reset(desired)
        assert self.command is not None
        desired_velocity = np.clip(
            self.position_gain * (desired - self.command),
            -self.max_speed,
            self.max_speed,
        )
        desired_acceleration = np.clip(
            self.velocity_gain * (desired_velocity - self.velocity),
            -self.max_acceleration,
            self.max_acceleration,
        )
        jerk = np.clip(
            (desired_acceleration - self.acceleration) / dt,
            -self.max_jerk,
            self.max_jerk,
        )
        self.acceleration = np.clip(
            self.acceleration + jerk * dt,
            -self.max_acceleration,
            self.max_acceleration,
        )
        self.velocity = np.clip(
            self.velocity + self.acceleration * dt,
            -self.max_speed,
            self.max_speed,
        )
        self.command = self.command + self.velocity * dt
        if lower_limits is not None or upper_limits is not None:
            if lower_limits is None or upper_limits is None:
                raise ValueError("both lower_limits and upper_limits are required")
            lower = np.asarray(lower_limits, dtype=np.float64)
            upper = np.asarray(upper_limits, dtype=np.float64)
            if lower.shape != (14,) or upper.shape != (14,):
                raise ValueError("joint limits must have shape (14,)")
            clipped = np.clip(self.command, lower, upper)
            hit_limit = clipped != self.command
            self.command = clipped
            self.velocity[hit_limit] = 0.0
            self.acceleration[hit_limit] = 0.0
        telemetry = JointFilterTelemetry(
            max_speed_rad_s=float(np.abs(self.velocity).max()),
            max_acceleration_rad_s2=float(np.abs(self.acceleration).max()),
            max_jerk_rad_s3=float(np.abs(jerk).max()),
            max_speed_ratio=float(np.max(np.abs(self.velocity) / self.max_speed)),
            max_acceleration_ratio=float(np.max(
                np.abs(self.acceleration) / self.max_acceleration
            )),
            max_jerk_ratio=float(np.max(np.abs(jerk) / self.max_jerk)),
        )
        return self.command.copy(), telemetry


def _body_tree_distance(model: mujoco.MjModel, body1: int, body2: int) -> int:
    ancestors: dict[int, int] = {}
    current, distance = body1, 0
    while current >= 0:
        ancestors[current] = distance
        parent = int(model.body_parentid[current])
        if parent == current:
            break
        current, distance = parent, distance + 1
    current, distance = body2, 0
    while current >= 0:
        if current in ancestors:
            return distance + ancestors[current]
        parent = int(model.body_parentid[current])
        if parent == current:
            break
        current, distance = parent, distance + 1
    return 10_000


def manipulator_contact_violations(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    phase: str = "free_space",
) -> tuple[str, ...]:
    """Return phase-aware collision categories for the current configuration.

    ``grasp`` and ``place`` allow Dex1-to-cube contact. All phases reject table,
    floor, torso/elbow, cross-arm and non-adjacent self contact. The known
    torso-to-shoulder-yaw overlap in the public G1 geometry is treated as a
    near-kinematic model contact, but is explicitly reported by this policy's
    documentation rather than being inferred as hardware-safe.
    """
    if phase not in {"free_space", "grasp", "place"}:
        raise ValueError("phase must be free_space, grasp, or place")

    def name(object_type, object_id) -> str:
        return mujoco.mj_id2name(model, object_type, object_id) or "world"

    def is_manipulator(body: str) -> bool:
        return any(part in body for part in ("shoulder", "elbow", "wrist", "dex1"))

    violations: set[str] = set()
    expected_model_pairs = {
        frozenset(("torso_link", "left_shoulder_yaw_link")),
        frozenset(("torso_link", "right_shoulder_yaw_link")),
    }
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body_id1 = int(model.geom_bodyid[geom1])
        body_id2 = int(model.geom_bodyid[geom2])
        body1 = name(mujoco.mjtObj.mjOBJ_BODY, body_id1)
        body2 = name(mujoco.mjtObj.mjOBJ_BODY, body_id2)
        if not (is_manipulator(body1) or is_manipulator(body2)):
            continue
        pair = frozenset((body1, body2))
        if pair in expected_model_pairs:
            continue
        cube_contact = (
            ("dex1" in body1 and body2.endswith("_cube"))
            or ("dex1" in body2 and body1.endswith("_cube"))
        )
        if cube_contact:
            if phase in {"grasp", "place"}:
                continue
            violations.add("dex_cube_outside_manipulation_phase")
            continue
        same_side = (
            (body1.startswith("left_") and body2.startswith("left_"))
            or (body1.startswith("right_") and body2.startswith("right_"))
        )
        if same_side and _body_tree_distance(model, body_id1, body_id2) <= 2:
            continue
        geom_name1 = name(mujoco.mjtObj.mjOBJ_GEOM, geom1)
        geom_name2 = name(mujoco.mjtObj.mjOBJ_GEOM, geom2)
        if "work_table" in (geom_name1, geom_name2) or "floor" in (geom_name1, geom_name2):
            violations.add("workspace_contact")
        elif "torso" in body1 or "torso" in body2:
            violations.add("torso_contact")
        elif (
            (body1.startswith("left_") and body2.startswith("right_"))
            or (body1.startswith("right_") and body2.startswith("left_"))
        ):
            violations.add("cross_arm_contact")
        elif same_side:
            violations.add("nonadjacent_self_contact")
        else:
            violations.add("other_manipulator_contact")
    return tuple(sorted(violations))


def forbidden_manipulator_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    phase: str = "free_space",
) -> bool:
    """Compatibility wrapper around the phase-aware contact classifier."""
    return bool(manipulator_contact_violations(model, data, phase))


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

    def check(
        self,
        source: mujoco.MjData,
        target_world_wxyz: np.ndarray,
        phase: str = "free_space",
    ) -> TargetSafetyResult:
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
        collision_reasons = manipulator_contact_violations(self.model, data, phase)
        collision_free = not collision_reasons
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
            position_error, rotation_error, reason, collision_reasons,
        )
