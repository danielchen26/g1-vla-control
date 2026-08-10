"""VLA-compatible dual end-effector action representation.

The checkpoint LGG100/stack-cube-eef-24k uses 16 values:
left xyz + quaternion, right xyz + quaternion, and two gripper values.
Checkpoint normalization statistics indicate an xyzw quaternion convention.
Conversion to MuJoCo's wxyz convention happens only at the simulator boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

ACTION_DIM = 16
LEFT_POS = slice(0, 3)
LEFT_QUAT = slice(3, 7)
RIGHT_POS = slice(7, 10)
RIGHT_QUAT = slice(10, 14)
GRIPPERS = slice(14, 16)


def mujoco_wxyz_to_vla_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q[[1, 2, 3, 0]]


def vla_xyzw_to_mujoco_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q[[3, 0, 1, 2]]


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-9:
        raise ValueError("Quaternion norm is zero")
    return q / norm


def slerp(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    """Shortest-path spherical interpolation between xyzw quaternions."""
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        return normalize_quaternion(q0 + fraction * (q1 - q0))
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    return (
        np.sin((1.0 - fraction) * theta) / sin_theta * q0
        + np.sin(fraction * theta) / sin_theta * q1
    )


@dataclass(frozen=True)
class EEFActionChunk:
    timestamps: np.ndarray
    actions: np.ndarray

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        actions = np.asarray(self.actions, dtype=np.float64)
        if timestamps.ndim != 1 or len(timestamps) < 2:
            raise ValueError("timestamps must contain at least two samples")
        if actions.shape != (len(timestamps), ACTION_DIM):
            raise ValueError(f"actions must have shape (T, {ACTION_DIM})")
        if not np.all(np.diff(timestamps) > 0):
            raise ValueError("timestamps must be strictly increasing")
        actions = actions.copy()
        for index in range(len(actions)):
            actions[index, LEFT_QUAT] = normalize_quaternion(actions[index, LEFT_QUAT])
            actions[index, RIGHT_QUAT] = normalize_quaternion(actions[index, RIGHT_QUAT])
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "actions", actions)

    def sample(self, timestamp: float) -> np.ndarray:
        """Interpolate an action, using SLERP for both orientations."""
        t = float(np.clip(timestamp, self.timestamps[0], self.timestamps[-1]))
        upper = int(np.searchsorted(self.timestamps, t, side="right"))
        if upper == 0:
            return self.actions[0].copy()
        if upper >= len(self.timestamps):
            return self.actions[-1].copy()
        lower = upper - 1
        fraction = (t - self.timestamps[lower]) / (
            self.timestamps[upper] - self.timestamps[lower]
        )
        result = (1.0 - fraction) * self.actions[lower] + fraction * self.actions[upper]
        result[LEFT_QUAT] = slerp(
            self.actions[lower, LEFT_QUAT], self.actions[upper, LEFT_QUAT], fraction
        )
        result[RIGHT_QUAT] = slerp(
            self.actions[lower, RIGHT_QUAT], self.actions[upper, RIGHT_QUAT], fraction
        )
        return result
