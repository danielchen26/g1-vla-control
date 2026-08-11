"""Deterministic VLA stand-in that emits the real 16-D EEF interface.

Only this module needs to be replaced when a remote OpenPI server is available.
"""

from __future__ import annotations

import numpy as np

from action_schema import EEFActionChunk, LEFT_POS, LEFT_QUAT, RIGHT_POS, RIGHT_QUAT


def minimum_jerk(progress: np.ndarray) -> np.ndarray:
    return 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5


def make_reach_chunk(
    left_start_position: np.ndarray,
    left_start_quaternion: np.ndarray,
    right_start_position: np.ndarray,
    right_start_quaternion: np.ndarray,
    left_target: np.ndarray,
    right_target: np.ndarray,
    *,
    duration: float = 4.0,
    frequency: float = 30.0,
) -> EEFActionChunk:
    count = int(duration * frequency) + 1
    timestamps = np.linspace(0.0, duration, count)
    blend = minimum_jerk(timestamps / duration)
    actions = np.zeros((count, 16), dtype=np.float64)
    actions[:, LEFT_POS] = (
        left_start_position[None, :] * (1.0 - blend[:, None])
        + left_target[None, :] * blend[:, None]
    )
    actions[:, RIGHT_POS] = (
        right_start_position[None, :] * (1.0 - blend[:, None])
        + right_target[None, :] * blend[:, None]
    )
    actions[:, LEFT_QUAT] = left_start_quaternion
    actions[:, RIGHT_QUAT] = right_start_quaternion
    # Synchronized dataset evidence: ~5.4 rad corresponds to visibly open.
    actions[:, 14] = 5.5
    actions[:, 15] = 5.5
    return EEFActionChunk(timestamps, actions)
