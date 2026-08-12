"""Deterministic trajectory fixture in the frozen G1 EDU policy contract.

This is not a VLA and is never presented as task-performance evidence.  It
exists only to regression-test the exact 16-D pelvis-frame path that a trained
G1 policy will later provide to the shared safety/IK/controller pipeline.
"""

from __future__ import annotations

import numpy as np

from action_schema import EEFActionChunk, LEFT_POS, LEFT_QUAT, RIGHT_POS, RIGHT_QUAT
from g1_policy_contract import (
    ACTION_DIM, ACTION_HORIZON, POLICY_RATE_HZ, validate_action_chunk,
)


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
    gripper_state_rad: np.ndarray,
    frequency: float = POLICY_RATE_HZ,
) -> EEFActionChunk:
    """Build one exact-horizon pelvis-frame fixture with no gripper transition."""
    count = ACTION_HORIZON
    timestamps = np.arange(count, dtype=np.float64) / frequency
    duration = float(timestamps[-1])
    blend = minimum_jerk(timestamps / duration)
    actions = np.zeros((count, ACTION_DIM), dtype=np.float64)
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
    actions[:, 14:16] = np.asarray(gripper_state_rad, dtype=np.float64)
    validate_action_chunk(
        actions, timestamps,
        expected_horizon=ACTION_HORIZON,
        expected_rate_hz=POLICY_RATE_HZ,
    )
    return EEFActionChunk(timestamps, actions)
