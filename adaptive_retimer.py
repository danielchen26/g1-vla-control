"""Stateful adaptive time reparameterization for VLA action chunks."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from action_schema import EEFActionChunk, LEFT_POS, RIGHT_POS


@dataclass
class RetimerConfig:
    min_scale: float = 0.50
    max_scale: float = 1.65
    near_distance: float = 0.025
    far_distance: float = 0.16
    max_scale_rate: float = 2.0       # scale units per second
    max_eef_speed: float = 0.65       # metres per second
    safety_floor: float = 0.20


@dataclass
class RetimingResult:
    chunk: EEFActionChunk
    scale_profile: np.ndarray
    original_duration: float
    retimed_duration: float


class AdaptiveChunkExecutor:
    """Online path-progress governor for closed-loop execution.

    The VLA path remains unchanged. At every control tick, measured EEF error and
    robot stability determine how quickly nominal path time advances.
    """

    def __init__(self, config: RetimerConfig | None = None):
        self.config = config or RetimerConfig()
        self.scale = 1.0
        self.path_time: float | None = None

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        return value * value * (3.0 - 2.0 * value)

    def reset(self, chunk: EEFActionChunk, scale: float = 1.0) -> None:
        self.path_time = float(chunk.timestamps[0])
        self.scale = float(scale)

    def step(
        self,
        chunk: EEFActionChunk,
        dt: float,
        current_left: np.ndarray,
        current_right: np.ndarray,
        target_left: np.ndarray,
        target_right: np.ndarray,
        *,
        stability: float = 1.0,
    ) -> tuple[np.ndarray, float, bool]:
        if self.path_time is None:
            self.reset(chunk)
        cfg = self.config
        distance = max(
            float(np.linalg.norm(np.asarray(current_left) - target_left)),
            float(np.linalg.norm(np.asarray(current_right) - target_right)),
        )
        ratio = (distance - cfg.near_distance) / (cfg.far_distance - cfg.near_distance)
        desired = cfg.min_scale + (cfg.max_scale - cfg.min_scale) * self._smoothstep(ratio)
        stability = float(np.clip(stability, cfg.safety_floor, 1.0))
        desired = cfg.min_scale + (desired - cfg.min_scale) * stability
        max_change = cfg.max_scale_rate * dt
        self.scale += float(np.clip(desired - self.scale, -max_change, max_change))
        self.scale = float(np.clip(self.scale, cfg.min_scale, cfg.max_scale))
        self.path_time = min(
            float(chunk.timestamps[-1]), self.path_time + self.scale * dt
        )
        done = self.path_time >= chunk.timestamps[-1]
        return chunk.sample(self.path_time), self.scale, done


class AdaptiveRetimer:
    """Changes path timing without changing its geometry.

    A scale greater than one executes path progress faster. Scale transitions are
    rate-limited and stateful, so adjacent VLA chunks cannot create a speed jump.
    """

    def __init__(self, config: RetimerConfig | None = None):
        self.config = config or RetimerConfig()
        self.previous_scale = 1.0

    def reset(self, scale: float = 1.0) -> None:
        self.previous_scale = float(scale)

    @staticmethod
    def _smoothstep(value: np.ndarray) -> np.ndarray:
        value = np.clip(value, 0.0, 1.0)
        return value * value * (3.0 - 2.0 * value)

    def retime(
        self,
        chunk: EEFActionChunk,
        target_left: np.ndarray,
        target_right: np.ndarray,
        *,
        stability: float = 1.0,
    ) -> RetimingResult:
        cfg = self.config
        stability = float(np.clip(stability, cfg.safety_floor, 1.0))
        target_left = np.asarray(target_left, dtype=np.float64)
        target_right = np.asarray(target_right, dtype=np.float64)

        left_error = np.linalg.norm(chunk.actions[:, LEFT_POS] - target_left, axis=1)
        right_error = np.linalg.norm(chunk.actions[:, RIGHT_POS] - target_right, axis=1)
        distance = np.maximum(left_error, right_error)
        progress = (distance - cfg.near_distance) / (cfg.far_distance - cfg.near_distance)
        desired = cfg.min_scale + (cfg.max_scale - cfg.min_scale) * self._smoothstep(progress)
        desired = cfg.min_scale + (desired - cfg.min_scale) * stability

        nominal_dt = np.diff(chunk.timestamps)
        left_step = np.linalg.norm(np.diff(chunk.actions[:, LEFT_POS], axis=0), axis=1)
        right_step = np.linalg.norm(np.diff(chunk.actions[:, RIGHT_POS], axis=0), axis=1)
        nominal_speed = np.maximum(left_step, right_step) / nominal_dt
        kinematic_cap = np.ones(len(chunk.timestamps)) * cfg.max_scale
        kinematic_cap[1:] = cfg.max_eef_speed / np.maximum(nominal_speed, 1e-8)
        desired = np.minimum(desired, kinematic_cap)
        desired = np.clip(desired, cfg.min_scale, cfg.max_scale)

        # Rate-limit scale in real time and carry state across chunk boundaries.
        scale = np.empty_like(desired)
        scale[0] = self.previous_scale
        for index in range(1, len(scale)):
            max_change = cfg.max_scale_rate * nominal_dt[index - 1]
            scale[index] = scale[index - 1] + np.clip(
                desired[index] - scale[index - 1], -max_change, max_change
            )
        self.previous_scale = float(scale[-1])

        # Keep action samples and path geometry unchanged; only replace timestamps.
        segment_scale = np.maximum(0.5 * (scale[:-1] + scale[1:]), 1e-6)
        retimed_dt = nominal_dt / segment_scale
        retimed_timestamps = np.concatenate(([0.0], np.cumsum(retimed_dt)))
        retimed = EEFActionChunk(retimed_timestamps, chunk.actions.copy())
        return RetimingResult(
            chunk=retimed,
            scale_profile=scale,
            original_duration=float(chunk.timestamps[-1] - chunk.timestamps[0]),
            retimed_duration=float(retimed_timestamps[-1]),
        )
