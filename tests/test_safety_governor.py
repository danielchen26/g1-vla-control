import sys
from pathlib import Path
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1_dual_arm_ik import G1DualArmIK
from safety_governor import (
    G1TargetPreflight, JerkLimitedActionFilter, MotionEnvelope,
)
from stack_scene import build_model, reset_to_stand


class SafetyGovernorTests(unittest.TestCase):
    def test_filter_enforces_translation_speed_acceleration_and_jerk(self):
        envelope = MotionEnvelope()
        initial = np.zeros(16)
        initial[3] = 1.0
        initial[10] = 1.0
        initial[14:] = 5.5
        desired = initial.copy()
        desired[0:3] = [0.5, 0.2, 0.3]
        desired[7:10] = [0.5, -0.2, 0.3]
        desired[14:] = 0.0
        action_filter = JerkLimitedActionFilter(envelope)
        action_filter.reset(initial)
        dt = 0.002
        previous_position = np.vstack((initial[0:3], initial[7:10]))
        previous_velocity = np.zeros((2, 3))
        previous_acceleration = np.zeros((2, 3))
        maxima = np.zeros(3)
        for _ in range(2000):
            output, _ = action_filter.step(desired, dt)
            position = np.vstack((output[0:3], output[7:10]))
            velocity = (position - previous_position) / dt
            acceleration = (velocity - previous_velocity) / dt
            jerk = (acceleration - previous_acceleration) / dt
            maxima = np.maximum(maxima, [
                np.linalg.norm(velocity, axis=1).max(),
                np.linalg.norm(acceleration, axis=1).max(),
                np.linalg.norm(jerk, axis=1).max(),
            ])
            previous_position = position
            previous_velocity = velocity
            previous_acceleration = acceleration
        self.assertLessEqual(maxima[0], envelope.max_eef_speed_m_s + 1e-8)
        self.assertLessEqual(maxima[1], envelope.max_eef_acceleration_m_s2 + 1e-8)
        self.assertLessEqual(maxima[2], envelope.max_eef_jerk_m_s3 + 1e-5)

    def test_current_reachable_pose_passes_conservative_preflight(self):
        model = build_model()
        data = mujoco.MjData(model)
        reset_to_stand(model, data)
        ik = G1DualArmIK(model, data)
        left = ik.pose("left")
        right = ik.pose("right")
        target = np.r_[left[0], left[1], right[0], right[1], 5.5, 5.5]
        result = G1TargetPreflight(model).check(data, target)
        self.assertTrue(result.reachable)
        self.assertTrue(result.joint_limits_ok)
        self.assertTrue(result.collision_free)
        self.assertTrue(result.accepted)


if __name__ == "__main__":
    unittest.main()
