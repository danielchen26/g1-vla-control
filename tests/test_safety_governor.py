import sys
from pathlib import Path
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1_dual_arm_ik import G1DualArmIK
from safety_governor import (
    G1TargetPreflight, JerkLimitedActionFilter, JerkLimitedJointFilter,
    JointMotionEnvelope, MotionEnvelope, manipulator_contact_violations,
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

    def test_joint_filter_enforces_per_joint_speed_acceleration_and_jerk(self):
        envelope = JointMotionEnvelope()
        max_speed, max_acceleration, max_jerk = envelope.arrays()
        action_filter = JerkLimitedJointFilter(envelope)
        initial = np.zeros(14)
        desired = np.linspace(-1.0, 1.0, 14)
        action_filter.reset(initial)
        previous = initial.copy()
        previous_velocity = np.zeros(14)
        previous_acceleration = np.zeros(14)
        dt = 0.002
        observed_speed = np.zeros(14)
        observed_acceleration = np.zeros(14)
        observed_jerk = np.zeros(14)
        for _ in range(3000):
            output, telemetry = action_filter.step(desired, dt)
            velocity = (output - previous) / dt
            acceleration = (velocity - previous_velocity) / dt
            jerk = (acceleration - previous_acceleration) / dt
            observed_speed = np.maximum(observed_speed, np.abs(velocity))
            observed_acceleration = np.maximum(observed_acceleration, np.abs(acceleration))
            observed_jerk = np.maximum(observed_jerk, np.abs(jerk))
            previous = output
            previous_velocity = velocity
            previous_acceleration = acceleration
        np.testing.assert_array_less(observed_speed, max_speed + 1e-7)
        np.testing.assert_array_less(observed_acceleration, max_acceleration + 1e-6)
        np.testing.assert_array_less(observed_jerk, max_jerk + 1e-3)
        self.assertLessEqual(telemetry.max_jerk_ratio, 1.0 + 1e-12)

    def test_dex_cube_contact_depends_on_task_phase(self):
        model = build_model()
        data = mujoco.MjData(model)
        reset_to_stand(model, data)
        eef = data.site_xpos[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "left_eef"
        )].copy()
        cube_joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_free"
        )
        cube_qpos = model.jnt_qposadr[cube_joint]
        data.qpos[cube_qpos : cube_qpos + 3] = eef + [0.025, 0.0, 0.0]
        data.qpos[cube_qpos + 3 : cube_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)
        free_space = manipulator_contact_violations(model, data, "free_space")
        grasp = manipulator_contact_violations(model, data, "grasp")
        self.assertIn("dex_cube_outside_manipulation_phase", free_space)
        self.assertNotIn("dex_cube_outside_manipulation_phase", grasp)

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
