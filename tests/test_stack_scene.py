import sys
from pathlib import Path
import unittest

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stack_scene import build_model, reset_to_stand, CAMERA_NAMES
from dex1_gripper import (
    Dex1Controller, motor_radians_to_jaw_position, JAW_MIN_M, JAW_MAX_M,
)


class StackSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_model()
        cls.data = mujoco.MjData(cls.model)
        reset_to_stand(cls.model, cls.data)

    def test_cameras_and_cubes_exist(self):
        for name in CAMERA_NAMES:
            self.assertGreaterEqual(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name), 0
            )
        for side in ("left", "right"):
            camera_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, f"cam_{side}_wrist"
            )
            parent_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.model.cam_bodyid[camera_id],
            )
            self.assertEqual(parent_name, f"{side}_dex1_base_link")
        for name in ("red_cube", "blue_cube", "yellow_cube"):
            self.assertGreaterEqual(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name), 0
            )

    def test_official_dex1_replaces_articulated_hands(self):
        self.assertEqual(
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_thumb_0_link"
            ),
            -1,
        )
        for side in ("left", "right"):
            self.assertGreaterEqual(
                mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    f"{side}_dex1_base_link",
                ),
                0,
            )
        controller = Dex1Controller(self.model)
        self.assertEqual(len(controller.actuators["left"]), 2)
        self.assertEqual(len(controller.actuators["right"]), 2)

    def test_dex1_motor_mapping_matches_limits(self):
        self.assertAlmostEqual(motor_radians_to_jaw_position(0.0), JAW_MIN_M)
        self.assertAlmostEqual(motor_radians_to_jaw_position(5.5), JAW_MAX_M)
        self.assertAlmostEqual(motor_radians_to_jaw_position(-10.0), JAW_MIN_M)
        self.assertAlmostEqual(motor_radians_to_jaw_position(10.0), JAW_MAX_M)

    def test_robot_and_cubes_remain_stable(self):
        reset_to_stand(self.model, self.data)
        for _ in range(250):
            mujoco.mj_step(self.model, self.data)
        self.assertGreater(self.data.qpos[2], 0.75)
        for name in ("red_cube", "blue_cube", "yellow_cube"):
            body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            self.assertGreater(self.data.xpos[body, 2], 0.61)


if __name__ == "__main__":
    unittest.main()
