#!/usr/bin/env python3
"""Open the G1 cube scene in MuJoCo and hold the robot in stand pose."""

import time
import mujoco
import mujoco.viewer

from stack_scene import build_model, reset_to_stand, CAMERA_NAMES, TASK_PROMPT


def main() -> None:
    model = build_model()
    data = mujoco.MjData(model)
    reset_to_stand(model, data)
    stand_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    stand_control = model.key_ctrl[stand_key].copy()
    print("Task:", TASK_PROMPT)
    print("VLA cameras:", ", ".join(CAMERA_NAMES))
    print("This scene provides observations only; no VLA policy is connected yet.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = (0.42, 0.0, 0.66)
        viewer.cam.distance = 2.3
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -22
        while viewer.is_running():
            started = time.time()
            data.ctrl[:] = stand_control
            mujoco.mj_step(model, data)
            viewer.sync()
            remaining = model.opt.timestep - (time.time() - started)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
