#!/usr/bin/env python3
"""Open the G1 cube scene in MuJoCo and hold the robot in stand pose."""

import time
import mujoco
import mujoco.viewer

from stack_scene import (
    build_model, reset_to_reference_pose, CAMERA_NAMES, TASK_PROMPT,
)


def main() -> None:
    model = build_model()
    data = mujoco.MjData(model)
    reset_to_reference_pose(model, data)
    hold_control = data.ctrl.copy()
    print("Task:", TASK_PROMPT)
    print("VLA cameras:", ", ".join(CAMERA_NAMES))
    print("This scene provides observations only; no VLA policy is connected yet.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = (0.35, 0.0, 0.82)
        viewer.cam.distance = 2.3
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -22
        while viewer.is_running():
            started = time.time()
            data.ctrl[:] = hold_control
            mujoco.mj_step(model, data)
            viewer.sync()
            remaining = model.opt.timestep - (time.time() - started)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
