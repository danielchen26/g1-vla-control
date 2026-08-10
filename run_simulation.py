#!/usr/bin/env python3
"""Run and measure the local G1 VLA-compatible adaptive-speed simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from action_schema import (
    LEFT_POS, LEFT_QUAT, RIGHT_POS, RIGHT_QUAT,
    mujoco_wxyz_to_vla_xyzw, vla_xyzw_to_mujoco_wxyz,
)
from adaptive_retimer import AdaptiveChunkExecutor, AdaptiveRetimer
from g1_dual_arm_ik import G1DualArmIK
from vla_stub import make_reach_chunk
from stack_scene import build_model, reset_to_stand
from dex1_gripper import Dex1Controller


def stability_score(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    rotation = data.xmat[pelvis].reshape(3, 3)
    body_up = rotation[:, 2]
    tilt = np.arccos(np.clip(body_up[2], -1.0, 1.0))
    angular_speed = np.linalg.norm(data.qvel[3:6])
    return float(np.clip(1.0 - tilt / 0.35 - angular_speed / 6.0, 0.2, 1.0))


def build_problem(model: mujoco.MjModel, data: mujoco.MjData, ik: G1DualArmIK):
    left_position, left_quaternion = ik.pose("left")
    right_position, right_quaternion = ik.pose("right")
    left_target = left_position + np.array([0.30, 0.055, 0.10])
    right_target = right_position + np.array([0.20, -0.035, 0.055])
    raw = make_reach_chunk(
        left_position,
        mujoco_wxyz_to_vla_xyzw(left_quaternion),
        right_position,
        mujoco_wxyz_to_vla_xyzw(right_quaternion),
        left_target,
        right_target,
    )
    retimer = AdaptiveRetimer()
    adaptive = retimer.retime(
        raw, left_target, right_target, stability=stability_score(model, data)
    )
    return raw, adaptive, left_target, right_target


def run(*, adaptive: bool, gui: bool, output: Path | None = None) -> dict:
    model = build_model()
    data = mujoco.MjData(model)
    stand_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    reset_to_stand(model, data)
    stand_control = model.key_ctrl[stand_key].copy()
    data.ctrl[:] = stand_control

    ik = G1DualArmIK(model, data)
    ik.reset()
    dex1 = Dex1Controller(model)
    raw_chunk, retimed, left_target, right_target = build_problem(model, data, ik)
    mode = "adaptive" if adaptive else "baseline"
    executor = AdaptiveChunkExecutor() if adaptive else None
    if executor:
        executor.reset(raw_chunk)
    print(f"Mode: {mode}")
    print(f"Nominal duration: {raw_chunk.timestamps[-1]:.3f} s")
    if adaptive:
        print("Online governor enabled: measured distance + live stability feedback")

    viewer_context = None
    if gui:
        import importlib
        mj_viewer = importlib.import_module("mujoco.viewer")
        viewer_context = mj_viewer.launch_passive(model, data)
        viewer_context.cam.lookat[:] = (0.25, 0.0, 0.8)
        viewer_context.cam.distance = 2.6
        viewer_context.cam.azimuth = 145
        viewer_context.cam.elevation = -15

    start_time = data.time
    max_joint_speed = 0.0
    max_joint_acceleration = 0.0
    previous_arm_velocity = np.zeros(14)
    arm_dofs = np.concatenate((ik.left["dof"], ik.right["dof"]))
    final_errors = {}
    settle_time = 1.0
    trajectory_done_at: float | None = None
    scale_history: list[float] = []

    try:
        while True:
            wall_start = time.time()
            elapsed = data.time - start_time
            if adaptive:
                measured_left, _ = ik.pose("left")
                measured_right, _ = ik.pose("right")
                action, scale, done = executor.step(
                    raw_chunk,
                    model.opt.timestep,
                    measured_left,
                    measured_right,
                    left_target,
                    right_target,
                    stability=stability_score(model, data),
                )
                scale_history.append(scale)
            else:
                action = raw_chunk.sample(elapsed)
                done = elapsed >= raw_chunk.timestamps[-1]
                scale_history.append(1.0)
            if done and trajectory_done_at is None:
                trajectory_done_at = elapsed
            if (
                trajectory_done_at is not None
                and elapsed >= trajectory_done_at + settle_time
                and viewer_context is None
            ):
                break
            data.ctrl[:] = stand_control
            dex1.set_motor_commands(data, action[14], action[15])
            final_errors = ik.step(
                np.concatenate((
                    action[LEFT_POS], vla_xyzw_to_mujoco_wxyz(action[LEFT_QUAT])
                )),
                np.concatenate((
                    action[RIGHT_POS], vla_xyzw_to_mujoco_wxyz(action[RIGHT_QUAT])
                )),
                model.opt.timestep,
            )
            mujoco.mj_step(model, data)

            arm_velocity = data.qvel[arm_dofs].copy()
            arm_acceleration = (arm_velocity - previous_arm_velocity) / model.opt.timestep
            max_joint_speed = max(max_joint_speed, float(np.max(np.abs(arm_velocity))))
            max_joint_acceleration = max(
                max_joint_acceleration, float(np.max(np.abs(arm_acceleration)))
            )
            previous_arm_velocity = arm_velocity

            if viewer_context is not None:
                if not viewer_context.is_running():
                    break
                viewer_context.sync()
                remaining = model.opt.timestep - (time.time() - wall_start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        if viewer_context is not None:
            viewer_context.close()

    left_position, _ = ik.pose("left")
    right_position, _ = ik.pose("right")
    result = {
        "mode": mode,
        "success": bool(
            np.linalg.norm(left_position - left_target) < 0.035
            and np.linalg.norm(right_position - right_target) < 0.035
            and data.qpos[2] > 0.70
        ),
        "duration": float(trajectory_done_at or elapsed),
        "scale_start": float(scale_history[0]),
        "scale_max": float(max(scale_history)),
        "scale_end": float(scale_history[-1]),
        "left_final_error_m": float(np.linalg.norm(left_position - left_target)),
        "right_final_error_m": float(np.linalg.norm(right_position - right_target)),
        "pelvis_height_m": float(data.qpos[2]),
        "stability_score": stability_score(model, data),
        "max_joint_speed_rad_s": max_joint_speed,
        "max_joint_acceleration_rad_s2": max_joint_acceleration,
        **final_errors,
    }
    print(json.dumps(result, indent=2))
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="disable adaptive retiming")
    parser.add_argument("--gui", action="store_true", help="open the MuJoCo viewer")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(adaptive=not args.baseline, gui=args.gui, output=args.output)
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
