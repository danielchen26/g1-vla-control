#!/usr/bin/env python3
"""Comprehensive local dynamics, Dex1, IK, and robustness validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import pyarrow.parquet as pq

from action_schema import pelvis_vla_action_to_world_mujoco
from dataset_contract_audit import ARM_JOINTS, CACHE, _fk_transform
from dex1_gripper import (
    Dex1Controller, JAW_MAX_M, JAW_MIN_M, jaw_position_to_motor_radians,
)
from g1_dual_arm_ik import G1DualArmIK, orientation_error
from stack_scene import build_model, reset_to_stand

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SIDES = ("left", "right")


def _id(model, object_type, name: str) -> int:
    result = mujoco.mj_name2id(model, object_type, name)
    if result < 0:
        raise RuntimeError(f"Missing {name}")
    return result


def _arm_addresses(model: mujoco.MjModel) -> np.ndarray:
    return np.asarray([
        model.jnt_qposadr[_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in ARM_JOINTS
    ])


def _tip_collision_geoms(model: mujoco.MjModel, side: str) -> np.ndarray:
    bodies = {
        _id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_dex1_Link1_3"),
        _id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_dex1_Link2_2"),
    }
    return np.asarray([
        geom for geom in range(model.ngeom)
        if model.geom_bodyid[geom] in bodies and model.geom_contype[geom] != 0
    ])


def dex1_sweep() -> dict:
    commands = np.linspace(0.0, 5.5, 12)
    samples = []
    for command in commands:
        model = build_model()
        data = mujoco.MjData(model)
        reset_to_stand(model, data)
        controller = Dex1Controller(model)
        for _ in range(700):
            controller.set_motor_commands(data, command, command)
            mujoco.mj_step(model, data)
        row = {"command_rad": float(command)}
        for side in SIDES:
            joints = []
            for finger in ("Joint1_1", "Joint2_1"):
                joint = _id(
                    model, mujoco.mjtObj.mjOBJ_JOINT,
                    f"{side}_dex1_{finger}",
                )
                joints.append(float(data.qpos[model.jnt_qposadr[joint]]))
            geoms = _tip_collision_geoms(model, side)
            gap = float(np.linalg.norm(data.geom_xpos[geoms[0]] - data.geom_xpos[geoms[1]]))
            row[side] = {"jaw_positions_m": joints, "tip_center_gap_m": gap}
        samples.append(row)
    left_gap = np.asarray([sample["left"]["tip_center_gap_m"] for sample in samples])
    right_gap = np.asarray([sample["right"]["tip_center_gap_m"] for sample in samples])
    all_jaws = np.asarray([
        sample[side]["jaw_positions_m"] for sample in samples for side in SIDES
    ])
    targets = np.repeat(
        (JAW_MAX_M - commands / 5.5 * (JAW_MAX_M - JAW_MIN_M))[:, None],
        4, axis=1,
    ).reshape(-1)
    return {
        "samples": samples,
        "gap_monotonically_increases": bool(
            np.all(np.diff(left_gap) > 0) and np.all(np.diff(right_gap) > 0)
        ),
        "left_right_max_gap_difference_m": float(np.max(np.abs(left_gap - right_gap))),
        "closed_gap_m": float(np.mean([left_gap[0], right_gap[0]])),
        "open_gap_m": float(np.mean([left_gap[-1], right_gap[-1]])),
        "max_actuator_tracking_error_m": float(
            np.max(np.abs(all_jaws.reshape(12, 4).ravel() - targets))
        ),
        "command_direction": "0 rad closed; 5.5 rad open",
    }


def _grasp_trial(side: str, force_n: float, seed: int) -> dict:
    model = build_model()
    data = mujoco.MjData(model)
    reset_to_stand(model, data)
    controller = Dex1Controller(model)
    tip_geoms = _tip_collision_geoms(model, side)
    base = _id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_dex1_base_link")
    cube_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "yellow_cube")
    cube_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "yellow_cube_free")
    qadr = model.jnt_qposadr[cube_joint]
    vadr = model.jnt_dofadr[cube_joint]
    rng = np.random.default_rng(seed)
    center = np.mean(data.geom_xpos[tip_geoms], axis=0)
    # Small deterministic placement jitter tests capture tolerance.
    center += rng.uniform(-0.002, 0.002, 3)
    center[2] += 0.01
    data.qpos[qadr : qadr + 3] = center
    data.qpos[qadr + 3 : qadr + 7] = data.xquat[base]
    data.qvel[vadr : vadr + 6] = 0.0
    mujoco.mj_forward(model, data)
    initial = data.qpos[qadr : qadr + 3].copy()
    max_relative = 0.0
    dex_cube_contacts = 0
    for step in range(3000):  # six seconds
        left = 0.0 if side == "left" else 5.5
        right = 0.0 if side == "right" else 5.5
        controller.set_motor_commands(data, left, right)
        data.xfrc_applied[:] = 0.0
        if 1000 <= step < 1100:
            data.xfrc_applied[cube_body, 0] = force_n
        mujoco.mj_step(model, data)
        midpoint = np.mean(data.geom_xpos[tip_geoms], axis=0)
        relative = float(np.linalg.norm(data.qpos[qadr : qadr + 3] - midpoint))
        max_relative = max(max_relative, relative)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            bodies = {
                int(model.geom_bodyid[contact.geom1]),
                int(model.geom_bodyid[contact.geom2]),
            }
            if cube_body in bodies and any(
                model.geom_bodyid[geom] in bodies for geom in tip_geoms
            ):
                dex_cube_contacts += 1
    final = data.qpos[qadr : qadr + 3].copy()
    midpoint = np.mean(data.geom_xpos[tip_geoms], axis=0)
    final_relative = float(np.linalg.norm(final - midpoint))
    held = bool(final_relative < 0.15 and final[2] > 0.2)
    return {
        "side": side,
        "disturbance_force_n": force_n,
        "held": held,
        "initial_cube_position": initial.tolist(),
        "final_cube_position": final.tolist(),
        "vertical_drop_m": float(final[2] - initial[2]),
        "final_relative_distance_m": final_relative,
        "max_relative_distance_m": max_relative,
        "dex_cube_contact_samples": dex_cube_contacts,
    }


def grasp_sweep() -> dict:
    forces = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
    trials = [
        _grasp_trial(side, force, seed=100 * side_index + force_index)
        for side_index, side in enumerate(SIDES)
        for force_index, force in enumerate(forces)
    ]
    thresholds = {}
    for side in SIDES:
        held_forces = [
            trial["disturbance_force_n"] for trial in trials
            if trial["side"] == side and trial["held"]
        ]
        thresholds[side] = max(held_forces) if held_forces else None
    return {
        "cube_full_width_m": 0.08,
        "trials": trials,
        "zero_force_bilateral_success": all(
            trial["held"] for trial in trials
            if trial["disturbance_force_n"] == 0.0
        ),
        "maximum_tested_held_force_n": thresholds,
        "note": "Synthetic pre-positioned grasp; this is not a VLA grasp episode.",
    }


def _load_dataset_samples(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    paths = sorted(CACHE.glob("episode_*.parquet"))
    if len(paths) < 100:
        raise RuntimeError("Run dataset_contract_audit.py first")
    states = np.concatenate([
        np.asarray(pq.read_table(path, columns=["observation.state"]).column(0).to_pylist())
        for path in paths
    ])
    actions = np.concatenate([
        np.asarray(pq.read_table(path, columns=["action"]).column(0).to_pylist())
        for path in paths
    ])
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(states), size=count, replace=False)
    return states[indices], actions[indices]


def episode_zero_dynamics_replay() -> dict:
    """Replay all episode-0 joint targets through standing G1 dynamics."""
    path = CACHE / "episode_000000.parquet"
    table = pq.read_table(
        path, columns=["observation.state", "action", "timestamp"]
    )
    states = np.asarray(table.column("observation.state").to_pylist())
    actions = np.asarray(table.column("action").to_pylist())
    timestamps = np.asarray(table.column("timestamp"))
    expected_eef = _fk_transform(states)
    model = build_model()
    data = mujoco.MjData(model)
    reset_to_stand(model, data)
    arm_qpos = _arm_addresses(model)
    arm_actuators = np.asarray([
        _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ARM_JOINTS
    ])
    data.qpos[arm_qpos] = states[0, :14]
    data.ctrl[arm_actuators] = actions[0, :14]
    controller = Dex1Controller(model)
    controller.set_motor_commands(data, states[0, 14], states[0, 15])
    for side, command in zip(SIDES, states[0, 14:], strict=True):
        target = JAW_MAX_M - command / 5.5 * (JAW_MAX_M - JAW_MIN_M)
        for finger in ("Joint1_1", "Joint2_1"):
            joint = _id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_dex1_{finger}"
            )
            data.qpos[model.jnt_qposadr[joint]] = target
    mujoco.mj_forward(model, data)
    hold_control = data.ctrl.copy()
    start_time = data.time
    joint_errors, arm_errors, gripper_errors = [], [], []
    eef_errors, pelvis_heights = [], []
    contact_frames = 0
    for frame, (state, action, timestamp) in enumerate(
        zip(states, actions, timestamps, strict=True)
    ):
        target_time = start_time + float(timestamp)
        while data.time + 1e-12 < target_time:
            data.ctrl[:] = hold_control
            data.ctrl[arm_actuators] = action[:14]
            controller.set_motor_commands(data, action[14], action[15])
            mujoco.mj_step(model, data)
        simulated = np.empty(16)
        simulated[:14] = data.qpos[arm_qpos]
        for side_index, side in enumerate(SIDES):
            positions = []
            for finger in ("Joint1_1", "Joint2_1"):
                joint = _id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_dex1_{finger}"
                )
                positions.append(data.qpos[model.jnt_qposadr[joint]])
            simulated[14 + side_index] = jaw_position_to_motor_radians(
                float(np.mean(positions))
            )
        joint_errors.append(float(np.sqrt(np.mean(np.square(simulated - state)))))
        arm_errors.append(float(np.sqrt(np.mean(np.square(simulated[:14] - state[:14])))))
        gripper_errors.append(float(np.sqrt(np.mean(np.square(simulated[14:] - state[14:])))))
        pelvis = _id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        position_error = []
        for side_index, side in enumerate(SIDES):
            site = _id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_eef")
            actual = data.xmat[pelvis].reshape(3, 3).T @ (
                data.site_xpos[site] - data.xpos[pelvis]
            )
            expected = expected_eef[frame, (0, 7)[side_index] : (3, 10)[side_index]]
            position_error.append(np.linalg.norm(actual - expected))
        eef_errors.append(float(max(position_error)))
        pelvis_heights.append(float(data.xpos[pelvis, 2]))
        contact_frames += int(_forbidden_manipulator_contact(model, data))
    joint_errors = np.asarray(joint_errors)
    arm_errors = np.asarray(arm_errors)
    gripper_errors = np.asarray(gripper_errors)
    eef_errors = np.asarray(eef_errors)
    return {
        "episode": 0,
        "frames": int(len(states)),
        "duration_s": float(timestamps[-1]),
        "joint_and_gripper_rmse": {
            "p50": float(np.quantile(joint_errors, 0.50)),
            "p95": float(np.quantile(joint_errors, 0.95)),
            "max": float(joint_errors.max()),
        },
        "arm_joint_rmse_rad": {
            "p50": float(np.quantile(arm_errors, 0.50)),
            "p95": float(np.quantile(arm_errors, 0.95)),
            "max": float(arm_errors.max()),
        },
        "gripper_motor_reconstruction_rmse_rad": {
            "p50": float(np.quantile(gripper_errors, 0.50)),
            "p95": float(np.quantile(gripper_errors, 0.95)),
            "max": float(gripper_errors.max()),
        },
        "eef_position_error_m": {
            "p50": float(np.quantile(eef_errors, 0.50)),
            "p95": float(np.quantile(eef_errors, 0.95)),
            "max": float(eef_errors.max()),
        },
        "minimum_pelvis_height_m": min(pelvis_heights),
        "manipulator_environment_contact_frame_rate": contact_frames / len(states),
        "finite": bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))),
        "scope": (
            "Replays public arm/gripper targets only. Dataset has no object state, "
            "so block trajectory or task success cannot be compared."
        ),
    }


def dataset_ik_replay(samples: int, seed: int) -> dict:
    raw_states, raw_actions = _load_dataset_samples(samples, seed)
    eef_actions = _fk_transform(raw_actions)
    position_errors = []
    orientation_errors = []
    joint_errors = []
    minimum_singular_values = []
    iterations = []
    bounded = []
    for state, raw_action, eef_action in zip(
        raw_states, raw_actions, eef_actions, strict=True
    ):
        model = build_model()
        data = mujoco.MjData(model)
        reset_to_stand(model, data)
        arm_qpos = _arm_addresses(model)
        data.qpos[arm_qpos] = state[:14]
        mujoco.mj_forward(model, data)
        solver = G1DualArmIK(model, data, damping=0.05, max_joint_speed=2.0)
        solver.reset()
        pelvis = _id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        target = pelvis_vla_action_to_world_mujoco(
            eef_action, data.xpos[pelvis], data.xquat[pelvis]
        )
        final_step = 0
        for step in range(250):
            solver.step(target[:7], target[7:14], 0.02)
            data.qpos[arm_qpos] = solver.q_target[arm_qpos]
            data.qvel[:] = 0.0
            mujoco.mj_fwdPosition(model, data)
            final_step = step + 1
            left_pose = solver.pose("left")
            right_pose = solver.pose("right")
            pos_error = max(
                np.linalg.norm(left_pose[0] - target[:3]),
                np.linalg.norm(right_pose[0] - target[7:10]),
            )
            rot_error = max(
                np.linalg.norm(orientation_error(target[3:7], left_pose[1])),
                np.linalg.norm(orientation_error(target[10:14], right_pose[1])),
            )
            if pos_error < 0.002 and rot_error < np.deg2rad(1.0):
                break
        position_errors.append(float(pos_error))
        orientation_errors.append(float(rot_error))
        for site, info in (
            (solver.left_site, solver.left), (solver.right_site, solver.right)
        ):
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, site)
            singular = np.linalg.svd(
                np.vstack((jacp[:, info["dof"]], jacr[:, info["dof"]])),
                compute_uv=False,
            )
            minimum_singular_values.append(float(singular[-1]))
        joint_errors.append(float(np.sqrt(np.mean(np.square(
            data.qpos[arm_qpos] - raw_action[:14]
        )))))
        iterations.append(final_step)
        in_range = True
        for name, qpos in zip(ARM_JOINTS, data.qpos[arm_qpos], strict=True):
            joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if model.jnt_limited[joint]:
                low, high = model.jnt_range[joint]
                in_range &= bool(low - 1e-9 <= qpos <= high + 1e-9)
        bounded.append(in_range)
    position_errors = np.asarray(position_errors)
    orientation_errors = np.asarray(orientation_errors)
    return {
        "samples": samples,
        "success_threshold": {"position_m": 0.002, "orientation_deg": 1.0},
        "success_rate": float(np.mean(
            (position_errors < 0.002) &
            (orientation_errors < np.deg2rad(1.0))
        )),
        "position_error_m": {
            "p50": float(np.quantile(position_errors, 0.5)),
            "p95": float(np.quantile(position_errors, 0.95)),
            "p99": float(np.quantile(position_errors, 0.99)),
            "max": float(position_errors.max()),
        },
        "orientation_error_deg": {
            "p50": float(np.rad2deg(np.quantile(orientation_errors, 0.5))),
            "p95": float(np.rad2deg(np.quantile(orientation_errors, 0.95))),
            "p99": float(np.rad2deg(np.quantile(orientation_errors, 0.99))),
            "max": float(np.rad2deg(orientation_errors.max())),
        },
        "joint_solution_rmse_rad": {
            "p50": float(np.quantile(joint_errors, 0.5)),
            "p95": float(np.quantile(joint_errors, 0.95)),
        },
        "iterations_p95": float(np.quantile(iterations, 0.95)),
        "joint_limit_bounded_rate": float(np.mean(bounded)),
        "jacobian_min_singular_value": {
            "p01": float(np.quantile(minimum_singular_values, 0.01)),
            "p50": float(np.quantile(minimum_singular_values, 0.50)),
            "near_singular_below_1e-3_rate": float(
                np.mean(np.asarray(minimum_singular_values) < 1e-3)
            ),
        },
        "note": "Targets come from real dataset joint actions transformed by FK.",
    }


def _forbidden_manipulator_contact(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    def manipulator(body_name: str) -> bool:
        return any(token in body_name for token in (
            "shoulder", "elbow", "wrist", "dex1"
        ))
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1 = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom1]
        ) or "world"
        body2 = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom2]
        ) or "world"
        geom_name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or ""
        geom_name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or ""
        if manipulator(body1) or manipulator(body2):
            # Any manipulator contact with the environment, torso, or opposite
            # chain is unsafe for this free-space workspace stress test.
            same_chain = (
                (body1.startswith("left_") and body2.startswith("left_")) or
                (body1.startswith("right_") and body2.startswith("right_"))
            )
            if not same_chain or "work_table" in (geom_name1, geom_name2):
                return True
    return False


def workspace_stress(samples: int, seed: int) -> dict:
    """Stress random dual-arm positions, including unreachable/OOD targets."""
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed + 17)
    model = build_model()
    data = mujoco.MjData(model)
    arm_qpos = _arm_addresses(model)
    successes = []
    per_arm_successes = []
    position_errors = []
    bounded = []
    saturated = []
    forbidden_contacts = []
    target_records = []
    bins_success = np.zeros((8, 8), dtype=float)
    bins_total = np.zeros((8, 8), dtype=float)
    for _ in range(samples):
        reset_to_stand(model, data)
        solver = G1DualArmIK(model, data, damping=0.06, max_joint_speed=2.0)
        solver.reset()
        left_pose = solver.pose("left")
        right_pose = solver.pose("right")
        left_target = np.r_[
            rng.uniform(0.10, 0.70), rng.uniform(0.02, 0.48),
            rng.uniform(0.62, 1.16), left_pose[1],
        ]
        right_target = np.r_[
            rng.uniform(0.10, 0.70), rng.uniform(-0.48, -0.02),
            rng.uniform(0.62, 1.16), right_pose[1],
        ]
        for _step in range(300):
            solver.step(left_target, right_target, 0.02)
            data.qpos[arm_qpos] = solver.q_target[arm_qpos]
            data.qvel[:] = 0.0
            mujoco.mj_fwdPosition(model, data)
        left_final = solver.pose("left")
        right_final = solver.pose("right")
        left_error = float(np.linalg.norm(left_final[0] - left_target[:3]))
        right_error = float(np.linalg.norm(right_final[0] - right_target[:3]))
        pos_error = max(left_error, right_error)
        arm_success = (left_error < 0.005, right_error < 0.005)
        success = bool(all(arm_success))
        in_range = True
        at_limit = False
        for name, qpos in zip(ARM_JOINTS, data.qpos[arm_qpos], strict=True):
            joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            low, high = model.jnt_range[joint]
            in_range &= bool(low - 1e-9 <= qpos <= high + 1e-9)
            at_limit |= bool(min(abs(qpos - low), abs(qpos - high)) < 1e-4)
        contact = _forbidden_manipulator_contact(model, data)
        successes.append(success)
        per_arm_successes.extend(arm_success)
        position_errors.extend((left_error, right_error))
        bounded.append(in_range)
        saturated.append(at_limit)
        forbidden_contacts.append(contact)
        target_records.append((left_target[:3], right_target[:3], success))
        for target, target_success in zip(
            (left_target, right_target), arm_success, strict=True
        ):
            xbin = min(7, max(0, int((target[0] - 0.10) / 0.60 * 8)))
            zbin = min(7, max(0, int((target[2] - 0.62) / 0.54 * 8)))
            bins_total[7 - zbin, xbin] += 1
            bins_success[7 - zbin, xbin] += target_success
    rates = np.divide(
        bins_success, bins_total, out=np.zeros_like(bins_success), where=bins_total > 0
    )
    image = Image.new("RGB", (512, 512), (8, 12, 22))
    draw = ImageDraw.Draw(image)
    for row in range(8):
        for column in range(8):
            rate = rates[row, column]
            color = (int(230 * (1 - rate)), int(210 * rate), 80)
            draw.rectangle(
                [column * 64 + 2, row * 64 + 2, (column + 1) * 64 - 2, (row + 1) * 64 - 2],
                fill=color,
            )
    heatmap = RESULTS / "workspace_success_heatmap.png"
    image.save(heatmap)
    return {
        "dual_target_samples": samples,
        "target_position_bounds_world_m": {
            "x": [0.10, 0.70], "abs_y": [0.02, 0.48], "z": [0.62, 1.16]
        },
        "success_threshold_position_m": 0.005,
        "dual_target_success_rate": float(np.mean(successes)),
        "per_arm_target_success_rate": float(np.mean(per_arm_successes)),
        "position_error_m": {
            "p50": float(np.quantile(position_errors, 0.50)),
            "p95": float(np.quantile(position_errors, 0.95)),
        },
        "joint_limit_bounded_rate": float(np.mean(bounded)),
        "joint_limit_saturation_rate": float(np.mean(saturated)),
        "forbidden_contact_rate": float(np.mean(forbidden_contacts)),
        "explicit_unreachable_target_rejection_present": False,
        "heatmap": str(heatmap.relative_to(ROOT)),
        "verdict": (
            "Joint limits remain bounded, but OOD targets can saturate or collide; "
            "an explicit feasibility/collision rejection gate is still required."
        ),
    }


def long_horizon_stability(seconds: float = 30.0) -> dict:
    model = build_model()
    data = mujoco.MjData(model)
    reset_to_stand(model, data)
    pelvis = _id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    cube_bodies = [
        _id(model, mujoco.mjtObj.mjOBJ_BODY, f"{color}_cube")
        for color in ("red", "blue", "yellow")
    ]
    heights, tilts, cube_heights = [], [], []
    steps = int(seconds / model.opt.timestep)
    for step in range(steps):
        mujoco.mj_step(model, data)
        if step % 50 == 0:
            heights.append(float(data.xpos[pelvis, 2]))
            # For a unit quaternion, 2*acos(|w|) is total root tilt magnitude.
            tilts.append(float(2 * np.arccos(np.clip(abs(data.xquat[pelvis, 0]), 0, 1))))
            cube_heights.append([float(data.xpos[body, 2]) for body in cube_bodies])
    values = np.asarray(cube_heights)
    return {
        "duration_s": seconds,
        "finite": bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))),
        "pelvis_height_m": {
            "min": min(heights), "max": max(heights), "final": heights[-1],
        },
        "root_rotation_deg_max": float(np.rad2deg(max(tilts))),
        "cube_height_drift_m_max": float(np.max(np.ptp(values, axis=0))),
        "stable": bool(min(heights) > 0.72 and np.max(np.ptp(values, axis=0)) < 0.003),
    }


def disturbance_sweep() -> dict:
    trials = []
    for force in (5.0, 10.0, 20.0, 40.0, 60.0):
        model = build_model()
        data = mujoco.MjData(model)
        reset_to_stand(model, data)
        pelvis = _id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        torso = _id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        minimum = float("inf")
        maximum_tilt = 0.0
        initial_xy = data.xpos[pelvis, :2].copy()
        for step in range(5000):  # ten seconds
            data.xfrc_applied[:] = 0.0
            if 1000 <= step < 1100:
                data.xfrc_applied[torso, 0] = force
            mujoco.mj_step(model, data)
            minimum = min(minimum, float(data.xpos[pelvis, 2]))
            tilt = 2 * np.arccos(np.clip(abs(data.xquat[pelvis, 0]), 0, 1))
            maximum_tilt = max(maximum_tilt, float(tilt))
        xy_displacement = float(np.linalg.norm(data.xpos[pelvis, :2] - initial_xy))
        recovered = bool(
            data.xpos[pelvis, 2] > 0.72
            and minimum > 0.62
            and xy_displacement < 0.10
            and maximum_tilt < np.deg2rad(15)
        )
        trials.append({
            "force_n": force,
            "impulse_ns": force * 100 * model.opt.timestep,
            "minimum_pelvis_height_m": minimum,
            "final_pelvis_height_m": float(data.xpos[pelvis, 2]),
            "final_xy_displacement_m": xy_displacement,
            "maximum_root_rotation_deg": float(np.rad2deg(maximum_tilt)),
            "recovered": recovered,
        })
    contiguous_threshold = None
    for trial in trials:
        if not trial["recovered"]:
            break
        contiguous_threshold = trial["force_n"]
    return {
        "trials": trials,
        "all_lower_forces_recovered_through_n": contiguous_threshold,
        "note": "Recovery also requires <0.10 m base drift and <15 deg root rotation.",
    }


def randomized_scene_sweep(trials: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    records = []
    for trial in range(trials):
        model = build_model()
        data = mujoco.MjData(model)
        reset_to_stand(model, data)
        masses, frictions = [], []
        stable = True
        for index, color in enumerate(("red", "blue", "yellow")):
            body = _id(model, mujoco.mjtObj.mjOBJ_BODY, f"{color}_cube")
            geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{color}_cube_geom")
            joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{color}_cube_free")
            mass = float(rng.uniform(0.08, 0.30))
            friction = float(rng.uniform(0.3, 1.3))
            inertia_scale = mass / model.body_mass[body]
            model.body_inertia[body] *= inertia_scale
            model.body_mass[body] = mass
            model.geom_friction[geom, 0] = friction
            qadr = model.jnt_qposadr[joint]
            data.qpos[qadr] = 0.43 + rng.uniform(-0.06, 0.06)
            data.qpos[qadr + 1] = (-0.18, 0.0, 0.18)[index] + rng.uniform(-0.025, 0.025)
            masses.append(mass)
            frictions.append(friction)
        mujoco.mj_forward(model, data)
        pelvis = _id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        for _ in range(2500):
            mujoco.mj_step(model, data)
        for color in ("red", "blue", "yellow"):
            body = _id(model, mujoco.mjtObj.mjOBJ_BODY, f"{color}_cube")
            stable &= bool(data.xpos[body, 2] > 0.60)
        stable &= bool(data.xpos[pelvis, 2] > 0.72 and np.all(np.isfinite(data.qpos)))
        records.append({"trial": trial, "masses_kg": masses, "frictions": frictions, "stable": stable})
    return {"trials": records, "stable_rate": float(np.mean([record["stable"] for record in records]))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ik-samples", type=int, default=250)
    parser.add_argument("--random-trials", type=int, default=20)
    parser.add_argument("--workspace-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20250810)
    args = parser.parse_args()
    report = {
        "dex1_command_sweep": dex1_sweep(),
        "dex1_grasp_sweep": grasp_sweep(),
        "episode_zero_dynamics_replay": episode_zero_dynamics_replay(),
        "dataset_ik_replay": dataset_ik_replay(args.ik_samples, args.seed),
        "workspace_stress": workspace_stress(args.workspace_samples, args.seed),
        "long_horizon_stability": long_horizon_stability(),
        "external_disturbance": disturbance_sweep(),
        "randomized_scene": randomized_scene_sweep(args.random_trials, args.seed),
        "scope": (
            "Local deterministic and Monte Carlo MuJoCo validation. No neural "
            "VLA checkpoint was loaded."
        ),
    }
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "comprehensive_sim_validation.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(path)
    print(json.dumps({
        "dex_monotonic": report["dex1_command_sweep"]["gap_monotonically_increases"],
        "grasp_zero_force": report["dex1_grasp_sweep"]["zero_force_bilateral_success"],
        "episode0_eef_p95_m": report["episode_zero_dynamics_replay"]["eef_position_error_m"]["p95"],
        "ik_success_rate": report["dataset_ik_replay"]["success_rate"],
        "workspace_success_rate": report["workspace_stress"]["dual_target_success_rate"],
        "long_horizon_stable": report["long_horizon_stability"]["stable"],
        "randomized_stable_rate": report["randomized_scene"]["stable_rate"],
    }, indent=2))


if __name__ == "__main__":
    main()
