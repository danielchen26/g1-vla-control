"""MuJoCo side of the frozen G1 EDU observation/action boundary.

The future hardware adapter must emit and consume the same policy-level values;
only this simulator boundary is replaced by Unitree camera/state/controller I/O.
"""

from __future__ import annotations

import hashlib
from typing import Any

import mujoco
import numpy as np

from action_schema import (
    mujoco_wxyz_to_vla_xyzw,
    normalize_quaternion,
    pelvis_vla_action_to_world_mujoco,
)
from dex1_gripper import Dex1Controller
from g1_policy_contract import (
    CONTRACT_ID,
    CONTRACT_SHA256,
    IMAGE_KEYS,
    PROMPT_KEY,
    STATE_KEY,
    preprocess_rgb_image,
    validate_observation,
    validate_state,
)
from stack_scene import CAMERA_NAMES, TASK_PROMPT, build_model, reset_to_reference_pose


def _quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(q)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_multiply_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = normalize_quaternion(a)
    bw, bx, by, bz = normalize_quaternion(b)
    return normalize_quaternion(np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]))


def policy_state_from_mujoco(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    gripper_state_rad: np.ndarray,
) -> np.ndarray:
    """Read both EEFs in the pelvis frame using the frozen 16-D order."""
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    pelvis_pos = data.xpos[pelvis_id].copy()
    pelvis_rot = data.xmat[pelvis_id].reshape(3, 3).copy()
    pelvis_quat = np.empty(4)
    mujoco.mju_mat2Quat(pelvis_quat, pelvis_rot.ravel())
    pelvis_inv = _quat_conjugate_wxyz(pelvis_quat)

    state = np.empty(16, dtype=np.float32)
    cursor = 0
    for side in ("left", "right"):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_eef")
        world_quat = np.empty(4)
        mujoco.mju_mat2Quat(world_quat, data.site_xmat[site_id])
        relative_quat = _quat_multiply_wxyz(pelvis_inv, world_quat)
        state[cursor:cursor + 3] = pelvis_rot.T @ (data.site_xpos[site_id] - pelvis_pos)
        state[cursor + 3:cursor + 7] = mujoco_wxyz_to_vla_xyzw(relative_quat)
        cursor += 7
    state[14:16] = np.asarray(gripper_state_rad, dtype=np.float32)
    return validate_state(state)


def policy_action_to_mujoco_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    action: np.ndarray,
) -> np.ndarray:
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    return pelvis_vla_action_to_world_mujoco(
        action,
        data.xpos[pelvis_id],
        data.xquat[pelvis_id],
    )


def build_sim_observation() -> tuple[dict[str, Any], dict[str, Any]]:
    """Render the exact policy observation used by the future G1 EDU adapter."""
    model = build_model()
    data = mujoco.MjData(model)
    reset_to_reference_pose(model, data)
    hold = data.ctrl.copy()
    for _ in range(250):
        data.ctrl[:] = hold
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)
    source_images: dict[str, np.ndarray] = {}
    try:
        for camera in CAMERA_NAMES:
            renderer.update_scene(data, camera=camera)
            source_images[camera] = renderer.render().copy()
    finally:
        renderer.close()

    images = {
        name: preprocess_rgb_image(image)
        for name, image in source_images.items()
    }
    state = policy_state_from_mujoco(
        model, data, Dex1Controller(model).motor_states(data)
    )
    observation = {
        IMAGE_KEYS[0]: images["cam_left_high"],
        IMAGE_KEYS[1]: images["cam_left_wrist"],
        IMAGE_KEYS[2]: images["cam_right_wrist"],
        STATE_KEY: state,
        PROMPT_KEY: TASK_PROMPT,
    }
    validate_observation(observation)
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    evidence = {
        "g1_policy_contract_id": CONTRACT_ID,
        "g1_policy_contract_sha256": CONTRACT_SHA256,
        "state": state.tolist(),
        "state_shape": list(state.shape),
        "state_finite": bool(np.all(np.isfinite(state))),
        "source_image_shape": list(source_images["cam_left_high"].shape),
        "policy_image_shape": list(images["cam_left_high"].shape),
        "image_shape": list(images["cam_left_high"].shape),
        "image_dtype": str(images["cam_left_high"].dtype),
        "preprocessing": "640x480_RGB_center_crop_480x480_bilinear_resize_224x224",
        "image_sha256": {
            name: hashlib.sha256(image.tobytes()).hexdigest()
            for name, image in images.items()
        },
        "pelvis_position_world_m": data.xpos[pelvis_id].tolist(),
        "pelvis_quaternion_world_wxyz": data.xquat[pelvis_id].tolist(),
    }
    return observation, evidence
