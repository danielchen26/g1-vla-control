#!/usr/bin/env python3
"""Output-only LGG100 smoke using real MuJoCo G1 images and EEF state.

No returned action is executed.  A passing report proves that the strict
candidate server loaded the real neural checkpoint and returned finite 16-D
chunks for the simulated observation.  It does not confirm the unpublished
author transforms or task success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import socket
from pathlib import Path
import time
from typing import Any

import mujoco
import numpy as np

from action_schema import mujoco_wxyz_to_vla_xyzw, normalize_quaternion
from stack_scene import (
    CAMERA_NAMES, REFERENCE_EP0_STATE, TASK_PROMPT, build_model,
    reset_to_reference_pose,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
HF_REPO = "LGG100/stack-cube-eef-24k"
HF_REVISION = "cced7a7ff7b454fdcac555457a1a2a3dc262ac77"
EXPECTED_ACTION_DIM = 16


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


def build_sim_observation() -> tuple[dict[str, Any], dict[str, Any]]:
    """Render all three cameras and reconstruct current pelvis-frame EEF state."""
    model = build_model()
    data = mujoco.MjData(model)
    reset_to_reference_pose(model, data)
    hold = data.ctrl.copy()
    for _ in range(250):
        data.ctrl[:] = hold
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, height=224, width=224)
    images: dict[str, np.ndarray] = {}
    try:
        for camera in CAMERA_NAMES:
            renderer.update_scene(data, camera=camera)
            images[camera] = renderer.render().copy()
    finally:
        renderer.close()

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
        state[cursor : cursor + 3] = pelvis_rot.T @ (data.site_xpos[site_id] - pelvis_pos)
        state[cursor + 3 : cursor + 7] = mujoco_wxyz_to_vla_xyzw(relative_quat)
        cursor += 7
    # The synchronized episode-0 motor values are the command/state contract used
    # to initialize this scene; jaw qpos uses a separate prismatic representation.
    state[14:] = REFERENCE_EP0_STATE[14:]

    observation = {
        "observation/cam_left_high": images["cam_left_high"],
        "observation/cam_left_wrist": images["cam_left_wrist"],
        "observation/cam_right_wrist": images["cam_right_wrist"],
        "observation/state": state,
        "prompt": TASK_PROMPT,
    }
    evidence = {
        "state": state.tolist(),
        "state_shape": list(state.shape),
        "state_finite": bool(np.all(np.isfinite(state))),
        "image_shape": list(images["cam_left_high"].shape),
        "image_dtype": str(images["cam_left_high"].dtype),
        "image_sha256": {
            name: hashlib.sha256(image.tobytes()).hexdigest()
            for name, image in images.items()
        },
        "pelvis_position_world_m": pelvis_pos.tolist(),
        "pelvis_quaternion_world_wxyz": pelvis_quat.tolist(),
    }
    return observation, evidence


def _infer_with_timeout(client, observation: dict[str, Any], timeout_ms: float):
    def handler(signum, frame):
        del signum, frame
        raise TimeoutError(f"LGG100 inference exceeded {timeout_ms:.0f} ms")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_ms / 1000.0)
    try:
        return client.infer(observation)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument("--warmup-calls", type=int, default=1)
    parser.add_argument("--call-timeout-ms", type=float, default=60_000.0)
    parser.add_argument("--connect-timeout-s", type=float, default=5.0)
    parser.add_argument("--action-rate-hz", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=RESULTS / "lgg100_vla_smoke_real.json")
    parser.add_argument("--chunk-output", type=Path, default=RESULTS / "lgg100_action_chunk_real.npz")
    args = parser.parse_args()
    if args.calls < 1 or args.warmup_calls < 0 or args.action_rate_hz <= 0:
        raise SystemExit("calls/rate must be positive and warmup-calls non-negative")

    try:
        from openpi_client import websocket_client_policy
    except ImportError as exc:
        raise SystemExit("Install pinned openpi-client; see LGG100_REAL_VLA.md") from exc
    with socket.create_connection((args.host, args.port), timeout=args.connect_timeout_s):
        pass
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = _json_safe(client.get_server_metadata())
    strict_neural_restore = bool(
        metadata.get("neural_checkpoint_loaded") is True
        and metadata.get("strict_parameter_tree_restore") is True
        and metadata.get("hf_repo") == HF_REPO
        and metadata.get("hf_revision") == HF_REVISION
    )
    observation, observation_evidence = build_sim_observation()

    warmup_errors = []
    for _ in range(args.warmup_calls):
        try:
            _infer_with_timeout(client, observation, args.call_timeout_ms)
        except Exception as exc:
            warmup_errors.append(f"{type(exc).__name__}: {exc}")
            break

    records = []
    chunks: list[np.ndarray] = []
    shapes: set[tuple[int, ...]] = set()
    latencies = []
    for index in range(args.calls):
        start = time.monotonic_ns()
        try:
            response = _infer_with_timeout(client, observation, args.call_timeout_ms)
            latency_ms = (time.monotonic_ns() - start) / 1e6
            actions = np.asarray(response.get("actions"), dtype=np.float64)
            finite = bool(actions.size and np.all(np.isfinite(actions)))
            shape_ok = actions.ndim == 2 and actions.shape[1] == EXPECTED_ACTION_DIM
            shape = tuple(int(x) for x in actions.shape)
            shapes.add(shape)
            latencies.append(latency_ms)
            if finite and shape_ok:
                chunks.append(actions.copy())
            left_norm = np.linalg.norm(actions[:, 3:7], axis=1) if shape_ok else np.array([])
            right_norm = np.linalg.norm(actions[:, 10:14], axis=1) if shape_ok else np.array([])
            records.append({
                "call": index,
                "latency_ms": latency_ms,
                "action_shape": list(shape),
                "finite": finite,
                "valid_16d_chunk": shape_ok,
                "action_min": float(actions.min()) if actions.size else None,
                "action_max": float(actions.max()) if actions.size else None,
                "left_quaternion_norm_range": [float(left_norm.min()), float(left_norm.max())] if left_norm.size else None,
                "right_quaternion_norm_range": [float(right_norm.min()), float(right_norm.max())] if right_norm.size else None,
                "action_sha256": hashlib.sha256(actions.tobytes()).hexdigest() if actions.size else None,
                "policy_timing": _json_safe(response.get("policy_timing", {})),
                "error": None,
            })
        except Exception as exc:
            latency_ms = (time.monotonic_ns() - start) / 1e6
            latencies.append(latency_ms)
            records.append({
                "call": index,
                "latency_ms": latency_ms,
                "action_shape": [],
                "finite": False,
                "valid_16d_chunk": False,
                "error": f"{type(exc).__name__}: {exc}",
            })

    valid_calls = sum(
        bool(record["finite"] and record["valid_16d_chunk"] and record["error"] is None)
        for record in records
    )
    passed = bool(
        strict_neural_restore and not warmup_errors and valid_calls == args.calls
        and len(shapes) == 1
    )
    latency_array = np.asarray(latencies, dtype=np.float64)
    report = {
        "scope": "Real LGG100 neural output-only inference from MuJoCo observation; no action execution.",
        "evidence_mode": "lgg100_real_weights_candidate_semantics",
        "neural_vla_claimed": strict_neural_restore,
        "author_config_claimed": False,
        "g1_execution_enabled": False,
        "adaptive_retimer_enabled": False,
        "checkpoint": {"repo": HF_REPO, "revision": HF_REVISION},
        "server_metadata": metadata,
        "observation": observation_evidence,
        "summary": {
            "passed": passed,
            "calls": args.calls,
            "valid_calls": valid_calls,
            "warmup_errors": warmup_errors,
            "unique_action_shapes": [list(shape) for shape in sorted(shapes)],
            "latency_ms": {
                "p50": float(np.quantile(latency_array, 0.50)),
                "p95": float(np.quantile(latency_array, 0.95)),
                "p99": float(np.quantile(latency_array, 0.99)),
                "max": float(latency_array.max()),
            },
        },
        "calls": records,
        "verdict": (
            "Real weights loaded under a strict candidate architecture and returned finite 16-D chunks. "
            "Author semantics remain unconfirmed; proceed only to offline/preflight gates."
            if passed else
            "Restore/inference gate failed. Do not execute the chunk in simulation or hardware."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(args.output)
    print(json.dumps(report["summary"], indent=2))

    if passed:
        selected = chunks[0]
        timestamps = np.arange(len(selected), dtype=np.float64) / args.action_rate_hz
        args.chunk_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.chunk_output,
            actions=selected,
            timestamps=timestamps,
            observation_state=np.asarray(observation["observation/state"]),
            action_sha256=np.asarray(hashlib.sha256(selected.tobytes()).hexdigest()),
            hf_revision=np.asarray(HF_REVISION),
            source_report=np.asarray(str(args.output)),
        )
        print(args.chunk_output)
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
