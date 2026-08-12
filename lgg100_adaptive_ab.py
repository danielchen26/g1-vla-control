#!/usr/bin/env python3
"""Paired baseline/adaptive MuJoCo replay of one real LGG100 action chunk.

This is deliberately a second gate after output-only neural inference.  Both
arms receive the exact same saved chunk from the exact same initial simulation
state.  The adaptive branch may change timestamps only.  One chunk can assess
retiming mechanics, not end-to-end block-stacking success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from action_schema import EEFActionChunk, pelvis_vla_action_to_world_mujoco
from adaptive_retimer import AdaptiveRetimer
from retiming_safety_validation import _run_scale
from safety_governor import G1TargetPreflight
from stack_scene import REFERENCE_EP0_STATE, build_model, reset_to_reference_pose

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
HF_REVISION = "cced7a7ff7b454fdcac555457a1a2a3dc262ac77"


def _load_verified_chunk(chunk_path: Path, report_path: Path, quaternion_tolerance: float):
    report = json.loads(report_path.read_text())
    if not (
        report.get("summary", {}).get("passed") is True
        and report.get("neural_vla_claimed") is True
        and report.get("g1_execution_enabled") is False
        and report.get("checkpoint", {}).get("revision") == HF_REVISION
    ):
        raise ValueError("Source smoke report is not a passing real-neural output-only gate")
    with np.load(chunk_path, allow_pickle=False) as payload:
        actions = np.asarray(payload["actions"], dtype=np.float64)
        timestamps = np.asarray(payload["timestamps"], dtype=np.float64)
        expected_hash = str(payload["action_sha256"].item())
        revision = str(payload["hf_revision"].item())
    actual_hash = hashlib.sha256(actions.tobytes()).hexdigest()
    if actual_hash != expected_hash or revision != HF_REVISION:
        raise ValueError("Chunk fingerprint/revision does not match the smoke artifact")
    if actions.ndim != 2 or actions.shape[1] != 16 or len(actions) < 2:
        raise ValueError(f"Expected [T,16] with T>=2, got {actions.shape}")
    if timestamps.shape != (len(actions),) or not np.all(np.diff(timestamps) > 0):
        raise ValueError("Chunk timestamps must be strictly increasing")
    if not np.all(np.isfinite(actions)):
        raise ValueError("Chunk contains NaN/Inf")
    norms = np.r_[
        np.linalg.norm(actions[:, 3:7], axis=1),
        np.linalg.norm(actions[:, 10:14], axis=1),
    ]
    norm_error = float(np.max(np.abs(norms - 1.0)))
    if norm_error > quaternion_tolerance:
        raise ValueError(
            f"Quaternion norm error {norm_error:.4f} exceeds {quaternion_tolerance:.4f}; refusing normalization"
        )
    return EEFActionChunk(timestamps, actions), actual_hash, norm_error


def _preflight(chunk: EEFActionChunk, phase: str) -> dict:
    model = build_model()
    data = mujoco.MjData(model)
    reset_to_reference_pose(model, data)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    gate = G1TargetPreflight(model)
    records = []
    for index, action in enumerate(chunk.actions):
        target = pelvis_vla_action_to_world_mujoco(
            action, data.xpos[pelvis], data.xquat[pelvis]
        )
        result = gate.check(data, target, phase=phase)
        records.append({
            "index": index,
            "accepted": result.accepted,
            "reachable": result.reachable,
            "collision_free": result.collision_free,
            "joint_limits_ok": result.joint_limits_ok,
            "position_error_m": result.position_error_m,
            "orientation_error_rad": result.orientation_error_rad,
            "reason": result.reason,
            "collision_reasons": list(result.collision_reasons),
        })
    return {
        "phase": phase,
        "checked": len(records),
        "accepted": sum(item["accepted"] for item in records),
        "all_accepted": all(item["accepted"] for item in records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=Path, default=RESULTS / "lgg100_action_chunk_real.npz")
    parser.add_argument("--smoke-report", type=Path, default=RESULTS / "lgg100_vla_smoke_real.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "lgg100_adaptive_ab.json")
    parser.add_argument("--phase", choices=("free_space", "grasp", "place"), default="grasp")
    parser.add_argument("--quaternion-norm-tolerance", type=float, default=0.15)
    parser.add_argument("--allow-sim-execution", action="store_true")
    args = parser.parse_args()
    if not args.allow_sim_execution:
        raise SystemExit(
            "Output-only by default. Re-run with --allow-sim-execution only after reviewing "
            "the passing smoke report; this flag still cannot enable G1 hardware."
        )

    chunk, chunk_hash, quaternion_norm_error = _load_verified_chunk(
        args.chunk, args.smoke_report, args.quaternion_norm_tolerance
    )
    preflight = _preflight(chunk, args.phase)
    evidence = {
        "scope": "Paired single-chunk MuJoCo retiming A/B; never G1 hardware.",
        "checkpoint_revision": HF_REVISION,
        "chunk_sha256": chunk_hash,
        "chunk_shape": list(chunk.actions.shape),
        "path_geometry_identical": True,
        "model_inference_reused_for_both_branches": True,
        "quaternion_max_norm_error_before_chunk_normalization": quaternion_norm_error,
        "preflight": preflight,
        "execution_performed": False,
        "baseline": None,
        "adaptive": None,
        "comparison": None,
    }
    if not preflight["all_accepted"]:
        evidence["verdict"] = (
            "Preflight rejected one or more real-model targets. No simulation execution was performed."
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
        print(args.output)
        raise SystemExit(1)

    initial_grippers = REFERENCE_EP0_STATE[14:16].copy()
    retiming = AdaptiveRetimer().retime(
        chunk,
        chunk.actions[-1, 0:3],
        chunk.actions[-1, 7:10],
        stability=1.0,
    )
    # Both branches use Gate-A and Gate-A.2 inside _run_scale.  Only the
    # adaptive chunk timestamps differ; action samples are byte-identical.
    baseline = _run_scale(
        chunk, initial_grippers, scale=1.0,
        use_filter=True, use_joint_filter=True,
    )
    adaptive = _run_scale(
        retiming.chunk, initial_grippers, scale=1.0,
        use_filter=True, use_joint_filter=True,
    )
    duration_delta = adaptive["simulated_duration_s"] - baseline["simulated_duration_s"]
    actual_jerk_baseline = baseline["maxima"]["actual_joint_jerk_rad_s3"]
    actual_jerk_adaptive = adaptive["maxima"]["actual_joint_jerk_rad_s3"]
    candidate_pass = bool(
        baseline["hard_command_limits_pass"]
        and adaptive["hard_command_limits_pass"]
        and adaptive["finite"]
        and adaptive["endpoint_error_m"] <= baseline["endpoint_error_m"] + 0.005
        and adaptive["manipulation_contact_step_rate"] <= baseline["manipulation_contact_step_rate"]
        and actual_jerk_adaptive <= actual_jerk_baseline
        and duration_delta < 0
    )
    evidence.update({
        "execution_performed": True,
        "baseline": baseline,
        "adaptive": adaptive,
        "comparison": {
            "adaptive_candidate_pass": candidate_pass,
            "duration_delta_s": duration_delta,
            "duration_change_percent": 100.0 * duration_delta / baseline["simulated_duration_s"],
            "endpoint_error_delta_m": adaptive["endpoint_error_m"] - baseline["endpoint_error_m"],
            "actual_joint_jerk_delta_rad_s3": actual_jerk_adaptive - actual_jerk_baseline,
            "scale_profile": {
                "min": float(retiming.scale_profile.min()),
                "median": float(np.median(retiming.scale_profile)),
                "max": float(retiming.scale_profile.max()),
            },
            "single_chunk_task_success_claimed": False,
        },
        "verdict": (
            "Adaptive timing improved this exact chunk under the conservative paired criteria. "
            "This is not yet a block-stacking success claim; multi-chunk randomized trials are required."
            if candidate_pass else
            "Adaptive timing did not pass the conservative paired criteria for this chunk. "
            "Do not claim that the adaptive module is beneficial."
        ),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    print(args.output)
    print(json.dumps(evidence["comparison"], indent=2))
    if not candidate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
