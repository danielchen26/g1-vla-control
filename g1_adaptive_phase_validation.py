#!/usr/bin/env python3
"""Cover far-speedup and near-slowdown regions on a G1-valid lift path.

This validates retiming mechanics through the full G1 MuJoCo command pipeline.
It deliberately does not claim LGG100 neural or task-level evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from action_schema import LEFT_POS, RIGHT_POS
from adaptive_retimer import AdaptiveRetimer, RetimerConfig
from g1_policy_contract import contract_metadata
from retiming_safety_validation import _run_scale
from run_simulation import build_contract_fixture, preflight_contract_chunk
from stack_scene import REFERENCE_EP0_STATE

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FAR_LIFT_OFFSET_M = np.array([0.0, 0.0, 0.170], dtype=np.float64)


def run(output: Path | None = None) -> dict:
    chunk, left_target, right_target = build_contract_fixture(FAR_LIFT_OFFSET_M)
    preflight = preflight_contract_chunk(chunk)
    config = RetimerConfig()
    retiming = AdaptiveRetimer(config).retime(chunk, left_target, right_target)
    distances = np.maximum(
        np.linalg.norm(chunk.actions[:, LEFT_POS] - left_target, axis=1),
        np.linalg.norm(chunk.actions[:, RIGHT_POS] - right_target, axis=1),
    )
    far_mask = distances >= config.far_distance
    near_mask = distances <= config.near_distance
    far_scales = retiming.scale_profile[far_mask]
    near_scales = retiming.scale_profile[near_mask]
    far_condition_observed = bool(np.any(far_mask))
    near_condition_observed = bool(np.any(near_mask))
    fast_when_far = bool(far_scales.size and np.max(far_scales) > 1.0)
    slow_when_near = bool(near_scales.size and np.max(near_scales) < 1.0)

    baseline = None
    adaptive = None
    if preflight["all_accepted"]:
        baseline = _run_scale(
            chunk, REFERENCE_EP0_STATE[14:16],
            scale=1.0, use_filter=True, use_joint_filter=True,
        )
        adaptive = _run_scale(
            retiming.chunk, REFERENCE_EP0_STATE[14:16],
            scale=1.0, use_filter=True, use_joint_filter=True,
        )
    dynamics_passed = bool(
        baseline
        and adaptive
        and baseline["hard_command_limits_pass"]
        and adaptive["hard_command_limits_pass"]
        and baseline["finite"]
        and adaptive["finite"]
        and baseline["manipulation_contact_step_rate"] == 0.0
        and adaptive["manipulation_contact_step_rate"] == 0.0
        and baseline["endpoint_error_m"] < 0.035
        and adaptive["endpoint_error_m"] < 0.035
        and adaptive["maxima"]["actual_joint_jerk_rad_s3"]
        <= baseline["maxima"]["actual_joint_jerk_rad_s3"] + 1e-6
    )
    local_phase_behavior_passed = bool(
        preflight["all_accepted"]
        and far_condition_observed
        and near_condition_observed
        and fast_when_far
        and slow_when_near
        and dynamics_passed
        and np.array_equal(retiming.chunk.actions, chunk.actions)
    )
    adaptive_faster_overall = bool(
        baseline and adaptive
        and adaptive["simulated_duration_s"] < baseline["simulated_duration_s"]
    )
    result = {
        **contract_metadata(verified=True),
        "scope": "G1-compatible deterministic far-to-near adaptive coverage; not neural VLA evidence.",
        "trajectory_source": "deterministic_g1_contract_fixture_not_lgg100",
        "real_lgg100_chunk_used": False,
        "g1_hardware_execution_enabled": False,
        "production_adaptive_enabled": False,
        "lift_offset_m": FAR_LIFT_OFFSET_M.tolist(),
        "preflight": preflight,
        "thresholds": {
            "near_distance_m": config.near_distance,
            "far_distance_m": config.far_distance,
            "min_scale": config.min_scale,
            "max_scale": config.max_scale,
        },
        "coverage": {
            "initial_distance_m": float(distances[0]),
            "far_sample_count": int(np.count_nonzero(far_mask)),
            "near_sample_count": int(np.count_nonzero(near_mask)),
            "far_scale_min": float(far_scales.min()),
            "far_scale_median": float(np.median(far_scales)),
            "far_scale_max": float(far_scales.max()),
            "near_scale_min": float(near_scales.min()),
            "near_scale_median": float(np.median(near_scales)),
            "near_scale_max": float(near_scales.max()),
            "fast_when_far": fast_when_far,
            "slow_when_near": slow_when_near,
            "path_actions_byte_identical": bool(
                np.array_equal(retiming.chunk.actions, chunk.actions)
            ),
        },
        "baseline": baseline,
        "adaptive": adaptive,
        "comparison": {
            "local_phase_behavior_passed": local_phase_behavior_passed,
            "adaptive_faster_overall": adaptive_faster_overall,
            "duration_delta_s": (
                adaptive["simulated_duration_s"] - baseline["simulated_duration_s"]
                if baseline and adaptive else None
            ),
            "real_lgg100_behavior_proven": False,
            "task_level_benefit_proven": False,
        },
        "success": local_phase_behavior_passed,
        "verdict": (
            "G1 mechanics covered both regions: faster while far and slower while near. Overall completion was not faster, and no real LGG100 chunk was used."
            if local_phase_behavior_passed else
            "G1 far/near adaptive coverage failed; do not enable adaptive retiming."
        ),
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=RESULTS / "g1_adaptive_phase_validation.json",
    )
    args = parser.parse_args()
    if not run(args.output)["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
