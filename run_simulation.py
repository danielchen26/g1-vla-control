#!/usr/bin/env python3
"""Run the G1 EDU production-contract simulation pipeline.

The input fixture is not a VLA.  It exercises only components intended to stay
identical after MuJoCo is replaced by G1 EDU I/O: 16-D pelvis-frame actions,
contract validation, preflight, EEF limits, dual-arm IK, joint limits and Dex1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from adaptive_retimer import AdaptiveRetimer
from dex1_gripper import Dex1Controller
from g1_contract_trajectory import make_reach_chunk
from g1_mujoco_bridge import policy_action_to_mujoco_world, policy_state_from_mujoco
from g1_policy_contract import (
    ACTION_HORIZON, POLICY_RATE_HZ, contract_metadata, validate_action_chunk,
)
from retiming_safety_validation import _run_scale
from safety_governor import G1TargetPreflight
from stack_scene import REFERENCE_EP0_STATE, build_model, reset_to_reference_pose


def build_contract_fixture():
    model = build_model()
    data = mujoco.MjData(model)
    reset_to_reference_pose(model, data)
    state = policy_state_from_mujoco(
        model, data, Dex1Controller(model).motor_states(data)
    )
    # Move upward without defining a synthetic task objective. The synchronized
    # reference pose starts near cubes, so contact is evaluated in grasp phase.
    left_target = state[0:3].astype(np.float64) + np.array([0.000, 0.000, 0.080])
    right_target = state[7:10].astype(np.float64) + np.array([0.000, 0.000, 0.080])
    chunk = make_reach_chunk(
        state[0:3], state[3:7], state[7:10], state[10:14],
        left_target, right_target,
        gripper_state_rad=state[14:16],
    )
    validate_action_chunk(
        chunk.actions, chunk.timestamps,
        expected_horizon=ACTION_HORIZON,
        expected_rate_hz=POLICY_RATE_HZ,
    )
    return chunk, left_target, right_target


def preflight_contract_chunk(chunk) -> dict:
    model = build_model()
    data = mujoco.MjData(model)
    reset_to_reference_pose(model, data)
    gate = G1TargetPreflight(model)
    reasons: dict[str, int] = {}
    accepted = 0
    for action in chunk.actions:
        target = policy_action_to_mujoco_world(model, data, action)
        result = gate.check(data, target, phase="grasp")
        reasons[result.reason] = reasons.get(result.reason, 0) + 1
        accepted += int(result.accepted)
    return {
        "phase": "grasp",
        "checked": len(chunk.actions),
        "accepted": accepted,
        "all_accepted": accepted == len(chunk.actions),
        "reason_counts": reasons,
    }


def run(*, adaptive: bool, output: Path | None = None) -> dict:
    raw_chunk, left_target, right_target = build_contract_fixture()
    preflight = preflight_contract_chunk(raw_chunk)
    if not preflight["all_accepted"]:
        result = {
            **contract_metadata(verified=True),
            "mode": "adaptive" if adaptive else "baseline",
            "trajectory_source": "deterministic_g1_contract_fixture_not_vla",
            "g1_hardware_execution_enabled": False,
            "preflight": preflight,
            "success": False,
            "verdict": "Contract fixture failed G1 preflight; no dynamics execution.",
        }
    else:
        retiming = None
        executed_chunk = raw_chunk
        if adaptive:
            retiming = AdaptiveRetimer().retime(
                raw_chunk, left_target, right_target, stability=1.0
            )
            executed_chunk = retiming.chunk
        dynamics = _run_scale(
            executed_chunk,
            REFERENCE_EP0_STATE[14:16],
            scale=1.0,
            use_filter=True,
            use_joint_filter=True,
        )
        success = bool(
            dynamics["finite"]
            and dynamics["hard_command_limits_pass"]
            and dynamics["endpoint_error_m"] < 0.035
            and dynamics["minimum_pelvis_height_m"] > 0.70
            and dynamics["manipulation_contact_step_rate"] == 0.0
        )
        result = {
            **contract_metadata(verified=True),
            "mode": "adaptive" if adaptive else "baseline",
            "trajectory_source": "deterministic_g1_contract_fixture_not_vla",
            "action_frame": "pelvis",
            "action_semantics": "absolute_16d_eef_target",
            "g1_hardware_execution_enabled": False,
            "production_path_components": [
                "g1_policy_contract_validation",
                "phase_aware_preflight",
                "EEF_command_filter",
                "pelvis_to_world_boundary",
                "G1_dual_arm_IK",
                "14_joint_command_filter",
                "Dex1_command_mapping",
            ],
            "preflight": preflight,
            "retiming": None if retiming is None else {
                "path_actions_byte_identical": bool(
                    np.array_equal(retiming.chunk.actions, raw_chunk.actions)
                ),
                "original_duration_s": retiming.original_duration,
                "retimed_duration_s": retiming.retimed_duration,
                "scale_min": float(retiming.scale_profile.min()),
                "scale_max": float(retiming.scale_profile.max()),
            },
            "dynamics": dynamics,
            "success": success,
            "verdict": (
                "Frozen G1 EDU contract pipeline passed in MuJoCo. This proves integration only, not VLA task success or hardware safety."
                if success else
                "G1 EDU contract pipeline failed; do not advance toward hardware."
            ),
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="disable optional adaptive retiming")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(adaptive=not args.baseline, output=args.output)
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
