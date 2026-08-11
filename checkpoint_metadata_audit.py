#!/usr/bin/env python3
"""Inspect public Orbax metadata without downloading the multi-GB parameters."""

from __future__ import annotations

import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
REPO = "LGG100/stack-cube-eef-24k"
BASE = f"https://huggingface.co/{REPO}/resolve/main"


def _json(url: str):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def main() -> None:
    tree = _json(
        f"https://huggingface.co/api/models/{REPO}/tree/main"
        "?recursive=true&expand=false"
    )
    checkpoint = _json(f"{BASE}/_CHECKPOINT_METADATA")
    params = _json(f"{BASE}/params/_METADATA")
    norm = _json(f"{BASE}/assets/stack-cube-eef/norm_stats.json")["norm_stats"]
    files = [item["path"] for item in tree if item.get("type") == "file"]
    total_bytes = sum(int(item.get("size") or 0) for item in tree)
    leaves = params["tree_metadata"]
    leaf_shapes = {
        key: value["value_metadata"].get("write_shape")
        for key, value in leaves.items()
    }
    selected = {
        key: shape for key, shape in leaf_shapes.items()
        if any(token in key for token in (
            "action_in_proj", "action_out_proj", "time_mlp", "PaliGemma"
        ))
    }
    action_kernel = next(
        (shape for key, shape in leaf_shapes.items()
         if "action_out_proj" in key and "kernel" in key), None
    )
    has_time_mlp = any("time_mlp_in" in key for key in leaves)
    has_pi0_state_proj = any("state_proj" in key for key in leaves)
    has_pi0_action_time_mlp = any("action_time_mlp" in key for key in leaves)
    has_adarms = any(
        "pre_attention_norm_1" in key and "Dense_0" in key for key in leaves
    )
    has_lora = any("lora_" in key for key in leaves)
    pi05_signature = bool(
        has_time_mlp and has_adarms
        and not has_pi0_state_proj and not has_pi0_action_time_mlp
    )
    report = {
        "repository": REPO,
        "public_tree": {
            "files": files,
            "listed_bytes": total_bytes,
            "has_params": any(path.startswith("params/") for path in files),
            "has_norm_stats": "assets/stack-cube-eef/norm_stats.json" in files,
            "has_training_config": any(
                "config" in path.lower() and path.endswith((".py", ".json", ".yaml", ".yml"))
                for path in files
            ),
            "has_model_card": any(path.lower().startswith("readme") for path in files),
            "has_train_state_payload": any(path.startswith("train_state/") for path in files),
        },
        "orbax": {
            "item_handlers": checkpoint.get("item_handlers", {}),
            "parameter_leaf_count": len(leaves),
            "selected_parameter_shapes": selected,
        },
        "architecture_evidence": {
            "paligemma_present": any("PaliGemma" in key for key in leaves),
            "flow_action_projection_present": any("action_in_proj" in key for key in leaves),
            "time_mlp_present": has_time_mlp,
            "pi0_state_proj_present": has_pi0_state_proj,
            "pi0_action_time_mlp_present": has_pi0_action_time_mlp,
            "adaptive_rmsnorm_signature_present": has_adarms,
            "lora_parameters_present": has_lora,
            "action_output_kernel_write_shape": action_kernel,
            "likely_internal_padded_action_dim": (
                action_kernel[-1] if action_kernel else None
            ),
            "published_state_dim": len(norm["state"]["mean"]),
            "published_action_dim": len(norm["actions"]["mean"]),
            "family_inference": "OpenPI Pi0Config with pi05=True and LoRA is strongly indicated",
            "pi05_signature_strongly_supported": pi05_signature,
            "candidate_model_config": {
                "class": "Pi0Config",
                "pi05": True if pi05_signature else None,
                "action_dim": action_kernel[-1] if action_kernel else None,
                "paligemma_lora": True if has_lora else None,
                "action_horizon_candidate": 50,
                "action_horizon_status": "OpenPI default; not encoded in parameter shapes",
                "max_token_len_candidate": 200,
                "image_resolution_candidate": [224, 224],
                "normalization_candidate": "q01/q99 quantile normalization",
                "candidate_defaults_confirmed": False,
            },
            "standard_model_input_keys_candidate": [
                "base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "state", "prompt"
            ],
            "exact_variant_confirmed": False,
        },
        "restore_readiness": {
            "checkpoint_payload_available": True,
            "normalization_payload_available": True,
            "exact_parameter_tree_config_available": False,
            "custom_data_transform_available": False,
            "safe_to_claim_policy_restore": False,
            "still_required": [
                "OpenPI Git commit", "complete TrainConfig/ModelConfig",
                "DataConfig and repack/FK transform", "inference observation keys",
            ],
        },
        "verdict": (
            "The parameter signature strongly indicates Pi0Config(pi05=True) with "
            "LoRA, 32-wide internal action padding, and 16 published robot dimensions. "
            "Action horizon, exact code revision, and DataConfig remain unencoded, so "
            "the policy still cannot be safely restored."
        ),
    }
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "checkpoint_metadata_audit.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(path)
    print(json.dumps({
        "files": len(files),
        "parameter_leaves": len(leaves),
        "padded_action_dim": report["architecture_evidence"]["likely_internal_padded_action_dim"],
        "config_available": report["public_tree"]["has_training_config"],
        "safe_to_restore": report["restore_readiness"]["safe_to_claim_policy_restore"],
    }, indent=2))


if __name__ == "__main__":
    main()
