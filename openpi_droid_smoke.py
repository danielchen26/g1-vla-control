#!/usr/bin/env python3
"""Output-only smoke auditor for the official OpenPI pi05_droid server.

This program never imports the G1 controller and never executes returned actions.
Mock mode validates the transport/audit contract without claiming neural inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import time
from typing import Any, Callable, Protocol

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
AUDITED_OPENPI_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
MODEL_CONFIG = "pi05_droid"
MODEL_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_droid"


class PolicyClient(Protocol):
    def infer(self, observation: dict[str, Any]) -> dict[str, Any]: ...
    def get_server_metadata(self) -> dict[str, Any]: ...


class MockDroidPolicy:
    """Deterministic protocol stub. It is deliberately not called a VLA."""

    def __init__(self, horizon: int = 10):
        self.horizon = horizon
        self.call_index = 0

    def get_server_metadata(self) -> dict[str, Any]:
        return {
            "mode": "deterministic_mock",
            "model_config": MODEL_CONFIG,
            "neural_checkpoint_loaded": False,
        }

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        del observation
        base = np.linspace(-0.2, 0.2, self.horizon * 8, dtype=np.float32)
        actions = base.reshape(self.horizon, 8) + self.call_index * 1e-3
        self.call_index += 1
        return {
            "actions": actions,
            "server_timing": {"infer_ms": 0.25},
            "policy_timing": {"model_ms": 0.20},
        }


def make_synthetic_droid_observation(
    prompt: str = "stack the colored cubes",
) -> dict[str, Any]:
    """Build the exact public DROID example schema with deterministic values."""
    y, x = np.indices((224, 224))
    exterior = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)
    wrist = np.stack(((x + 31) % 256, (2 * y) % 256, (x // 2) % 256), axis=-1).astype(np.uint8)
    return {
        "observation/exterior_image_1_left": exterior,
        "observation/wrist_image_left": wrist,
        "observation/joint_position": np.zeros(7, dtype=np.float32),
        "observation/gripper_position": np.array([0.5], dtype=np.float32),
        "prompt": prompt,
    }


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


def _infer_with_timeout(
    client: PolicyClient,
    observation: dict[str, Any],
    timeout_ms: float,
) -> dict[str, Any]:
    """Enforce a hard inference deadline on the supported macOS/Linux clients."""
    if timeout_ms <= 0:
        raise ValueError("timeout_ms must be positive")
    if not hasattr(signal, "setitimer"):
        return client.infer(observation)

    def timeout_handler(signum, frame):
        del signum, frame
        raise TimeoutError(f"policy inference exceeded {timeout_ms:.0f} ms")

    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_ms / 1000.0)
    try:
        return client.infer(observation)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _latency_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.quantile(array, 0.50)),
        "p95_ms": float(np.quantile(array, 0.95)),
        "p99_ms": float(np.quantile(array, 0.99)),
        "max_ms": float(array.max()),
    }


def run_smoke_audit(
    client: PolicyClient,
    observation_factory: Callable[[], dict[str, Any]],
    calls: int = 20,
    warmup_calls: int = 2,
    stale_threshold_ms: float = 1000.0,
    call_timeout_ms: float = 5000.0,
    evidence_mode: str = "real_remote_vla",
) -> dict[str, Any]:
    if calls < 1 or warmup_calls < 0:
        raise ValueError("calls must be positive and warmup_calls non-negative")
    if stale_threshold_ms <= 0 or call_timeout_ms <= 0:
        raise ValueError("stale threshold and call timeout must be positive")
    warmup_errors: list[str] = []
    for _ in range(warmup_calls):
        try:
            _infer_with_timeout(client, observation_factory(), call_timeout_ms)
        except Exception as exc:
            warmup_errors.append(f"{type(exc).__name__}: {exc}")
            break

    records: list[dict[str, Any]] = []
    shapes: set[tuple[int, ...]] = set()
    latencies: list[float] = []
    fingerprints: list[str] = []
    for index in range(calls):
        observation = observation_factory()
        request_wall_ns = time.time_ns()
        request_monotonic_ns = time.monotonic_ns()
        try:
            response = _infer_with_timeout(client, observation, call_timeout_ms)
            response_monotonic_ns = time.monotonic_ns()
            response_wall_ns = time.time_ns()
            latency_ms = (response_monotonic_ns - request_monotonic_ns) / 1e6
            actions = np.asarray(response.get("actions"))
            finite = bool(actions.size and np.all(np.isfinite(actions)))
            valid_rank = actions.ndim == 2
            valid_action_dim = valid_rank and actions.shape[1] == 8
            shape = tuple(int(value) for value in actions.shape)
            shapes.add(shape)
            latencies.append(latency_ms)
            fingerprint = hashlib.sha256(actions.tobytes()).hexdigest()
            fingerprints.append(fingerprint)
            records.append({
                "call": index,
                "request_wall_time_ns": request_wall_ns,
                "response_wall_time_ns": response_wall_ns,
                "latency_ms": latency_ms,
                "stale": latency_ms > stale_threshold_ms,
                "action_shape": list(shape),
                "finite": finite,
                "valid_rank": valid_rank,
                "valid_droid_action_dim": valid_action_dim,
                "action_min": float(np.min(actions)) if actions.size else None,
                "action_max": float(np.max(actions)) if actions.size else None,
                "action_sha256": fingerprint,
                "server_timing": _json_safe(response.get("server_timing", {})),
                "policy_timing": _json_safe(response.get("policy_timing", {})),
                "error": None,
            })
        except Exception as exc:  # Preserve evidence before returning failure.
            response_monotonic_ns = time.monotonic_ns()
            latency_ms = (response_monotonic_ns - request_monotonic_ns) / 1e6
            latencies.append(latency_ms)
            records.append({
                "call": index,
                "request_wall_time_ns": request_wall_ns,
                "response_wall_time_ns": time.time_ns(),
                "latency_ms": latency_ms,
                "stale": True,
                "action_shape": [],
                "finite": False,
                "valid_rank": False,
                "valid_droid_action_dim": False,
                "action_sha256": None,
                "error": f"{type(exc).__name__}: {exc}",
            })

    valid_calls = sum(
        record["finite"] and record["valid_rank"] and record["valid_droid_action_dim"]
        and record["error"] is None
        for record in records
    )
    stale_calls = sum(bool(record["stale"]) for record in records)
    passed = valid_calls == calls and len(shapes) == 1 and not warmup_errors
    is_real = evidence_mode == "real_remote_vla"
    return {
        "scope": (
            "Output-only OpenPI pi05_droid policy-server smoke test. Returned DROID "
            "actions are never mapped to or executed by G1."
        ),
        "evidence_mode": evidence_mode,
        "neural_vla_claimed": is_real,
        "g1_execution_enabled": False,
        "g1_action_compatible": False,
        "openpi": {
            "audited_commit": AUDITED_OPENPI_COMMIT,
            "config": MODEL_CONFIG,
            "checkpoint": MODEL_CHECKPOINT,
            "server_metadata": _json_safe(client.get_server_metadata()),
        },
        "observation_contract": {
            "source": "deterministic synthetic DROID-shaped observation",
            "image_keys": [
                "observation/exterior_image_1_left",
                "observation/wrist_image_left",
            ],
            "joint_position_dim": 7,
            "gripper_position_dim": 1,
            "warning": "This is a transport/shape smoke input, not a G1 observation contract.",
        },
        "summary": {
            "passed": passed,
            "calls": calls,
            "valid_calls": valid_calls,
            "stale_calls": stale_calls,
            "stale_threshold_ms": stale_threshold_ms,
            "call_timeout_ms": call_timeout_ms,
            "warmup_errors": warmup_errors,
            "unique_action_shapes": [list(shape) for shape in sorted(shapes)],
            "unique_chunk_fingerprints": len(set(fingerprints)),
            "latency": _latency_stats(latencies),
        },
        "calls": records,
        "verdict": (
            "Protocol and evidence contract passed in mock mode; no neural model was loaded."
            if not is_real and passed else
            "Real remote neural inference passed output-only validation; DROID actions remain unsafe for G1."
            if is_real and passed else
            "Smoke validation failed; inspect per-call errors and do not execute actions."
        ),
    }


def _real_client(
    host: str,
    port: int,
    api_key: str | None,
    connect_timeout_s: float,
) -> PolicyClient:
    try:
        from openpi_client import websocket_client_policy
    except ImportError as exc:
        raise RuntimeError(
            "openpi-client is not installed. Install it from the pinned OpenPI "
            "packages/openpi-client directory; see OPENPI_DROID_SMOKE.md."
        ) from exc
    # The official client retries refused connections indefinitely. Fail fast
    # here so automation produces a clear error rather than hanging forever.
    probe_host = host.removeprefix("ws://").removeprefix("wss://").split("/")[0]
    with socket.create_connection((probe_host, port), timeout=connect_timeout_s):
        pass
    return websocket_client_policy.WebsocketClientPolicy(
        host=host, port=port, api_key=api_key
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--warmup-calls", type=int, default=2)
    parser.add_argument("--stale-threshold-ms", type=float, default=1000.0)
    parser.add_argument("--call-timeout-ms", type=float, default=5000.0)
    parser.add_argument("--connect-timeout-s", type=float, default=5.0)
    parser.add_argument("--prompt", default="stack the colored cubes")
    parser.add_argument("--api-key-env", default="OPENPI_API_KEY")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.mock:
        client: PolicyClient = MockDroidPolicy()
        mode = "deterministic_mock_transport"
        default_output = RESULTS / "openpi_droid_smoke_mock.json"
    else:
        client = _real_client(
            args.host, args.port, os.environ.get(args.api_key_env) or None,
            args.connect_timeout_s,
        )
        mode = "real_remote_vla"
        default_output = RESULTS / "openpi_droid_smoke_real.json"
    report = run_smoke_audit(
        client,
        lambda: make_synthetic_droid_observation(args.prompt),
        calls=args.calls,
        warmup_calls=args.warmup_calls,
        stale_threshold_ms=args.stale_threshold_ms,
        call_timeout_ms=args.call_timeout_ms,
        evidence_mode=mode,
    )
    output = args.output or default_output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(output)
    print(json.dumps(report["summary"], indent=2))
    if not report["summary"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
