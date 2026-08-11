# OpenPI π0.5-DROID · output-only remote smoke test

This stage proves that a real neural OpenPI server can be loaded and queried. It does **not** claim that DROID actions are compatible with Unitree G1.

## Evidence boundary

| Item | Meaning |
|---|---|
| `--mock` | Deterministic protocol/evidence regression only; no neural model |
| Real server | Genuine π0.5-DROID inference, but with synthetic DROID-shaped observations |
| Returned action | DROID 8-D action chunk; log/audit only |
| G1 execution | Hard-disabled by architecture; this client imports no G1 controller |

The audited upstream OpenPI revision is:

```text
15a9616a00943ada6c20a0f158e3adb39df2ccac
```

Official model selection:

```text
config:     pi05_droid
checkpoint: gs://openpi-assets/checkpoints/pi05_droid
```

## 1. Ubuntu NVIDIA server

OpenPI documents Ubuntu 22.04 and more than 8 GB of GPU memory for inference. Pin the audited revision:

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
git checkout 15a9616a00943ada6c20a0f158e3adb39df2ccac
GIT_LFS_SKIP_SMUDGE=1 uv sync
uv run scripts/serve_policy.py --env DROID
```

The official `DROID` default resolves to `pi05_droid` and `gs://openpi-assets/checkpoints/pi05_droid`. The server listens on port 8000 by default.

Do not expose port 8000 directly to the public Internet. Prefer an SSH tunnel from the MuJoCo machine:

```bash
ssh -N -L 8000:127.0.0.1:8000 USER@GPU_HOST
```

## 2. Client dependency

Install the small official client from the same pinned checkout into the client environment:

```bash
python -m pip install -e /path/to/openpi/packages/openpi-client
```

No OpenPI training/JAX dependencies are required on the MuJoCo client.

## 3. Local contract regression

This is safe to run anywhere and never claims neural inference:

```bash
python openpi_droid_smoke.py --mock --calls 20
```

Evidence:

```text
results/openpi_droid_smoke_mock.json
```

## 4. Real output-only smoke

With the SSH tunnel active:

```bash
python openpi_droid_smoke.py \
  --host 127.0.0.1 \
  --port 8000 \
  --calls 30 \
  --warmup-calls 2 \
  --stale-threshold-ms 1000 \
  --call-timeout-ms 5000 \
  --connect-timeout-s 5
```

Evidence:

```text
results/openpi_droid_smoke_real.json
```

The report records:

- server metadata;
- request/response wall and monotonic timestamps;
- P50/P95/P99 round-trip latency;
- stale-call count plus hard per-call and connection deadlines;
- action shape, finite checks, range and SHA-256 per chunk;
- server and policy timing fields;
- explicit `g1_execution_enabled: false` and `g1_action_compatible: false`.

## Observation contract

The first real smoke intentionally uses the exact public DROID example schema:

```text
observation/exterior_image_1_left  uint8 [224,224,3]
observation/wrist_image_left       uint8 [224,224,3]
observation/joint_position         float [7]
observation/gripper_position       float [1]
prompt                             string
```

The values are deterministic synthetic inputs. This isolates model loading, transport, output structure and latency from unresolved G1 semantics.

## Pass gate

A real run passes only when all calls:

1. return one stable rank-2 chunk shape;
2. use DROID action dimension 8;
3. contain no NaN/Inf;
4. return without protocol errors;
5. preserve output-only isolation.

Passing this gate proves a real OpenPI VLA was queried. It still does not permit mapping or executing the 8-D DROID actions on G1. The next model stage is a separately named `pi05_base` G1 LoRA with an owned 16-D EEF contract.
