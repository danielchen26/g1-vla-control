#!/usr/bin/env python3
"""Run local validation, save machine-readable evidence, and rebuild HTML."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def run(command: list[str]) -> subprocess.CompletedProcess:
    print("+", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    tests = run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    output = (tests.stdout or "") + (tests.stderr or "")
    match = re.search(r"Ran (\d+) tests?", output)
    test_count = int(match.group(1)) if match else 0
    test_summary = {
        "passed": test_count if tests.returncode == 0 else 0,
        "failed": 0 if tests.returncode == 0 else 1,
        "total": test_count,
        "success": tests.returncode == 0,
    }
    (RESULTS / "test_summary.json").write_text(
        json.dumps(test_summary, indent=2) + "\n"
    )
    if tests.returncode != 0:
        run([sys.executable, "generate_report.py", "--record", "Unit tests failed; dynamics runs skipped."])
        raise SystemExit(tests.returncode)

    baseline = run([
        sys.executable, "run_simulation.py", "--baseline", "--output", "results/baseline.json"
    ])
    adaptive = run([
        sys.executable, "run_simulation.py", "--output", "results/adaptive.json"
    ])
    cameras = run([sys.executable, "render_camera_observations.py"])
    success = (
        baseline.returncode == 0
        and adaptive.returncode == 0
        and cameras.returncode == 0
    )
    detail = (
        f"{test_count}/{test_count} 项自动测试、G1 EDU contract baseline/adaptive 全链路和三路相机渲染全部通过。"
        if success else
        "一个或多个动力学测试失败，请检查 JSON 证据和控制台输出。"
    )
    run([sys.executable, "generate_report.py", "--record", detail])
    if not success:
        raise SystemExit(1)
    print(f"Report updated: {ROOT / 'validation_report.html'}")


if __name__ == "__main__":
    main()
