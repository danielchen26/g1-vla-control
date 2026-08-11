#!/usr/bin/env python3
"""Run every locally feasible audit and regenerate the evidence dashboard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(*command: str) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    python = sys.executable
    run(python, "dataset_contract_audit.py")
    run(python, "checkpoint_metadata_audit.py")
    run(python, "render_camera_observations.py")
    run(python, "visual_fidelity_audit.py")
    run(python, "retiming_safety_validation.py")
    run(
        python, "comprehensive_sim_validation.py",
        "--ik-samples", "250", "--random-trials", "20",
    )
    run(python, "-m", "unittest", "discover", "-s", "tests", "-v")
    run(
        python, "generate_report.py", "--record",
        "完成 comprehensive deep validation：100 episodes/95,966 帧数据审计、"
        "episode-0 全轨迹回放、250 个真实数据 IK 目标、1000 组宽域工作空间压力测试、"
        "Dex1 指尖/抓持、5 档 Gate-A.2 EEF/Joint 安全调速、168 个唯一 target 的 phase-aware preflight、"
        "30 秒动力学、20 组随机化、5 档外部扰动、三路视觉域对照及 Orbax metadata 审计。",
    )
    print(f"Deep-validation report: {ROOT / 'validation_report.html'}")


if __name__ == "__main__":
    main()
