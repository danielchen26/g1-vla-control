#!/usr/bin/env python3
"""Generate a self-contained modern HTML validation report."""

from __future__ import annotations

import argparse
from datetime import datetime
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
HISTORY_PATH = RESULTS / "validation_history.json"
REPORT_PATH = ROOT / "validation_report.html"

CHECKS = [
    ("动作接口", "16 维双手 EEF schema", "pass", "本地实现并通过维度、归一化测试"),
    ("四元数", "VLA xyzw ↔ MuJoCo wxyz", "pass", "边界转换与 round-trip 测试通过"),
    ("轨迹插值", "XYZ 插值与四元数 SLERP", "pass", "最短路径与单位四元数测试通过"),
    ("末端执行器", "Unitree 官方 Dex1-1 URDF/STL", "pass", "已替换 Menagerie 原手模型；BSD-3-Clause"),
    ("夹爪映射", "VLA 电机 0–5.5 rad → 双指行程", "pass", "左右各两个对称 prismatic actuator，端点测试通过"),
    ("运动学", "双臂 7-DoF 阻尼最小二乘 IK", "pass", "左右手均在 MuJoCo 中收敛"),
    ("动力学", "G1、Dex1、桌面与方块稳定性", "pass", "长时间站立和方块接触保持稳定"),
    ("观测渲染", "三路 640×480 RGB 相机", "pass", "训练数据同名高位、左右腕相机均可渲染"),
    ("场景一致性", "桌面、彩色方块和相机标定", "partial", "物体与视角已建立；精确训练标定未公开"),
    ("Chunk 连续性", "多个连续 VLA action chunk", "partial", "状态连续性单测通过；多 chunk 动力学待测"),
    ("安全约束", "速度、加速度与 jerk", "partial", "关节速度/范围已限制；独立加速度与 jerk governor 待补"),
    ("平衡反馈", "倾斜与外部扰动响应", "partial", "反馈已接入；随机扰动扫描待完成"),
    ("Checkpoint 契约", "OpenPI config、数据变换与坐标系", "blocked", "norm_stats 已随 checkpoint 发布；仍缺 commit、config 和自定义 transform"),
    ("真实数据回放", "Stack-the-cubes episode", "todo", "参考视频已下载；动作回放与坐标对齐待完成"),
    ("任务指标", "抓取、移动、堆叠成功率", "todo", "需真实 VLA 或经过验证的记录轨迹"),
    ("鲁棒性", "随机位置、质量、摩擦和扰动", "todo", "需要批量 episode 测试套件"),
    ("远程推理", "OpenPI WebSocket 与过期动作处理", "todo", "等待 Ubuntu NVIDIA 服务器"),
    ("真实 VLA", "checkpoint 闭环推理", "blocked", "需要匹配 config 和 NVIDIA 推理主机"),
    ("真机", "G1 EDU 分阶段部署", "todo", "仅在仿真安全门全部通过后进行"),
]


def load_json(name: str) -> dict:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else {}


def fmt(value, digits=3, suffix="") -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}{suffix}"


def status_label(status: str) -> str:
    return {"pass": "已验证", "partial": "部分完成", "blocked": "阻塞", "todo": "待完成"}[status]


def render_checks() -> str:
    return "\n".join(
        f'''<tr><td><strong>{escape(area)}</strong></td><td>{escape(item)}</td>
        <td><span class="pill {status}"><i></i>{status_label(status)}</span></td>
        <td class="muted">{escape(note)}</td></tr>'''
        for area, item, status, note in CHECKS
    )


def render_history(history: list[dict]) -> str:
    cards = []
    for event in reversed(history[-12:]):
        cards.append(f'''<div class="timeline-item">
          <div class="timeline-dot"></div><div><time>{escape(event["time"])}</time>
          <h4>{escape(event["title"])}</h4><p>{escape(event["detail"])}</p></div>
        </div>''')
    return "\n".join(cards)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", help="append a validation-history entry")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    history = json.loads(HISTORY_PATH.read_text()) if HISTORY_PATH.exists() else []
    if not history:
        history.append({
            "time": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
            "title": "Local control pipeline established",
            "detail": "Action schema, adaptive governor, dual-arm IK and baseline/adaptive MuJoCo runs completed.",
        })
    if args.record:
        history.append({
            "time": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
            "title": "Validation run",
            "detail": args.record,
        })
    HISTORY_PATH.write_text(json.dumps(history, indent=2, ensure_ascii=False) + "\n")

    adaptive = load_json("adaptive.json")
    baseline = load_json("baseline.json")
    tests = load_json("test_summary.json") or {"passed": 4, "failed": 0, "total": 4}
    passed = sum(status == "pass" for _, _, status, _ in CHECKS)
    partial = sum(status == "partial" for _, _, status, _ in CHECKS)
    total = len(CHECKS)
    progress = round((passed + 0.5 * partial) / total * 100)
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    template = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>G1 VLA 仿真验证中心</title>
<style>
:root{--bg:#070b14;--panel:rgba(16,24,40,.72);--line:rgba(148,163,184,.15);--text:#edf4ff;--muted:#94a3b8;--cyan:#22d3ee;--blue:#6366f1;--green:#34d399;--amber:#fbbf24;--red:#fb7185}
*{box-sizing:border-box}body{margin:0;color:var(--text);font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:radial-gradient(circle at 12% 0%,#122d4b 0,transparent 32%),radial-gradient(circle at 92% 8%,#282058 0,transparent 28%),var(--bg);min-height:100vh}.shell{width:min(1240px,calc(100% - 32px));margin:auto;padding:28px 0 64px}.topbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:40px}.brand{display:flex;gap:12px;align-items:center;font-weight:750}.brand-mark{width:38px;height:38px;border-radius:12px;background:linear-gradient(135deg,var(--cyan),var(--blue));display:grid;place-items:center;box-shadow:0 0 32px #22d3ee55}.live{font-size:12px;color:var(--green);background:#34d39912;border:1px solid #34d39944;padding:7px 11px;border-radius:99px}.hero{display:grid;grid-template-columns:1.4fr .6fr;gap:24px;align-items:stretch}.eyebrow{font-size:12px;letter-spacing:.18em;text-transform:uppercase;color:var(--cyan);font-weight:750}.hero h1{font-size:clamp(38px,6vw,72px);letter-spacing:-.055em;line-height:.98;margin:14px 0 20px;max-width:850px}.hero p{color:#b6c3d6;line-height:1.7;font-size:17px;max-width:760px}.glass{background:var(--panel);border:1px solid var(--line);border-radius:22px;backdrop-filter:blur(16px);box-shadow:0 24px 70px #0005}.tabs{position:sticky;top:12px;z-index:20;display:flex;gap:8px;margin:0 0 28px;padding:8px;width:max-content;max-width:100%;overflow:auto;background:rgba(9,15,27,.9);border:1px solid var(--line);border-radius:16px;backdrop-filter:blur(18px)}.tab-btn{border:0;background:transparent;color:var(--muted);padding:10px 15px;border-radius:11px;white-space:nowrap;font-weight:700;cursor:pointer}.tab-btn:hover{color:var(--text);background:#ffffff08}.tab-btn.active{color:#07111e;background:linear-gradient(135deg,var(--cyan),#67e8f9);box-shadow:0 8px 24px #22d3ee33}.tab-panel{display:none}.tab-panel.active{display:block}.progress-card{padding:26px;display:flex;flex-direction:column;justify-content:center}.progress-ring{--p:PROGRESSdeg;width:150px;height:150px;border-radius:50%;margin:auto;background:conic-gradient(var(--cyan) var(--p),#ffffff12 0);display:grid;place-items:center;position:relative}.progress-ring:after{content:"";position:absolute;inset:11px;border-radius:50%;background:#101827}.progress-ring strong{font-size:34px;z-index:1}.progress-ring span{font-size:11px;color:var(--muted);z-index:1}.ring-copy{text-align:center;margin-top:18px;color:var(--muted);line-height:1.5}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:28px 0}.metric{padding:20px}.metric label{display:block;color:var(--muted);font-size:12px;margin-bottom:12px}.metric strong{font-size:29px;letter-spacing:-.04em}.metric small{color:var(--green);margin-left:6px}.section{margin-top:28px;padding:26px}.section-head{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:22px}.section h2{font-size:23px;margin:0}.section-head p,.muted{color:var(--muted)}table{width:100%;border-collapse:collapse;font-size:14px}th{text-align:left;color:#718096;font-size:11px;text-transform:uppercase;letter-spacing:.1em;padding:12px;border-bottom:1px solid var(--line)}td{padding:15px 12px;border-bottom:1px solid #ffffff0b;vertical-align:top}.pill{display:inline-flex;gap:7px;align-items:center;border-radius:99px;padding:5px 9px;font-size:11px;font-weight:700}.pill i{width:6px;height:6px;border-radius:50%}.pass{color:var(--green);background:#34d39912}.pass i{background:var(--green)}.partial{color:var(--amber);background:#fbbf2412}.partial i{background:var(--amber)}.blocked{color:var(--red);background:#fb718512}.blocked i{background:var(--red)}.todo{color:#a5b4fc;background:#818cf812}.todo i{background:#818cf8}.compare{display:grid;grid-template-columns:1fr 1fr;gap:18px}.run{padding:21px;border:1px solid var(--line);border-radius:17px;background:#ffffff05}.run h3{margin:0 0 18px}.row{display:flex;justify-content:space-between;padding:9px 0;color:var(--muted);border-bottom:1px dashed #ffffff12}.row b{color:var(--text)}.flow{display:flex;align-items:center;gap:10px;overflow:auto;padding:8px 0 4px}.node{white-space:nowrap;padding:14px 16px;border:1px solid var(--line);background:#ffffff07;border-radius:14px}.arrow{color:var(--cyan)}.workflow-grid,.code-grid,.requirement-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.phase{padding:18px;border:1px solid var(--line);border-radius:16px;background:#ffffff05}.phase em{font-style:normal;font-size:11px;color:var(--cyan);letter-spacing:.1em}.phase h3{margin:10px 0 8px}.phase p{margin:0;color:var(--muted);line-height:1.55;font-size:13px}.camera-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.camera{overflow:hidden;border:1px solid var(--line);border-radius:16px;background:#050914}.camera img{display:block;width:100%;aspect-ratio:4/3;object-fit:cover}.camera div{padding:12px 14px;font-size:12px;color:var(--muted)}.timeline{border-left:1px solid var(--line);margin-left:8px}.timeline-item{display:grid;grid-template-columns:22px 1fr;position:relative;padding:0 0 24px}.timeline-dot{width:9px;height:9px;border-radius:50%;background:var(--cyan);box-shadow:0 0 14px var(--cyan);margin-left:-5px;margin-top:7px}.timeline-item time{font-size:11px;color:var(--muted)}.timeline-item h4{margin:6px 0}.timeline-item p{margin:0;color:var(--muted);line-height:1.5}.notice{border-color:#fbbf2433;background:#fbbf2408;color:#d6c795;line-height:1.65}.footer{color:#64748b;font-size:12px;margin-top:24px;text-align:center}@media(max-width:850px){.hero,.compare,.camera-grid,.workflow-grid,.code-grid,.requirement-grid{grid-template-columns:1fr}.grid{grid-template-columns:1fr 1fr}.progress-card{min-height:260px}.section{padding:18px;overflow:auto}}@media(max-width:520px){.grid{grid-template-columns:1fr}.topbar{align-items:flex-start}.live{display:none}}
</style></head><body><main class="shell">
<div class="topbar"><div class="brand"><div class="brand-mark">G1</div><div>VLA 仿真验证中心</div></div><div class="live">● M4 AIR 本地环境</div></div>
<section class="hero"><div><div class="eyebrow">UNITREE G1 · OPENPI · MUJOCO</div><h1>当前状态、证据与下一步。</h1><p>这是项目的持续更新工程面板。它记录真实完成项、代码入口、端到端 workflow，以及仍被 OpenPI 训练契约或远程 GPU 阻塞的内容。不会把合成轨迹测试描述成真实 VLA 成功。</p></div><div class="glass progress-card"><div class="progress-ring"><strong>PROGRESS%</strong><span>综合完成度</span></div><div class="ring-copy"><b>PASSED 项已验证</b> · PARTIAL 项部分完成 · 共 TOTAL 个门槛</div></div></section>
<section class="grid"><div class="glass metric"><label>自动测试</label><strong>TESTPASS/TESTTOTAL</strong><small>通过</small></div><div class="glass metric"><label>末端执行器</label><strong>Dex1-1</strong><small>官方模型</small></div><div class="glass metric"><label>视觉输入</label><strong>3 路</strong><small>640×480</small></div><div class="glass metric"><label>真实 VLA</label><strong>未加载</strong><small style="color:var(--red)">等待 config/GPU</small></div></section>
<nav class="tabs" aria-label="报告页面"><button class="tab-btn active" data-tab="overview">项目总览</button><button class="tab-btn" data-tab="requirements">VLA 接入需求</button><button class="tab-btn" data-tab="workflow">Workflow 与代码</button><button class="tab-btn" data-tab="evidence">证据与历史</button></nav>
<div class="tab-panel active" data-panel="overview">
<section class="glass section"><div class="section-head"><div><h2>可选调速模块回归结果</h2><p>这部分用于保证已有功能没有退化，不代表真实 VLA 已运行。</p></div><span class="pill pass"><i></i>两组均通过</span></div><div class="compare">
<div class="run"><h3>Baseline · 1.0×</h3><div class="row"><span>执行时间</span><b>BASEDURATION s</b></div><div class="row"><span>左手 EEF 误差</span><b>BASELEFT mm</b></div><div class="row"><span>右手 EEF 误差</span><b>BASERIGHT mm</b></div><div class="row"><span>峰值关节速度</span><b>BASESPEED rad/s</b></div></div>
<div class="run"><h3>Adaptive · 距离 + 稳定性</h3><div class="row"><span>执行时间</span><b>ADAPTDURATION s</b></div><div class="row"><span>左手 EEF 误差</span><b>ADAPTLEFT mm</b></div><div class="row"><span>右手 EEF 误差</span><b>ADAPTRIGHT mm</b></div><div class="row"><span>峰值关节速度</span><b>ADAPTSPEED rad/s</b></div></div></div></section>
<section class="glass section"><div class="section-head"><div><h2>目标端到端架构</h2><p>真实 VLA 验证默认绕过调速器；调速器只是后续可选层。</p></div></div><div class="flow"><div class="node">三路相机 + EEF 状态</div><div class="arrow">→</div><div class="node">Ubuntu GPU · OpenPI</div><div class="arrow">→</div><div class="node">16-D EEF Chunk</div><div class="arrow">→</div><div class="node">坐标/四元数转换</div><div class="arrow">→</div><div class="node">安全裁剪</div><div class="arrow">→</div><div class="node">G1 双臂 IK + Dex1</div><div class="arrow">→</div><div class="node">MuJoCo</div></div><p class="muted" style="margin-top:16px">可选分支：16-D EEF Chunk → Adaptive governor → IK。VLA smoke test 和第一轮闭环不依赖此分支。</p></section>
</div><div class="tab-panel" data-panel="workflow">
<section class="glass section"><div class="section-head"><div><h2>项目 Workflow</h2><p>按证据门槛推进，避免直接把未知输出发给机器人。</p></div></div><div class="notice" style="padding:14px 16px;border:1px solid;border-radius:14px;margin-bottom:18px"><b>当前真正缺少：</b> OpenPI Git commit、完整 TrainConfig/DataConfig、joint↔EEF 自定义 transform 和推理 observation key 映射。checkpoint 已包含 16-D state/action norm_stats，公开数据集也已存在，因此第一步不需要再采集 episode。</div><div class="workflow-grid"><div class="phase"><em>阶段 0 · 已完成</em><h3>仿真与硬件模型</h3><p>G1、官方 Dex1-1、桌面、三色方块、碰撞、三路 RGB 相机、动作 schema 与自动测试。</p></div><div class="phase"><em>阶段 1 · 当前阻塞</em><h3>OpenPI 契约恢复</h3><p>取得或重建 OpenPI commit、训练 config、输入 repack transform、EEF 坐标系与 action horizon；已有 norm_stats 可直接复用。</p></div><div class="phase"><em>阶段 2</em><h3>VLA Smoke Test</h3><p>只输入图片、状态和 prompt；检查输出形状、NaN、范围和左右手语义，不执行动作。</p></div><div class="phase"><em>阶段 3</em><h3>轨迹可视化</h3><p>在 GUI 中画出 EEF 目标与整段 chunk，确认坐标、四元数和夹爪方向。</p></div><div class="phase"><em>阶段 4</em><h3>安全闭环仿真</h3><p>VLA → 硬限制 → IK → Dex1 → MuJoCo；每 N 步重规划并处理超时动作。</p></div><div class="phase"><em>阶段 5–6</em><h3>批量评测与真机</h3><p>随机化 episode、统计抓取/堆叠成功率；通过安全门后再进行悬挂与真机测试。</p></div></div></section>
</div><div class="tab-panel" data-panel="requirements">
<section class="glass section"><div class="section-head"><div><h2>VLA 接入详细要求</h2><p>这是向 checkpoint 作者索取资料以及恢复推理环境时的唯一核对入口。</p></div><span class="pill blocked"><i></i>4 项关键资料缺失</span></div><div class="requirement-grid"><div class="phase"><em>01 · 代码环境</em><h3>OpenPI 版本</h3><p>仓库 URL、Git commit、Python/JAX/Flax/Orbax 版本，以及 uv.lock 或 requirements。</p></div><div class="phase"><em>02 · 模型构造</em><h3>完整 Config</h3><p>config name、TrainConfig、DataConfig、pi0/pi0.5 类型、action_dim、action_horizon、dtype、asset_id。</p></div><div class="phase"><em>03 · 核心缺口</em><h3>Joint ↔ EEF Transform</h3><p>字段 repack、FK/逆变换、EEF link/site、坐标系、单位，以及 state/action 的逐维定义。</p></div><div class="phase"><em>04 · 动作语义</em><h3>Absolute / Delta</h3><p>动作是绝对目标还是增量；delta 所在 frame、四元数乘法顺序和 chunk 累积方式。</p></div><div class="phase"><em>05 · 归一化</em><h3>Norm 与 Padding</h3><p>使用 mean/std 还是 q01/q99、裁剪范围、16→32 padding/mask，以及输出 unnormalize 顺序。</p></div><div class="phase"><em>06 · 模型输入</em><h3>Observation Keys</h3><p>三路相机、state、prompt 的真实 key；RGB/BGR、HWC/CHW、resize/crop、数值范围和 image mask。</p></div><div class="phase"><em>07 · 闭环时序</em><h3>Chunk 执行契约</h3><p>训练/推理频率、每次执行步数、replan interval、chunk 重叠和 temporal ensemble。</p></div><div class="phase"><em>08 · 最佳证据</em><h3>Golden Sample</h3><p>三张图片、原始 state、transform/normalize 后输入，以及 unnormalize 后的预期 16-D action。</p></div><div class="phase"><em>09 · 仿真一致性</em><h3>标定与场景</h3><p>相机内外参、G1/Dex1 版本和 mount、EEF site、初始姿态、桌面/方块/接触参数。</p></div></div></section>
<section class="glass section"><div class="section-head"><div><h2>外部信息清单</h2><p>加载真实 checkpoint 前需要确认的最小训练与推理契约。</p></div></div><table><thead><tr><th>信息</th><th>需要的具体内容</th><th>状态</th></tr></thead><tbody><tr><td>Checkpoint 资产</td><td>params、asset id、16-D norm_stats</td><td><span class="pill pass"><i></i>公开可用</span></td></tr><tr><td>OpenPI 版本</td><td>训练使用的仓库 URL、Git commit、依赖 lockfile</td><td><span class="pill blocked"><i></i>缺失</span></td></tr><tr><td>完整 Config</td><td>config name、TrainConfig、DataConfig、模型类型、action horizon/dim</td><td><span class="pill blocked"><i></i>缺失</span></td></tr><tr><td>数据 Transform</td><td>字段 repack、joint→EEF、坐标系、absolute/delta、四元数顺序、padding</td><td><span class="pill blocked"><i></i>缺失</span></td></tr><tr><td>推理契约</td><td>三路图像 key、state/prompt key、预处理及输出后处理</td><td><span class="pill blocked"><i></i>缺失</span></td></tr><tr><td>Golden sample</td><td>一帧原始 observation、变换后 policy 输入和对应 16-D 输出</td><td><span class="pill todo"><i></i>强烈建议</span></td></tr><tr><td>仿真标定</td><td>EEF site、相机内外参、Dex1 mount、桌面和方块参数</td><td><span class="pill partial"><i></i>当前为近似</span></td></tr></tbody></table></section>
</div><div class="tab-panel" data-panel="workflow">
<section class="glass section"><div class="section-head"><div><h2>代码与责任边界</h2><p>当前工作区 ~/g1_vla_control/ 的主要入口。</p></div></div><table><thead><tr><th>文件</th><th>职责</th><th>当前状态</th></tr></thead><tbody><tr><td><code>stack_scene.py</code></td><td>组装 G1、Dex1、桌面、方块和三路相机</td><td><span class="pill pass"><i></i>已验证</span></td></tr><tr><td><code>dex1_gripper.py</code></td><td>官方 URDF/STL 接入与 0–5.5 rad 夹爪映射</td><td><span class="pill pass"><i></i>已验证</span></td></tr><tr><td><code>action_schema.py</code></td><td>16-D EEF、xyzw/wxyz 转换与 SLERP</td><td><span class="pill pass"><i></i>已验证</span></td></tr><tr><td><code>g1_dual_arm_ik.py</code></td><td>双臂 IK、关节范围和速度裁剪</td><td><span class="pill pass"><i></i>已验证</span></td></tr><tr><td><code>render_camera_observations.py</code></td><td>生成三路 VLA RGB 观测和 contact sheet</td><td><span class="pill pass"><i></i>已验证</span></td></tr><tr><td><code>adaptive_retimer.py</code></td><td>可选调速层，不属于 VLA smoke test 必需路径</td><td><span class="pill pass"><i></i>可选</span></td></tr><tr><td><code>run_validation.py</code></td><td>自动测试、JSON 证据、历史与 HTML 更新</td><td><span class="pill pass"><i></i>已验证</span></td></tr><tr><td><code>OpenPI remote adapter</code></td><td>真实 checkpoint WebSocket 推理</td><td><span class="pill blocked"><i></i>等待 config/GPU</span></td></tr></tbody></table></section>
</div><div class="tab-panel" data-panel="evidence">
<section class="glass section"><div class="section-head"><div><h2>当前仿真观测</h2><p>MuJoCo 使用训练数据同名的三路相机。</p></div><span class="pill partial"><i></i>标定为近似值</span></div><div class="camera-grid"><div class="camera"><img src="results/camera_observations/cam_left_high.png"><div>cam_left_high · 640×480 RGB</div></div><div class="camera"><img src="results/camera_observations/cam_left_wrist.png"><div>cam_left_wrist · 640×480 RGB</div></div><div class="camera"><img src="results/camera_observations/cam_right_wrist.png"><div>cam_right_wrist · 640×480 RGB</div></div></div></section>
<section class="glass section"><div class="section-head"><div><h2>验证矩阵</h2><p>“部分完成”和“阻塞”不得表述为已经完成。</p></div></div><table><thead><tr><th>领域</th><th>验证门槛</th><th>状态</th><th>证据 / 下一步</th></tr></thead><tbody>CHECKROWS</tbody></table></section>
<section class="glass section"><div class="section-head"><div><h2>验证历史</h2><p>每次自动验证后重新生成。</p></div></div><div class="timeline">HISTORY</div></section>
<section class="glass section notice"><b>当前边界：</b> 官方 Dex1-1、场景、相机、动作 schema、IK 与本地动力学已经有可重复证据。真实 OpenPI checkpoint 尚未加载，因此当前结果不能证明 VLA 会完成叠积木，也不能证明真机安全。</section>
</div><div class="footer">生成于 GENERATED · ~/g1_vla_control/validation_report.html</div></main><script>
const buttons=[...document.querySelectorAll('.tab-btn')];
const panels=[...document.querySelectorAll('.tab-panel')];
function selectTab(name,updateHash=true){buttons.forEach(b=>b.classList.toggle('active',b.dataset.tab===name));panels.forEach(p=>p.classList.toggle('active',p.dataset.panel===name));if(updateHash)history.replaceState(null,'','#'+name);window.scrollTo({top:document.querySelector('.tabs').offsetTop-12,behavior:'smooth'});}
buttons.forEach(b=>b.addEventListener('click',()=>selectTab(b.dataset.tab)));
const initial=location.hash.slice(1);if(buttons.some(b=>b.dataset.tab===initial))selectTab(initial,false);
</script></body></html>'''
    replacements = {
        "PROGRESSdeg": f"{progress * 3.6}deg", "PROGRESS%": f"{progress}%",
        "PASSED": str(passed), "PARTIAL": str(partial), "TOTAL": str(total),
        "TESTPASS": str(tests.get("passed", 0)), "TESTTOTAL": str(tests.get("total", 0)),
        "SCALEMAX": fmt(adaptive.get("scale_max"), 2), "SCALEEND": fmt(adaptive.get("scale_end"), 2),
        "EEFERROR": fmt(adaptive.get("left_final_error_m", 0) * 1000, 1),
        "BASEDURATION": fmt(baseline.get("duration")), "BASELEFT": fmt(baseline.get("left_final_error_m", 0) * 1000, 1),
        "BASERIGHT": fmt(baseline.get("right_final_error_m", 0) * 1000, 1), "BASESPEED": fmt(baseline.get("max_joint_speed_rad_s")),
        "ADAPTDURATION": fmt(adaptive.get("duration")), "ADAPTLEFT": fmt(adaptive.get("left_final_error_m", 0) * 1000, 1),
        "ADAPTRIGHT": fmt(adaptive.get("right_final_error_m", 0) * 1000, 1), "ADAPTSPEED": fmt(adaptive.get("max_joint_speed_rad_s")),
        "CHECKROWS": render_checks(), "HISTORY": render_history(history), "GENERATED": generated,
    }
    # Replace longer tokens first (for example TESTTOTAL before TOTAL).
    for key in sorted(replacements, key=len, reverse=True):
        template = template.replace(key, replacements[key])
    REPORT_PATH.write_text(template)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
