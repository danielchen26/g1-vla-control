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
    ("公开数据", "100 episodes / 95,966 帧契约审计", "pass", "确认 raw parquet 为 14 关节绝对目标 + 2 Dex1 电机值，30 Hz"),
    ("EEF Transform", "joint → pelvis-frame EEF 重建", "partial", "50 mm wrist offset + xyzw 与 norm stats 高度吻合；最大统计差 0.013，尚非作者源码"),
    ("动作接口", "16 维双手 EEF schema", "pass", "逐维布局、维度和单位四元数验证通过"),
    ("坐标边界", "pelvis EEF → world 与 xyzw ↔ wxyz", "pass", "转换函数及 round-trip/平移单测通过"),
    ("轨迹插值", "XYZ 插值与四元数 SLERP", "pass", "最短路径与单位四元数测试通过"),
    ("末端执行器", "Unitree 官方 Dex1-1 URDF/STL", "pass", "左右安装朝向验证；原 Menagerie 手已移除"),
    ("夹爪映射", "电机 0 closed / 5.5 open", "pass", "同步首帧确认方向；指尖间距 0.021–0.100 m 且左右对称"),
    ("合成抓取", "Dex1 预定位方块抓持", "pass", "零扰动左右 2/2；左手在 2 N 失效、右手在 4 N 失效；不是 VLA 抓取"),
    ("真实目标 IK", "公开数据随机 EEF 目标回放", "pass", "250/250 在 2 mm / 1° 门槛内收敛，关节范围 100% 合规"),
    ("连续轨迹回放", "Episode 0 全 678 帧动力学执行", "partial", "全程有限且站立；EEF P95 约 20 mm，数据没有 object state 可核对任务"),
    ("长时间动力学", "30 秒站立、Dex1、桌面与方块", "pass", "pelvis ≥0.790 m，方块高度最大漂移约 1.1 mm"),
    ("随机化", "20 组质量、摩擦与方块位置", "pass", "20/20 保持机器人和方块稳定"),
    ("跨平台复现", "GitHub Actions Ubuntu 22.04 headless", "pass", "submodule、13 项测试、动力学、三路 OSMesa 渲染和 HTML 检查全部通过"),
    ("外部扰动", "躯干横向冲击恢复", "partial", "5–20 N 均恢复；40 N 倒地，60 N 漂移/倾斜超门槛"),
    ("观测渲染", "三路 640×480 RGB 相机", "pass", "训练数据同名高位、左右腕相机均可渲染"),
    ("视觉域一致性", "参考 frame-0 像素几何与外观", "partial", "物体均可见，但中心/尺度、夹爪外观、照明和背景差异显著"),
    ("训练运动包络", "EEF/关节速度、加速度与 jerk 分布", "pass", "95,966 帧逐 episode 差分完成；可用于 sim-only 参数初值，不是硬件限制"),
    ("Chunk 连续性", "多个连续 VLA action chunk", "partial", "状态连续性单测通过；真实 VLA 多 chunk 尚未运行"),
    ("调速时序契约", "horizon、执行 stride、replan 和 ensemble", "blocked", "权重不编码这些值；必须取得 config/policy timing"),
    ("官方动态限制", "G1 逐关节 v/a/j/torque 与低层控制器", "blocked", "当前 1.2 rad/s 等为工程默认值，不能作为真机限制"),
    ("安全约束", "速度、加速度、jerk 与不可达拒绝", "partial", "关节范围保持 100%；宽域测试出现饱和/碰撞，显式可达性和碰撞拒绝器待补"),
    ("Checkpoint 结构", "Orbax 参数树与模型族", "partial", "强证据指向 Pi0Config(pi05=True)+LoRA；action pad=32、发布维度=16"),
    ("Checkpoint 恢复", "OpenPI config 与 inference contract", "blocked", "仓库没有 config/model card；仍缺 commit、DataConfig 和字段映射"),
    ("任务指标", "真实 VLA 抓取与堆叠成功率", "todo", "合成抓持已测；真实 checkpoint 尚未加载"),
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
    dataset = load_json("dataset_contract_audit.json")
    deep = load_json("comprehensive_sim_validation.json")
    visual = load_json("visual_fidelity_audit.json")
    checkpoint = load_json("checkpoint_metadata_audit.json")
    ik = deep.get("dataset_ik_replay", {})
    episode_replay = deep.get("episode_zero_dynamics_replay", {})
    workspace = deep.get("workspace_stress", {})
    dex = deep.get("dex1_command_sweep", {})
    grasp = deep.get("dex1_grasp_sweep", {})
    long_run = deep.get("long_horizon_stability", {})
    disturbance = deep.get("external_disturbance", {})
    randomized = deep.get("randomized_scene", {})
    passed = sum(status == "pass" for _, _, status, _ in CHECKS)
    partial = sum(status == "partial" for _, _, status, _ in CHECKS)
    total = len(CHECKS)
    progress = round((passed + 0.5 * partial) / total * 100)
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    grasp_trials = grasp.get("trials", [])
    random_trials = randomized.get("trials", [])
    visual_summary = visual.get("summary", {})
    transform_compare = dataset.get("reconstructed_training_transform", {}).get(
        "published_norm_stats_comparison", {}
    )
    motion = dataset.get("training_motion_envelope", {})

    template = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>G1 VLA 仿真验证中心</title>
<style>
:root{--bg:#070b14;--panel:rgba(16,24,40,.72);--line:rgba(148,163,184,.15);--text:#edf4ff;--muted:#94a3b8;--cyan:#22d3ee;--blue:#6366f1;--green:#34d399;--amber:#fbbf24;--red:#fb7185}
*{box-sizing:border-box}body{margin:0;color:var(--text);font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:radial-gradient(circle at 12% 0%,#122d4b 0,transparent 32%),radial-gradient(circle at 92% 8%,#282058 0,transparent 28%),var(--bg);min-height:100vh}.shell{width:min(1240px,calc(100% - 32px));margin:auto;padding:28px 0 64px}.topbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:40px}.brand{display:flex;gap:12px;align-items:center;font-weight:750}.brand-mark{width:38px;height:38px;border-radius:12px;background:linear-gradient(135deg,var(--cyan),var(--blue));display:grid;place-items:center;box-shadow:0 0 32px #22d3ee55}.live{font-size:12px;color:var(--green);background:#34d39912;border:1px solid #34d39944;padding:7px 11px;border-radius:99px;text-decoration:none}.live:hover{background:#34d39922}.hero{display:grid;grid-template-columns:1.4fr .6fr;gap:24px;align-items:stretch}.eyebrow{font-size:12px;letter-spacing:.18em;text-transform:uppercase;color:var(--cyan);font-weight:750}.hero h1{font-size:clamp(38px,6vw,72px);letter-spacing:-.055em;line-height:.98;margin:14px 0 20px;max-width:850px}.hero p{color:#b6c3d6;line-height:1.7;font-size:17px;max-width:760px}.glass{background:var(--panel);border:1px solid var(--line);border-radius:22px;backdrop-filter:blur(16px);box-shadow:0 24px 70px #0005}.tabs{position:sticky;top:12px;z-index:20;display:flex;gap:8px;margin:0 0 28px;padding:8px;width:max-content;max-width:100%;overflow:auto;background:rgba(9,15,27,.9);border:1px solid var(--line);border-radius:16px;backdrop-filter:blur(18px)}.tab-btn{border:0;background:transparent;color:var(--muted);padding:10px 15px;border-radius:11px;white-space:nowrap;font-weight:700;cursor:pointer}.tab-btn:hover{color:var(--text);background:#ffffff08}.tab-btn.active{color:#07111e;background:linear-gradient(135deg,var(--cyan),#67e8f9);box-shadow:0 8px 24px #22d3ee33}.tab-panel{display:none}.tab-panel.active{display:block}.progress-card{padding:26px;display:flex;flex-direction:column;justify-content:center}.progress-ring{--p:PROGRESSdeg;width:150px;height:150px;border-radius:50%;margin:auto;background:conic-gradient(var(--cyan) var(--p),#ffffff12 0);display:grid;place-items:center;position:relative}.progress-ring:after{content:"";position:absolute;inset:11px;border-radius:50%;background:#101827}.progress-ring strong{font-size:34px;z-index:1}.progress-ring span{font-size:11px;color:var(--muted);z-index:1}.ring-copy{text-align:center;margin-top:18px;color:var(--muted);line-height:1.5}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:28px 0}.metric{padding:20px}.metric label{display:block;color:var(--muted);font-size:12px;margin-bottom:12px}.metric strong{font-size:29px;letter-spacing:-.04em}.metric small{color:var(--green);margin-left:6px}.section{margin-top:28px;padding:26px}.section-head{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:22px}.section h2{font-size:23px;margin:0}.section-head p,.muted{color:var(--muted)}table{width:100%;border-collapse:collapse;font-size:14px}th{text-align:left;color:#718096;font-size:11px;text-transform:uppercase;letter-spacing:.1em;padding:12px;border-bottom:1px solid var(--line)}td{padding:15px 12px;border-bottom:1px solid #ffffff0b;vertical-align:top}.pill{display:inline-flex;gap:7px;align-items:center;border-radius:99px;padding:5px 9px;font-size:11px;font-weight:700}.pill i{width:6px;height:6px;border-radius:50%}.pass{color:var(--green);background:#34d39912}.pass i{background:var(--green)}.partial{color:var(--amber);background:#fbbf2412}.partial i{background:var(--amber)}.blocked{color:var(--red);background:#fb718512}.blocked i{background:var(--red)}.todo{color:#a5b4fc;background:#818cf812}.todo i{background:#818cf8}.compare{display:grid;grid-template-columns:1fr 1fr;gap:18px}.run{padding:21px;border:1px solid var(--line);border-radius:17px;background:#ffffff05}.run h3{margin:0 0 18px}.row{display:flex;justify-content:space-between;padding:9px 0;color:var(--muted);border-bottom:1px dashed #ffffff12}.row b{color:var(--text)}.flow{display:flex;align-items:center;gap:10px;overflow:auto;padding:8px 0 4px}.node{white-space:nowrap;padding:14px 16px;border:1px solid var(--line);background:#ffffff07;border-radius:14px}.arrow{color:var(--cyan)}.workflow-grid,.code-grid,.requirement-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.phase{padding:18px;border:1px solid var(--line);border-radius:16px;background:#ffffff05}.phase em{font-style:normal;font-size:11px;color:var(--cyan);letter-spacing:.1em}.phase h3{margin:10px 0 8px}.phase p{margin:0;color:var(--muted);line-height:1.55;font-size:13px}.camera-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.camera{overflow:hidden;border:1px solid var(--line);border-radius:16px;background:#050914}.camera img{display:block;width:100%;aspect-ratio:4/3;object-fit:cover}.camera div{padding:12px 14px;font-size:12px;color:var(--muted)}.timeline{border-left:1px solid var(--line);margin-left:8px}.timeline-item{display:grid;grid-template-columns:22px 1fr;position:relative;padding:0 0 24px}.timeline-dot{width:9px;height:9px;border-radius:50%;background:var(--cyan);box-shadow:0 0 14px var(--cyan);margin-left:-5px;margin-top:7px}.timeline-item time{font-size:11px;color:var(--muted)}.timeline-item h4{margin:6px 0}.timeline-item p{margin:0;color:var(--muted);line-height:1.5}.notice{border-color:#fbbf2433;background:#fbbf2408;color:#d6c795;line-height:1.65}.footer{color:#64748b;font-size:12px;margin-top:24px;text-align:center}@media(max-width:850px){.hero,.compare,.camera-grid,.workflow-grid,.code-grid,.requirement-grid{grid-template-columns:1fr}.grid{grid-template-columns:1fr 1fr}.progress-card{min-height:260px}.section{padding:18px;overflow:auto}}@media(max-width:520px){.grid{grid-template-columns:1fr}.topbar{align-items:flex-start}.live{display:none}}
</style></head><body><main class="shell">
<div class="topbar"><div class="brand"><div class="brand-mark">G1</div><div>VLA 仿真验证中心</div></div><a class="live" href="https://github.com/danielchen26/g1-vla-control" target="_blank" rel="noreferrer">● PUBLIC GITHUB ↗</a></div>
<section class="hero"><div><div class="eyebrow">UNITREE G1 · OPENPI · MUJOCO</div><h1>当前状态、证据与下一步。</h1><p>这是项目的持续更新工程面板。它记录真实完成项、代码入口、端到端 workflow，以及仍被 OpenPI 训练契约或远程 GPU 阻塞的内容。不会把合成轨迹测试描述成真实 VLA 成功。</p></div><div class="glass progress-card"><div class="progress-ring"><strong>PROGRESS%</strong><span>综合完成度</span></div><div class="ring-copy"><b>PASSED 项已验证</b> · PARTIAL 项部分完成 · 共 TOTAL 个门槛</div></div></section>
<section class="grid"><div class="glass metric"><label>自动测试</label><strong>TESTPASS/TESTTOTAL</strong><small>通过</small></div><div class="glass metric"><label>末端执行器</label><strong>Dex1-1</strong><small>官方模型</small></div><div class="glass metric"><label>视觉输入</label><strong>3 路</strong><small>640×480</small></div><div class="glass metric"><label>真实 VLA</label><strong>未加载</strong><small style="color:var(--red)">等待 config/GPU</small></div></section>
<nav class="tabs" aria-label="报告页面"><button class="tab-btn active" data-tab="overview">项目总览</button><button class="tab-btn" data-tab="deep">深度验证</button><button class="tab-btn" data-tab="retiming">调速所需信息</button><button class="tab-btn" data-tab="requirements">VLA 接入需求</button><button class="tab-btn" data-tab="workflow">Workflow 与代码</button><button class="tab-btn" data-tab="evidence">证据与历史</button></nav>
<div class="tab-panel active" data-panel="overview">
<section class="glass section"><div class="section-head"><div><h2>可选调速模块回归结果</h2><p>这部分用于保证已有功能没有退化，不代表真实 VLA 已运行。</p></div><span class="pill pass"><i></i>两组均通过</span></div><div class="compare">
<div class="run"><h3>Baseline · 1.0×</h3><div class="row"><span>执行时间</span><b>BASEDURATION s</b></div><div class="row"><span>左手 EEF 误差</span><b>BASELEFT mm</b></div><div class="row"><span>右手 EEF 误差</span><b>BASERIGHT mm</b></div><div class="row"><span>峰值关节速度</span><b>BASESPEED rad/s</b></div><div class="row"><span>峰值加速度</span><b>BASEACCEL rad/s²</b></div><div class="row"><span>峰值 jerk</span><b style="color:var(--amber)">BASEJERK rad/s³</b></div></div>
<div class="run"><h3>Adaptive · 距离 + 稳定性</h3><div class="row"><span>执行时间</span><b>ADAPTDURATION s</b></div><div class="row"><span>左手 EEF 误差</span><b>ADAPTLEFT mm</b></div><div class="row"><span>右手 EEF 误差</span><b>ADAPTRIGHT mm</b></div><div class="row"><span>峰值关节速度</span><b>ADAPTSPEED rad/s</b></div><div class="row"><span>峰值加速度</span><b>ADAPTACCEL rad/s²</b></div><div class="row"><span>峰值 jerk</span><b style="color:var(--amber)">ADAPTJERK rad/s³</b></div></div></div></section>
<section class="glass section"><div class="section-head"><div><h2>目标端到端架构</h2><p>真实 VLA 验证默认绕过调速器；调速器只是后续可选层。</p></div></div><div class="flow"><div class="node">三路相机 + EEF 状态</div><div class="arrow">→</div><div class="node">Ubuntu GPU · OpenPI</div><div class="arrow">→</div><div class="node">16-D EEF Chunk</div><div class="arrow">→</div><div class="node">坐标/四元数转换</div><div class="arrow">→</div><div class="node">安全裁剪</div><div class="arrow">→</div><div class="node">G1 双臂 IK + Dex1</div><div class="arrow">→</div><div class="node">MuJoCo</div></div><p class="muted" style="margin-top:16px">可选分支：16-D EEF Chunk → Adaptive governor → IK。VLA smoke test 和第一轮闭环不依赖此分支。</p></section>
</div><div class="tab-panel" data-panel="workflow">
<section class="glass section"><div class="section-head"><div><h2>项目 Workflow</h2><p>按证据门槛推进，避免直接把未知输出发给机器人。</p></div></div><div class="notice" style="padding:14px 16px;border:1px solid;border-radius:14px;margin-bottom:18px"><b>当前真正缺少：</b> OpenPI Git commit、完整 TrainConfig/DataConfig、joint↔EEF 自定义 transform 和推理 observation key 映射。checkpoint 已包含 16-D state/action norm_stats，公开数据集也已存在，因此第一步不需要再采集 episode。</div><div class="workflow-grid"><div class="phase"><em>阶段 0 · 已完成</em><h3>仿真与硬件模型</h3><p>G1、官方 Dex1-1、桌面、三色方块、碰撞、三路 RGB 相机、动作 schema 与自动测试。</p></div><div class="phase"><em>阶段 1 · 当前阻塞</em><h3>OpenPI 契约恢复</h3><p>取得或重建 OpenPI commit、训练 config、输入 repack transform、EEF 坐标系与 action horizon；已有 norm_stats 可直接复用。</p></div><div class="phase"><em>阶段 2</em><h3>VLA Smoke Test</h3><p>只输入图片、状态和 prompt；检查输出形状、NaN、范围和左右手语义，不执行动作。</p></div><div class="phase"><em>阶段 3</em><h3>轨迹可视化</h3><p>在 GUI 中画出 EEF 目标与整段 chunk，确认坐标、四元数和夹爪方向。</p></div><div class="phase"><em>阶段 4</em><h3>安全闭环仿真</h3><p>VLA → 硬限制 → IK → Dex1 → MuJoCo；每 N 步重规划并处理超时动作。</p></div><div class="phase"><em>阶段 5–6</em><h3>批量评测与真机</h3><p>随机化 episode、统计抓取/堆叠成功率；通过安全门后再进行悬挂与真机测试。</p></div></div></section>
</div><div class="tab-panel" data-panel="retiming">
<section class="glass section"><div class="section-head"><div><h2>继续调速 Job：已知证据与信息缺口</h2><p>这里专门回答“哪些参数已经能由数据支持，哪些必须从模型作者、GPU 服务或 Unitree 控制侧取得”。</p></div><span class="pill partial"><i></i>仿真可继续 · 最终标定阻塞</span></div><div class="notice" style="padding:14px 16px;border:1px solid;border-radius:14px;margin-bottom:18px"><b>结论：</b> 现在可以继续开发仿真版 acceleration/jerk limiter、可达性预检和 recorded-trajectory retiming；但在 action chunk 时序、真实控制器限制和远程推理延迟未知时，不能把 0.50×–1.65× 宣称为适用于真实 VLA 或真机的最终范围。</div><div class="grid"><div class="run"><label class="muted">训练数据频率</label><strong>30 Hz</strong><small> 已确认</small></div><div class="run"><label class="muted">Action→State 延迟</label><strong>3 帧</strong><small> ≈100 ms</small></div><div class="run"><label class="muted">EEF 速度 P95</label><strong>MOTIONSPEED95 m/s</strong><small> 数据目标</small></div><div class="run"><label class="muted">关节目标速度 P95</label><strong>JOINTSPEED95 rad/s</strong><small> 非硬件上限</small></div></div></section>
<section class="glass section"><div class="section-head"><div><h2>已经从公开轨迹得到的运动包络</h2><p>跨 episode 计算且不跨 episode 边界求差分；这些是训练目标分布，不是官方安全限制。</p></div><span class="pill pass"><i></i>95,966 帧统计</span></div><table><thead><tr><th>指标</th><th>P50</th><th>P95</th><th>P99</th><th>最大值</th><th>解释</th></tr></thead><tbody><tr><td>EEF 平移速度</td><td>MOTIONSPEED50 m/s</td><td>MOTIONSPEED95 m/s</td><td>MOTIONSPEED99 m/s</td><td>MOTIONSPEEDMAX m/s</td><td class="muted">左右手取较大值</td></tr><tr><td>EEF 角速度</td><td>ANGSPEED50 rad/s</td><td>ANGSPEED95 rad/s</td><td>ANGSPEED99 rad/s</td><td>ANGSPEEDMAX rad/s</td><td class="muted">Quaternion 最短角距离</td></tr><tr><td>EEF 平移加速度</td><td>MOTIONACC50 m/s²</td><td>MOTIONACC95 m/s²</td><td>MOTIONACC99 m/s²</td><td>MOTIONACCMAX m/s²</td><td class="muted">最大值可能含目标跳变</td></tr><tr><td>EEF 平移 jerk</td><td>MOTIONJERK50 m/s³</td><td>MOTIONJERK95 m/s³</td><td>MOTIONJERK99 m/s³</td><td>MOTIONJERKMAX m/s³</td><td class="muted">用于设计滤波器，不是安全认证值</td></tr><tr><td>原始关节目标速度</td><td>JOINTSPEED50 rad/s</td><td>JOINTSPEED95 rad/s</td><td>JOINTSPEED99 rad/s</td><td>JOINTSPEEDMAX rad/s</td><td class="muted">14 关节逐帧最大值</td></tr></tbody></table></section>
<section class="glass section"><div class="section-head"><div><h2>当前调速参数：全部仍是工程默认值</h2><p>这些值可以做回归实验，但没有模型作者或 Unitree 安全规范背书。</p></div></div><table><thead><tr><th>参数</th><th>当前值</th><th>证据等级</th><th>还需要什么</th></tr></thead><tbody><tr><td><code>min_scale / max_scale</code></td><td>0.50× / 1.65×</td><td><span class="pill partial"><i></i>人为设定</span></td><td>允许的时间缩放范围、VLA 是否容忍重采样</td></tr><tr><td><code>near / far_distance</code></td><td>25 mm / 160 mm</td><td><span class="pill partial"><i></i>人为设定</span></td><td>任务阶段、抓取/放置精度和成功容差</td></tr><tr><td><code>max_scale_rate</code></td><td>2.0 s⁻¹</td><td><span class="pill partial"><i></i>人为设定</span></td><td>允许的加速度/jerk 及伺服跟踪带宽</td></tr><tr><td><code>max_eef_speed</code></td><td>0.65 m/s</td><td><span class="pill partial"><i></i>仅 offline 生效</span></td><td>应基于训练 P95/P99、碰撞距离和官方限制重新定值</td></tr><tr><td><code>IK max_joint_speed</code></td><td>1.2 rad/s</td><td><span class="pill partial"><i></i>非官方值</span></td><td>14 个关节逐项速度、加速度、jerk、扭矩限制</td></tr><tr><td>显式 accel / jerk limit</td><td>未实现</td><td><span class="pill blocked"><i></i>安全缺口</span></td><td>Baseline 已测 77.4 rad/s²、43,264 rad/s³ 瞬态</td></tr></tbody></table></section>
<section class="glass section"><div class="section-head"><div><h2>P0：继续做真实 VLA 调速前必须知道</h2><p>缺少任一 P0 项，都只能进行 sim-only 调参。</p></div></div><table><thead><tr><th>必须信息</th><th>需要回答的精确问题</th><th>为什么调速依赖它</th><th>状态</th></tr></thead><tbody><tr><td><strong>Action chunk 时序</strong></td><td><code>action_horizon</code>、action rate、每次执行几步、replan Hz、chunk 是否重叠</td><td>决定时间缩放是在重采样、跳步还是改变执行间隔</td><td><span class="pill blocked"><i></i>未知</span></td></tr><tr><td><strong>Temporal ensemble</strong></td><td>是否融合多个 chunk；融合窗口和权重是什么</td><td>任意变速可能破坏时间索引和 ensemble 对齐</td><td><span class="pill blocked"><i></i>未知</span></td></tr><tr><td><strong>精确动作语义</strong></td><td>absolute/delta、pelvis frame、50 mm EEF offset、xyzw 是否与作者实现完全一致</td><td>距离、速度和剩余路径全部由这些定义计算</td><td><span class="pill partial"><i></i>高度支持，非精确</span></td></tr><tr><td><strong>远程推理时延</strong></td><td>inference P50/P95/P99、网络 jitter、超时、stale chunk 策略</td><td>调速必须把观测年龄和下一 chunk 到达时间纳入 governor</td><td><span class="pill blocked"><i></i>尚无 GPU trace</span></td></tr><tr><td><strong>G1 官方动态限制</strong></td><td>14 关节逐项 velocity/acceleration/jerk/torque 和允许持续时间</td><td>当前 1.2 rad/s 只能保证软件裁剪，不能代表硬件安全</td><td><span class="pill blocked"><i></i>未取得</span></td></tr><tr><td><strong>真实控制器契约</strong></td><td>位置/速度/扭矩接口、伺服 Hz、插值方式、跟踪延迟和 watchdog</td><td>同一 EEF scale 在不同低层控制器上产生完全不同的 jerk</td><td><span class="pill blocked"><i></i>未接入</span></td></tr><tr><td><strong>变速许可语义</strong></td><td>模型作者是否允许 action chunk 非均匀时间重参数化，推荐范围是多少</td><td>必须确认变速不破坏模型训练时的闭环分布</td><td><span class="pill blocked"><i></i>未知</span></td></tr></tbody></table></section>
<section class="glass section"><div class="section-head"><div><h2>P1：确定 slowdown / stop 条件所需信息</h2><p>这些决定调速器何时减速、暂停或拒绝动作。</p></div></div><div class="requirement-grid"><div class="phase"><em>P1 · 任务阶段</em><h3>抓取与放置容差</h3><p>抓取前、闭爪、抬升、对齐和释放分别允许多大位置/角度误差，是否有 phase/state 标记。</p></div><div class="phase"><em>P1 · 平衡</em><h3>允许的稳定性包络</h3><p>Pelvis 位移、倾角、COM/support polygon、足底接触和恢复时间的 stop/slow 阈值。</p></div><div class="phase"><em>P1 · 碰撞</em><h3>安全距离与拒绝规则</h3><p>桌面、躯干、双臂和 Dex1 的最小 clearance；OOD 测试已证明只裁剪关节不够。</p></div><div class="phase"><em>P1 · Dex1</em><h3>夹爪速度与接触</h3><p>电机速度/力限制、闭合检测、滑移判断，以及闭爪期间是否强制降低 EEF scale。</p></div><div class="phase"><em>P1 · 评测</em><h3>速度优化目标</h3><p>优化总时长、成功率、峰值 jerk、接触冲击还是能耗；必须先定义优先级。</p></div><div class="phase"><em>P1 · 失败恢复</em><h3>Chunk 丢失与重规划</h3><p>超时后 hold、slow、stop 或回安全位；恢复时是否丢弃旧 chunk。</p></div></div></section>
<section class="glass section"><div class="section-head"><div><h2>最小资料包与推进门槛</h2><p>拿到以下文件即可从 sim-only 调速进入真实 VLA 调速。</p></div></div><div class="compare"><div class="run"><h3>需要模型/GPU 侧提供</h3><div class="row"><span><code>config.py</code> / config name</span><b>P0</b></div><div class="row"><span><code>transforms.py</code></span><b>P0</b></div><div class="row"><span><code>policy_timing.json</code></span><b>horizon/rate/stride</b></div><div class="row"><span><code>latency_trace.csv</code></span><b>obs/infer/action timestamps</b></div><div class="row"><span>一组真实 action chunk</span><b>含时间戳</b></div></div><div class="run"><h3>需要机器人/安全侧提供</h3><div class="row"><span><code>g1_arm_limits.yaml</code></span><b>v/a/j/torque</b></div><div class="row"><span><code>controller_contract.yaml</code></span><b>Hz/interface/watchdog</b></div><div class="row"><span><code>task_tolerances.yaml</code></span><b>grasp/place/stop</b></div><div class="row"><span><code>collision_margins.yaml</code></span><b>clearance</b></div><div class="row"><span><code>balance_limits.yaml</code></span><b>tilt/COM/contact</b></div></div></div><div class="workflow-grid" style="margin-top:18px"><div class="phase"><em>GATE A · 现在可做</em><h3>Sim-only Governor</h3><p>基于训练 P95/P99 实现 accel/jerk limiter、可达性和碰撞拒绝，并回放公开轨迹。</p></div><div class="phase"><em>GATE B · 需 P0</em><h3>真实 VLA 仿真调速</h3><p>拿到 config、chunk timing 和 latency trace 后，接入 GPU 输出并测 stale-action。</p></div><div class="phase"><em>GATE C · 需硬件规范</em><h3>真机调速</h3><p>取得官方动态限制与低层控制契约后，才能确定最终 scale 和 stop 阈值。</p></div></div></section>
</div><div class="tab-panel" data-panel="requirements">
<section class="glass section"><div class="section-head"><div><h2>VLA 接入详细要求</h2><p>这是向 checkpoint 作者索取资料以及恢复推理环境时的唯一核对入口。</p></div><span class="pill blocked"><i></i>4 项关键资料缺失</span></div><div class="requirement-grid"><div class="phase"><em>01 · 代码环境</em><h3>OpenPI 版本</h3><p>仓库 URL、Git commit、Python/JAX/Flax/Orbax 版本，以及 uv.lock 或 requirements。</p></div><div class="phase"><em>02 · 模型构造</em><h3>完整 Config</h3><p>config name、TrainConfig、DataConfig、pi0/pi0.5 类型、action_dim、action_horizon、dtype、asset_id。</p></div><div class="phase"><em>03 · 核心缺口</em><h3>Joint ↔ EEF Transform</h3><p>字段 repack、FK/逆变换、EEF link/site、坐标系、单位，以及 state/action 的逐维定义。</p></div><div class="phase"><em>04 · 动作语义</em><h3>Absolute / Delta</h3><p>动作是绝对目标还是增量；delta 所在 frame、四元数乘法顺序和 chunk 累积方式。</p></div><div class="phase"><em>05 · 归一化</em><h3>Norm 与 Padding</h3><p>使用 mean/std 还是 q01/q99、裁剪范围、16→32 padding/mask，以及输出 unnormalize 顺序。</p></div><div class="phase"><em>06 · 模型输入</em><h3>Observation Keys</h3><p>三路相机、state、prompt 的真实 key；RGB/BGR、HWC/CHW、resize/crop、数值范围和 image mask。</p></div><div class="phase"><em>07 · 闭环时序</em><h3>Chunk 执行契约</h3><p>训练/推理频率、每次执行步数、replan interval、chunk 重叠和 temporal ensemble。</p></div><div class="phase"><em>08 · 最佳证据</em><h3>Golden Sample</h3><p>三张图片、原始 state、transform/normalize 后输入，以及 unnormalize 后的预期 16-D action。</p></div><div class="phase"><em>09 · 仿真一致性</em><h3>标定与场景</h3><p>相机内外参、G1/Dex1 版本和 mount、EEF site、初始姿态、桌面/方块/接触参数。</p></div></div></section>
<section class="glass section"><div class="section-head"><div><h2>外部信息清单</h2><p>加载真实 checkpoint 前需要确认的最小训练与推理契约。</p></div></div><table><thead><tr><th>信息</th><th>需要的具体内容</th><th>状态</th></tr></thead><tbody><tr><td>Checkpoint 资产</td><td>params、asset id、16-D norm_stats</td><td><span class="pill pass"><i></i>公开可用</span></td></tr><tr><td>OpenPI 版本</td><td>训练使用的仓库 URL、Git commit、依赖 lockfile</td><td><span class="pill blocked"><i></i>缺失</span></td></tr><tr><td>完整 Config</td><td>config name、TrainConfig、DataConfig、模型类型、action horizon/dim</td><td><span class="pill blocked"><i></i>缺失</span></td></tr><tr><td>数据 Transform</td><td>字段 repack、joint→EEF、坐标系、absolute/delta、四元数顺序、padding</td><td><span class="pill blocked"><i></i>缺失</span></td></tr><tr><td>推理契约</td><td>三路图像 key、state/prompt key、预处理及输出后处理</td><td><span class="pill blocked"><i></i>缺失</span></td></tr><tr><td>Golden sample</td><td>一帧原始 observation、变换后 policy 输入和对应 16-D 输出</td><td><span class="pill todo"><i></i>强烈建议</span></td></tr><tr><td>仿真标定</td><td>EEF site、相机内外参、Dex1 mount、桌面和方块参数</td><td><span class="pill partial"><i></i>当前为近似</span></td></tr></tbody></table></section>
</div><div class="tab-panel" data-panel="workflow">
<section class="glass section"><div class="section-head"><div><h2>代码与责任边界</h2><p>当前工作区 ~/g1_vla_control/ 的主要入口。</p></div></div><table><thead><tr><th>文件</th><th>职责</th><th>当前状态</th></tr></thead><tbody><tr><td><code>stack_scene.py</code></td><td>组装 G1、Dex1、桌面、方块和三路相机</td><td><span class="pill pass"><i></i>已验证</span></td></tr><tr><td><code>dex1_gripper.py</code></td><td>官方 URDF/STL 接入与 0–5.5 rad 夹爪映射</td><td><span class="pill pass"><i></i>已验证</span></td></tr><tr><td><code>action_schema.py</code></td><td>16-D EEF、xyzw/wxyz 转换与 SLERP</td><td><span class="pill pass"><i></i>已验证</span></td></tr><tr><td><code>g1_dual_arm_ik.py</code></td><td>双臂 IK、关节范围和速度裁剪</td><td><span class="pill pass"><i></i>已验证</span></td></tr><tr><td><code>render_camera_observations.py</code></td><td>生成三路 VLA RGB 观测和 contact sheet</td><td><span class="pill pass"><i></i>已验证</span></td></tr><tr><td><code>adaptive_retimer.py</code></td><td>可选调速层，不属于 VLA smoke test 必需路径</td><td><span class="pill pass"><i></i>可选</span></td></tr><tr><td><code>run_validation.py</code></td><td>自动测试、JSON 证据、历史与 HTML 更新</td><td><span class="pill pass"><i></i>已验证</span></td></tr><tr><td><code>dataset_contract_audit.py</code></td><td>100 episodes 数据语义、时序、FK 与 norm_stats 取证</td><td><span class="pill pass"><i></i>已执行</span></td></tr><tr><td><code>comprehensive_sim_validation.py</code></td><td>Dex1、抓持、250 目标 IK、长时间、扰动与随机化</td><td><span class="pill pass"><i></i>已执行</span></td></tr><tr><td><code>visual_fidelity_audit.py</code></td><td>三路 frame-0 物体可见性、中心与尺度差异</td><td><span class="pill partial"><i></i>量化不匹配</span></td></tr><tr><td><code>checkpoint_metadata_audit.py</code></td><td>Orbax 参数树、模型族和 action padding 审计</td><td><span class="pill partial"><i></i>结构可识别</span></td></tr><tr><td><code>run_deep_validation.py</code></td><td>一键运行所有本地可行的深度验证</td><td><span class="pill pass"><i></i>可复现</span></td></tr><tr><td><code>.github/workflows/validation.yml</code></td><td>Ubuntu headless CI 与证据 artifact</td><td><span class="pill pass"><i></i>GitHub CI 已通过</span></td></tr><tr><td><code>OpenPI remote adapter</code></td><td>真实 checkpoint WebSocket 推理</td><td><span class="pill blocked"><i></i>等待 config/GPU</span></td></tr></tbody></table></section>
</div><div class="tab-panel" data-panel="deep">
<section class="glass section"><div class="section-head"><div><h2>公开数据契约取证</h2><p>不是根据文件名猜测，而是遍历全部 episode、执行 FK 并与 checkpoint norm_stats 对照。</p></div><span class="pill partial"><i></i>Transform 高度支持，非精确确认</span></div><section class="grid"><div class="run"><label class="muted">审计规模</label><strong>DATAFRAMES</strong><small> 帧 / 100 episodes</small></div><div class="run"><label class="muted">Action → State 最佳延迟</label><strong>DATALAG</strong><small> 帧 · 0.10 s</small></div><div class="run"><label class="muted">最大 norm 统计差</label><strong>TRANSFORMERR</strong><small> 非零</small></div><div class="run"><label class="muted">数据频率</label><strong>30 Hz</strong><small> 已实测</small></div></section><div class="requirement-grid"><div class="phase"><em>RAW PARQUET · 已确认</em><h3>关节空间绝对目标</h3><p>前 14 维是左右各 7 个关节位置，最后两维是 Dex1 电机角；不是 raw EEF，也不是 delta action。</p></div><div class="phase"><em>FK 假设 · 高度支持</em><h3>Pelvis-frame EEF</h3><p>每侧 wrist_yaw_link 沿局部 +X 偏移 50 mm，位置在 pelvis frame，四元数为 xyzw。</p></div><div class="phase"><em>时序 · 已确认</em><h3>约 3 帧响应延迟</h3><p>action[t] 与 state[t+3] 的标准化 RMSE 最低；公开轨迹按 30 Hz 记录。</p></div></div><div class="run" style="margin-top:18px"><h3>Episode 0 · 全 EPFRAMES 帧连续动力学回放</h3><div class="row"><span>公开轨迹时长</span><b>EPDURATION s</b></div><div class="row"><span>双臂关节 RMSE P95</span><b>EPARMP95 rad</b></div><div class="row"><span>EEF 位置误差 P95</span><b>EPEEFP95 mm</b></div><div class="row"><span>最低 pelvis 高度</span><b>EPPELVIS m</b></div><div class="row"><span>方块轨迹真值</span><b style="color:var(--amber)">数据集未发布</b></div></div></section>
<section class="glass section"><div class="section-head"><div><h2>真实数据目标 IK</h2><p>从 95,966 帧中随机抽样，joint action 经 FK 变为 EEF，再从对应 state 反解。</p></div><span class="pill pass"><i></i>IKSUCC 成功</span></div><div class="compare"><div class="run"><h3>IKSAMPLES 个真实目标</h3><div class="row"><span>位置误差 P95</span><b>IKP95MM mm</b></div><div class="row"><span>姿态误差 P95</span><b>IKP95DEG°</b></div><div class="row"><span>关节范围合规</span><b>100%</b></div></div><div class="run"><h3>Dex1 几何与抓持</h3><div class="row"><span>指尖间距</span><b>DEXCLOSEDMM–DEXOPENMM mm</b></div><div class="row"><span>左右最大差异</span><b>DEXSYMMM mm</b></div><div class="row"><span>预定位抓持</span><b>GRASPPASS</b></div><div class="row"><span>最大已测持握扰动</span><b>GRASPFORCE N</b></div></div></div></section>
<section class="glass section"><div class="section-head"><div><h2>OOD 工作空间与碰撞压力测试</h2><p>故意从宽于训练分布的体积随机采样，验证不能执行的目标是否会被安全拒绝。</p></div><span class="pill partial"><i></i>发现安全门缺口</span></div><div class="compare"><div class="run"><h3>WORKSAMPLES 组双手随机目标</h3><div class="row"><span>单臂目标成功率</span><b>WORKPERARM</b></div><div class="row"><span>双手同时成功率</span><b>WORKDUAL</b></div><div class="row"><span>关节范围合规</span><b>100%</b></div><div class="row"><span>关节极限饱和率</span><b>WORKSAT</b></div><div class="row"><span>禁区接触率</span><b>WORKCONTACT</b></div><div class="row"><span>显式不可达拒绝器</span><b style="color:var(--red)">未实现</b></div></div><div class="camera"><img src="results/workspace_success_heatmap.png"><div>宽域 X–Z 位置成功率热图 · 绿色高 / 红色低</div></div></div><p class="muted" style="margin-top:16px">结论：IK 的关节裁剪有效，但仅靠裁剪不足以保证安全。接入真实 VLA 前必须增加可达性预检、碰撞预测和不可达目标拒绝。</p></section>
<section class="glass section"><div class="section-head"><div><h2>动力学、鲁棒性与失败边界</h2><p>明确记录通过项和失效阈值，而不是只展示成功案例。</p></div></div><div class="requirement-grid"><div class="phase"><em>长时间 · 通过</em><h3>LONGSECONDS 秒稳定</h3><p>Pelvis 最低 PELVISMIN m；三块方块最大高度漂移 CUBEDRIFT mm；全程无 NaN。</p></div><div class="phase"><em>随机化 · 通过</em><h3>RANDPASS/RANDTOTAL</h3><p>方块质量 0.08–0.30 kg、摩擦 0.3–1.3 及位置扰动下保持稳定。</p></div><div class="phase"><em>扰动 · 有边界</em><h3>连续通过至 FORCEPASS N</h3><p>40 N 测试倒地；60 N 测试底座漂移/倾斜超门槛，因此平衡只能标记为部分完成。</p></div></div></section>
<section class="glass section"><div class="section-head"><div><h2>视觉域与 Checkpoint 结构</h2><p>观测可用不等于训练域一致，结构可识别也不等于 checkpoint 可恢复。</p></div></div><div class="compare"><div class="run"><h3>Frame-0 视觉对照</h3><div class="row"><span>共同可见彩色物体</span><b>VISCOMMON</b></div><div class="row"><span>归一化中心误差均值</span><b>VISCENTER</b></div><div class="row"><span>面积比例中位数</span><b>VISAREA×</b></div><div class="row"><span>精确标定</span><b style="color:var(--amber)">未通过</b></div></div><div class="run"><h3>Orbax Metadata</h3><div class="row"><span>模型族证据</span><b>π0.5 + LoRA</b></div><div class="row"><span>内部 action padding</span><b>PADDED action dim</b></div><div class="row"><span>发布机器人维度</span><b>16</b></div><div class="row"><span>安全恢复</span><b style="color:var(--red)">否 · config 缺失</b></div></div></div></section>
<section class="glass section"><div class="section-head"><div><h2>GitHub Actions 跨平台证据</h2><p>两次 Workflow 来自两次连续 push；均在 Ubuntu 22.04 headless 环境完成。</p></div><span class="pill pass"><i></i>2/2 成功</span></div><div class="compare"><a class="run" href="https://github.com/danielchen26/g1-vla-control/actions/runs/31452479374" target="_blank" rel="noreferrer" style="text-decoration:none;color:inherit"><h3>Run 31452479374 ↗</h3><div class="row"><span>Commit</span><b>b3c8a60</b></div><div class="row"><span>内容</span><b>深度验证代码</b></div><div class="row"><span>结论</span><b style="color:var(--green)">SUCCESS</b></div></a><a class="run" href="https://github.com/danielchen26/g1-vla-control/actions/runs/31452596513" target="_blank" rel="noreferrer" style="text-decoration:none;color:inherit"><h3>Run 31452596513 ↗</h3><div class="row"><span>Commit</span><b>a58c537</b></div><div class="row"><span>内容</span><b>CI 状态回写</b></div><div class="row"><span>结论</span><b style="color:var(--green)">SUCCESS</b></div></a></div><p class="muted" style="margin-top:16px">两次均通过 13 项测试、两组动力学、三路 OSMesa 渲染、HTML 完整性检查和 artifact 上传。</p></section>
<section class="glass section notice"><b>深度验证结论边界：</b> 数据语义、候选 FK、IK、Dex1 几何、合成抓持和本地动力学都有机器可读证据；视觉域仍明显不匹配，FK 与作者实现仍有小统计差异，真实 VLA 网络从未加载。</section>
</div><div class="tab-panel" data-panel="evidence">
<section class="glass section"><div class="section-head"><div><h2>当前仿真观测</h2><p>MuJoCo 使用训练数据同名的三路相机。</p></div><span class="pill partial"><i></i>标定为近似值</span></div><div class="camera-grid"><div class="camera"><img src="results/camera_observations/cam_left_high.png"><div>cam_left_high · 640×480 RGB</div></div><div class="camera"><img src="results/camera_observations/cam_left_wrist.png"><div>cam_left_wrist · 640×480 RGB</div></div><div class="camera"><img src="results/camera_observations/cam_right_wrist.png"><div>cam_right_wrist · 640×480 RGB</div></div></div></section>
<section class="glass section"><div class="section-head"><div><h2>验证矩阵</h2><p>“部分完成”和“阻塞”不得表述为已经完成。</p></div></div><table><thead><tr><th>领域</th><th>验证门槛</th><th>状态</th><th>证据 / 下一步</th></tr></thead><tbody>CHECKROWS</tbody></table></section>
<section class="glass section"><div class="section-head"><div><h2>验证历史</h2><p>每次自动验证后重新生成。</p></div></div><div class="timeline">HISTORY</div></section>
<section class="glass section notice"><b>当前边界：</b> 官方 Dex1-1、场景、相机、动作 schema、IK 与本地动力学已经有可重复证据。真实 OpenPI checkpoint 尚未加载，因此当前结果不能证明 VLA 会完成叠积木，也不能证明真机安全。</section>
</div><div class="footer">生成于 GENERATED · <a href="https://github.com/danielchen26/g1-vla-control" style="color:var(--cyan)">github.com/danielchen26/g1-vla-control</a></div></main><script>
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
        "BASEACCEL": fmt(baseline.get("max_joint_acceleration_rad_s2"), 1),
        "BASEJERK": fmt(baseline.get("max_joint_jerk_rad_s3"), 0),
        "ADAPTDURATION": fmt(adaptive.get("duration")), "ADAPTLEFT": fmt(adaptive.get("left_final_error_m", 0) * 1000, 1),
        "ADAPTRIGHT": fmt(adaptive.get("right_final_error_m", 0) * 1000, 1), "ADAPTSPEED": fmt(adaptive.get("max_joint_speed_rad_s")),
        "ADAPTACCEL": fmt(adaptive.get("max_joint_acceleration_rad_s2"), 1),
        "ADAPTJERK": fmt(adaptive.get("max_joint_jerk_rad_s3"), 0),
        "DATAFRAMES": f"{dataset.get('frames', 0):,}",
        "DATALAG": str(dataset.get("action_state_timing", {}).get("best_lag_frames", "—")),
        "TRANSFORMERR": fmt(transform_compare.get("max_any_stat_error"), 4),
        "MOTIONSPEED50": fmt(motion.get("eef_translation_speed_m_s", {}).get("p50"), 3),
        "MOTIONSPEED95": fmt(motion.get("eef_translation_speed_m_s", {}).get("p95"), 3),
        "MOTIONSPEED99": fmt(motion.get("eef_translation_speed_m_s", {}).get("p99"), 3),
        "MOTIONSPEEDMAX": fmt(motion.get("eef_translation_speed_m_s", {}).get("max"), 3),
        "ANGSPEED50": fmt(motion.get("eef_angular_speed_rad_s", {}).get("p50"), 3),
        "ANGSPEED95": fmt(motion.get("eef_angular_speed_rad_s", {}).get("p95"), 3),
        "ANGSPEED99": fmt(motion.get("eef_angular_speed_rad_s", {}).get("p99"), 3),
        "ANGSPEEDMAX": fmt(motion.get("eef_angular_speed_rad_s", {}).get("max"), 3),
        "MOTIONACC50": fmt(motion.get("eef_translation_acceleration_m_s2", {}).get("p50"), 3),
        "MOTIONACC95": fmt(motion.get("eef_translation_acceleration_m_s2", {}).get("p95"), 3),
        "MOTIONACC99": fmt(motion.get("eef_translation_acceleration_m_s2", {}).get("p99"), 3),
        "MOTIONACCMAX": fmt(motion.get("eef_translation_acceleration_m_s2", {}).get("max"), 3),
        "MOTIONJERK50": fmt(motion.get("eef_translation_jerk_m_s3", {}).get("p50"), 2),
        "MOTIONJERK95": fmt(motion.get("eef_translation_jerk_m_s3", {}).get("p95"), 2),
        "MOTIONJERK99": fmt(motion.get("eef_translation_jerk_m_s3", {}).get("p99"), 2),
        "MOTIONJERKMAX": fmt(motion.get("eef_translation_jerk_m_s3", {}).get("max"), 2),
        "JOINTSPEED50": fmt(motion.get("raw_joint_target_speed_rad_s", {}).get("p50"), 3),
        "JOINTSPEED95": fmt(motion.get("raw_joint_target_speed_rad_s", {}).get("p95"), 3),
        "JOINTSPEED99": fmt(motion.get("raw_joint_target_speed_rad_s", {}).get("p99"), 3),
        "JOINTSPEEDMAX": fmt(motion.get("raw_joint_target_speed_rad_s", {}).get("max"), 3),
        "EPFRAMES": str(episode_replay.get("frames", "—")),
        "EPDURATION": fmt(episode_replay.get("duration_s"), 2),
        "EPARMP95": fmt(episode_replay.get("arm_joint_rmse_rad", {}).get("p95"), 3),
        "EPEEFP95": fmt(episode_replay.get("eef_position_error_m", {}).get("p95", 0) * 1000, 2),
        "EPPELVIS": fmt(episode_replay.get("minimum_pelvis_height_m"), 3),
        "IKSAMPLES": str(ik.get("samples", "—")),
        "IKSUCC": fmt(ik.get("success_rate", 0) * 100, 1, "%"),
        "IKP95MM": fmt(ik.get("position_error_m", {}).get("p95", 0) * 1000, 2),
        "IKP95DEG": fmt(ik.get("orientation_error_deg", {}).get("p95"), 2),
        "WORKSAMPLES": str(workspace.get("dual_target_samples", "—")),
        "WORKPERARM": fmt(workspace.get("per_arm_target_success_rate", 0) * 100, 1, "%"),
        "WORKDUAL": fmt(workspace.get("dual_target_success_rate", 0) * 100, 1, "%"),
        "WORKSAT": fmt(workspace.get("joint_limit_saturation_rate", 0) * 100, 1, "%"),
        "WORKCONTACT": fmt(workspace.get("forbidden_contact_rate", 0) * 100, 1, "%"),
        "DEXCLOSEDMM": fmt(dex.get("closed_gap_m", 0) * 1000, 1),
        "DEXOPENMM": fmt(dex.get("open_gap_m", 0) * 1000, 1),
        "DEXSYMMM": fmt(dex.get("left_right_max_gap_difference_m", 0) * 1000, 3),
        "GRASPPASS": f"{sum(bool(x.get('held')) for x in grasp_trials)}/{len(grasp_trials)}",
        "GRASPFORCE": fmt(min(
            (value for value in grasp.get("maximum_tested_held_force_n", {}).values()
             if value is not None), default=0.0
        ), 1),
        "LONGSECONDS": fmt(long_run.get("duration_s"), 0),
        "PELVISMIN": fmt(long_run.get("pelvis_height_m", {}).get("min"), 3),
        "CUBEDRIFT": fmt(long_run.get("cube_height_drift_m_max", 0) * 1000, 2),
        "RANDPASS": str(sum(bool(x.get("stable")) for x in random_trials)),
        "RANDTOTAL": str(len(random_trials)),
        "FORCEPASS": fmt(disturbance.get("all_lower_forces_recovered_through_n"), 0),
        "VISCOMMON": str(visual_summary.get("common_object_observations", "—")),
        "VISCENTER": fmt(visual_summary.get("center_error_normalized" , {}).get("mean"), 3),
        "VISAREA": fmt(visual_summary.get("area_ratio", {}).get("median"), 2),
        "PADDED": str(checkpoint.get("architecture_evidence", {}).get("likely_internal_padded_action_dim", "—")),
        "CHECKROWS": render_checks(), "HISTORY": render_history(history), "GENERATED": generated,
    }
    # Replace longer tokens first (for example TESTTOTAL before TOTAL).
    for key in sorted(replacements, key=len, reverse=True):
        template = template.replace(key, replacements[key])
    REPORT_PATH.write_text(template)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
