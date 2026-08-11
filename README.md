# Unitree G1 · Dex1 · VLA 仿真验证

[![MuJoCo validation](https://github.com/danielchen26/g1-vla-control/actions/workflows/validation.yml/badge.svg)](https://github.com/danielchen26/g1-vla-control/actions/workflows/validation.yml)

面向 Apple M4 Air 24 GB 的本地 MuJoCo 验证环境。真实 OpenPI VLA 计划在 Ubuntu NVIDIA 主机运行，本机负责场景、三路相机、动作契约、IK、Dex1 和评测。

> 当前边界：验证输入分为确定性 Stub 和公开 episode 的 recorded-policy proxy，两者都不是真实 VLA 在线输出。真实 checkpoint 仍缺 OpenPI config 和精确 joint↔EEF transform。

## 安装

```bash
git clone --recurse-submodules https://github.com/danielchen26/g1-vla-control.git
cd g1-vla-control
conda create -n mujoco python=3.12 -y
conda activate mujoco
python -m pip install -r requirements.txt
```

如果已经 clone 但没有下载第三方模型：

```bash
git submodule update --init --recursive
```

模型默认从 `third_party/mujoco_menagerie` 加载；也可以通过环境变量覆盖：

```bash
export MUJOCO_MENAGERIE_PATH=/path/to/mujoco_menagerie
```

## 当前已完成

- Unitree G1 29-DoF 官方 MuJoCo Menagerie 模型
- Unitree 官方 Dex1-1 URDF/STL（BSD-3-Clause）替换原 rubber/articulated hand
- 左右 Dex1 各两个对称 prismatic finger actuator
- VLA 夹爪电机值到 URDF 指爪的反向映射：`0 rad` 闭合、`5.5 rad` 张开（由同步 episode-0 首帧验证方向）
- 白色桌面、红/蓝/黄动态方块、摩擦与碰撞
- `cam_left_high`、`cam_left_wrist`、`cam_right_wrist` 三路 640×480 RGB
- 16-D 双手 EEF action schema
- VLA `xyzw` 与 MuJoCo `wxyz` 四元数边界转换
- XYZ 插值、Quaternion SLERP、G1 双臂 IK
- 自动测试、JSON 证据、验证历史和现代化 HTML App
- 可选 adaptive speed 模块；真实 VLA smoke test 可以完全绕过它
- Gate-A sim-only EEF v/a/jerk safety filter
- Gate-A.2 14-D 逐关节 command v/a/jerk filter 与 phase-aware target preflight
- OpenPI π0.5-DROID output-only WebSocket smoke client、Mock evidence 与固定版本部署说明

## 尚未完成

- π0.5-DROID 的 Ubuntu NVIDIA 真实 output-only inference（当前真实调用数为 0）
- `LGG100/stack-cube-eef-24k` 的真实神经网络推理
- 模型作者的 OpenPI training config、repack transform 和 EEF 坐标定义
- 训练仿真器的精确相机内外参与视觉域
- 真实 VLA action chunk 的离线可视化与闭环执行
- 能约束 actual qvel acceleration/jerk 的真实低层控制器接口与反馈层
- 真实 VLA 抓取/堆叠成功率
- 真机 G1 EDU 测试

## 深度验证结果

- 审计全部 100 episodes / 95,966 帧：raw parquet 是 14-D 关节绝对目标 + 2-D Dex1 电机值
- `action[t]` 与 `state[t+3]` 最吻合，对应约 0.10 秒响应延迟
- 候选训练 transform：pelvis frame、wrist +X 50 mm、quaternion `xyzw`；与 norm stats 高度吻合但非精确相等
- Episode 0 全 678 帧、22.57 秒连续动力学回放；站立保持稳定，EEF P95 误差约 20 mm
- 250 个随机真实数据 EEF 目标：IK 在 2 mm / 1° 门槛下 250/250 收敛，未发现近奇异样本
- 1000 组宽于训练域的双手位置压力测试：关节范围始终合规，但大量目标饱和/碰撞，确认需要显式可达性拒绝器
- Dex1 指尖间距约 21–100 mm；左右几何最大差约 0.039 mm
- 左右预定位合成抓持零扰动 2/2；左手在 2 N 失效、右手在 4 N 失效，明确记录不对称边界
- 30 秒稳定性通过；20 组质量/摩擦/位置随机化通过
- 躯干冲击连续通过至 20 N；40 N 倒地，60 N 漂移/倾斜超门槛
- 视觉域量化显示仍有明显中心、尺度、背景、照明和夹爪外观差异
- Orbax 参数签名强烈指向 `Pi0Config(pi05=True)` + LoRA，内部 action padding 32、发布动作维度 16；仍不能无 config 安全恢复

## Gate-A 调速安全层

使用公开训练目标 P99 作为 sim-only 初值：

```text
EEF speed       0.227 m/s
EEF acceleration 2.760 m/s²
EEF jerk        125.67 m/s³
Angular speed   1.044 rad/s
Gripper speed   5.793 rad/s
```

Episode 0 在 `0.50× / 0.75× / 1.00× / 1.25× / 1.50×` 五档回放中，EEF 与 14-D joint command 均满足各自训练 target P99 limits，最终 endpoint 误差均小于 0.91 mm。Phase-aware preflight 对 100 个 OOD 目标全部拒绝。公开数据没有 phase 标签：free-space 视图接受 44/68；假定全部为 grasp 的宽松视图接受 59/68。后者不是已恢复的真实任务阶段，并包含明确记录的 torso/shoulder 模型重叠 allowlist。

但 command 合规不等于 actual dynamics 合规：1.25× 下 MuJoCo actual joint jerk 出现约 100k rad/s³。尖峰发生在 `right_shoulder_roll_joint`、t=10.538 s，事件帧没有当前策略定义的 forbidden contact，指向 position actuator/controller transient，而不是 joint command 超限。因此当前不建议把 `≥1.25×` 作为候选安全档，也不能把这些数据解释为真机限制。

### 为什么没加载 VLA 仍能做 Gate A/A.2

公开 episode 的 joint action 经候选 FK 重建为 16-D EEF，作为 **recorded-policy proxy** 输入同一个下游链路。这能验证 schema、坐标转换、filter、IK、command limits 和拒绝逻辑，但不能验证真实 VLA 的 chunk timing、延迟、输出分布、视觉闭环或叠积木成功率。Gate B 会用带时间戳的真实 OpenPI chunk 替换该输入源并重跑全部测试。

## 第一真实 VLA Smoke：π0.5-DROID

已按官方 OpenPI revision `15a9616a00943ada6c20a0f158e3adb39df2ccac` 建立 output-only 客户端：

```text
config:     pi05_droid
checkpoint: gs://openpi-assets/checkpoints/pi05_droid
```

本地 Mock 只验证 WebSocket 审计契约、rank-2/8-D action 检查、NaN fail-closed、timestamp、latency、stale 标记和 SHA-256。它明确记录：

```text
neural_vla_claimed: false
g1_execution_enabled: false
g1_action_compatible: false
```

Ubuntu NVIDIA endpoint 到位后运行：

```bash
python openpi_droid_smoke.py --host 127.0.0.1 --port 8000 --calls 30
```

真实证据将写入 `results/openpi_droid_smoke_real.json`。DROID 8-D action 仍然只记录，绝不通过补零或人工复制映射到 G1。完整步骤见 [`OPENPI_DROID_SMOKE.md`](OPENPI_DROID_SMOKE.md)。

一键重跑所有本地可行验证：

```bash
python run_deep_validation.py
```

结果写入 `results/*_audit.json`、`results/comprehensive_sim_validation.json` 和 HTML 的“深度验证”Tab。

## 一键验证并更新 HTML

```bash
conda activate mujoco
cd ~/g1_vla_control
python run_validation.py
open validation_report.html
```

## 查看场景与三路图像

```bash
conda activate mujoco
cd ~/g1_vla_control
mjpython run_stack_scene.py
python render_camera_observations.py
open results/camera_observations/camera_contact_sheet.png
```

## 目标 Workflow

```text
MuJoCo 三路 RGB + EEF 状态 + prompt
                  ↓
Ubuntu NVIDIA · OpenPI checkpoint
                  ↓
16-D EEF action chunk
                  ↓
坐标系 / quaternion / normalization 逆变换
                  ↓
硬安全裁剪（第一轮不启用 adaptive speed）
                  ↓
G1 双臂 IK
                  ↓
14-D Joint command v/a/jerk filter + Dex1 映射
                  ↓
MuJoCo 闭环重规划与任务评测
```

## 第三方资源

- `third_party/mujoco_menagerie/`：Google DeepMind MuJoCo Menagerie（Git submodule）
- `third_party/dex1_1_service/`：Unitree 官方 Dex1-1 URDF/STL（Git submodule，BSD-3-Clause）

各第三方资源继续适用其上游许可证；本仓库未包含 VLA checkpoint 和公开数据集视频。

## 结论边界

当前证据证明场景、官方 Dex1、三路相机、动作接口、IK 和本地动力学链路可运行。它尚不证明真实 VLA 能完成叠积木，也不证明系统已达到真机安全标准。
