# Unitree G1 · Dex1 · VLA 仿真验证

面向 Apple M4 Air 24 GB 的本地 MuJoCo 验证环境。真实 OpenPI VLA 计划在 Ubuntu NVIDIA 主机运行，本机负责场景、三路相机、动作契约、IK、Dex1 和评测。

> 当前边界：仓库中的轨迹是确定性 Stub，并非真实 VLA 输出。真实 checkpoint 仍缺 OpenPI config 和 joint↔EEF transform。

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

## 尚未完成

- `LGG100/stack-cube-eef-24k` 的真实神经网络推理
- 模型作者的 OpenPI training config、repack transform 和 EEF 坐标定义
- 训练仿真器的精确相机内外参与视觉域
- 真实 action chunk 的离线可视化与闭环执行
- 批量抓取/堆叠成功率和随机化鲁棒性
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
G1 双臂 IK + Dex1 夹爪映射
                  ↓
MuJoCo 闭环重规划与任务评测
```

## 第三方资源

- `third_party/mujoco_menagerie/`：Google DeepMind MuJoCo Menagerie（Git submodule）
- `third_party/dex1_1_service/`：Unitree 官方 Dex1-1 URDF/STL（Git submodule，BSD-3-Clause）

各第三方资源继续适用其上游许可证；本仓库未包含 VLA checkpoint 和公开数据集视频。

## 结论边界

当前证据证明场景、官方 Dex1、三路相机、动作接口、IK 和本地动力学链路可运行。它尚不证明真实 VLA 能完成叠积木，也不证明系统已达到真机安全标准。
