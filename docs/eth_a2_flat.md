# ETH A2 平地适配（验证进行中）

独立任务 `Unitree-A2-ETH-Flat` 深复制现有平地配置，不修改原 Flat/Rough。

## 控制契约与证据

来源：本地 ETH 提交 da6fbdd2f17ef27fea6c3f1dce033f3cfd0e5fcb 的 Python 控制器。
权重 model_2000.pt，SHA256 8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722。

Actor 45→512→256→128→12，ELU。输入为机身角速度、机身投影重力、vx/vy/wz、相对默认关节角、关节速度、原始上一动作。归一化使用 checkpoint std，下限1e-5，归一化后裁剪±5。严格加载13个字段。与本地参考 ActorNet 的128条测试输入输出最大绝对误差0。

顺序 FL/FR/RL/RR，每腿 hip/thigh/calf；左右髋-0.1/+0.1，大腿0.9、小腿-1.8 rad。目标角=默认角+0.25×原始动作。物理步长0.005s，4次物理步执行一次策略，50Hz。

使用目标项目 A2 XML、内置位置执行器；PD为100/100/150与4/4/6，力矩上限120/120/180，armature 0.03。ETH 仓库未发现原始模型与完整训练配置，无法证明动力学严格一致。C++默认加载另外的ONNX，不视为同一策略证据。

## 基线测试

1环境、seed42、CUDA；根高度0.4m、无旋转、零速度、默认关节角。关闭仅ETH任务的噪声、摩擦/质心/编码器随机化和推扰，保持失败终止条件。命令不得直接写机器人速度。四项为零指令30秒、vx0.3持续30秒、vx0.6持续30秒、0→0.3→0每段10秒；vy/wz均零。

首次终止停止该项，保存重置前状态，不能以自动重置后的轨迹宣称通过。JSON/NPZ记录状态。当前尚未完成实际测试及录像，禁止据此宣称步行或楼梯能力。

## 分层

- config/a2/eth_env_cfg.py：任务配置。
- rl/eth_actor.py：仅Actor严格推理加载。
- scripts/play.py：ETH专属加载分支。
- scripts/evaluate_eth_a2.py：确定性评估与重置前记录。

## 未来续训

Critic 72维的含义、归一化更新、动作分布、奖励、终止/超时、随机化、课程与优化器兼容均需另行核对。当前不提供断点续训兼容保证，不运行PPO。

## 实际结果（2026-11-10）

正式结果：`logs/eth_a2_flat/eval_20261110_02`。四项各30秒、无终止、无重置、记录全部有限。平均vx分别 -0.000552、0.272595、0.550400、0.088193 m/s；两项直行横向位移分别+0.702和-0.615m，不能宣称精准跟踪。停走各10秒平均vx分别-0.001659、0.262867、0.003373m/s。各项最大roll/pitch分别0.330/0.339、0.699/0.935、1.456/1.159、0.992/1.107度。

`logs/eth_a2_flat/video_smoke_20261110` 含四项各2秒的离屏预览录像；预览不替代30秒正式测试。无DISPLAY，桌面窗口未实测。

### 桌面回放

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
source /home/xxs/anaconda3/etc/profile.d/conda.sh
conda activate a2
PYTHONPATH=/home/xxs/dog/unitree_rl_mjlab_a2 python scripts/play.py Unitree-A2-ETH-Flat --agent trained --checkpoint-file /home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt --num-envs 1 --device cuda:0 --seed 42 --vx 0.3 --vy 0.0 --wz 0.0 --duration 30 --viewer native --output-dir logs/eth_a2_flat/desktop
```

### 无头四项评估

输出目录必须不存在。speeds依次指定站立、慢行、直行速度；停走使用第一、第二、第一速度，每段duration/3。

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
source /home/xxs/anaconda3/etc/profile.d/conda.sh
conda activate a2
MUJOCO_GL=egl python scripts/evaluate_eth_a2.py --task Unitree-A2-ETH-Flat --checkpoint /home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt --num-envs 1 --device cuda:0 --seed 42 --speeds 0.0 0.3 0.6 --vy 0.0 --wz 0.0 --duration 30 --output-dir "logs/eth_a2_flat/eval_$(date +%Y%m%d_%H%M%S)" --video
```
