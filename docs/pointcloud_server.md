# A2 点云服务器操作

代码工程交付与训练验收分开。本文命令对应仓库实际入口；2026-11-11用户已批准P5有效执行器增益修订，启动阻止已解除；当前48项本地测试通过（含有限无学习物理检查与ONNX）。服务器训练与资源容量尚未实测。本文没有执行任何训练、试训、训练型压测或多种子训练。仅在服务器由使用者主动执行含学习更新的命令。一次仅用一张RTX4090，不启用DDP；8096不是8192，资源不够时停止报告，不自动改配方。

## 环境和外部文件

在服务器建立独立Python3.11环境，保留本机已安装版本组合，不以nvidia-smi显示的CUDA版本替代实际兼容检查。本机关键版本为torch2.15.0、mjlab1.2.0、mujoco3.5.0、mujoco-warp3.5.0、warp-lang1.12.0、rsl-rl-lib5.0.1、numpy2.4.6、tensordict0.15.5、onnx1.23.2、onnxscript0.7.2。ONNX检查使用项目实际导出后端，不能因存在onnx包就认为验证通过。先在本机导出完整安装清单并与源码一起传输（以下生成文件不会自动进入Git）：

```bash
/home/xxs/anaconda3/envs/a2/bin/python -m pip freeze > /tmp/a2-pointcloud-requirements.txt
```

服务器示例，先把源码和上述文件放到目标位置；需求文件若含本机editable/file路径，应转换为对应已传输源码路径，不静默升级版本：

```bash
conda create --name a2-pointcloud --yes python=3.11
conda activate a2-pointcloud
python -m pip install --requirement /srv/a2-pointcloud-requirements.txt
cd /srv/unitree_rl_mjlab_a2_pointcloud
export PYTHONPATH="$PWD"
python -m pip check
nvidia-smi
python -c 'import torch,mjlab,mujoco,warp; print(torch.__version__,torch.version.cuda,mujoco.__version__); print(torch.cuda.is_available())'
```

ETH初始权重现已按用户授权纳入Git：`weights/eth_a2/model_2000.pt`，来源为原ETH控制器 `/home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt`，不能使用同名未知文件替代。拉取仓库后，可直接将命令中的权重路径改为 `$PWD/weights/eth_a2/model_2000.pt`；若保留本文命令路径，先复制：

```bash
mkdir -p /srv/weights/eth_a2
cp weights/eth_a2/model_2000.pt /srv/weights/eth_a2/model_2000.pt
```

然后校验：

```bash
sha256sum /srv/weights/eth_a2/model_2000.pt
```

期望SHA256：`8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722`。训练保存的checkpoint、轨迹、录像、压测输出和ONNX亦不默认进入Git。恢复checkpoint仅使用可信本地文件。

## 不训练的预检与调试

```bash
python scripts/pointcloud_check.py --eth-checkpoint /srv/weights/eth_a2/model_2000.pt --eth-sha256 8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722 --seed 42 --batch-size 16 --device cpu --variant full
python -m unittest discover -s tests -p 'test_pointcloud*.py' -v
python scripts/pointcloud_debug.py --eth-checkpoint /srv/weights/eth_a2/model_2000.pt --physical-gpu 0 --num-envs 1 --max-steps 50 --seed 42 --scan-seed 42 --stage P1 --vx 0.2 --output-dir logs/pointcloud/debug_p1_seed42
```

debug硬限制1～4环境、1～5000步，不构造优化器，采用原ETH确定性动作，首次任何环境结束即停止。输出逐步`trajectory.npz`（点云/mask/年龄/本体历史/足端/三类结束）、`pointcloud_xz.svg`和`summary.json`。点云仍是各采样时刻机身坐标，SVG不会伪装成配准地图。修改`--stage P4`可检查最终楼梯推理，但原ETH表现不代表训练后能力。输出目录必须是新目录。

## 服务器资源门槛（会执行学习更新）

必须由用户在服务器主动运行；不是本地冒烟测试。完整P5几何、传感器和联合更新，依序1024/2048/4096/8096，每档预热10、测量200次。峰值21GiB门槛，接触溢出与吞吐同时查看输出及stderr；不能把缺失指标当零。

```bash
python scripts/pointcloud_resource_test.py --eth-checkpoint /srv/weights/eth_a2/model_2000.pt --eth-sha256 8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722 --output-dir logs/pointcloud/resource_seed42 --physical-gpu 0 --seed 42 --num-envs 1024 2048 4096 8096 --warmup-iterations 10 --measure-iterations 200 --memory-limit-gib 21 --rollout 24 --epochs 5 --mini-batches 64
```

## 正式训练与恢复（会执行学习更新）

8096×24=194304转移/迭代，5 epochs、64 minibatches（3036样本/批），首轮总上限4000。新分支/Critic/分布学习率3e-4，旧Actor前25次冻结、之后1e-5；固定学习率。每25次保存、每100次阶段验证。子阶段预算：P1=200；P2a/b/c各200；P3a/b/c各400；P4=1600；P5=400。连续两次通过才晋级，不自动加时或挪预算。CLI固定参数与SDS冲突会报错。

```bash
python scripts/pointcloud_train.py --eth-checkpoint /srv/weights/eth_a2/model_2000.pt --eth-sha256 8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722 --output-dir logs/pointcloud/full_seed42 --physical-gpu 0 --num-envs 8096 --max-iterations 4000 --rollout 24 --epochs 5 --mini-batches 64 --seed 42 --save-interval 25 --validation-interval 100 --vx-min 0.2 --vx-max 0.6 --scan-seed 42 --variant full
```

恢复时使用实际保存的checkpoint路径；以下路径仅表示用户选定文件，不表示已经产生训练成果。恢复训练状态后统一环境reset，清空上一动作/感知历史/延迟队列，记录resume_reset，不保证逐轨迹连续。

```bash
python scripts/pointcloud_train.py --eth-checkpoint /srv/weights/eth_a2/model_2000.pt --eth-sha256 8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722 --output-dir logs/pointcloud/full_seed42_resume --physical-gpu 0 --num-envs 8096 --max-iterations 4000 --rollout 24 --epochs 5 --mini-batches 64 --seed 42 --save-interval 25 --validation-interval 100 --vx-min 0.2 --vx-max 0.6 --scan-seed 42 --variant full --resume /srv/checkpoints/selected_resume.pt
```

## 最终保留集、对照与导出

五档各200次、单独标准集与感知退化集；确定性推理，无PPO更新。使用未用于训练及checkpoint选择的种子。速度检查实际世界水平速度投影，失败episode不得剔除。标准每档95%、退化每档90%，同时速度门槛；程序输出不等于门槛必然通过。

```bash
python scripts/pointcloud_evaluate.py --checkpoint /srv/checkpoints/selected_full.pt --method full --condition standard --heldout-seed 900001 --physical-gpu 0 --episodes-per-speed 200 --speeds 0.2 0.3 0.4 0.5 0.6 --output-dir logs/pointcloud/heldout_standard
python scripts/pointcloud_evaluate.py --checkpoint /srv/checkpoints/selected_full.pt --method full --condition degraded --heldout-seed 910001 --physical-gpu 0 --episodes-per-speed 200 --speeds 0.2 0.3 0.4 0.5 0.6 --output-dir logs/pointcloud/heldout_degraded
python scripts/pointcloud_evaluate.py --eth-checkpoint /srv/weights/eth_a2/model_2000.pt --method eth --condition standard --heldout-seed 900001 --physical-gpu 0 --episodes-per-speed 200 --speeds 0.2 0.3 0.4 0.5 0.6 --output-dir logs/pointcloud/heldout_eth
python scripts/pointcloud_evaluate.py --checkpoint /srv/checkpoints/selected_full.pt --method occlusion --condition standard --heldout-seed 900001 --physical-gpu 0 --episodes-per-speed 200 --speeds 0.2 0.3 0.4 0.5 0.6 --output-dir logs/pointcloud/heldout_occluded
python scripts/pointcloud_export.py --checkpoint /srv/checkpoints/selected_full.pt --output /srv/exports/a2_pointcloud.onnx --batch-size 4 --seed 42 --tolerance 0.000001
```

无点云训练对照：完整训练命令仅把`--variant full`改为`--variant blind`、输出改为`logs/pointcloud/blind_seed42`，保持所有其他显式参数。对应评估`--method blind`并使用blind checkpoint。该变体保留本体修正与辅助头而移除点云依赖，必须在比较中披露架构差异，不能称为严格原ETH同架构训练。

多种子不自动启动。先完成42并达标后，按相同完整命令分别显式改为`--seed 43 --output-dir logs/pointcloud/full_seed43`、然后`--seed 44 --output-dir logs/pointcloud/full_seed44`；保留`--scan-seed 42`以固定扫描划分，其余配置一致，顺序运行且只用物理GPU0。每个对照或种子独立预算4000，不挤入首轮42预算。三种子实际完成前不宣称稳定收敛。

## 状态边界

- 代码实现和CLI存在不等于环境集成验证通过；接口整合状态以主审核与SDS状态表为准。
- 本地允许有限几何/传感器/推理、合成梯度、序列化和ONNX一致性验证。
- 8096完整资源、全部课程、成功率、速度、对照和三种子稳定性均待服务器实际执行。
- 合成双雷达不等于真实JT128规格复现，ONNX一致性不等于实机部署验证。

## 最终奖励与控制解释

新任务保持dt=0.005、decimation=4、50Hz、动作目标默认角+0.25×原始策略动作，以及原执行器控制限位路径；PPO log-prob使用采样的原始动作。未采用X30参数。

| 奖励 | 权重 | 变化 |
|---|---:|---|
| 航向xy速度跟踪 | 1 | 去掉旧z速度误差 |
| 角速度跟踪 | 1 | 保留 |
| 局部法线姿态 | 1 | 负的机身轴与世界局部法线偏差，替代旧姿态 |
| 关节姿态偏好 | 1 | 保留 |
| 机身角速度 | -0.05 | 保留 |
| 角动量 | -0.025 | 保留 |
| failure | -200 | 替代合并terminated；成功/超时为0 |
| 关节加速度 | -2.5e-7 | 保留 |
| 关节限位 | -10 | 保留 |
| 动作变化 | -0.05 | 保留 |
| 足相位 | 0.5 | 保留周期0.6/stance0.56 |
| 局部摆动轨迹平方误差 | -10 | 替代世界固定足高，摆动参考冻结 |
| 足滑动 | -0.25 | 保留 |
| 落地 | -0.001 | 保留 |
| 站立 | -1 | 保留 |
| 支撑边缘无效比例 | 0.5 | 负值项新增，11采样/足 |

全部贡献保留原环境的dt=0.02缩放，因此failure单次实际贡献-4。局部支撑查询只进入奖励/监督/Critic，不进入Actor。P5已按批准方案每episode同步缩放gain、bias及力矩限幅0.9～1.1；这是有效PD增益变化。P1～P4和标准/感知退化评估使用名义值1.0。资源入口阻塞已解除，但未执行压测。
