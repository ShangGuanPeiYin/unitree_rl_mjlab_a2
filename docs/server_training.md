# A2 v1.0 服务器文件

Linux、Python 3.11、NVIDIA GPU/兼容驱动、GitHub CLI（gh）。依赖版本见 requirements-server.txt；系统需提供 EGL/OpenGL 运行库。

## 下载与安装

```bash
git clone https://github.com/ShangGuanPeiYin/unitree_rl_mjlab_a2.git
cd unitree_rl_mjlab_a2
gh release download a2-v10-server-20261128 --repo ShangGuanPeiYin/unitree_rl_mjlab_a2 --dir server_bundle --skip-existing
conda create --name a2 python=3.11 --yes
conda activate a2
python -m pip install --requirement requirements-server.txt
python -m pip install --editable .
python scripts/restore_server_bundle.py --root "$PWD"
```

公开fork不支持Git LFS上传，因此所需权重通过同仓库Release附件提供，main包含SHA256清单与校验还原脚本。

保留ETH盲走教师（4.5MiB）与S2楼梯教师（8.5MiB），以及压缩的历史学习/评估费用台账。第50轮学生checkpoint及其2940MiB replay按用户要求移除，不再上传、下载或恢复。中间评估轨迹、视频、调试缓存和其他历史模型不包含在迁移包中。

## 从第0轮开始训练

先安装上述Python3.11依赖并激活a2环境。在已经下载权重的服务器上更新代码并运行：

```bash
git pull --ff-only origin main
conda activate a2
python -m pip install --requirement requirements-server.txt
python -m pip install --editable .
A2_PYTHON=python bash scripts/run_sds_manual.sh --mode all --device cuda:0 --num-envs 2048 --rollout 48 --seed 42 --resume-reason "服务器从第0轮训练或继续本次训练"
```

首次运行以S2教师参数初始化学生，优化器和replay从空状态开始；两份教师权重保持冻结。通过原教师门禁后开始D1。从头训练使用独立logs/a2_v10/server_fresh目录，历史50轮及其费用台账保留但不作为本次起点，也不恢复旧checkpoint。再次执行同一命令会恢复本次训练，而非重复从0轮开始。

D1/D2/D3轮数上限500/1000/1500，PPO上限1000（前100轮P0），每50轮保存，rollout为48步。学习并行数由--num-envs设置；教师评估每组20环境、最终评估100环境。奖励、损失、A1纠偏、原门禁未改变。新训练使用自己的完整轮数预算及计费台账，原费用不清除、不退款。不同N在本次已有学习后不能当成新的免费起点。

仅检查文件与配置、不启动物理采样：

```bash
A2_PYTHON=python bash scripts/run_sds_manual.sh --mode check --device cuda:0 --num-envs 2048 --rollout 48 --seed 42 --resume-reason "服务器启动检查"
```

物理协议基准来自原真实环境contract（doc/sds/reviews/v10_server_physical_contract.json），用于实际环境参数一致性校验，不是服务器资源通过证书。终端持续显示子进程日志及进度。服务器资源与教师行为门禁仍按实际结果执行。

恢复脚本拒绝覆盖不同内容的现有文件。已下载旧版本的用户可删除 server_bundle/resume50.pt.part000、part001、part002；请勿覆盖已有更新的费用台账。
