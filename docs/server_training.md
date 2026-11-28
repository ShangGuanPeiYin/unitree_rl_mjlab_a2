# A2 v1.0 服务器训练

Linux、Python 3.11、NVIDIA GPU/兼容驱动、GitHub CLI（gh）。当前可用环境的直接依赖版本固定在 requirements-server.txt；系统需提供 EGL/OpenGL 运行库。权重分片最大1 GiB，恢复前后用SHA256校验。

权重位于同仓库Release `a2-v10-server-20261128`，代码与SHA清单位于main。公开fork不支持LFS上传，因此采用Release附件分发。

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

包含ETH盲走教师、S2楼梯教师、D1第50轮完整checkpoint（Adam/replay/RNG一起保留）、原学习与评估费用台账。不携带旧机器controller/profile及旧通过证书；服务器首次启动按当前源码与当地路径创建配置，重新执行教师门禁。历史已发生费用保留，不退款。不要把服务器的新旧run与本机同时跑在同一份预算下。

## 检查与启动

```bash
A2_PYTHON=python bash scripts/run_sds_manual.sh --mode check --device cuda:0 --num-envs 2048 --rollout 48 --seed 42 --resume-reason "服务器迁移：保留D1第50轮完整状态"
A2_PYTHON=python bash scripts/run_sds_manual.sh --mode all --device cuda:0 --num-envs 2048 --rollout 48 --seed 42 --resume-reason "服务器迁移：保留D1第50轮完整状态"
```

D1/D2/D3轮数上限500/1000/1500，PPO上限1000（前100轮P0），每50轮保存。已完成D1的50轮不重跑；并行数可由参数设置，完整rollout为48步。教师评估每组20环境、最终评估100环境，与学习N分开。奖励、损失、教师A1纠偏、课程门禁未改变。日志实时输出；各评估目录memory_groups.jsonl记录组间内存。

完整教师门禁失败时停止；历史已有S_ROUTE组16/20失败记录，未将它声明为通过。服务器资源未在本机替代验证。restore脚本拒绝覆盖任何不同内容的现有checkpoint/台账；只在新克隆目录执行迁移。后续恢复训练直接运行all，不再次还原旧台账。
