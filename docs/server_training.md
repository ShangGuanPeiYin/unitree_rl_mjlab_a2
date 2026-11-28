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

当前 run_sds_manual.sh 是旧第50轮续训入口，不能用于删除该checkpoint后的从头训练。从第0轮训练需要另行调整启动入口；本次删除没有修改训练流程、奖励、损失、教师参数或预算，也没有自动启动训练。

恢复脚本拒绝覆盖不同内容的现有文件。已下载旧版本的用户可删除 server_bundle/resume50.pt.part000、part001、part002；请勿覆盖已有更新的费用台账。
