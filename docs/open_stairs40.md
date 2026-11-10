# A2 40级开放楼梯：两条感知路线的公共地形

本阶段只有地形和静态场景预览，不运行策略、不训练，也未配置 LiDAR。
代码通过 mjlab 的 `SubTerrainCfg` / `TerrainGenerator` 扩展接口创建独立 box，未修改依赖包或已有 Flat/Rough/ETH 任务。

## 几何约定

楼梯朝 +x，第一阶前缘 x=0，宽度中心 y=0，底部地面 z=0。

- 40 块独立实心踏板，踏高0.20m、踏深0.20m、宽0.80m、厚0.01m。
- 第 i 阶 x 范围 `[0.2(i-1),0.2i]`，上表面 z=0.2i，下表面 z=0.2i-0.01。
- 楼梯主体 x=0…8m，总升高8m，坡度45°。
- 接近区 x=-2…0m；底部地面覆盖 x=-2…10m、y=-2…2m，上表面 z=0，下表面 z=-0.10m。
- 顶部平台 x=8…10m、y=-0.4…0.4m，上表面 z=8m，厚0.01m，与第40级共面。
- 无竖直立板、侧挡板、扶手、支撑梁或隐形侧墙。静态踏板固定在世界坐标，不模拟结构变形。
- 所有踏板、平台、地面同时参与显示与碰撞，摩擦系数为(0.6,0.005,0.0001)，不修改机器人参数。
- 预览 A2 位置(-1,0,0.4)，朝+x，使用ETH默认关节姿态作为静态比例参照。

`OpenStairsTerrainCfg` 可直接作为原生地形生成器的子地形使用，其原点在接近区。原生多网格坐标按 mjlab 居中规则处理。
`build_open_stairs_spec` 用于独立场景，将单块地形平移至上述 x=0 首阶约定，并返回地面出生点(-1,0,0)。

## 桌面自由视角预览

在有 DISPLAY 的桌面终端执行；相机可自由旋转，不跟随锁定机器人。显示的是静态姿态，不执行重力推进或控制策略。

```bash
source /home/xxs/anaconda3/etc/profile.d/conda.sh
conda activate a2
cd /home/xxs/dog/unitree_rl_mjlab_a2
PYTHONPATH="$PWD" python scripts/preview_open_stairs.py \
  --step-count 40 --step-height 0.20 --step-depth 0.20 \
  --stair-width 0.80 --tread-thickness 0.01 \
  --approach-length 2.0 --landing-length 2.0 --ground-width 4.0 \
  --seed 42 --robot a2
```

## 无头检查和离屏图像

输出目录必须是新目录。输出场景参数JSON、自包含XML与资源目录，以及PNG概览；纯编译检查可以去掉`--render`。

```bash
source /home/xxs/anaconda3/etc/profile.d/conda.sh
conda activate a2
cd /home/xxs/dog/unitree_rl_mjlab_a2
MUJOCO_GL=egl PYTHONPATH="$PWD" python scripts/preview_open_stairs.py \
  --step-count 40 --step-height 0.20 --step-depth 0.20 \
  --stair-width 0.80 --tread-thickness 0.01 \
  --approach-length 2.0 --landing-length 2.0 --ground-width 4.0 \
  --seed 42 --robot a2 --headless --render \
  --output-dir "logs/open_stairs40/preview_$(date +%Y%m%d_%H%M%S)"
PYTHONPATH="$PWD" python -m unittest discover -s tests -p test_open_stairs.py -v
```

## 两条路线的工作区

- 高程图：`/home/xxs/dog/unitree_rl_mjlab_a2`，分支 `main`。
- 点云：`/home/xxs/dog/unitree_rl_mjlab_a2_pointcloud`，分支 `a2-pointcloud`。
- 两处从同一个楼梯基线提交开始。共用`a2` Conda环境和已有ETH权重目录，不重复安装 editable 包；每次先切换到目标目录，并使用`PYTHONPATH="$PWD"`选择该工作区的源码。
- checkpoint、录像、轨迹均不进Git，修改日志例外。点云分支只在本地创建，本阶段不推送它。

## 验证边界

几何、射线和接触测试不代表 A2 能爬楼。没有添加新的 PPO 任务、传感器、奖励、课程或续训逻辑。
桌面窗口是否实测及最终测试结果以 `logs/codex_changes.md` 为准。
