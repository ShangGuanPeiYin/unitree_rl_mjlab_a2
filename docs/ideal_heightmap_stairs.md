# A2 理想高程图开放楼梯：本地实施与运行记录

依据 [SDS v1.3](../doc/sds/SDS_A2_IdealHeightMap_Stairs40.md)。当前实施和验收状态以 `logs/a2_ideal_stairs/implementation/` 审核及实际运行结果为准，组件通过不等于爬楼通过。

## 系统与训练逻辑

独立任务 `Unitree-A2-IdealHeightMap-Stairs` 通过 `scripts/train.py` 路由至本地监督器，旧任务保留原入口。6类24×8m静态图集显式分配平地及S1–S5；楼梯首阶坐标与terrain origin分离，平地使用独立出生参考系。真实碰撞仍为0.01m薄板，地图是同几何的向下第一表面查询，不替代碰撞。

Actor接收45维ETH本体、5帧历史、48×32×3理想地图，地图10Hz、策略50Hz、物理200Hz。零初始化融合使迁移初始均值与ETH一致；全部Actor可训练，ETH统计冻结，探索std约束[.05,.30]。Critic独立地图与历史编码器，加入35维特权状态，独立统计。真实geom接触区分足球、小腿与主体；顶面法向、点高、向上力及两次控制采样共同决定支撑，不能用普通接触代替。

本任务严格采用SDS连续奖励，`r=.02×Σ(wf)+10×成功事件−4×失败事件`。相对旧任务，XY跟踪不再惩罚爬升vz，允许−15°至55°俯仰，默认姿态权重从1降至.2，固定步态与世界足高项关闭，并新增居中、航向、支撑边缘和非足接触项。其他保留项按原公式。没有任何实测结果证明这些设计改善了学习效果。

PPO每更新48步、5epochs、4minibatches；Adam初始1e-4，目标KL=.01，LR限制1e-5–3e-4。S0–S5上限500/2000/2000/2500/3000/4000，总计14000/种子；每100更新保存，每200更新独立验证。连续两次通过阶段门槛才能晋级，耗尽阶段预算停止。按seed42→43→44顺序执行，42最终验收失败不继续后续种子。

## 资源与恢复

独立进程按1024→512→258探测；每档5更新预热+50更新测量，完整地图rollout/优化器常驻、保存后退出、串行完整评估，再新进程加载并更新。probe专用初态覆盖中高阶接触及混合课程，正式训练重新从ETH初始化。batch分别49152/24576/12384，minibatch12288/6144/3096，不是等价采样预算。

NVML每20ms采集整卡及进程显存，包括Warp；PyTorch记录allocator高水位，主存采集available/RSS。整卡≤7168MiB，进程≤min(5120MiB,7168−启动其他占用)，主存available≥4096MiB。NVML峰值是20ms采样观测峰值，不能保证捕获采样间更短尖峰；异常及OOM仍使候选失败，不放宽保护。

训练按独立进程块运行，checkpoint保存模型、优化器、统计、全局与分阶段预算、随机数（含环境独立Generator）。恢复从新回合开始，不恢复物理轨迹。核心行为源码哈希、ETH来源哈希、控制/地图/网络等契约不兼容即拒绝加载。资源降档新run记录父checkpoint和已消耗预算；即使模型退回旧完整checkpoint也不返还已消耗更新数。验证节点恢复先验证再继续。

## 证据位置

- `logs/a2_ideal_stairs/implementation/review_model_runner.md`：模型/PPO独立审核。
- `logs/a2_ideal_stairs/implementation/review_map.md`：图集/地图独立审核。
- `logs/a2_ideal_stairs/implementation/review_environment.md`：接触与任务独立审核。
- `logs/a2_ideal_stairs/implementation/review_orchestration.md`：运行调度及修复复核。
- `logs/a2_ideal_stairs/integration_smoke/`：2环境完整48步PPO、GPU保存恢复及评估/视频冒烟。

正式运行命令与最终结果将在审核通过且实际启动后追加。未完成的正式资源探测、训练成功率及多种子复验不标为通过。不要将smoke.pt作为正式训练初始化或最终模型。

## 独立审核状态（2026-11-10）

最终审核26项CPU检查通过，并独立读取GPU更新/恢复与视频证据。审核批准进入资源探测，见 `logs/a2_ideal_stairs/implementation/final_review.md` 和 `approval.json`。approval包含源码指纹，任何运行核心代码变化都会使监督器拒绝开始，必须先复核。子进程退出后实际查询CUDA PID释放再进入下一档。

## 当前执行命令

2026-11-10启动run `local_sds_20261110_221417`。实际监督PID914699，执行会话97583，状态索引 `logs/a2_ideal_stairs/implementation/active_run.json`。首次nohup包装器未保留进程，改为持续执行会话；不要仅凭旧PID判断运行。

以下是真实通过--help且已启动的接口；重启整套实验须换新output-dir，禁止覆盖现有结果：

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
/home/xxs/anaconda3/envs/a2/bin/python -u scripts/train.py Unitree-A2-IdealHeightMap-Stairs \
  --output-dir /home/xxs/dog/unitree_rl_mjlab_a2/logs/a2_ideal_stairs/local_sds_20261110_221417 \
  --device cuda:0 --num-envs-candidates 1024 512 258 --seeds 42 43 44 \
  --save-interval 100 --eval-interval 200 --eval-num-envs 8 --max-iterations 14000 \
  --eth-checkpoint /home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt \
  --review-report /home/xxs/dog/unitree_rl_mjlab_a2/logs/a2_ideal_stairs/implementation/approval.json
```

训练worker命令显式记录vx范围0.2–0.6、vy=0、wz=0；平地槽位半数站立按SDS分配，非全环境固定前进命令。正式checkpoint产生后再给出对应精确SHA的回放与恢复命令。

## 资源实测结论

正式档位512环境，batch24576、minibatch6144。1024在首个完整更新后观察到进程5178MiB，超过5120MiB预算，按保护退出；确认CUDA PID消失后才启动512。512完成5预热+50测量，整卡观测峰值4839.25MiB、进程3172MiB、PyTorch allocated1887.35MiB/reserved2598MiB、RSS2998.34MiB、主存available最低16752.18MiB。更新中位2.591s/P95 2.717s，吞吐9445.17环境步/s，checkpoint 8822551字节。全部loss/梯度/动作finite。

串行S5评估完整60回合耗时999.823s，各档0/20成功，仅用于验证资源生命周期；随后新进程probe-resume完整更新通过，才选择512。258按首个通过即采用的规则未运行。正式初始化不使用probe权重。

## 2026-11-11 最终结果：B类预算停止

本节覆盖上述历史进行中状态。正式run已正常终止，监督器退出码0；不是训练成功。seed42通过S0，S1用尽2000次更新仍未晋级。最终每速度100回合测试尚未进入，seed43/44按先42通过的规则未启动。没有合格40阶最佳模型，以下checkpoint仅用于失败诊断/严格恢复。

| 工作包 | 实际完成情况 |
|---|---|
| W0 | SDS资源策略同步、ETH来源/哈希、独立配置与基线完成 |
| W1 | 新训练/回放任务、首阶坐标与出生坐标分离、静态图集及阶段分配完成 |
| W2 | 理想地图、缓存时序/环境隔离、历史观测完成；真实ray与解析地图验证通过 |
| W3 | 严格ETH迁移、可训练Actor、新Critic、初始均值一致性完成 |
| W4 | 真实geom接触/顶面支撑、奖励/终止/成功与timeout区分完成 |
| W5 | PPO、门槛晋级、保存/严格恢复与资源生命周期完成 |
| W6 | 独立验证、最终测试接口、逐回合指标和录像接口完成；40阶最终测试未进入 |
| W7 | 本地资源探测、seed42正式训练执行至S1预算停止；S2–S5及43/44未运行 |
| W8 | 失败结果、证据索引与可执行复现文档交付；不标注能力验收通过 |

| 阶段 | 正式更新 | 环境步样本 | 更新计算墙钟 | 晋级结果 |
|---|---:|---:|---:|---|
| S0 | 400 | 9,830,400 | 961.62秒 | global200和400连续gate通过，401进入S1 |
| S1 | 2000 | 49,152,000 | 4944.61秒 | global600至2400每200验证，10轮全部失败；按预算停止 |
| 合计 | 2400 | 58,982,400 | 5906.23秒 | 未达到40阶目标 |

总流程墙钟11804.08秒（约3小时16分44秒，含探测、基线、正式训练、评估及初始化）；正式首chunk开始至终止约10172.74秒（2小时49分33秒）。更新计算墙钟仅collect+PPO update，不包含验证/初始化。无正式中途资源降档、模型回退或重放更新。

| 训练种子 | 0.2m/s | 0.4m/s | 0.6m/s | 结论 |
|---|---|---|---|---|
| 42 | S1终点0/20 | S1终点0/20 | S1终点0/20 | S1预算耗尽；不是最终100回合测试 |
| 43 | 未运行 | 未运行 | 未运行 | 42未通过，不启动 |
| 44 | 未运行 | 未运行 | 未运行 | 42未通过，不启动 |

终点S1是4阶、踏高0.08m、踏深0.20m、宽0.80m的开放薄板。三档平均vx分别0.019164/0.031619/0.039954m/s，全部timeout，最高有效台阶0。十轮共600个验证回合均未成功。S0的success事件为0不表示S0 gate失败：S0按30秒无失败、站立速度及行走速度误差判定；global400站立均速0.001883m/s，三档均速0.212870/0.399570/0.597175m/s。与ETH基线相比速度跟踪改善，但部分侧偏指标更差，不能声称整体运动性能改善。

末期只读诊断：S1前50与后期50更新，非足接触原项约0.006526→0.0000558、失败事件均值−0.02350→−0.0000846，速度跟踪项0.9191→0.6442；global2088附近末1000楼梯回合971timeout、20非足、9主体接触，平均vx约0.03325。表明碰撞减少而登阶未形成；不能仅凭这些数据判定地图未使用、具体碰撞部位或停滞因果。未据此改变奖励/探索/速度/课程。

独立审核发现并修复了std约束、缺档/空验证门槛、checkpoint行为哈希与RNG、预算累计及待验证节点恢复、资源监测结束时异常和探测恢复生命周期等问题。修复后26项CPU检查及真实GPU更新/保存/恢复通过。终态独立复核确认1..2400更新连续、Adam step48000、12次gate全部消费、56份checkpoint存在且哈希一致、30个子进程释放记录无重叠、15份批准核心源码哈希未变。最终checkpoint模型张量与global2400受验证模型完全相同，差异仅为已消费的gate/终态元数据。

证据：
- [逐速度CSV](../logs/a2_ideal_stairs/final_report_local_sds_20261110_221417/validation_results.csv) 与 [机器可读完整报告](../logs/a2_ideal_stairs/final_report_local_sds_20261110_221417/summary.json)。
- [终点原始60回合](../logs/a2_ideal_stairs/local_sds_20261110_221417/seed_42/validation_002400_e8/episodes.jsonl)。
- [终态独立审核](../logs/a2_ideal_stairs/implementation/final_results_review.md)。
- [最终课程账本](../logs/a2_ideal_stairs/local_sds_20261110_221417/final.json)。

最终checkpoint：`/home/xxs/dog/unitree_rl_mjlab_a2/logs/a2_ideal_stairs/local_sds_20261110_221417/seed_42/chunk_0012_n512/checkpoints/budget_exhausted.pt`

SHA256：`83cbfb292b2f7fbbde9994fac4c1251c8a818d8f4500ce195461f047ccdc8cb2`

## 最终模型回放与恢复接口

无DISPLAY环境必须显式`MUJOCO_GL=egl`。首次遗漏该环境变量导致OpenGL初始化失败，日志保留在`logs/a2_ideal_stairs/final_replay_20261111/vx_02_failed_display.log`；没有修改模型或训练配方。以下回放接口已实际运行，输出目录必须更换为不存在的新路径。分别将`--vx`设为0.2、0.4、0.6运行，不能省略速度参数：

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
MUJOCO_GL=egl /home/xxs/anaconda3/envs/a2/bin/python -u scripts/play.py Unitree-A2-IdealHeightMap-Stairs \
  --checkpoint /home/xxs/dog/unitree_rl_mjlab_a2/logs/a2_ideal_stairs/local_sds_20261110_221417/seed_42/chunk_0012_n512/checkpoints/budget_exhausted.pt \
  --checkpoint-sha256 83cbfb292b2f7fbbde9994fac4c1251c8a818d8f4500ce195461f047ccdc8cb2 \
  --stage 1 --num-envs 1 --seed 1000042 --device cuda:0 \
  --vx 0.2 --vy 0.0 --wz 0.0 --max-steps 2100 \
  --output-dir "logs/a2_ideal_stairs/replay_reproduce_$(date +%Y%m%d_%H%M%S)" --video
```

最终模型的S1预算已经耗尽。恢复入口会检查哈希、模型、优化器和课程状态，随后保持global2400/S1 2000并以budget_exhausted退出，不会进行第2401次更新。`--max-iterations 200`只是worker上限，不会覆盖阶段预算。未经新授权不能通过重置课程或删除预算继续训练。

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
MUJOCO_GL=egl /home/xxs/anaconda3/envs/a2/bin/python -u scripts/train_a2_ideal_stairs.py \
  --task Unitree-A2-IdealHeightMap-Stairs --num-envs 512 --seed 42 --device cuda:0 \
  --mode train --max-iterations 200 --save-interval 100 --eval-interval 200 \
  --checkpoint /home/xxs/dog/unitree_rl_mjlab_a2/logs/a2_ideal_stairs/local_sds_20261110_221417/seed_42/chunk_0012_n512/checkpoints/budget_exhausted.pt \
  --checkpoint-sha256 83cbfb292b2f7fbbde9994fac4c1251c8a818d8f4500ce195461f047ccdc8cb2 \
  --eth-checkpoint /home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt \
  --vx-min 0.2 --vx-max 0.6 --vy 0.0 --wz 0.0 \
  --output-dir "logs/a2_ideal_stairs/resume_reproduce_$(date +%Y%m%d_%H%M%S)"
```

限制：S2–S5策略能力、40阶每速度≥95/100、43/44复验均未验证；S0–S4重复固定条件不代表独立鲁棒工况。地图缓存最大80ms无补偿的影响已有契约测试记录，但学习层面影响未做消融。NVML20ms观测峰值不保证捕获更短尖峰。回放视频的可解码检查不等同于完整人工视觉审查；外部vision.js识图被自动审批拒绝，未上传图像。

## 终态回放与恢复补证

2026-11-11实际运行以上回放接口，三个独立串行进程退出码均0。

| 指令vx | 终止/最高有效阶 | 回放控制步数 | MP4帧数/时长 |
|---|---|---:|---|
| 0.2 | timeout / 0 | 2031 | 407帧 / 40.7秒 |
| 0.4 | timeout / 0 | 1266 | 254帧 / 25.4秒 |
| 0.6 | timeout / 0 | 1011 | 203帧 / 20.3秒 |

录像均320×240、10fps；全部帧本地解码通过。视频时长与物理回合时长有10fps离散采样取整差异。它们是失败回放，不是新增100回合最终测试，也不参与选模。

- [0.2m/s录像](../logs/a2_ideal_stairs/final_replay_20261111/vx_02_egl/replay.mp4)
- [0.4m/s录像](../logs/a2_ideal_stairs/final_replay_20261111/vx_04_egl/replay.mp4)
- [0.6m/s录像](../logs/a2_ideal_stairs/final_replay_20261111/vx_06_egl/replay.mp4)
- [逐视频哈希及本地解码证据](../logs/a2_ideal_stairs/final_replay_20261111/local_video_validation.json)

实际执行恢复命令至新目录 `logs/a2_ideal_stairs/final_resume_check_20261111`，退出码0；[result.json](../logs/a2_ideal_stairs/final_resume_check_20261111/result.json)仍为budget_exhausted/global2400/stage1=2000/consumed=[400,2000,0,0,0,0]，没有updates.jsonl、没有增加预算。恢复另存checkpoint SHA为70fed5fcca64a885f324245c8b35419f6d1a54760095d5c5e294d64e17cdb556；新的manifest导致文件哈希不同，不替代原始终态模型作为报告主checkpoint。所有GPU计算进程已退出。
