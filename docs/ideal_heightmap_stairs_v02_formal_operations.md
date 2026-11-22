# A2 理想高程图楼梯 Formal 0.2：运行、续训与回放

本页用于操作已有正式实验。任务为 `Unitree-A2-IdealHeightMap-Stairs-V02-Formal`，协议 `v02-formal-climb-v1`，代码版本0.2.1、checkpoint schema3。运行中的主实验必须沿用原模型、预算和W&B身份，不能从C重新启动。

## 当前运行快照

截至 **2026-11-12 21:11:18（Asia/Shanghai），2400更新边界**：

|项目|实际记录|
|---|---|
|master PID|2092734；运行期间以 `active_supervisor.json` 为准|
|当前目录|`logs/a2_ideal_stairs_v02_formal/monitor_counter_restart_20261112_200716`|
|阶段/正式更新|R0；global_iteration=2400，stage_iteration=2400|
|实际累计环境步|58,982,400；训练并行512环境|
|当前操作|2400更新后的diagnostics；楼梯验证已完成，flat验证进行中|
|2400楼梯结果|vx=0.2/0.4/0.6各20/20攀爬成功|
|晋级状态|快照中的连续joint通过次数为0；不能仅凭楼梯结果宣布R0通过|
|2400模型|`formal_seed42/chunk_0002_n512/checkpoints/latest.pt`|
|2400 SHA256|`0af62be2be4372da0486022b5351a1003256dd51db05c9c0c5dc83c69243a392`|

此快照没有预填flat结论、R0晋级、最终验收或终态。后续进度以运行目录的新记录为准；三速首阶攀爬20/20不代表40级、停车或实机部署已通过。

实际已执行入口及参数来自 [launch_curriculum_counter_fix.json](../logs/a2_ideal_stairs_v02_formal/implementation/launch_curriculum_counter_fix.json)。当前绑定批准文件是 [approval_curriculum_counter_fix.json](../logs/a2_ideal_stairs_v02_formal/implementation/approval_curriculum_counter_fix.json)，不要替换成旧approval或改写已绑定源码摘要。

## 只读查看状态

当前master仍存活时，只查看状态，不执行第二个监督器、独立评估或GPU回放。

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
ps -p 2092734 -o pid,ppid,lstart,stat,args
tail -n 30 logs/a2_ideal_stairs_v02_formal/monitor_counter_restart_20261112_200716_supervisor.log
```

下面的命令只读本地JSON，不导入W&B SDK，不创建run。supervisor主状态通常在chunk边界更新；worker运行时，最新 `budget.json` 和遥测 `live_progress.json` 会更及时。恢复仍由脚本核对完整账本，不以日志行数人工补写消费。

```bash
/home/xxs/anaconda3/envs/a2/bin/python - <<'PY'
from pathlib import Path
import json
run = Path('logs/a2_ideal_stairs_v02_formal/monitor_counter_restart_20261112_200716')
state = json.loads((run/'supervisor_state.json').read_text())
curriculum = state['curriculum']
print('supervisor:', {key: state.get(key) for key in
      ('phase', 'status', 'iteration', 'env_steps', 'num_envs', 'checkpoint', 'sha256')})
print('curriculum:', {key: curriculum.get(key) for key in
      ('stage', 'stage_iteration', 'consumed', 'consecutive_passes', 'r0_extra_passed')})
print('master:', json.loads((run/'active_supervisor.json').read_text()))
workers = sorted((run/'formal_seed42').glob('*/budget.json'))
if workers:
    print('latest worker budget:', workers[-1], json.loads(workers[-1].read_text())['curriculum']['iteration'])
progress = run/'wandb_telemetry/live_progress.json'
if progress.exists():
    current = json.loads(progress.read_text())
    print('live:', {key: current.get(key) for key in
          ('global_iteration', 'stage_iteration', 'env_steps', 'num_envs', 'curriculum/stage')})
telemetry = json.loads((run/'wandb_telemetry/state.json').read_text())
print('monitor:', {
    'run_id': telemetry['run_id'], 'local_events': telemetry['sequence'],
    'acked_events': len(set(telemetry['acked'])),
    'pending_updates': len(telemetry.get('pending_updates', [])),
    'pending_episodes': len(telemetry.get('pending_episodes', [])),
    'online_status': telemetry.get('online_status'),
    'monitoring_blocked': telemetry.get('monitoring_blocked', False),
})
PY
```

[W&B运行页面](https://forge.coreweave.com/wandb/xingshenxu-/a2-ideal-stairs-v02-formal/runs/37b559428d704af09bf16d7ed646595d) · [八组监控面板](https://forge.coreweave.com/xingshenxu-/a2-ideal-stairs-v02-formal/reports/v02-formal-climb-v1-seed42-20261112_135450-·-Formal-SDS-1.2--VmlldzoxODI5Mzg3NA==)

同一实验的W&B ID固定为 `37b559428d704af09bf16d7ed646595d`，project为 `a2-ideal-stairs-v02-formal`，entity为 `xingshenxu-`。链接来自当前 `wandb_manifest.json`。监督器是唯一W&B写入者；不要另外调用 `wandb.init`，也不要手改ACK、SDK游标、事件ID或历史错误记录。

## 普通续训：恢复同一个目录

**仅在当前master及其训练/评估/录像子进程已退出、GPU上下文已释放后执行。** 本页给出的是launch工件中已实际执行过的普通resume接口；它不是邀请在当前活进程旁再启动一个副本。锁与PID检查会拒绝并发，不能删除锁或杀其他用户进程来绕过。

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
export MUJOCO_GL=egl
/home/xxs/anaconda3/envs/a2/bin/python scripts/run_a2_ideal_stairs_v02_formal.py \
  --task Unitree-A2-IdealHeightMap-Stairs-V02-Formal \
  --output-dir /home/xxs/dog/unitree_rl_mjlab_a2/logs/a2_ideal_stairs_v02_formal/monitor_counter_restart_20261112_200716 \
  --device cuda:0 --seed 42 \
  --num-envs-candidates 1024 512 258 --eval-num-envs 8 \
  --init-checkpoint logs/a2_ideal_stairs_v02/local_sds_20261111_1621/comparisons/C/chunk_0010_n512/checkpoints/budget_exhausted.pt \
  --init-checkpoint-sha256 7d94810bb86e6997fe2a0f03883a6a0e4ee594140bb3ff65deb86642c3938d9a \
  --max-iterations 26000 --save-interval 100 --eval-interval 200 \
  --vx-min 0.2 --vx-max 0.6 --vy 0.0 --wz 0.0 \
  --wandb-project a2-ideal-stairs-v02-formal --wandb-entity xingshenxu- \
  --wandb-run-name v02-formal-climb-v1-seed42-20261112_135450 --wandb-mode online \
  --review-report /home/xxs/dog/unitree_rl_mjlab_a2/logs/a2_ideal_stairs_v02_formal/implementation/approval_curriculum_counter_fix.json \
  --resume-run
```

这里的init-checkpoint参数保留原完整配置和C来源校验；`--resume-run`实际采用当前run保存的checkpoint和课程/消费账本，**不会从C重训**。`--max-iterations 26000`是原正式总上限，包含已经消费的更新，不是现在再增加26000。保持参数原始写法和批准文件路径，不额外加zero-preflight或monitor迁移的一次性参数，也不重跑已经完成的工程迁移helper。

首次从C策略迁移时，仅迁移完整Actor及其既有统计/std，Critic和Adam按正式任务新建。此后内部resume恢复Actor、Critic、Adam、归一化/统计、学习计数、随机状态、课程和预算；物理环境从新回合开始，地图/历史/接触等回合缓存重置，再恢复相应训练RNG。它保护恢复边界之后的训练随机序列，不宣称物理轨迹逐帧连续。

源码/完整配置、checkpoint SHA、协议/schema、父run或账本不匹配时应停止定位，不能关闭校验。监控连续120秒未恢复会要求当前PPO更新结束后保存暂停；恢复须同run ID补传并实际确认在线可见，不能静默offline或人工把待发事件标为ACK。

## 训练与验收参数

当前seed42、并行512；速度指令vx均匀[0.2,0.6]m/s，vy=0、wz=0。物理/控制/地图为200/50/10Hz，action scale=0.25；每次PPO更新48步rollout、5 epochs、4 minibatches。初始学习率1e−4、允许范围[1e−5,3e−4]、adaptive KL目标0.01；不在操作命令中调整奖励、探索或课程。

|阶段|级数|踏高m|该阶段更新上限|
|---|---:|---:|---:|
|R0|1|0.08|3000|
|S1|4|0.08|3000|
|S2|8|0.12|4000|
|S3|12|0.16|4000|
|S4|20|0.20|5000|
|S5|40|0.20|7000|

总上限26000；通过可提前晋级，未用预算不转移。每100更新保存，每200更新验证，阶段起点/晋级/正常终态/预算耗尽另存。当前2400边界仍在R0，本阶段剩余上限600更新。

每个joint gate使用同一checkpoint：楼梯vx=0.2/0.4/0.6各20回合，每档至少18/20；平地vx=0/0.2/0.4/0.6各20回合、每回合30秒，全无真实失败，站立平均三维速度模长<0.05m/s，行走平均机体vx相对误差≤20%。楼梯和平地同时通过且连续两次才算晋级；R0还需固定轻扰动案例每速度20回合、每档至少18/20。停车是独立诊断，不参与晋级或训练回报。

S5两次joint通过后固定触发模型，最终独立扰动测试每速度100回合、每档至少95/100，并做同条件平地四档验收；不能换模型反复试测试集。最终结果以真实报告为准。

资源档位固定 **1024→512→258（不是256）**，首个完整生命周期通过即选用；当前已选512，普通resume不自动升回1024。仅真实资源故障才按顺序降档，学习失败/速度误差不能触发降档。评估8→4→1环境只作资源降档，案例数不减。训练、评估、停车和录像使用单GPU串行队列，前一进程退出并释放CUDA后才能启动下一项。

资源守卫要求整卡显存≤7168MiB，进程显存≤min(5120MiB, 7168MiB−启动时其他占用)，host available≥4096MiB。NVML以20ms间隔采样，报告的是观察峰值，不保证捕捉采样间隔内的瞬时峰值；不能终止其他用户进程来满足守卫。

阶段预算耗尽未通过、最终验收失败、所有资源档位失败或不可恢复数值/接口错误都应停止并交付实际终态。不得自动加预算、改seed、改速度/奖励/终止条件或强行晋级；已经有final.json的终态run不能当作新训练追加更新。

## 独立回放/录像：接口模板与2400模型示例

**当前主队列运行时不要执行以下GPU命令。** 回放必须由root在阶段通过、终态或已安全停稳的串行空档安排；输出目录必须新建。当前仅用CUDA禁用的真实 `--help` 和源码核对了接口，下面的录像命令没有在本次文档工作中执行，不能称为已验证GPU录像。真实mp4、资源关闭及画面内容将在阶段通过或终态执行后补验。

通用模板：checkpoint、对应完整SHA与模型阶段必须匹配；vx只能0.2/0.4/0.6，num-envs固定1，回放seed固定200142，max-steps固定10000。

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
FORMAL_REPLAY_CHECKPOINT='<checkpoint绝对路径>'
FORMAL_REPLAY_SHA256='<对应64位SHA256>'
FORMAL_REPLAY_STAGE='R0'  # 按模型替换为R0/S1/S2/S3/S4/S5
FORMAL_REPLAY_VX='0.2'   # 按需替换为0.2/0.4/0.6
FORMAL_REPLAY_OUTPUT="logs/a2_ideal_stairs_v02_formal/replay_$(date +%Y%m%d_%H%M%S)"
MUJOCO_GL=egl PYTHONUNBUFFERED=1 /home/xxs/anaconda3/envs/a2/bin/python \
  scripts/play_a2_ideal_stairs_v02_formal.py \
  --task Unitree-A2-IdealHeightMap-Stairs-V02-Formal \
  --checkpoint "$FORMAL_REPLAY_CHECKPOINT" --checkpoint-sha256 "$FORMAL_REPLAY_SHA256" \
  --stage "$FORMAL_REPLAY_STAGE" --num-envs 1 --seed 200142 --device cuda:0 \
  --vx "$FORMAL_REPLAY_VX" --vy 0.0 --wz 0.0 --max-steps 10000 \
  --output-dir "$FORMAL_REPLAY_OUTPUT" --video
```

当前2400边界的真实R0模型示例（仍为**未执行的录像示例**）：

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
MUJOCO_GL=egl PYTHONUNBUFFERED=1 /home/xxs/anaconda3/envs/a2/bin/python \
  scripts/play_a2_ideal_stairs_v02_formal.py \
  --task Unitree-A2-IdealHeightMap-Stairs-V02-Formal \
  --checkpoint /home/xxs/dog/unitree_rl_mjlab_a2/logs/a2_ideal_stairs_v02_formal/monitor_counter_restart_20261112_200716/formal_seed42/chunk_0002_n512/checkpoints/latest.pt \
  --checkpoint-sha256 0af62be2be4372da0486022b5351a1003256dd51db05c9c0c5dc83c69243a392 \
  --stage R0 --num-envs 1 --seed 200142 --device cuda:0 \
  --vx 0.2 --vy 0.0 --wz 0.0 --max-steps 10000 \
  --output-dir "logs/a2_ideal_stairs_v02_formal/replay_r0_i2400_vx02_$(date +%Y%m%d_%H%M%S)" \
  --video
```

脚本采用独立环境确定性动作，`--video`写离屏 `replay.mp4`（10fps），并保留manifest、逐步诊断、result及资源记录；不是交互GUI播放器。它只回放一个代表回合，不等于每速度20/100回合的验证，也不代替独立停车诊断。三个速度的阶段通过/终态代表录像由监督器串行安排；不能把首个示例成功当成完整验收。
