# A2理想高程图0.2执行记录

依据 `doc/sds/SDS_A2_IdealHeightMap_Stairs40_v0.2.md` 文档修订1.1。代码版本0.2.0、训练配方0.2、checkpoint schema2独立记录。当前为工程整合/独立审核阶段；尚未开始正式对照，不存在0.2成功结论。

## 结构与设计

新任务 `Unitree-A2-IdealHeightMap-Stairs-V02` 由专用入口路由，旧任务与旧checkpoint契约不变。实现位于 `src/tasks/velocity/v02/`。十tile依次flat/P1/P2/P3/P4/S1/S2/S3/S4/S5，统一metadata驱动地形、解析地图、真实geom接触索引、出生及边界，保留24×8m布局、首阶坐标与terrain origin的1m偏移；2cm阶段仍为1cm厚开放薄板。

显式S0迁移只加载完整Actor（含已学习的地图/历史/fusion）、冻结归一化和std。来源 `logs/a2_ideal_stairs/local_sds_20261110_221417/seed_42/chunk_0002_n512/checkpoints/stage_1_start.pt`，SHA256 `7db7ea4f65d94aa762c6101bb1a8a06f61aa26e78af0622b7420712854f5efdb`。Critic及统计、Adam、课程、环境缓存均新建。内部resume必须校验版本、配方、模式、输入/行为源码、PPO和来源身份，恢复模型/优化器/账本/RNG，从新回合开始，不宣称物理轨迹连续。

所有组统一修正停车：已锁存且当前机身x≥Nd+0.8、四足平台安全支撑、|y|≤.2、速度模长<.05、无失败，连续150个控制周期才成功。退线清零计数、保持零命令，重新满足后重新计150步。该定义与旧版历史训练不同，不把A组称为完全同一历史任务。

## 四组和公式

A单级8cm、旧奖励/探索；B按2→4→6→8cm课程、旧奖励/探索；C相同课程加新奖励；D在C基础上调整std。四组seed42、同一资源档、各最多2000更新、独立S0初始化，顺序执行。

B/C/D更新1–400为2cm，401–800为4cm，801–1200为6cm，1201–2000为8cm；比较阶段10%平地+90%当前，最大余数分配，平地内站立/行走各半（奇数确定性分配）。切换必须清空回合及所有缓存，不重置优化器。

A/B保留上一版连续奖励和+10成功/−4失败。C/D对active命令的线速项减exp(−‖cmdxy‖²/.25)、角速项减1、pose项减1（原权重分别1/1/.2），停车/站立不扣基线。连续奖励仍乘dt=.02；该变化可能改变可变终止任务权衡，不能称为无影响平移。

C/D另加每足每级首次有效落足+0.25：独立同(tile,step)连续两控制采样计数，真实顶面接触合力>5N且安全余量≥.042m，末阶/平台共面合并。地面/前阶计数不能继承，失接触/余量不足清零，重复踩同级不再领奖，失败帧不发正奖。公共support保持各组一致。

C/D进度：`delta=min(max(min(x,Nd+.8)−x_max,0),max(vx_cmd,0)*.02)`，楼梯active、≥2足有效支撑、未锁存且无失败时奖`2*delta`，不再乘dt。不论是否发奖均更新x_max，禁止后退重领、空中越过后补领、超过限额后原地补领。其他奖励与终止不变。

A/B/C std继承S0并限制[.05,.30]；D初始化max(S0std,.20)、全程[.10,.30]。均不改动作均值、动作缩放或力矩。PPO48步、5epochs、4minibatches、lr1e-4、KL目标.01与lr[1e-5,3e-4]、gamma.99、lambda.95、clip.2、entropy.005、梯度范数1。

## 验证和停止

对照每100保存，在0/1800/2000统一P4（1级8cm）每速度.2/.4/.6各20回合。0仅基线；末两次均各≥18/20才进入第2000权重的扰动验证（seed300042，各20）及平地回归（站立/三速度各20×30s）。案例预生成保存并四组复用；固定初态重复不称独立工况。全部A–D完成后按A→B→C→D选择满足全部门槛的最小改动；总奖励不参与选择。

全部失败即停止。若选中配方，从同一S0重新初始化一次seed42正式训练，不能续接对照权重。A从P4开始最多25000更新；B/C/D从P1开始最多28000。阶段上限1000/1000/1000/2000/3000/4000/4000/5000/7000；每阶段局部200验证、连续两次各速度≥18/20晋级，100保存。阶段耗尽不达标停止。仅S5通过后做独立seed400042每速100回合，各≥95成功；不利用测试选模型，不自动换checkpoint重测，不训练43/44。

## 资源与证据

独立进程1024→512→258探测，5完整更新预热+50测量、复杂S5接触、完整rollout/Adam、保存、串行60回合评估和新进程恢复更新。首个通过档供四组共用；probe不进入正式初始化。batch49152/24576/12384，minibatch12288/6144/3096，不同规模不是等价比较。整卡≤7168MiB，进程≤min(5120,7168−启动其他占用)MiB，host available≥4096MiB，NVML20ms采样包含Warp但不保证捕捉更短尖峰。

四组资源失配停止本轮比较，不偷偷降档或重置预算；正式阶段允许有账本的资源降档、新目录/父权重哈希、新回合恢复。学习失败/逻辑错误不能靠降档掩盖。训练与评估始终串行，退出后核对GPU PID释放。

实施/审核证据在 `logs/a2_ideal_stairs_v02/implementation/`：progress.md映射条款与文件归属，baseline.json保存原dirty源码指纹；environment_report.md、rewards_report.md、model_runner_report.md及review_*.md记录实现/独立检查。初步58项CPU测试通过，另2项案例隔离/进程去重检查通过；2env完整GPU PPO保存恢复一致；eval_smoke三速各20全0成功，仅接口检查，不作为四组训练证据。录像可解码不等同视觉审阅。

## 已核对入口

`python scripts/train.py Unitree-A2-IdealHeightMap-Stairs-V02 --help` 与play路由help已执行。正式启动命令、run/PID/资源和结果在独立审核闭环后追加。任何output-dir必须新建；不覆盖旧结果。训练supervisor使用完整SDS固定预算，worker的max-iterations仅每块上限，不能覆盖课程总预算。

补充迁移证据：`implementation/migration_actual_1000.json`记录全十阶段实际仿真各2环境×50控制采样，共1000条观测；A/B/C/D确定性均值与来源S0最大误差均0，重力单位范数检查通过，未执行PPO更新。此前CPU随机合法形状测试与这份实际物理观测证据分开记录。旧任务原checkpoint经旧入口真实执行1步通过，保护兼容性。

## 2026-11-11 本机执行记录

独立Sol/high三模块审核与R1–R4复核通过；approval.json绑定29项行为源码。run `logs/a2_ideal_stairs_v02/local_sds_20261111_1621`，supervisor PID 1129815。当前开始1024→512→258完整生命周期探测，尚未宣布任一档通过或学习成功。

```bash
cd /home/xxs/dog/unitree_rl_mjlab_a2
MUJOCO_GL=egl /home/xxs/anaconda3/envs/a2/bin/python scripts/run_a2_ideal_stairs_v02.py --task Unitree-A2-IdealHeightMap-Stairs-V02 --output-dir logs/a2_ideal_stairs_v02/local_sds_20261111_1621 --device cuda:0 --seed 42 --num-envs-candidates 1024 512 258 --eval-num-envs 8 --comparison-max-iterations 2000 --save-interval 100 --eval-interval 200 --init-s0 logs/a2_ideal_stairs/local_sds_20261110_221417/seed_42/chunk_0002_n512/checkpoints/stage_1_start.pt --init-s0-sha256 7db7ea4f65d94aa762c6101bb1a8a06f61aa26e78af0622b7420712854f5efdb --review-report logs/a2_ideal_stairs_v02/implementation/approval.json
```

恢复同一未结束run使用上述参数另加 `--resume-run`；必须先确认原supervisor及子进程已停止，禁止并行重复启动。运行状态以该目录process记录、updates和final.json为准。

## 2026-11-11 — 512环境完整生命周期探测通过
- 证据：local_sds_20261111_1621/selected_resources.json及probe目录。1024首次触发整卡7315.19MiB/进程5188MiB限制，失败退出并确认CUDA释放后尝试512；原始采样进程最终峰值5196MiB。
- 512：5预热+50测量，median2.99889s/P95 3.26789s、8138.70环境步/s；训练整卡峰5409.19MiB、进程3172MiB、PyTorch allocated1891.23/reserved2598MiB、RSS3007MiB、host可用最低14304.81MiB。串行S5各速度20回合完成，恢复优化器再更新1次，均无资源错误。NVML20ms为目标采样周期，并非连续硬件峰值。
- 采用512，batch24576、minibatch6144。探测含恢复共56更新1376256样本，独立于对照；四组总上限8000更新196608000样本，正式A上限614400000，B/C/D上限688128000。不继续探测258。
- A组新初始化从规定S0，开始训练前基线评估；目前没有首阶学习通过结论。

## 2026-11-11 — A组完成2000更新，末期验证失败
- 证据：local_sds_20261111_1621/comparisons/A。512环境，2000更新=49152000样本；fixed1800、fixed2000的0.2/0.4/0.6m/s均0/20。第1800验证60回合全timeout、最高有效支撑0。
- fixed1800 SHA256 0c3a0eb23cedb2ce5f5cd6b3998bd829678cefd460d1c655d9236a2cb456e75e；fixed2000 SHA256 dce49466b6373b0c5bdfb8d39faebda49c51d2ac04c57a3f264a8b6ff0bb4ff3。预算终态另存21c927347b4ff782af725ea0bacd44910fe1434eaf8442ccfc3c613e4eb279f4用于三档录像，独立审核核对状态一致性。
- A未达到前置门槛，不进入扰动/平地附加门槛，不具备胜出资格；没有延长预算或改配方。B已从S0重新初始化、相同512规模开始训练前评估。全部对照尚未结束。

## 2026-11-11 — B组完成2000更新，低速门槛失败
- 证据：local_sds_20261111_1621/comparisons/B。512环境，2000更新49152000样本，按1–400/401–800/801–1200/1201–2000分别2/4/6/8cm。
- 实测：fixed1800在0.2/0.4/0.6m/s为1/20、20/20、20/20；fixed2000为0/20、20/20、20/20。低速门槛失败，不能按中高速或平均率选B，不进入附加扰动/flat门槛；固定初态重复不称独立随机工况。无预算扩展或配方变动。
- 三速录像正在完成；Sol/high独立结果审核已调度，C/D尚待执行，不宣称总流程结束。

## 2026-11-11 — C组完成，附加门槛失败，D组开始
- 证据：local_sds_20261111_1621/comparisons/C。512环境2000更新49152000样本，固定1800和2000三速度均20/20；同2000模型SHA256 3f198c52ce6fce4194e2711e42e797ce53080c109932c417fcd9da9fafe9378d执行扰动和flat。
- 扰动seed300042三速度17/20、20/20、18/20，低速未达18/20，不能选C。
- flat完整80例：站立mean_speed .0028475通过；0.2/0.4/0.6命令下mean_vx .263095/.308133/.471664，相对误差31.55%/22.97%/21.39%均超20%。前60例完整30s；高速8次持续非足接触、12次越界，平均21.026s，是实际提前失败而非缩短horizon。
- 录像已生成，独立Sol/high审核正在收尾；D已同512从S0新初始化并开始基线评估。A/B/C均不满足全部候选门槛，不更换checkpoint或扩大预算，尚待D结果。

## 2026-11-12 — 终态与最终证据

实际run已于11月11日23:57前结束：`all_comparisons_failed`，supervisor退出0、所有子进程GPU释放。D两次末期均0/20、20/20、20/20；低速上台后退到停车线前，按严格停车契约失败。四组各2000更新/512环境/seed42，总196608000 PPO样本；无合格配方，未启动正式课程或最终测试，没有扩大预算。

最终完整报告：`logs/a2_ideal_stairs_v02/results_20261111_2358/report.md`；同目录summary.json、evaluations.csv、checkpoints.json包含真实资源/耗时/逐速度/全SHA。最终独立审核：`implementation/comparison_D_review.md`、`implementation/terminal_review.md`（相对logs/a2_ideal_stairs_v02）。恢复/回放参数见报告；原run已终态，不可通过resume重置或扩展预算。前述“开始/正在/尚待”段落是保留的历史进度，最终状态以本节和final.json为准。

终态独立审核统计补正：上文“探测含恢复共56更新1376256样本”仅指512通过档位。1024失败探测此前已完整更新1次、49152样本；全部探测合计57次、1425408样本，仍独立于8000次对照预算，不能遗漏失败探测工作量。
