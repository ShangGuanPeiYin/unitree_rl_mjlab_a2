# A1最终工程验收与SDS证据映射（2026-11-27）

## 结论和精确身份

A1工程实现经统一CPU回归及分工独立复核通过，可进入已批准的限额物理验证。**这不是资源补证、教师能力或行为验收通过；签本文时新增物理0、正式学习0，旧失败run仍stopped。** 旧平台0/20和原始result/NPZ原样保留。

- 最终实际源码：`2f82589e805b2579843a4381089467d40931300aff8a80f466becf90dcc4fbcb`。
- 冻结SDS v1.1：`eb95905ed24bf0674f94a1b02dab02256ee11e40a20ed1cd96eb24daeb1b49a0`，不改旧SDS。新增A1契约及执行裁定以v10_A1_blind_yaw_contract_20261127.md和v10_execution_rulings.md关联。
- 旧实际probe源码b48与新源码共295文件：12项A1已有文件修改、11项A1新增、272项保持实际原字节。将12项还原旧Git字节并排除11新增，保留所有既有dirty/untracked实际字节后，精确重建旧b48；没有把Git07fe等同旧dirty工作区。
- 对动力学/模型/观测/地图/奖励/回放/优化等42项核对：41项字节相同；env整文件SHA明确不同，仅精确白名单recording-only frame.update节点及位置允许插入，移除后其余完整AST同。完整env字节并未宣称相同。

## 改动及设计理由

旧blind直接读取原学生45维proprio中的(vx,0,0)。新blind仍冻结原ETH网络/归一化，只有私有clone命令6:9改成(vx,0,wz)：

```
y=.2*route0; psi=atan2(route1,route2)
psi_d=clip(-atan2(.2*y,max(vx,.1)),-.15,.15)
wz=clip(.5*atan2(sin(psi_d-psi),cos(psi_d-psi)),-.15,.15)*clip(vx/.3,0,1)
```

vx<=0全部为零；F/P/posttop所有blind调用同规则，不看隐藏task/phase、不增privileged反馈或状态、不另增vy。学生请求、同帧/history/上一实际动作和S2教师不变。新增经验参数待物理验证；原ETH权重缺原训练cfg，normalizer统计不能证明训练命令上下界。没有声称纠偏已改善行为。

新增teacher身份包含两权重SHA和A1配置SHA；checkpoint/cache/评估/控制器按相同身份校验。只凭原两权重或旧code SHA的结果不沿用。资源采用规划批准的显式关联协议，旧完整证据+新短测必须精确关联，短测不伪装原20测量完整复验。原完整probe规则保持1fill+5warm+20measure。

trace v2保留原评分rows20，分开记录action-time请求/私有命令/route/yaw和post-step yaw/feet/raw/top/独立failure masks；首终止冻结，不记录padding。采用按上界紧凑预分配数组，避免逐帧字典导致内存增加。根独立执行精确新增记录节点，在同一合成状态下验证张量内容/version、物理位置、proprio/history、教师输入及Python/NumPy/Torch RNG不变；这是CPU节点证明，没有冒称新物理轨迹重放。

容量预留与Q1实际step扣费分离；同ledger stage锁独占，stale/pending拒绝自动恢复。只扣除持久SHA/资源通过/模型含A1/精确病例与协议身份完整的已完成评估，容量节省和实际消费共用同一判断。partial或错配不减容量，完整行为失败仍停止。课程提前通过同SHA A/B时，以单promotion事件原子记课程完成和未启动的后续本课程门禁取消；保存不可变grade证据，幂等恢复、共享teacher cache保留，不退费用或转移学习预算。

## 实际验证与边界

最终全部15个test_v10*.py纳入CPU统一回归：159 passed、5 skipped、1上游torch.jit.script弃用warning，87.45s。显式移除V10_RUN_GPU_TESTS/V10_FULL_REPLAY_CPU开关并禁CUDA可用检测；三项GPU环境、一项3GB完整CPU replay恢复、一项CUDA microbatch内存测试跳过，不能算通过。原裸pytest/全项目历史收集冲突仍未整体验证，依规划不扩修旧测试。

定向RED/GREEN、最终完整CPU日志及文件SHA清单在logs/a2_v10/A1_engineering_20261127。8daf/146和d90/156均为中间证据，已单独归档，不当作最终2f版本。根审发现并修复过跨目录重复诊断、trace逐帧对象内存、证书拼接/缺引用、容量扣减身份不全及取消事件遗漏；最终独立审核须以2f实际源码为准。

## SDS→实现→测试映射

| 条款/裁定 | 最终实现 | 证据 |
|---|---|---|
| A1精确公式、统一observable rule | teachers.py私有clone；无hidden task门控 | teacher_yaw20项、真实ETH适配输入手算parity、第三方教师审核 |
| 学生帧时序/历史/S2不变 | env既有observe/TaskState、DualTeacher原stairs路径 | learning_core/runtime/env_contract、42项不变证据及旧实际b48重建 |
| 几何/地图/安全终止/奖励/损失 | geometry/contacts/tasks/rewards/dagger/ppo/runtime原字节 | exact文件SHA、env仅记录节点AST等价；未新物理 |
| Q1/Q4保存恢复与身份 | checkpoint新teacher身份；其余完整保存/校验/释放/恢复共用旧执行体 | a1_identity/runtime_memory/probe_budget；实际满库恢复待短测 |
| Q2评估padding、冻结和统计 | eval_worker/TraceBatchV2，原rows20不变 | trace8项、真实run_suite假环境16=10有效+6padding |
| A1限额resource关联 | resource_a1显式独立协议，完整真实新两栈结果/旧SHA/差异和不变证据关联 | resource_a1篡改/缺引用/重复stack/原始result校验、单次全局claim |
| 同cap授权承继与跨run费用 | probe_budget严格父SHA/id/cap/ledger/SDS+全局journal | supersedes错误/并发/幂等测试；生产账本只读3293184且余额540672 |
| 固定112诊断与预算 | diagnostic_a1+evaluation_capacity；resource gate先行、O_EXCL once claim | 112原case SHA/max_steps=251288；错resource、竞争、stale/pending和跨目录再试拒绝 |
| 门禁/终验/恢复 | orchestration同身份消费/容量、AB成对预留、final开集前查额 | controller错身份/重复缺失病例/partial/行为失败/合法仅剩reference边界 |
| 晋级取消事件 | _promote_course单EventJournal事务、不可变同SHA AB证据 | 新3项含仅未启动、本课scope、0扣/退/转、重复及旧快照恢复、末门禁不取消 |

## 已批准的实际执行流程和关键参数

1. N1024两栈针对性资源补证，各1fill+2warm+2measure=245760，共491520。全196608 FP32 replay、Adam、真实save/CRC/SHA/free/load和峰值不清零，原GPU/host阈值保持；任栈失败停止，不凑余量重测/回退。原probe cap3833856，历史3293184保留，完成两栈后最大3784704、余49152。
2. 固定112诊断：平地10组+H2/G宽1.2/.8各9平台组，每组原A病例index0..3，上界251288。逐组4/4才继续，不替代正式18/20，不调增益/重复。
3. 新教师D1/D2/D3 A/B全套准入：D1各7225680上界（A/B14451360），D2合21392720，D3合45539120；全部需要通过。原平地200/200不复用。
4. 正式学习沿原SDS：N1024、rollout48；DAgger D1最多500轮/24576000样本（门禁200/400/500），D2最多1000轮/49152000（400/800/1000），D3最多1500轮/73728000（600/1200/1500）；DAgger独立终验通过后PPO总最多1000轮/49152000，含P0冻结Actor100轮、后续P1，门禁250/500/750/1000。总学习196608000，不跨课转额；vx请求.3/.5/.6，学生vy/wz仍0，LR仍3e-5。每2457600学习样本（50完整rollout更新）保存，入口/出口/门禁/终验即时保存与周期去重。
5. 原eval cap1685390840、已耗304100保留，所有新评估共享旧账本。固定完整清单含新诊断及历史的零缓存最坏1685946228，潜在缺555388；规划允许按阶段容量继续，不保证全程足额。只认实证缓存或持久取消事件，不足即停、不删组减例降门槛。

正式物理/教师/学习结果由后续运行报告追加；此工程验收不会将尚未运行的门禁写成通过。
