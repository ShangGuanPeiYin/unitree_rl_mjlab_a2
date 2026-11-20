# v0.9 S1 资源、证据与后续学习独立审核（旧 447 / Q009–Q010）

审核日期：2026-11-20。此文件是新增范围的独立审核，**不改写或扩展原 S1.0 批准**，不是资源通过、学习批准或行为验收。审核员未修改生产代码、阈值、病例、冻结源码、既有签字、账本或 Git index；此次只新增本报告和本机身份清单，修改日志由主 agent 汇总。

## 1. 身份和当前结论

- 项目：`/home/xxs/dog/unitree_rl_mjlab_a2`；实现提交 `c5b2878cc747ef675e0a637a8146138d69cd8e7a`。
- SDS SHA256：`500ad32aaa946c927459427f3a65565b15d4b77fca4be214817121d15c742976`。
- 旧执行源码：`44780cd5d6f9ee4029fb38b2f9348a872268e7bedf995d4b36e2e30ec2f2163f`，冻结文件 `logs/a2_v09/s1_seed42_r3/source_freeze_s1_q009_final.json`，322 个文件。
- run：`a2-v09-s1-seed42-r3`，目录 `logs/a2_v09/s1_seed42_r3`。
- 完整预留计划 SHA：`fa439aac08c47fda060413cbae2942ce92ef39194d49bf850dd68a9f5ea7b418`；非学习预留 124,142,190，其中 S1_RESOURCE 16,030,340，工程额度未预留余额 25,857,810。校准义务 limit=1,000,000、600 个病例完整覆盖 target=945,800。
- 唯一教师 6400 SHA：`30ed23ba16f7e490afca001fbbecadb4a11ff4b11733a5ae7ac7d0c841f9a8b2`。

**当前不签发资源执行批准，也不签发 S1.1/S1.2 学习批准。** 已确认旧资源 CLI 存在确定性入口阻断 R-RES-ENTRY-01。真实 S1.0 基线、校准及 finalize 尚在执行流程内，真实资源证据尚不存在。原批准只允许原教师五套基线和 600 病例校准；其范围和有效性不因本报告改变。

主 agent 已转达规划 Q010：先用旧 447 完成全部已批 S1.0、结账 finalize 并封存，再修新范围。旧运行进行中保持源冻结；这也是本次不修代码、不发新资源批准的依据。Q010 尚待后续记录进入新快照，本报告只记录收到的正式执行条件，不声称它已属于旧 447 源码。

## 2. 确认阻断：R-RES-ENTRY-01

`scripts/probe_a2_v09_s1.py` 读取 gate_A 后筛选 L40、0.3 m/s 源病例，却要求筛选结果数量等于 `EVAL_ENVS=100`。冻结 gate_A 该组实际有 **20 个唯一源病例**，每个 hard timeout 为 5005。Q009 的 100 是扩展后的物理并行槽数；`resource_assessment_cases` 才负责将 20 个源病例各复制 5 次。CLI 在调用该函数之前已经抛错，dry-run 和真实 `--execute` 均不可达实际探测。

独立零物理复现与磁盘证据 `logs/a2_v09/s1_seed42_r3/resource_entry_reproduction.json` 一致：`gate_A resource assessment count/timeout differs`。复现仅将 calibration loader 换为身份 fixture，没有真实 sV、没有执行环境构造或物理步，不能算资源准入测试。该错误不在教师评估或校准生成路径内，不因此撤销原 S1.0 准入。

后续修复验收必须直接覆盖真实 CLI：20 个源病例的 SHA/病例身份唯一、L40/0.3、每个完整 5005 步；扩展后 100 槽、20 源各出现 5 次；1 槽录像使用指定源病例。拒绝缺失、重复源或错误 timeout；不能通过放宽为任意 20 条记录完成验收。新源、新真实校准或经 Q010 严格绑定的旧校准通过后，才能重新审核资源执行准入。

## 3. Q009 资源执行合同及审核证据

已逐段读取 `resources.py`、资源主 CLI、`a2_v09_resource_worker.py`，并复核 runner 的 offload/restore 对应实现。已有 CPU 证据支持以下代码合同，但不证明真实显存或吞吐满足：

1. 候选按 2048、1536、1024、512 递减。每档 5 次热身、50 次测量；一次是完整 `N×48` 控制样本和 `5×4=20` 次逻辑 Adam，而非一次前向。
2. 保存恢复后额外完整 N×48/20 更新切到 S1.2，检查所有可训练参数梯度与 Adam 一二阶状态；std 不进入 Adam。原地图动量、冻结统计保留。
3. 训练 env/buffers 释放，原模型和 Adam 状态移到 CPU，独立子进程执行 100×5005 评估和 1×5005 录像，包含终止后的物理 padding。它们是工程负载，100 槽仅来自 20 源复制，不是新的 100 个独立行为病例。
4. 子进程只读取并验证父进程已预留的精确 obligation/work-item、模式、checkpoint、槽数、步数和版本；不 reserve/commit。父进程单次结算，异常未知按上界付费，不能子父双付或免费重跑。
5. 子进程退出后恢复同模型/Adam 数字指纹及 Python、NumPy、Torch、shuffle RNG，开启新的物理回合并保留单调 role IDs，再执行完整 N×48/20。普通保存不触发回合重置。
6. 全卡 NVML 与 CUDA UUID 规范化比较完整 128 位身份；轮询默认 20 ms。训练、完整 S1.2、卸载后的评估、录像、恢复训练分别记 phase。峰值比较互斥布局的最大值，不能将互斥阶段错误相加。
7. 要求全卡保守峰值 ≤7168 MiB、主机 MemAvailable 最低 ≥4 GiB，全部规定负载、恢复与监控成立。监控失败、非有限、接触 overflow、工程一致性错误停止；只有 OOM 或确认资源阈值超限允许降档。
8. 结果、source snapshot 和总 admission 先持久化，才关闭选中 probe obligation，再写独立 settlement。中途输出失败保留已付费与未闭合状态，不以缺报告为理由自动重跑。

每档计划为 55 个 N×48，再加 N×48、100×5005、1×5005、N×48，59 个 work items。四档总计 16,030,340。资源义务独立属于 S1_RESOURCE，不能消耗 S1.0 的 50,000,000 上限或学习预算。

本轮使用 `CUDA_VISIBLE_DEVICES=''` 独立运行 resources、diagnostics、s1_runner、video 四个测试文件：**62 passed，25.14 s**，一条 torch.jit.script 弃用警告。最初误写不存在的 `test_v09_runner.py`，该次未收集任何测试；纠正为 `test_v09_s1_runner.py` 后取得上述结果。测试涵盖 mock 物理接口、调度、状态和伪造错误反例；审核员本轮没有真实 GPU 负载、MuJoCo 物理或真 env renderer 执行，不能声称资源稳定或策略效果改善。

## 4. 新学习准入之前必须完成的真实证据

S1.0 必须具有两套 gate 教师缓存各 1500 例、三套 sealed final 各 7500 例，共 25,500 个首终态记录；600 个校准病例的真实首回合有效 value 样本，`sV=max(1, population_std(V_teacher))`，padding 不进入 value 分母。原始 batch、geometry certificate、episodes、summary、校准文件及其引用内容 SHA 需要与实际账本闭合、finalize 绑定。不能用 fixture 的 sV 或未闭合校准替代。

S1_RESOURCE 需在入口修复的新源码、合法旧证据桥接和单独独审执行批准下运行；选中最大稳定 N 的真实完整报告、恢复指纹、视频验证、phase 峰值、无工程异常和外部 obligation closure 必须独立验收。只读 NVML 查询、CPU mock、编码器的独立短视频不能代替该证据。

学习 CLI 的独审批准必须严格绑定新 code SHA、SDS、唯一 6400、run_id 和 resolved run_dir，并明确同时包含 `S1.1_learning`、`S1.2_learning`。已查验其拒绝 S1.0 engineering-only 批准；不能拿原 S1.0 JSON 启动自动两课程 driver。未来有条件批准 S1.2 工程入口仍不等于其行为通过，S2/S3 继续未授权。

## 5. 初始、S1.1 结束及后续证据流程

Q008 初始在任何正式学习前保存精确迁移 checkpoint；独立 S1_DIAG 1500 个固定病例记录真实初始学生状态及同状态 6400 标签，并分别为 6400、初始化学生录像每组 case0 共 75 段。完成证据再开始学习。已完成的零学习恢复只可核验并复用既有 journal、checkpoint、summary 和三项闭合义务，不能再次采集收费；唯一保存名保护旧证据引用。

S1.1 上限 20,000,000，只进行地图分支适配；教师/学生执行比例按回合 50%/50%，归一化和 std 冻结，教师永久 eval/no-grad。损失保持 `Σ_bΣ_j(a_student-a_teacher)²/(B×12) + 0.1×Σ_b((V_student-V_teacher)/sV)²/B`；每个逻辑 minibatch 按其完整样本数 B 归一，末尾不均匀微批按真实样本加权，一次累积完成才 clip=1 和 Adam 一次。学习率固定 3e-5，每迭代 N×48、20 次逻辑更新；未加入 PPO、奖励或配方修改。

S1.1 到达该 N 最大完整 rollout 数后，后测使用同一 S1_DIAG 1500 病例；两模型交叉计算同一 before 数据和同一 after 数据上的损失，另报各自轨迹与配对行为。不能用不同状态的轨迹损失下降代替固定数据比较。初始权重逐 tensor 核对指定 6400 迁移，compiled geometry/contact index 证书需要实际通过；finite/overflow 检查均是实际证据条件，不能单写 `contacts_valid=True` 代替。S1.1 工程数值完整后才解冻 S1.2，新回合教师比例降至 10%，保留原地图 Adam 动量、全部 normalizer 和 std。

S1.2 主段上限 630,000,000，最多两次各 100,000,000 恢复，未用课程额度不借用。每档 cap 取 `floor(cap/(N×48))` 个完整迭代；N 未实测选定前不预写某一档迭代数。checkpoint 每跨 5,000,000 样本完整迭代边界保存，1亿归档。主段 1/2/3/4/5/6亿及 6.3亿末尾，恢复各 0.5/1亿为原门禁槽，最多 11 次；A/B 相邻通过必须同一冻结 candidate SHA，确认期禁止学习；只在此前置成立后解封下一 sealed final。每 final 75×100=7500 固定分母，最多三套；Q001 教师有效窗为空必须阻断规划，不能自动按失败恢复或改阈值。未获真实完整通过不得写 S1_RESTORED。

## 6. Q010 兼容桥接的独立验收要求

收到的规划要求只批准本次单向复用，必须满足以下可核验条件后才能签对应新准入：

1. 先完成旧 447 的所有已批 S1.0 操作，外部结账、finalize、完整封存；旧 447 源和所有旧证据的原生产者字段原样保留，sealed final 不提前解封、不再计费。
2. 基于旧源码逐文件字节归档和新快照做真实 diff。允许差异限资源入口、消费端兼容校验、Q010 记录及不改变旧生成结果的明确项。逐项证明教师基线/校准生成和评分路径、依赖/资产/动力学、6400、病例配置、SDS 行为裁定不变；不能仅凭实现者称“无行为变化”。
3. 新快照形成后，独立生成不自引用的直接旧→新绑定清单：完整旧/新 code SHA、run_id、明确允许证据路径及内容 SHA、manifest、外部账本封存身份、逐文件差异及结论。不能用匹配目录名、仅短 SHA 或可任意扩展的证书代替。
4. 所有消费旧证据的入口验证同一独审清单。旧证据验其真实原生产者 447 和内容；当前执行、资源/学习审批、候选 checkpoint、新评估严格验新 SHA。禁止通用忽略 SHA、通配/传递兼容、重写旧 JSON 冒充新证据；旧资源结果不在该旧 S1.0 复用范围。
5. 回归必须包含合法桥接、错旧/新版本、内容篡改、病例变化、旧资源冒用拒绝。不能证明的项只阻断受影响下游；预算内重跑对应证据，只有派生汇总受影响时允许从已验证原始证据重算，不重写原始数据。

当前只是列明验收条件，没有生成兼容批准清单，没有授权新源码执行。待旧 S1.0 完成后需按真实磁盘数据逐项完成审核。

## 7. 旧运行依赖/资产身份的只读封存

旧 `code_snapshot()` 的后缀包含 Python/XML 等，但不包含 A2 STL，也不包含环境外的 mjlab 等依赖。已查 `a2_constants.py`：A2 XML 通过 `update_assets` 读取同目录资产，XML 引用 17 个 STL。因此本轮另外生成本机清单，不把它们塞回正在运行的旧 source snapshot：

| 清单 | 文件 SHA256 | 覆盖 |
| --- | --- | --- |
| `logs/a2_v09/s1_seed42_r3/old_runtime_identity_447_20261120.json` | `2375af6769d864a586cbe85396f9cfea89c614a542b12358b7e90534f774c7c5` | 15,406 文件，5,597,352,515 字节；A2 全部资产；mjlab、mujoco、mujoco_warp、warp、torch、numpy、PIL、imageio_ffmpeg 八关键包非缓存文件；所有已装分发 metadata；活跃教师进程实际 mapped 常规库；Python native extension、conda metadata、执行绑定 |
| `logs/a2_v09/s1_seed42_r3/old_runtime_python_sources_447_20261120.json` | `a3f4c4bf410b6196407190cec1d453e6afb2d0d3e973d69d293b4b6fe7544400` | a2 Python3.11 stdlib/site-packages 全部非缓存 .py/.pyi，23,611 文件、251,206,241 字节；hash 过程中零变化 |

第一份内容身份摘要 `de1b36905f6820f2c1d350d00f8ba9387d198e1c0182c549849e796e76348140`，第二份 `41ddaffe0af88d3deb5eb568d51db5e0deba314d5fb41efc8daa916db106b91d`。第一份采样实际教师 PID 3018111；同时重验全部 322 源码文件匹配旧 447。

第一份 `errors` 记录 10 个非普通文件例外：7 条 `/dev/zero (deleted)` 匿名设备映射，以及 `/dev/nvidia-uvm`、`/dev/nvidia0`、`/dev/nvidiactl` 字符设备不能作为常规文件 hash。这不表示丢失 10 个依赖代码文件；这些设备不能通过普通文件内容 SHA 证明驱动内核状态。其余被枚举的常规文件均成功 hash。

覆盖边界必须保留：清单不是进程 `sys.modules` 转储；八关键包以外未被采样进程映射的其他 native 文件、后续才加载的非 Python 资源及运行生成缓存尚未形成完整实际闭包证明。纯 Python 源完整补录也不等于所有可能条件分支的非 Python 资产都已执行验证。S1.0 finalize 后须复核已记录身份不变、补查剩余实际依赖并绑定最终证据，不能仅凭当前采样宣称 Q010 全环境兼容成立。

主 agent 另已归档旧 322 源码完整字节至 `source_archive_44780.tar.gz`。本轮只读独立重算归档 SHA `f62f12d5e9513072c2d315d06fa05bb637a82a856515af7c78118a8410b00d28`，并逐条读取 tar 中常规文件，文件名集合及全部 322 个内容 SHA 与旧冻结清单严格相等；没有解包写盘。该归档可以作为后续真实逐文件 diff 的旧端。

## 8. 本次范围与后续裁定

本轮新增的静态/CPU证据支持继续准备新资源修复与严格兼容审查，尚无策略效果或真实资源改善结论。资源执行批准待 R-RES-ENTRY-01 修复、新源 freeze、Q010 绑定、真实闭合校准等证据齐全后另签。学习批准继续等待全部真实基线、校准、资源和初始/后续证据合同核验。原 S1.0 报告和批准文件保持原样。
