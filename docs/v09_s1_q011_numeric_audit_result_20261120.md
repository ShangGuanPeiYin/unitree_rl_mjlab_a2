# Q011 标签数值审计独立结果（2026-11-20）

结论：**Q011规定的数值核验条件全部通过，解除本次Q011首梯度等待。** 既有 `s1_learning_approval_q010_c9ee.json`（SHA256 `9d4c515d799ecfbd586e57ba7ec75d436e610dca4738a9ab1086ccb2580d689f`）继续有效，无须重新签发学习准入。此结论不跳过剩余Q008学前视频及其工程条件，不代表S1行为恢复或S2晋级。

独审只用CPU读取实际GPU输出、原始保存观测/标签、冻结清单、配置收据和活账本，重新计算全部比较；没有再次运行GPU、物理、学习或发进程信号。当前父PID3228141仍T态，start_ticks145922371，正式学习样本0；迁移初始75视频尚未开始。

## 身份、安全边界及核验范围

绑定固定run `a2-v09-s1-seed42-r3`、目录 `logs/a2_v09/s1_seed42_r3`、源码 `c9ee8789f6051bc6c36cbfc24f076deba6121d8ffedb48ce5d689e0a6b665ea4`、原SDS和唯一6400。与既有学习批准相同的只读Context清单SHA256为 `0d158b266dde7daeefaf81bc5b270a4294a94f473203015ea8fd9d6df6380408`；重新构造真实Context并检查当前账本合法前缀，未扩大旧证据复用范围。

原学前DIAG已完成75批/1500例，先前独审的几何证书、首终态、raw metrics、命令/action历史和同状态非地图输入一致性保持有效。控制样本539740，其中首回合438030、padding101710。原CPU数值例外没有删除，也未重写任何标签或修改行为/数值容差。

GPU审计由root在Q009卸载态和父进程暂停后执行。安全证明 `q011_gpu_replay_safe_boundary_20261120.json` SHA256 `1e02d8dfff8ee5145e13041cf78693c0abc1447d6f63a0d3f435e50ff16d9f09` 绑定父PID/tick、录像子自然exit_code0、子CUDA上下文退出、env/双缓冲已释放、原账本及零新增环境步。独审核对该证明与实际报告/脚本/清单/契约SHA、当前T态和账本一致；原6400视频75例闭合，完整视频解码仍按原安排待init视频完成后合并独审，未据此提前宣称全部Q008通过。

## 事前冻结的比较合同

- v3输入清单SHA256：`a1ba44e537dacc088ddf84523f6b09a6c01f1c1734efdddcdbcae4adc3396324`。
- 数值配置契约SHA256：`147933ed10de30fa3f0a453fa50a09090523575181c9b80fad50ad74f96aab6b`。
- 实际执行审计脚本SHA256：`86f1c9521b806367459b36b3ddc125e864e78fa68db43645301eb9d0debb9ace`。
- 实际GPU报告 `q011_gpu_replay_v3_r1/replay_report.json` 文件SHA256：`f657e49d9d93ee45879646ef70ef8188dc4515aaf1b926ed4978f5000328425a`。

340个完整20槽异常快照，加按冻结顺序选定的非异常flat/stairs完整对照2个，均保留原批形状、原slot、时间索引、字段字节及顺序。56个部分异常全部覆盖，包含batch70/index100的2活跃槽情形；只把原active字节放回原slot、恢复固定N20，其余槽分别复制预选模板A或B。这是**固定形状补位对照，不是历史重建**，没有恢复或伪造未保存inactive观测。

模板A为batch0/index0/slot0，B为batch74/index50/slot19；按task、完整性及冻结顺序选择，不依据标签或GPU误差。不同case/frame/字段SHA事前固定，heightmap RMS差0.134715、proprio RMS差0.731870。教师Actor均值和Critic前向由逐样本卷积/线性/ELU、固定特征归一化与特征拼接组成，无BatchNorm或批轴归约，eval/无梯度/normalizer只读；CPU全部56双模板及33 unique masks检查后完整state摘要不变。实际GPU同样核对冻结6400 state和全部53项模型布局。

全部33种缺槽mask都在预先指定的完整原批控制上人工遮盖，包含2活跃槽；两种补位分别与原保存标签及同次未遮盖完整GPU输出比较。每个全新独立GPU进程执行342完整批+112部分双填+66遮盖双填=520次前向；两个独立进程共1040次，不使用active子批冒充原N20。

所有比较仍使用原逐分量公式：`abs(replay-saved) <= 1e-5 + 1e-4*abs(saved)`，覆盖12维μ和1维V；两次互比、双填法影响和遮盖控制同容差。v2的NaN/Inf及广播shape放行漏洞已在独立复现后于v3统一阻断；所有比较要求finite/equal nonempty shape，JSON禁止非有限值。CPU测试16失败/1通过→17通过，fresh复审两Important闭合，原v1/v2及证据保留。

## 实际结果的独立重算

两次新进程PID3276157、3276203及独立token不同，均在同一GPU UUID `5fd3145a-6ae6-4777-33c0-bb1a32cac8b9`，frozen teacher/eval，分别520次GPU输入布局硬核检查。原文件/观测字段/标签SHA、完整case顺序、slot与actor_index、FP32 dtype/stride/layout、全部模型布局均按事前契约核对。

独审核对两个实际PT与收据摘要，并独立重算双方合计221728个保存标签比较分量；重新计算520组两次互比、每pass89组双填影响及66组遮盖/未遮盖比较。所有重新比较张量有限且形状精确一致。

| 比较 | 最大绝对差 | RMS | 原容差例外数 |
| --- | ---: | ---: | ---: |
| 两pass各自μ/V对原保存标签 | 0 | 0 | 0 |
| 两pass相互μ/V | 0 | 0 | 0 |
| 两种填法对active μ/V影响 | 0 | 0 | 0 |
| 遮盖控制对原未遮盖完整批 | 0 | 0 | 0 |

两收据文件SHA256分别为 `4ebd5b7273b800594150ae8dea4ed3f9b1b6e6a8f0a01bb67e063994e83ca5f8`、`feeab2493ccce8aaeacda2b3b03e44a31ec24799a05f0ed33681f5026f2ebdd0`。两PT文件SHA256分别为 `f4547b280d52e6545bb48b094faa23889cf64e5971ab7640c5e45ef9219918c5`、`7c010c138b198ef7c78193fc1d3d71f4144c423ad7b02a483ba54bd6c3fb8831`。

事前配置和实际重放逐项相同：Torch2.15.0/CUDA13.2/cuDNN92600、同Python/依赖和驱动；FP32、autocast关闭、matmul TF32=false/最高精度、cuDNN TF32=true、benchmark=false、deterministic=false。相关新API支持状态也被记录并核对。原生产backend flags没有被直接保存；预期配置依据冻结C9启动/构造源码、实际无CUDA导入追踪、seed_rng只传seed且deterministic默认false、软件默认值及暂停生产父进程数值环境事前确定。重放前后没有修改flags，未根据GPU结果挑选配置或模板。

在Q011覆盖的模型、地图输入、时序、槽位和数值配置合同内，未发现错误；规定全部异常/对照和补位独立性核验均满足。独立证明 `q011_actual_gpu_independent_proof.json` SHA256 `f67ae1ffd3719b1b5898cff4a16c59f69f53c33f4e0913ce30e390d39d69b020` 保存实际文件摘要与逐项结论。

## 保留限制及继续条件

原CPU FP32对保存CUDA结果仍有动作2407/112392、value10/9366容差例外，最大差分别1.58667564e-4、7.853507996e-4；CPU FP64参考和固定第三Conv TF32模拟记录全部保留。**本次GPU通过不能证明某个具体算子是CPU偏差根因**；未记录的历史CUDA中间层/flags不能事后声称已观测。GPU核验覆盖按Q011冻结的异常与控制，未声称重放全部9366个保存状态。56部分情形仅验原保存active行的固定形状补位，无历史inactive重建。

本次GPU网络重放没有构造物理环境或优化器，新增物理步0、收费0。活账本始终seq1464/event `c9d5690b8c8f2739f2d213448e90c94f318d23fb1891c9e561ea50b04ce61423`，文件SHA256 `9478b5ef2ac71388872d6b0272f84a393bcbd97ba98db458e0378a1862ff44de`，正式学习0。

因此仅解除Q011数字核验等待。root仍须在读本报告/clearance并完成日志、提交推送后，严格复核同PID/start_ticks/argv/账本，只对原暂停父进程CONT一次；独审不发送信号。原driver继续完成init75视频、保留全部学前证据后，才可依既有学习批准开始S1.1；S1.2、门禁、final及S2边界保持原样。没有声明学习效果改善或S1恢复成功，没有读取真实final成绩。
