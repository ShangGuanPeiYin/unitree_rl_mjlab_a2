# Q008适配前证据独立验收（2026-11-20）

结论：**Q008适配前工程证据完整性通过。** 正式初始checkpoint、DIAG全部1500例、6400及迁移初始各75个固定视频，以及Q011数值核验链均完整。三项证据义务先闭合，随后才出现首个学习rollout和20次Adam更新；当前S1.1已按既有批准开始。本结论不代表初始行为合格、loss改善、S1恢复成功或S1.2工程晋级通过。

独审只用CPU；本轮仅解码新增init75视频，教师75沿用已完成独审，不重复解码。没有另起GPU、物理或learner，没有修改源码、阈值、旧报告、catalog、journal或账本，也未读取final成绩。日志、提交与推送由root统一处理。

## 身份和正式初始化

固定run为 `a2-v09-s1-seed42-r3`，目录为 `logs/a2_v09/s1_seed42_r3`；当前源码 `c9ee8789f6051bc6c36cbfc24f076deba6121d8ffedb48ce5d689e0a6b665ea4`，唯一6400 SHA256 `30ed23ba16f7e490afca001fbbecadb4a11ff4b11733a5ae7ac7d0c841f9a8b2`。SDS及只读Context沿用既定合同；同一兼容清单SHA256为 `0d158b266dde7daeefaf81bc5b270a4294a94f473203015ea8fd9d6df6380408`。

正式course_entry checkpoint SHA256为 `32c962f28352b13f81c1c9b39e4c5a9bbe01f5e3f6545d98f8b46414492e6c13`，学前卸载/恢复checkpoint为 `4893c425476368592488bde8e155117194178e0d266d8438bdeb5f5be09f3042`。此前独立逐张量验算：53项中51项非变化参数/buffer与6400精确相同，两个FC按独立Fraction区间重叠计算6→5汇聚后精确相同；两checkpoint的模型state SHA相同，std/归一化冻结，Adam管理45参数且初始state为空。正式模型从6400 fresh迁移，没有继承资源probe或旧训练状态。

相关独立初始化证明 `q008_formal_initial_checkpoint_independent_proof.json` SHA256：`a5a802c78e6ecf5f708baf992329f02d33069ffdd782a4420743ebb42ae2eba1`。

## DIAG原始证据及数值链

独立S1_DIAG固定75组、各20例，共1500唯一病例。75份PT及75份几何证书完整；每批231个编译地形对象、原地形表、接触索引和真实几何一致。首回合轨迹/首次终态、完整metrics、命令及action历史已按原数据复算，教师/学生非地图输入精确相同，地图分别为原生48×32/40×30；保存教师标签来自同一真实执行状态，非幻想历史。

全部9366个保存教师状态做过CPU重算和FP64参考。原CPU/GPU逐元素容差例外保持原样，没有重写标签或放宽容差。Q011又按事前冻结合同验证全部指定异常与对照、56部分异常的固定形状双模板补位和33种mask控制；两独立同GPU重放与保存μ/V、两次互比、填法影响和未遮盖完整控制均为零误差，Q011已签字解除。56部分情形仅验证原保存active，不声称历史inactive重建，也不由GPU通过归因某个具体算子。

原pending证明SHA256 `b42aa20485791cc7fd1f0a3b36ec212e85105b044710fd8f958cfa01b7a55397` 作为历史记录保留；解除凭据 `q011_numeric_clearance_c9ee.json` SHA256 `22139fcb8ce685adb7191c67cc8ed6a35822607d04bf35b35185bb1cf1edbfe3`。这两项共同解释本次汇总结论，未覆盖旧记录。

## 两套固定视频和费用

两模型均覆盖同一DIAG每组case_id=0的75例，失败录像没有过滤。逐例核对manifest/case/模型/code/SDS身份、同Context、元数据及视频摘要、640×480/25fps、初始帧和每2控制步帧、精确首终止帧及reset前来源、实际Actor命令和阶段。flat ramp按Q002检查DECEL/HOLD索引，楼梯初始APPROACH与原始出生位置一致。所有MP4已累计独立完整解码，保留精确末帧。

| 学前证据 | 病例/批次 | 独立解码帧 | 实际控制样本 |
| --- | --- | ---: | ---: |
| 初始DIAG | 1500例/75批 | 不适用 | 539740 |
| 原6400固定视频 | 75例/75文件 | 18070 | 35954 |
| 迁移初始固定视频 | 75例/75文件 | 10895 | 21605 |
| 合计 | 225项物理操作；150视频 | 28965 | 597299 |

DIAG首回合有效轨迹438030控制样本、padding101710，二者合计539740；视频为1环境首次终止，不存在多槽等待。三项均为S1_EVIDENCE工程费用、各75个已知complete操作，无unknown、重复费用或未闭合预留。累计学前物理子步为2389196。Q011离线GPU核验没有物理步或新增控制样本费用；学习消费另记，未混入此表。

教师75证明 `q008_teacher75_videos_independent_proof.json` SHA256为 `f6c7799195f09c7170f5ae2b4e107901053a075db992454d2492dee26089e1bd`；新增init75证明 `q008_init75_videos_independent_proof.json` SHA256为 `9aea51581788224c72a8b678ba7f29542bdf2ac49318f3c43cc712e13d26eb78`。本轮确认既有文件保留及summary/source身份；教师视频未重复解码。帧来源结论依据已审源码hook、元数据和完整解码，不声称逐像素检查了画面姿态。

## 闭合早于学习的权威顺序

只读共享锁捕获同一账本image并验证完整链/state；分别确认diag_before于seq1313、video_6400于1464、video_init于1615闭合且完整绑定各自checkpoint。首个学习reserve为1618，首个48步complete commit为1623，晚于三项闭合；未发现学前证据完成前的梯度学习或重复视频结算。

完整journal前缀中Q008_before_evidence仅一次，seq5时全部学习消费0；Q009恢复结束seq6，optimizer未重放；首个optimizer_complete为seq8，对应24576学习样本及20次Adam。首条training_iterations原始行摘要与journal引用吻合，loss、梯度均有限。这仅验证完整学习更新已经开始，不能把首loss或初始轨迹当作效果改善/能力恢复。

汇总证明 `q008_before_complete_independent_proof.json` SHA256为 `10295a9b4e10d58197358117b9e73667f5c91d7a536b48826543b65bd19917ca`，记录各独立证明SHA、闭合事件、当前只读账本快照和完整journal前缀。捕获时正式S1.1已累计1351680学习样本；运行仍在追加，不能把学前checkpoint中的learning0误作当前消费。

后续继续原N512、48步、5epochs×4逻辑批=20次Adam、microbatch512、LR3e-5、每5000000学习样本保存的纯蒸馏配方，S1.1最多813完整迭代。当前尚未验收完整适配量及after1500例、适配后75视频和同数据双模型loss/行为对照；进入S1.2仍须满足这些既定工程条件。正式行为门禁与final封存边界保持，未生成或批准 `S1_RESTORED`。
