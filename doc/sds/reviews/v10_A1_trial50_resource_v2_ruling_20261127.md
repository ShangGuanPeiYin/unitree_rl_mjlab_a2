# A1限额试训50轮与资源补证v2独立裁定（2026-11-27）

## 来源、历史及状态

用户要求统一纠偏并授权规划直接技术裁定，强调原教师平台0/20是持续侧漂掉边，不能混作不会基础行走。规划线程01a217c8-d825-7922-bd23-eb6ba2e37bfb在A1工程提交c6a76db后明确调整顺序，并于真实资源失败后批准本硬帽/新资源方案。

本文件独立关联冻结SDS v1.1 SHA eb95905ed24bf0674f94a1b02dab02256ee11e40a20ed1cd96eb24daeb1b49a0及已提交2f82589e805b2579843a4381089467d40931300aff8a80f466becf90dcc4fbcb来源，不覆盖旧SDS/已签裁定/证据。2f工程159 CPU通过，实际v1资源失败；学习0。当前仅批准本文件的最小工程、限额验证及条件满足后的试训，不将旧source或教师准入冒称已通过。新source必须另冻结、独立核验与显式关联证书；runtime整文件变化如实记录。

## 最新阶段顺序

新资源v2两栈通过 → 固定112例教师诊断每组4/4全部通过 → 原D1预算内限额试训50轮 → 保存停止、向规划报告。完整教师A/B前置门禁显式后移，不取消最终验收；不得自行运行完整课程或PPO。任资源或112诊断失败均停止，不以试训绕过；不搜索增益、不改奖励/路线/速度/损失/随机化或教师权重。

试训及其checkpoint明确pre_gate_adjusted_trial、teacher_admitted=false、accepted=null、training_accepted=false；不能生成V10_ACCEPTED、课程通过或PPO准入。loss下降/训练采集轨迹均不是独立行为验收。到限报告checkpoint原文件SHA、实际费用、per-task失败率/真实完成轨迹长度/标签覆盖及teacher分项/损失/已有行为证据，不把未记录或未测内容写作结果。

## 试训精确硬帽与恢复

N1024、rollout48、course仅D1、目标最多50完整更新，即2457600控制样本；全部计入原D1 24576000及总学习196608000，不新增或转借额度。原vx请求.3/.5/.6、A1 yaw公式、学生原请求/history、S2、奖励/损失/LR/保存周期不变。

trial cap对executed+uncertain charged共同约束。worker自身必须强制min(原D1剩余,trial剩余)，每完整rollout启动前剩余至少49152；Q1逐step只计一次。父调度检查不能代替worker内部硬帽。稳定trial-id、canonical guard及同一学习ledger绑定，不能通过新run/新guard/新ledger赠50轮。恢复沿原checkpoint/优化器/replay/RNG/计数，既耗不退；无resume但已有费用拒绝，trial checkpoint省略trial参数不能无帽续跑。

正常50保留学习更新完成后stop_reason=trial_limit。若崩溃/uncertain/丢弃rollout或回到较旧checkpoint使charged不足剩余完整更新，则trial_budget_exhausted，不补跑，不伪称原D1整个预算耗尽。checkpoint内retained_completed_updates与外账historical executed/updated分开，不用外账已丢失梯度更新伪称当前checkpoint含50更新。

直接trial worker首次step前保存完整retained0 initial checkpoint作为可恢复起点；不新增周期。第50轮跨原SAVE_SAMPLES=2457600时真实保存，边界与周期重合按原hard-link去重，原文件SHA保持。保存失败必须非零退出，不能开始/宣称完成训练。完成既有有效CPU恢复/保存证据可复用，仅缺口做最小回归。

## 已发生v1失败与不退款

2f资源v1为1024 DAgger，1fill+1update后停止；本次charged=executed98304、updated49152、uncertain0。全局probe charged3391488、updated3170304、余442368，原cap3833856不变。PPO、112、教师A/B及50试训均未开始。

最低MemAvailable3353313280B低于原4GiB=4294967296，发生于首update内部；初始fill区间min3697303552、首采样约4110860288也越限。新worker首update RSS/HWM比旧已过实测略低，初始机器可用内存却少约1.77GiB，现证据更支持背景余量差异，缺同期父子PSS不能归因具体进程。probe未实例化eval trace缓冲，不能无证据把约2GB归到trace/teacher clone。不改内存实现，不结束任何其他进程，不清理他人状态。

## 唯一资源v2协议与门槛

规划批准1024两栈各1fill+1warm+2measure=196608，合393216；预期global3391488→3784704、余49152。原两栈1+2+2需491520比余额缺49152，原完整26轮双栈需2555904比余额缺2113536；不擅自扩额。固定v2新protocol/contract及唯一claim，任栈失败停止，不用余量重试，不降N/容量/精度/批次/阈值。

保留六×32768=196608条完整FP32 replay、Adam、真实保存/CRC/SHA/释放/加载与下一采样/RNG等价检查，全程真实峰值不清零。标明这是短程补证，不是5warm+20measure完整复验。新contract绑定旧b48完整通过证据、2f失败仅作对照、完整新source差异、本裁定及独立审核；不得把2f失败当通过证据。trial guard只改变显式D1试训预算分支，probe实际使用的collect/save/模型/优化路径须逐AST/无副作用核验，不能假称runtime全文件byte相同。

启动前MemAvailable须连续60s每个样本≥12GiB；该只读前置检查在claim和物理前，不达则无物理停止报告。child填库前须≥9.5GiB，不达亦不得采样。运行原4GiB硬门槛保持，不能只看after-update或均值忽略瞬时低点。12GiB/9.5GiB是根据实测启动约2.54GB和后续约4.9GB峰差加余量的前置条件，不保证其他进程之后不改变负载。不得kill或调整他进程以满足条件。

## 同上限承继与原评估计划

正常同cap授权承继、同全局ledger，历史费用与uncertain完整沿用，新旧不可并行占用且旧授权只作历史。严格父SHA/id/cap/ledger/source与新裁定关联；自动审批拒绝时如实报告，不更换入口绕过。学习/eval/probe不互借。

112诊断沿已批28组×4原manifest，上界251288，原eval cap1685390840和旧304100不变，资源失败时不能启动。完整教师A/B门禁后移而非通过或无条件取消，最终验收不取消；50结束后由规划裁定后续，不自动扩试训或进入PPO。
