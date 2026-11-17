# Q005 v0.6：trial02 update100 浮点一致性异常与同 run 恢复裁决

日期：2026-11-16；裁决者：主 Codex。本文补充 Q004，不替代原 SDS 与 Q001–Q004。
问题来源：用户转发的 DSH trial02 update100 中止报告；非新的行为配方提案。

## 1. 裁决
批准修复本次证据一致性检查，完成下列验收后，从 trial02 的 boundary_000100.pt 恢复同一 run。不新建 trial03，不 ETH fresh，不重做前100次 PPO。验收通过后自动恢复，无需再次申请开训。
本次工程异常不是 F0 门禁不通过，也不是奖励/课程失败证据。尚未完成 update100 完整门禁，不能报告通过。

## 2. 已核实证据
- 源 checkpoint：logs/a2_ideal_stairs_v06/train_seed42_trial02/checkpoints/boundary_000100.pt。
- 主 Codex 复算 SHA256：8ca0760e8c7a67da2c2143927e6ba5baf2459deba1967b0756a2aa601e58e805。
- CPU 加载确认 update=100、optimizer_step=adam_steps=2000、control_samples=2457600、physics_substeps=9830400；512环境、rollout48、epochs5、minibatches4。
- 已保存自适应 learning_rate=8.888888888888888e-05，恢复必须读取它，不能因 CLI 初始值为0.0003而重置。
- ledger.pending_boundary：phase=F0、generation=0、update=100；validation 与 diagnostic 两项、results=[]、curriculum_decided=false、committed=true；pending_update=null、stop_state=null、lifecycle_state=running。
- validation case_table SHA b1803de2663f770d25a90de0bd1e85c97091fe35032ec0d5e9f6e32478d08320；diagnostic SHA 61f343b3366634093cae06c9ed0a1b7d7585d01bcdbfedcab02b095b92dbc25a。
- checkpoint 内 source_closure_sha256=3ff5b4b5711a2774701055244031fe389e0dc495b8280379014126a3fa049f4e。报告中的冻结闭包27c42ece…可能是另一层manifest摘要；验收须列出各摘要的对象和算法，不因字符串不同自行宣判不一致，也不能混称同一个SHA。
- 主 Codex 独立纯浮点复现：float32起点 -1.100000023841858，终点2.390000104904175，float32净量3.4900002479553223；回算终点2.390000343322754，再以Python浮点减起点得到3.490000367164612，差1.1920928955078125e-7。此例证明1e-9不能作为这条跨精度路径的统一一致性误差限；不是对全部生产case的复算。

## 3. 精确修复范围
1. 仅 F0/flat retention 净位移跨来源一致性检查使用绝对容差1e-6 m、相对容差0。接受 abs(D_env-D_end_start)<=1e-6，超出则仍抛 EvaluationError。不要使用库默认rtol，也不要把所有1e-9全局替换。
2. 参与比较的起点、终点、净量必须全部有限；NaN/Inf直接报证据无效，不能利用NaN比较为false穿过检查。
3. 门禁公式保持 D_net>=5*c-1e-9，500步、横向/航向界限和每速通过数全部不变。1e-6不能加到门禁阈值，不能四舍五入数据使case过线，不能取两个净量的较大者。
4. Q003规定的规范量仍为本回合reset后起点与第500步reset前终点相减。核查发现env多个非终止证据出口用 start+forward 回算end，这不是独立的终点观测。仅限评估证据出口修正：从同一帧state.local[:,0]保存真实终点，配对同一episode起点；terminal/auto-reset同样先冻结。不得为消除误差反过来用D_env构造“校验”终点。此修复不改变reward、训练观测或物理状态；门禁边缘case以真实端点规范量裁定并记录。
5. 保留原始D_env、start、end、规范净量、绝对差与容差，至少在本次回归证据及后续一致性异常中可定位到case/环境/步数。仍保留符号，不改为高水位/累计正向距离。

## 4. 验收与一次性工程运行授权
- 完整CPU；新增真实float32运算路径回归，证明旧1e-9失败、新1e-6通过；覆盖正负误差、超过1e-6的真实不一致、NaN/Inf、三档速度阈值上下、前进再倒退、8环境各自端点及自动reset。门禁下方5e-7仍须失败，不能因一致性容差而通过。
- 实际runner→evaluation→recovery组合核验第100次边界，包含真实非空Adam、真实case表与净位移字段；只测试50诊断或mock gate不足以覆盖本次缺陷。工程重放使用隔离目录，结果不算正式门禁，不得挑最好的一次结果填回run。
- 授权新冻结源码上的一轮smoke（8env/H00/2更新）和一轮原56更新probe（512+8env，5+50+1），保留Q004真实50诊断/快照/55→56恢复要求；不扩大学习预算来凑100步。100门禁路径由上述边界集成与待恢复checkpoint的零学习验证覆盖。CPU先通过再GPU；单GPU worker串行。若工程验收再失败，保留证据上报，不无限重复。
- 独立验收必须解释旧验收为何没覆盖float32实际证据以及真实validation路径，验证规范端点和门禁容差分离；完整CPU、GPU和独立验收必须绑定同一新冻结源码，blocking=0后恢复。
- 本裁决不要求重跑训练寻找好结果，不允许修改奖励、PPO、课程、速度、随机化、终止条件、门禁、资源限额。

## 5. 跨源码恢复与事务
本次明确授权一个受限的旧checkpoint→新修复源码恢复桥。旧checkpoint原字节/sidecar/source摘要均保留，不伪造旧模型由新源码训练。
新manifest记录：旧checkpoint完整SHA、旧source清单及两种闭包对象、新source完整清单与闭包、本裁决SHA、实际diff和独立验收SHA。独立审核确认学习配方、模型/Adam/normalizer/schema兼容，仅含本次已授权证据修复及相应测试/文档；发现额外行为改动则阻塞并提问。不得关闭源码校验或开放任意来源恢复。
恢复时保持模型、Adam含当前lr、自适应状态、normalizer、RNG、课程累计/窗口、门禁streak和台账；按Q004允许重建新episode，不声称物理轨迹逐位连续。
先用checkpoint中同一case表身份完成update100 validation/diagnostic及一次课程决策，再开始学习101。已提交结果按Q004去重；部分失败评估原件保留并标未完成，不计作失败门禁、不拼接其他轮case、不双增streak。不得覆盖原boundary_000100.pt来掩盖来源；恢复证据使用追加记录/独立恢复子目录。
若补完边界后触发真实行为/预算停止，按原规则停机，不能保证一定进入101或再次自动重试。

## 6. 成本、回执与限制
trial02保持单run30000更新上限，已消费100、最多剩29900；trial01的50另保留，两次正式尝试累计上限30050。恢复不重新计费100；smoke/probe/零学习评估成本单列。
原例行保存100、诊断50、门禁100、到期评估前pending checkpoint保存继续执行。R2 seed43保持用户停止；不commit/push。
实施与独立验收继续分工。回执本裁决SHA，完成后回送新冻结/验收/恢复桥SHA及实际启动命令；按条件自动恢复trial02。任何新的fresh run仍须另行裁决。
主Codex本轮仅做源码/现场文件只读核查、CPU checkpoint加载与标量浮点复现，未执行生产全量CPU、GPU、仿真或训练；上述实施效果尚未验证。
