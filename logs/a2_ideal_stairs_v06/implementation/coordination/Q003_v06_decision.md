# Q003 v0.6：F0前进距离裁决

日期：2026-11-16；裁决者：主Codex；结论：选择B，净前进位移。
问题来源：本轮用户转交DSH消息；归档快照Q003_v06_question_snapshot.md，SHA256 268e7ec75b204b0b9bcef22b526113ae14a2625eeef59806d12e424375d44369。
基础合同：SDS 1ed4bf5fa8f83d4276e278c7d161be3b6d9c0032cbdc1e3540dade9606e6ad60 + Q001 5f3f2aa397fb5cf5ce286788e76384e8b19914c758e0ae29e50a67aa4d71a4b1 + Q002 3ebc7441f9be457cb135846366fe23926d02ec69e2232e0502d6c318ce5ccd52。本文仅补充距离定义，原文件/旧SHA不追改。

## 1. 唯一生效口径

F0完整case恰好500个控制步（dt=.02，共10s）：

D_net = x_local[500, pre_reset] - x_local[0, after_reset_before_first_action]
required = .5 * c * 10 = 5*c

x为该case所属tile坐标系机身前向坐标。起点取完成出生写入和sim.forward/sense后、首次动作前的实际位置，不取常数−1、世界x=0或第1步之后的位置。终点为第500步物理推进完成后的同一episode状态，必须在任何自动reset之前冻结。

forward_distance字段从现在起统一表示有符号D_net。可以为负；不clamp_min(0)，不取绝对值，不用积分速度代替位置差。数学判据D_net≥5*c；保留既有实现的1e−9比较容差，不扩大容差。F0三档门槛为.3→1.5m、.5→2.5m、.8→4.0m。若诊断指定其他c，仍按5*c通式，不能硬编码三档查表。

case_pass仍须同时满足恰好500步、全窗口无failure、max|y|≤.20、max|yaw|≤10°等原条件；距离合格不是单独通过。发生提前failure时该case失败并使用真实终止前快照，不能自动reset后拼接到500步；无failure但采样缺帧/不足500属于不完整评估，不能通过或把时间分母缩短。

## 2. 修改范围

- 修env.forward_distance及所有F0 verdict/逐例快照/诊断/评估汇总，保证env和speed/evaluation消费同一净位移。
- flat保持集使用同一个F0判据，因此也采用净位移；不改变其15例/12例/连续2次规则。
- 其他楼梯门禁不新增距离条件；H/D/L/V/R仍按原真实平台接触成功判定。
- 若共享forward_distance字段出现在楼梯诊断中，其字段含义也统一为净位移，但只改变该诊断数值、不参与楼梯成功判定。
- max_body_x保留原统计含义。若需要展示最大前进位移，另用max_forward_excursion字段；不得把旧max值偷偷留在forward_distance里。累计正向路程若保留，必须另名positive_path_length且不得用于门禁。
- 奖励progress仍为每步.4*dx，所有奖励和相关历史最大位置/停滞状态机不因本裁决改变。
- 不把500步截断改为新的环境物理终止条件，不改变平地训练1500步上限。

不要只改getter后让评估在reset之后读取机器人当前位置。建议本帧以state.local[:,0]和本episode起点计算批量净位移，先写入不可变per-env终止前快照，再允许_reset_ids。自动reset后可产生下一episode自己的0起点，但不能覆盖已冻结case结果。实现形式由owner按此数据契约完成，不能用env0标量冒充全部8个评估环境。

## 3. expected与必须新增的测试

以下case均500步，未特别说明时其余F0条件合格；c=.3、起点−1、门槛1.5m：

| 情形 | 最大x | 最终x | D_net | expected |
|---|---:|---:|---:|---|
| 恰好走够 | .5 | .5 | 1.5 | 通过 |
| 先走远再退回 | 1.0 | .0 | 1.0 | 不通过，A原算法会误过 |
| 原地往返，累计正向路程>1.5 | -.5 | -1.0 | 0 | 不通过，C会误过 |
| 先退后前进并真的走够 | .6 | .6 | 1.6 | 通过，不能增加禁止倒退条件 |
| 净后退 | -1.0 | -1.2 | -.2 | 不通过，字段必须保留负号 |
| 峰值很高但终点差1e−6 | 1.0 | .499999 | 1.499999 | 不通过 |

另测：
1. 起点改为−.9/−1.1但D_net相同，判定相同；全tile世界平移后判定不变。
2. c=.5/.8恰好对应2.5/4m门槛；足够距离但y/yaw或failure不合格仍失败。
3. 第500步终止/自动reset，info保存旧episode终点与起点，而非新出生点；499步提前失败不拼接下一episode。
4. batch至少2个env，一例往返失败、一例净前进通过；部分reset不污染另一例或缓存。
5. env.flat_pass与独立speed.flat_case_pass逐例一致；从逐帧trace的x0/x500复算与汇总一致。
6. 原test_f0_case_scalars只检查类型，不足以验收；新增确定性轨迹/状态快照测试覆盖上表，不能仅靠策略恰好不动的500步样例。

## 4. 冻结、验收与执行

本裁决不要求改奖励、PPO、预算、速度、seed。DSH追加补充冻结及manifest，记录forward_distance_definition='net_displacement_pre_reset_500_steps'并纳入评估协议/配置SHA；不要修改旧冻结。

CPU工作继续。GPU smoke/probe是否允许仍按Q001/Q002和SDS已有工程准入决定，本裁决不是额外GPU放行。若先前工程检查已在旧源码SHA完成，必须说明哪些证据仍适用并在最终冻结SHA上重做受影响的验证；旧结果不得自动冒充新SHA结果。正式训练前必须完成修正、完整CPU及独立复验，最终验收仍需绑定一致SHA且blocking=0。

回执请含本裁决SHA、改动文件SHA、上表逐例净位移/旧max位移/判定对照、自动reset和batch证据，以及完整CPU/独立验收结果。R2 seed43保持停止。

主Codex本轮只读核对候选env/speed与测试，并做独立净位移反例算术验证；未执行生产测试、仿真或训练，不宣称效果改善。
