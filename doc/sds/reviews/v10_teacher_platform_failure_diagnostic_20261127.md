# A2教师平台准入失败：只读诊断（2026-11-27）

**结论：当前盲走教师不满足此平台任务准入，流程维持停止。没有修改策略、命令、门禁、奖励或训练参数，也没有新增物理。**

## 真实结果与费用

1024 DAgger/PPO完整资源探测已通过；随后D1-A教师suite执行到第11组早停。普通平地10组（3速度行走、静止、3速度×2走停）均20/20；首个平台P_WALK（H=2m、W=1.2m、G、vx=.3、10m目标）0/20，要求18/20。已测220/计划3080，complete=false；B及其余课程未开。正式学习0、无合格DAgger/PPO模型。

probe全局charged=executed3293184（含旧737280），updated3121152、uncertain0；A cap3833856余540672。独立eval物理304100=有效302074+padding2026，uncertain0。费用域不混加，历史不退。

## 终止边界与可判定范围

20/20首次机身走廊越界=首次failure=trace最后一帧，840–1005步（16.8–20.1s）；全部从+y侧超过0.6m。末y=.6006155～.6165791m，x=5.4566～6.3881m，净前进4.4511～5.3922m而非10m；timeout全false。

末roll −19.68～−35.27°、pitch −19.96～2.00°；全程未超过70°。G侧面地面H_measurement=0，末clearance2.2174～2.3580须视为相对底地的净空；按已知primitive及XY重建z_base=clearance，减H_route=2后仍+.2174～+.3580m，没有触发z_base<H_route−.5。不能把clearance的跳变误认为机身瞬间升高。

因此bounds条件足以解释所有终止；不是旧0.4m宽度判据误杀，也不是70°/timeout/route-fall触发。原trace未记录body/nonfoot具体触发mask，不能排除同帧还有接触条件共同触发。

## 同速度平地对照与趋势

两组均由相同冻结盲走策略控制，初态分布相同但实际case不同；这是分组对照，不能宣称严格配对因果。表中均为20例同一时刻均值，未删除失败病例（16秒前全20仍在）。

| 时间s | 平地y m | 平台y m | 平地roll ° | 平台roll ° | 平地safe足数 | 平台safe足数 |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.1198 | 0.1206 | 0.232 | 0.207 | 2.15 | 2.05 |
| 10 | 0.2509 | 0.2525 | 0.296 | 0.284 | 2.20 | 2.20 |
| 15 | 0.3738 | 0.3764 | 0.332 | 0.318 | 2.35 | 2.40 |
| 16 | 0.3971 | 0.3998 | 0.190 | 0.182 | 2.10 | 1.25 |

5–15s两组均持续向+y漂移（body vy约+.02m/s，world vy约+.024m/s），姿态均值仍接近水平；16s平台safe_support足数下降，但safe_support并非raw接触，不能据此声称哪足已脱离碰撞面。20例平台在16.60～19.92s首次出现|roll|>5°（排除起步2s），当时y=.4623～.5043m；同速平地20例没有这类roll>5°。之后平台发生明显倾斜并在y>.6时终止。

证据支持“原有持续侧漂→接近窄平台边缘时支撑质量/姿态恶化→机身走廊越界”，不支持把一开始就存在的平台姿态失稳或错误边界作为唯一解释。不能将侧漂全部归因于yaw：旧trace没有yaw，而且body vy本身已正偏。

## 几何与观测核对

病例H2/W1.2/G经terrain_from_case生成独立平台：x[0,14]、y[−.6,.6]、top2、厚.01；底地x[0,14]、y[−2,2]、top0。corridor使用同W/2、边界相等允许；H_route恒2，不随侧落地面下降。实际constructor逐primitive按位置/尺寸唯一匹配MuJoCo geom，能运行到这些病例意味着该运行的构造校验通过；保存的terrain摘要不是完整原始geom快照，不能将它说成额外碰撞重放实验。

BlindTeacher仅使用45维proprio，不读地图或route_error。因此本次教师漂移不能归因于教师读取G高程图；它没有显式绝对路线y/yaw误差信号。

## 数据缺口及下一步边界

20列trace有XY、body/world XY速度、roll/pitch、测量面净空、4个safe_support及事件等；无yaw、原始足力/接触、body/nonfoot mask、完整45维obs/动作历史。不能从这些trace精确还原yaw或离线重标教师动作。原NPZ和原result均未改；派生JSON保留每例首次事件与趋势。

只读接口/方案影响另见v10_teacher_route_adapter_interface_readonly_20261127.md。路线反馈教师wrapper超出原SDS零vy/wz及不做回中边界，尚待用户明确范围确认；未选择增益/限幅，未实现/未重跑，不新增第三教师或借学习预算。

独立核验：v10_teacher_platform_failure_independent_20261127.md，SHA fb12ae95c7b1b1ec028f92bb05caabee50720351ad22b1e9a1eb3432e9b86324。原证据：当前run/evaluation/teacher_admission_D1_A/result.json及20例trace_*.npz；派生统计logs/a2_v10/teacher_platform_failure_diagnostic_20261127.json。
