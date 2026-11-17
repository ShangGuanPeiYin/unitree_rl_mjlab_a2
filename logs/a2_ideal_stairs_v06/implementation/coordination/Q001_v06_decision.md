# Q001 v0.6 实施裁决

日期：2026-11-16。裁决者：主 Codex。状态：批准以下明确修正；不豁免任何工程验收。

问题文件 SHA256：d4422224403f78f3151bcef8e24f39644bbfe346f896e4daade75f02ffd1c84a
原批准 SDS SHA256：1ed4bf5fa8f83d4276e278c7d161be3b6d9c0032cbdc1e3540dade9606e6ad60

本裁决是原SDS的有约束力补充；冲突处以本裁决为准。原SDS、W0冻结与旧测试证据保留，不追改旧SHA。DSH须将补充纳入新合同/修订清单并重冻，实现者与独立验收者引用同一组SHA。不修改R1/R2，不启动或恢复旧seed43。

## Q001-1：采纳raw累计、filtered边沿结算，保留原数值锚点

按控制帧采样定义腾空时长，不声称这是连续物理接触时长。dt=.02时，已接触地面后15个连续raw=False控制帧对应.30s。

逐足更新顺序必须如下（所有右值的prev/has/air均为本次更新前状态）：

```python
filtered = raw | previous_raw_contact
landing = (raw & filtered & ~previous_filtered_contact
           & has_contacted_once & (air_time > 0))
bonus = where(landing, air_time - 0.10, 0.0)
next_air = where(raw, 0.0,
                 where(has_contacted_once, air_time + dt, 0.0))
next_has = has_contacted_once | raw
previous_raw_contact = raw
previous_filtered_contact = filtered
has_contacted_once = next_has
air_time = next_air
# 最终单足金额 = 0.2 * bonus * dt；对足求和；外层不得再次乘dt。
```

解释：raw失联首帧也累计；短暂失联可能始终不产生filtered上升沿，因而不付款。raw恢复时无论是否付款都清零，禁止把单帧失联时间累加到下次有效腾空。首次出生触地不付款；出生后尚未首次触地期间air始终0。落地帧不再追加dt，不支付滤波下降沿。slip/clearance仍用原filtered接触，不随本裁决改为raw。

必须增加真正经过状态机的验收，而非只测bonus纯函数：
- 初次True→15个False→True：最后单足+.0008；再连续True不重复。
- True→False→True：全程0，最后air=0。
- 重复多次单帧失联后再15帧失联：仍+.0008，不累计旧失联。
- 出生15个False→首次True：0；随后真实腾空再落地才支付。
- True→3个False→True：.06s，单足−.00016。
- .05s→−.0002保留为dt=.02的纯函数锚点；不能称.02步长下存在2.5个状态机帧。
- 四足异步、部分env reset、两帧失联、raw力阈值等于5N/略大于5N、序列中途state_dict恢复，均逐帧验证。
- previous_raw/previous_filtered/has/air均保存/加载；元数据或shape缺失不得静默补零。

## Q001-2：批准必需foot_surface_height字段

RewardFrame增加foot_surface_height: float32[B,4]，单位m，必须与foot_position使用同一世界z基准；env由当前实际tile的同一TerrainSpec查询H(xfoot,yfoot)。缺失/非finite必须拒绝。禁止平滑、前方台阶、起跳地表替代。部分reset和混合tile要测，原冻结字段不删除。

## Q001-3：批准新坐标约定，但更正“完全沿用旧atlas”措辞

v06采用frame_origin_world=(0, tile_id*8.0, 0)，楼梯bounds=(-2,N*d+2,-2,2)，flat=(-2,22,-4,4)。这是v06明确的世界布局；旧v02 frame的x是−10且y有整体偏移，不得称数值完全相同。只允许一致平移，不改变局部几何/任务。

first tread edge local_x=0；平台起点=N*d；所有geom、出生点、地图、接触归属、奖励、观测、终止必须一致转换。验证实际盒体与metadata、tile间接触归属，以及整体平移下奖励/路线观测/成功判定的数学不变性；不要求不同物理运行逐位轨迹一致。不得因此增加平地/楼梯之间的穿越训练。

## Q001-4：批准资产来源，禁止硬限位端点直接乘.9

q0按FL/FR/RL/RR及hip/thigh/calf顺序：
(-.1,.9,-1.8, +.1,.9,-1.8, -.1,.9,-1.8, +.1,.9,-1.8)。
软限位严格使用现有mjlab语义：m=(lo+hi)/2，r=(hi-lo)/2；soft=(m-.9*r,m+.9*r)。
检查到当前v06/spec.py::_soft_limits已经用中点公式，保持此实现，修正问题说明中“硬限位×.9”的错误简写。数值：hip[-.909,.909]，前腿thigh[-2.0655,2.8755]，后腿thigh[-1.285,3.665]，calf[-2.6585,-.6515]。独立验收须与运行时robot.data.default_joint_pos和soft_joint_pos_limits逐关节按名称对照，误差≤1e−6；不要只让两个手写常量互相证明。

## Q001-5：批准本次固定fresh输出与验收路径

fresh输出固定logs/a2_ideal_stairs_v06/train_seed42_trial01；验收路径固定logs/a2_ideal_stairs_v06/implementation/V06_IMPLEMENTATION_ACCEPTANCE.md。允许项目根目录解析后的等价绝对路径，不允许指向另一个run或报告。路径相同不足以放行，验收文件内容必须绑定当前SDS+补充裁决+生产源码+配置SHA，并且blocking=0。目录已存在时拒绝fresh启动，不清空、不覆盖；恢复只走--resume-run并按原合同与状态授权校验。此裁决不授予失败后再开trial02权限。

## Q001-6：批准公式锚点

L40 h=d=.20：c=.2→7257，c=.6→2753。主Codex已独立算术复核；公式及其他速度计算不变，不将两项当成查表替代通式。

## Q001-7：批准连续2次，仅针对继承的停滞collapse

在spec显式定义DIAGNOSTIC_COLLAPSE_STREAK=2，并纳入配置/SHA；speed/runner读取必需常量，删对此项静默fallback。缺失时预检报错，不自行用默认值。

继承停滞判定：每50更新15例、三档各5；每例stall_samples/active_samples≥.80的数量≥12；楼梯还要求无有效首阶案例数≥12，两项计数是各自达到12，不擅改成交集。平地无首阶条件。当前generation至少100更新才有资格；连续2次合格诊断命中→behavior_collapse。新generation清该streak；有效不命中清零；缺例/零分母/缺字段是评估错误，不能当不命中或通过。诊断速度使用当前宏阶段协议组（早期.3/.5/.8，V/R .2/.4/.6）。停滞状态仍使用已继承的r2定义，只关闭付款，不重定义成任意低奖励或失败。

不得把这个2应用于“任何诊断异常”。flat与H00保持集各自连续2次≥12/15失败的behavior_regression计数、最低子级连续4窗口F_EMA≥.80计数独立保存，不能互相累计/清零。

## 放行条件与交付

CPU工作可继续。DSH修正实现、补测试并独立复验后，更新冻结清单；180 passed是旧代码证据，保留但不得替代修正后的完整CPU结果。请回执本裁决SHA、实现涉及文件SHA、状态机逐帧证据、软限位运行时对照、完整CPU与独立复验报告。

在上述修正与复验通过前不进入GPU smoke/probe/训练。通过后按既有SDS继续GPU工程检查；只有所有工程步骤完成且独立最终验收blocking=0、SHA一致，才自动启动新v06训练。本裁决不豁免后续验收、不要求重新询问已授权的后续步骤。奖励权重、PPO、课程、预算、seed均不调整。

主Codex本轮验证：问题SHA匹配；只读核对生产候选与mjlab源码；独立标量状态机7组断言及2项timeout锚点通过。未运行生产完整CPU套件、仿真或GPU，未验证训练效果改善。
