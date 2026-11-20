# Q012 after 数值独立验收（2026-11-20）

独立结论：Q012 裁定规定的 after 数值阻断解除条件全部满足。本结论只解除该数值等待；原 9d4 学习批准继续有效，不新增学习范围，不代表行为恢复。production 四路 comparison、adapt 75 视频及运行时 S1.2 工程条件仍须沿原流程完成，尚未由本报告验收。独审未执行 GPU、环境构造、物理步、梯度更新、信号或 Git 操作，未读取封存 final 成绩。

本次从两份实际输出 PT、75 份 after 原始 PT 和冻结清单独立重算，未以程序汇总声明代替 tensor 核验。原式 `abs(x-ref) <= 1e-5 + 1e-4*abs(ref)`、方向和数据均未改变；每个比较先严格检查 FP32、有限性、非空、等 shape，再以 FP64 求差、max/RMS/超差计数。原始文件 SHA、case 顺序、actor_index、原 slots、观测每字段摘要及布局与标签摘要均匹配。原件共 845 个教师快照、15,110 个有效状态；异常 inventory 的 703 个快照与 571 完整异常及 132 部分异常逐项对应。另有事前冻结的平地/楼梯完整正常对照各 1，以及适配学生全部 75 个满 20 槽固定初始批。

每个独立进程核验 1,130 次 N20 前向（教师 1,055、学生 75），输入共 22,600 行。两实际 PID 为 3446918、3447298，独立 token 不同；同原 GPU UUID。教师使用自身旧图 `[20,3,48,32]`，学生使用自身新图 `[20,3,40,30]`，没有把教师标签当作学生保存输出。两次输出对保存参考及两次互比的独立结果如下（每次均相同）：

| 对象 | μ 分量分母 | V 分量分母 | max abs / RMS / 超差数 |
| --- | ---: | ---: | --- |
| 教师 | 196,680 | 16,390 | 全部 0 / 0 / 0 |
| 适配学生 | 18,000 | 1,500 | 全部 0 / 0 / 0 |
| 合计 | 214,680 | 17,890 | 全部 0 / 0 / 0 |

两个事前固定真实模板的输入行摘要与 case/frame 身份匹配且不同，active 原字节回原 slots，完整输入 N20 布局保持。全部 109 种实际部分 mask（最少 1 活跃槽）均独立从全部原 PT 提取复核。每进程 241 项双填比较、2,465 行；218 项遮盖对未遮盖 GPU 原批比较、2,152 行，逐 μ/V 比较全部 max/RMS/超差为零。132 部分快照只证明保存 active 在两种固定补位下满足合同，不声称重建历史 inactive 观测。

6400 文件 SHA、适配 checkpoint 345335…d800、两模型完整状态摘要和全部 53 项布局均核对。实际 receipt 的 eval/无梯度、前后 state 不变合同与实现检查相符；normalizer/stat/std 沿只读 forward/evaluate 路径，未执行统计更新或 optimizer。原始源码 C9、SDS、6400 及 Context 0d158…0408 均通过实际只读身份校验。两进程的 expected/actual 软件、驱动、精度 flags、环境及每模型原生完整输入布局完全一致；冻结的配置来源文件仍逐项匹配。采用的是 GPU 输出产生前固定的 contract，没有按结果选 flags。

历史生产 flags 未直接逐批保存，这一事实不变。原 CPU FP32/FP64 例外及 pending 原报告继续原封保留：教师 CPU 对保存输出 μ 4,991 分量超差、V 244 分量超差；学生固定初始 μ 92 分量超差、V 无超差。GPU 同合同重复通过不证明具体算子原因，也不抹除 CPU 证据。

独审用共享锁读取活 ledger 的同一 image 并验证完整链/state；审查前后文件 SHA 均为 369a89df9fe57943c3b1657dcb6abaec415beaa53b987d4c2223045dc5f18730，sequence 5067，事件摘要 6ae5048d1f4cd2c3636c41a05419cbcd71ffc2dc9aa74818b670b047d743315c。无在途未结费用，累计学习仍 19,980,288；此次重放没有新增物理或费用。安全证明绑定 after 子自然 exit0、已卸载训练环境/buffers、无子 CUDA 上下文；核验时原父 3228141/start_ticks 145922371 仍为 T。仅 root 在提交推送后再次严格核原 PID/tick/argv、子退出及账本身份，并按裁定 CONT 原父一次；本独审不发送 CONT。

固定证据：

- 独立实际输出 proof：`q012_actual_gpu_independent_proof_20261120.json`，SHA 691a781b94e0ba0cb32d9e58a1e0893f857b9b6e69ddb514eefb0bdd2d85d5b0。
- 冻结 manifest：162449d01f1ebfca84d266b45e51641b650fba5e5f112204895065d506b88fbb；script：7018023ef54aa892970173033acb29363fd88417ecfc846cd9c7a277f6b213f4；contract：e06483543b084780e6beaeee80ae1bb9ac7b6b7db8261d496b710c078a52dd88。
- 实际 report：eda2000ff458fdbcbc3d875f8af4977d41c32cc47383cfa70b354c40c5138cf0；pass 1 JSON：04b5e35c5a71e7de903309785fd097176126e7c290585c923605347e9040e1f9；pass 2 JSON：7ad176e688f8dc26c3480c41115e8ac55c17ab603e500236208bf5f9b9c9d889。两 PT 内容 SHA 和全部原件 SHA 见 proof。
- 正式裁定：1bbdaa97b8eb79905e4520480c73a49c055d33b983de69f6a51329d6bfc5138c；fresh CPU 复审：f46ae4cd837dee3ccab782b2ecc7ce6176a8fdc9675a629d23177038857bb846；root 安全证明：814de236f61a354b8d81f813f825483c5be569418df06918cbd545d50b72c776。

原 9d4 JSON 已经实际 `validate_learning_approval` 连同同一 Context 通过；外部 Q012 clearance 只有数值解除 scope，不能替代学习批准。生产源码、测试、既有报告/catalog、journal 和账本未修改；共同变更日志与提交由 root 统一追加维护。
