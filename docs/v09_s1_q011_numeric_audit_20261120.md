# Q011：Q008跨后端标签数值核验与首梯度等待

2026-11-20，规划线程`01a1f385-ceea-74a0-a77c-85f66b2ccd77`裁定及补充①②③的执行记录。绑定当前C9完整源码`c9ee8789f6051bc6c36cbfc24f076deba6121d8ffedb48ce5d689e0a6b665ea4`、r3及原学习批准；本文处于源码snapshot之外，不修改正在运行的rulings、源、参数、旧数据或清单。**当前Q011未解除，正式学习仍0。**

## 触发事实

初始DIAG1500例已闭合，控制收费539740；独审首回合438030、padding101710。75几何、1500首次终态、raw metrics、actual命令/action历史及teacher/student非地图输入检查通过，但跨后端标签数值核验仍pending。9366个保存教师状态的CPU FP32重算，原逐元素容差有动作2407/112392分量、value10/9366例外；最大绝对差分别1.58667564e-4和7.853507996e-4。原件有限，未重写标签。

CPU FP64参考接近CPU FP32；固定第三个map Conv按TF32量化的CPU反事实能解释大部分差异，但剩1动作例外，且没有直接记录生产CUDA中间层/flags，不能据此声称某个GPU算子是具体原因。原CPU对照结论和原容差均保留。

563个保存快照中396个有CPU容差例外：340个完整20槽，56个仅保存active子集（21个批）。最严重value差为batch70/index100，仅保存2槽；缺失非活跃槽不能声称已历史重建。v1 manifest/脚本保留，v2将在GPU结果出现前重新固定严格证据范围与补位对照。

## 规划给定的核验合同

1. 同生产GPU、软件、dtype、eval/normalizer及数值设置做零物理重放；核文件/模型SHA、帧时刻、原槽顺序、输入字段及原N=20批形状。生产整批推理后才取active，不能直接用active子批冒充原批。
2. 340个完整异常与按冻结顺序选定的非异常满槽平地/楼梯对照，按原批直接重放。每条两次独立前向，μ全部12维及V均用原式`abs(replay-saved) <= 1e-5 + 1e-4*abs(saved)`；记录逐批max/RMS、例外数与两次相互差异。不能从观察到的max反推新容差。
3. 56个部分异常全部覆盖，包含2槽最坏点。先独审核冻结教师实际eval前向按样本独立、normalizer只读、无批统计更新/跨样本混合。固定两条不同case/frame且输入足够不同的真实完整教师观测行，记录字段/文件SHA；不得看标签挑模板。active原字节放回原slot，其余槽分别确定性复制两条模板。记录为“固定形状补位对照”，不称历史恢复，不改原数据。
4. 每种补位两次独立前向；全部active对保存标签仍用同一原容差。两填法之间、重复前向之间亦比较active；至少一个有完整原批和标签的控制快照按缺槽掩码人为遮盖（含2活跃槽），与未遮盖原批作同容差对照；v2准备覆盖所有unique masks。参考方向及模板在GPU结果前冻结。
5. 仅在全部指定完整异常/对照、56部分两填法、遮盖控制及槽独立性均通过，且独审确认无模型/地图/时序/槽位/数值配置错误后，解除Q011；既有学习批准继续有效，无须再次请求同项授权。CPU跨后端单独超差不阻断，但不能用GPU通过宣称某具体CPU偏差来源。任一GPU超差、填法改变active到超限或身份未明，继续阻断并报首分歧层/证据。
6. 不重跑1500物理病例；只有离线手段不能解决时，先列具体受影响批次、最短前缀、成本与预留，另经规划批准才定向物理回采。旧标签、原CPU误差与原阈值保持。

## 安全串行执行与等待

已预留的6400及迁移初始固定视频可按原费用采集；GPU重放必须等待当前录像子进程自然退出、输出落盘且GPU可用，保持Q009单GPU外部任务串行。

为保证首个梯度等待，root在发信号前核父PID3228141的完整批准argv/run、wchan=do_wait及唯一video6400子PID3239514，仅向父发SIGSTOP并核T态；未停进程组/子进程、未kill/重启/解锁。子stdout/stderr直写常规日志文件，无须父排空管道或应答，仍可自然闭合。运行journal证实before证据会话`env_released=true`、`buffers_released=true`、complete_rollouts=0、adapt/main/retry学习费用均0。

启动tick在信号后补录并与launch时间/完整命令一致性核对，未冒称在信号前已保存：`145922371 @100Hz`，parent PPID3228116，与launch时间差-0.808秒（boot时间为秒精度）。未来CONT前再次严格核同PID/tick/argv、子退出码、账本及Q011签字，只CONT一次并确认子只结算一次。

本机暂停收据`q011_parent_hold_20261120.json` SHA `269e60d17965594c19a8e8ba2f147cc36c68c7bff8e7596e755ffff3519b6a19`；身份/FD guard `q011_parent_hold_identity_guard_20261120.json` SHA `b9d719c17cefd9faf7e86aad664fa8a7c61cb7727a03a9d160f7a86253c6fe3d`。这是工程核验中的进程等待，整个授权目标仍在推进。未改变物理/训练配方，未声称行为改善。

## 已有独审材料（本机）

- pending DIAG证明`q008_before_diagnostic_independent_pending_Q011_proof.json`：`b42aa20485791cc7fd1f0a3b36ec212e85105b044710fd8f958cfa01b7a55397`。
- 首差异CPU参考`q008_first_label_difference_cpu_reference.json`：`6b5dd20b67d77cb41e937ff8cd34cada6b44ae797e31bf5ae6361e158a3a34b7`。
- 固定TF32 CPU反事实`q008_all_labels_fixed_tf32_cpu_reference.json`：`3c15a0217f4b9ef1fd9bafedfba72b4de2a4552b5a46468a2c9316578a0a14f6`。
- 保留v1完整批清单`q011_gpu_label_replay_manifest.json`：`ce46807233953a92948581d2901e1337c2c981cafced97953c1e0004fedf60a6`；v1不授予忽略56部分异常的放行资格。

v2清单/脚本及实际GPU结果尚在准备，未执行GPU重放，未解除Q011。后续结果单独追加记录，以上既有材料保持原样。
