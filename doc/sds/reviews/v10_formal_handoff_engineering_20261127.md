# A2 v1.0 手动正式训练交付验收

日期：2026-11-27。目标是交付可在用户终端完成正式流程的脚本，本轮不执行物理或学习。

## 实现与流程

`bash scripts/run_sds_manual.sh --mode all --device cuda:0 --num-envs 1024 --rollout 48 --seed 42` 进入完整正式编排。入口固定原运行目录和学习、评估、probe 三本外账，不再创建试训。`--mode check` 只读预检，`--mode teachers` 仅完成教师准入；完整运行才在六套教师 A/B 全通过后生成交接凭据。

交接绑定原第50轮checkpoint SHA `74d9d2de82860fcd806521e4eef20ef74add7899582c987844e80350557010e7`、封存trial guard及结果、账本前缀、A1教师配置、资源和112诊断、六套正式教师结果及新旧源码。原文件/历史标记不变；正式后继另存provenance。模型、Adam、replay、RNG、任务轮转恢复，物理回合重新开始。普通resume仍拒绝不带有效凭据的trial。

D1跳过零轮entry，从已有50轮继续到原绝对门禁200/400/500；D2门禁400/800/1000；D3门禁600/1200/1500。唯一DAgger终验通过后进入P0前100轮，再进入P1；PPO总上限1000轮，门禁250/500/750/1000，原保持与最多两次回退不变。最终通过才导出部署。门禁失败或预算不足仍停止，不等于保证训练成功。

N=1024，每轮48步，每2,457,600样本（50轮）及门禁边界保存。D1/D2/D3/PPO原额度为24,576,000 / 49,152,000 / 73,728,000 / 49,152,000。原50轮2,457,600样本继续计费，不重复赠送。六任务各32768条FP32、微批256、LR3e-5、速度集合0.3/0.5/0.6均不变。奖励、损失、教师A1增益、物理、观测、模型、采样、优化与资源测量路径前后不变。

正式worker完整保存后新增轻量恢复manifest，绑定operation/target、同账和CP SHA。恢复仅采用匹配manifest的保存点；未保存成果可能丢失，但费用不退。事件重放不清除已完成teacher/operation，不将新CP倒退到原50轮。旧Ctrl+C留下的评估pending只在controller与GPU排他检查后记为uncertain，不退款。

## 源码与证据链

- 冻结SDS SHA：`eb95905ed24bf0674f94a1b02dab02256ee11e40a20ed1cd96eb24daeb1b49a0`。
- 原源码：`d54265108c298d5920ae8ea9316fd3d48be4b75ce989ac0507bdbf9a2b91498e`。
- 当前最终源码：`cf4d3a6495569ae8a785a00288acfa1faecb3204c6a5ed675eb1dcd4c3ec4db4`。
- 精确旧新源码兼容只允许列明artifact、purpose和operation；旧passed文件不修改。旧完整资源+v2补证+112通过是明确继承，未重跑。
- 原controller与旧execution_plan归档保留，新源码操作日志使用独立handoff命名子目录。当前源码教师结果重新严格检查，不把旧中断partial视为通过。
- 四个既有dirty文件逐一SHA核对未变，未夹带提交；source指纹包含这些实际本机字节。该手动入口依赖本机已有checkpoint与归档，不是脱离原数据的全新空仓启动器。

## 验证证据

完整CPU最终结果与只读真实预检在交付前追加于下方。独立审核见 `v10_formal_handoff_independent_20261127.md`。

首轮全V10 CPU：217通过、2失败、5跳过；两个失败来自旧v2测试错误地要求新增formal worker仍满足旧授权范围。限定修订这两个测试，生产AST授权白名单未扩大；现在分别断言当前源码旧scope拒绝、原d542档案通过、新显式兼容链通过。原失败JUnit保留在 `logs/a2_v10/formal_handoff_validation_20261127/pytest_initial_failed.xml`。

独立审核另发现公共checkpoint loader初版可接受伪造caller dict；已修为重新读取完整凭据并校验SHA，伪造路径/缺前置拒绝。CPU模型/Adam/replay/RNG/环境角色状态等价、100轮保存中断后恢复、同账/前缀/篡改/事件幂等均有针对性回归。

本轮不重跑昂贵物理、满3GB replay资源测试或整个旧项目测试集合；前者沿用已验证证据并限定继承，后者历史收集冲突未在本范围修复。新增恢复测试采用小CPU模型/小replay/虚拟费用，不能称作学生独立行为验收。

## 尚未执行的行为阶段

本轮新增物理样本=0，新增学习样本=0。完整正式教师A/B、学生课程门禁、DAgger/PPO最终验收与部署尚待用户运行脚本。未生成可执行正式handoff或任何学生通过证书；只有条件满足时脚本才生成。

## 最终 CPU 验收结果

完整 V10 CPU 回归 **219 passed / 0 failed / 5 skipped**，178.57秒；1个上游Torch弃用warning。跳过3个显式物理测试、1个CUDA专用测试和1个满3GB replay测试。最终源码与测试前指纹完全一致。JUnit及SHA见 `logs/a2_v10/formal_handoff_cpu_20261127.json`。独立工程审核已通过，报告SHA `02db630f4cbccce9cdd8b2d1bb637d78e144e2884221a4624e059877d06d22fb`。

## 真实手动入口只读预检

已实际执行显式1024/48/seed42/cuda:0的 `bash scripts/run_sds_manual.sh --mode check`，exit=0、preflight_passed=true。原50轮CP、trial guard、controller及学习/评估/probe三账前后SHA完全相同，未生成可执行handoff，新增物理和学习均为0。结果及完整前后摘要见 `logs/a2_v10/formal_handoff_validation_20261127/preflight.json`，SHA `4257ab7817c4cf15eab626d93260d0af26c2be937431e99e7f255121d7310719`。

最终结论：正式手动入口工程验收通过，当前证据和预备计划可用；用户可运行 --mode all，教师准入通过后自动交接并继续全部课程。实际行为门禁仍须运行，不提前宣称通过。
