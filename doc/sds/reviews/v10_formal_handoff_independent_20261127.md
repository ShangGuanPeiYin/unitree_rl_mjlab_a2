# A2 v1.0 正式手动入口与试训交接独立工程审核（2026-11-27）

**结论：本报告绑定的源码工程独审通过，未遗留已识别阻断。** 旧新源码兼容仅限明确归档和批准的产物/操作，默认严格检查不豁免。本轮仅交付代码与用户手动正式运行sh，审核未运行正式sh、物理仿真、资源探测、教师/学生评估或训练；不声称正式行为验收通过。

受审当前源码指纹：`cf4d3a6495569ae8a785a00288acfa1faecb3204c6a5ed675eb1dcd4c3ec4db4`。精确来源为冻结`d54265108c298d5920ae8ea9316fd3d48be4b75ce989ac0507bdbf9a2b91498e`归档，逐文件差异如下。后续任何指纹源码变化不自动继承本结论。

## 实际执行流程与参数核对

| 要求 | 已核实际接线 | 独立判断与边界 |
|---|---|---|
| 用户手动启动，不再试训 | `scripts/run_sds_manual.sh`前台exec固定a2 Python；`scripts/run_a2_v10_formal.py`只调Orchestrator完整run | 通过。set-euo-pipefail、显式check/all/teachers，固定1024/48/seed42/cuda:0，其他N/配方参数在argparse拒绝。本轮未启动该入口 |
| check不执行物理 | launcher先`preflight_formal_handoff`，check直接返回；`orchestration.py:65`预检兼容/工程证据/plan | 通过源码及CPU seam。真实最终证据生成后的预检由根代理另行记录，本报告不伪造已完成的真实preflight |
| 资源与112诊断精确继承 | `source_compatibility.validate_compatible_resource`及`handoff.validate_plan`验证原资源/诊断身份、SHA、完整分母、resource receipt | 通过工程接线。是旧证据限定继承，不称为新测或完整20轮重测；旧失败上下文不抹除 |
| 先完整教师A/B | `Orchestrator.teacher_admission`检查D1/D2/D3各A/B，当前源码+同教师身份+完整20例组/有效资源receipt | 通过；缺病例、错source、失败组、无resource均拒绝。18/20门槛保持；112诊断4/4不冒充正式AB |
| 全部教师齐后才交接 | `handoff.py:161`–175要求六份完整native teacher receipts并grade；`run`在teacher_admission后`_finalize_formal_handoff`再学习 | 通过。teachers模式只预准入，不训练、不生成handoff；all才幂等生成，符合根代理明确的模式边界 |
| 保留原50轮CP/费用 | plan绑定原CP路径+SHA、原guard/result、learning/evaluation账本前缀；`bind_handoff`核retained50、费用2457600、完整state | 通过。原CP不改字节，trial历史false/null不涂改；派生CP另写新source及formal_handoff来源。原CP声明SHA为74d9d2de82860fcd806521e4eef20ef74add7899582c987844e80350557010e7，完整196608 FP32/Adam/RNG结构验证复用根既有JSON，不在本轮重复3GB加载 |
| 从50继续原D1 | `_train_dagger`仅有handoff时跳过D1 target0 entry，直接原spec首门禁200 | 通过。D1门禁200/400/500，非125；最多还450轮，首门禁还150轮。D2=400/800/1000，D3=600/1200/1500 |
| 原配方与预算不变 | spec/环境/教师/DAgger/PPO/奖励等关键文件与d542归档字节相同 | 通过。D1 24576000、D2 49152000、D3 73728000、PPO 49152000，总196608000；每轮49152、保存2457600样本=50轮。LR3e-5、微批256、六任务各32768 FP32不变 |
| 课程到最终export | 仍执行课程A/B同SHA、唯一DAgger终验→P0→P1双参考保持/最多2回退→唯一PPO终验→证书export | 通过源码路径审核；PPO总1000轮，P0前100轮，PPO门禁250/500/750/1000。当前未实际完成任何新正式行为门禁，预算上限不保证成功 |

## 旧新源码兼容与实际资源验证差异

`source_compatibility.py`固定ORIGIN_SOURCE=d542，当前to source必须等实际全仓指纹；baseline manifest逐项重算，差异列表须完全一致，允许文件范围仅handoff/恢复/入口/验证及CPU测试。每份继承产物须绝对路径、原SHA、明确purpose和operation-id；独立review及批准引用须绑定hash。默认None兼容不放宽，旧receipt不会被改写成新source。

实际逐文件核验：`spec.py/env.py/teachers.py/dagger.py/ppo.py/rewards.py/resources.py/geometry.py/tasks.py`均与归档字节相同。当前rulings SHA `4de1477cbabb7b652cfa2b5bc21a589a2e6282568c1eec715d917673846fd898`与原v2 selection一致。

`resource_a1.py`仅为只读重算提供明确source_root/兼容验证分支，以冻结原source复验既有资源关联证书，并逐引用SHA核验。独立读取真实archive比较四函数`run_a1_probes/a1_probe_stack/run_a1_v2_probes/a1_v2_probe_stack` AST，全部与d542相同；生产测量/更新/保存恢复路径没有被这次兼容改动改变。不能用allowed文件名本身代替逐diff审查，本报告审的是本次实际字节。

## 发现并修复的阻断

初版`checkpoint_load(formal_handoff=dict)`直接信任caller字段。独立CPU最小复现实证：默认严格拒绝伪造旧source，但没有任何JSON凭证的裸dict可令相同文件被接受（`UNAPPROVED_RAW_DICT_ACCEPTED True`）。已报根代理和实现者，后经RED→修复。

最终`checkpoint.py:102`–119在任何兼容加载前按path重新读取`validate_handoff`，核六AB/资源/plan/bridge引用，并核提供snapshot SHA；仅使用新验证context。随后仅对exact origin路径+SHA+oldsource+SDS/recipe/teacher/N/physics匹配的CP例外；current-source后继仍走严格metadata验证。裸dict缺路径拒绝，凭据变化拒绝，默认None严格不变。独立最终12项测试已通过，原漏洞关闭。

## 正常恢复、中断和事件幂等

- controller-source迁移要求旧归档snapshot与当前旧state完全一致，兼容旧controller与frozen plan引用均指向不可变证据。新operations/evaluation使用handoff_<source前16位>子目录，保留旧费用/旧失败输出，不能把旧source评估当新teacher准入。
- hash-chain事件使用唯一ID、引用先验证、事件/证书先于controller状态；source-transition在已新snapshot重放时不清新operations/teacher；handoff-adopt重放不把最新CP倒退为原50。对应RED复现后修复及独立CPU回归已核。
- 原共享evaluation ledger中的pending仅在非阻塞GPU flock证明无孤儿worker后reconcile为uncertain；charged不变。learning同原ledger，checkpoint较旧不能退款；计费始终含executed与uncertain。
- 正式worker每次完整CP保存后原子写manifest，绑定operation/target/course/N/physics/source/同ledger/handoff SHA及CP SHA。controller只选同操作的本目录真实CP；manifest不符或篡改拒绝。100周期保存后101采样前中断，CPU实测恢复100再1轮到101，Adam/replay/RNG与环境任务轮转恢复等价。
- 对于保存后尚未落盘的后续工作，外账费用仍保留，恢复最近完整CP会损失该段未保存成果而不获得退款；不能将累计物理费用解释为当前CP保留了同等数量的更新。

## 独立实际CPU证据

1. `python -m pytest tests/test_v10_formal_orchestration.py tests/test_v10_source_compatibility.py tests/test_v10_manual_launcher.py -q`，使用`/home/xxs/anaconda3/envs/a2/bin/python`：**18 passed，1个上游warning，10.16 s**。覆盖手动接线、只读check、首200、失败teacher不得交接/训练、source迁移/事件重放、同操作manifest、计费不退、兼容篡改拒绝与真实baseline资源AST。
2. `python -m pytest tests/test_v10_handoff.py -q`，同a2 Python：**12 passed，1个上游warning，65.26 s**。覆盖六AB完整case身份与grade、计划不是handoff、缺AB/错source/无资源拒绝、不可变幂等、账本prefix接受append拒绝改史、默认旧source拒绝/正式来源限定、裸dict攻击拒绝、模型/Adam/replay/全局RNG/env游标恢复、后继正常恢复、周期保存中断恢复。
3. `bash -n scripts/run_sds_manual.sh`及`git diff --check`均通过。

本代理首先独立实跑上述30 CPU测试通过；最终又在仅两个历史测试变化、生产字节不变的版本实跑`tests/test_v10_resource_a1.py tests/test_v10_resource_v2.py tests/test_v10_source_compatibility.py`：30 passed、1个上游warning、16.27 s（source compatibility测试与前次有重复，不称为60个不同测试）。测试使用小模型/小replay/假环境/虚拟费用及真实checkpoint序列化，不冒充1024环境正式训练、满replay资源或真实行为评估。本轮主代理全V10 CPU全套仍在运行，结果由其另工程报告绑定，本报告不声称该全套已完成。唯一warning为上游torch.jit.script弃用。

## 最终边界与待执行事项

当前实际正式教师A/B、学生A/B、DAgger/PPO终验、最终export均未因本轮工程测试完成。主代理还需用本报告及最终源码生成明确批准的compatibility/engineering/plan产物，执行只读真实preflight并记录结果；没有凭据时入口会失败，不默认为兼容。用户此次只要代码与手动sh，根代理不得在交付过程中启动物理。

修改日志及提交推送由根代理统一处理；本代理仅新增本独立报告。正式操作说明见`doc/sds/reviews/v10_manual_formal_training_20261127.md`。

## 历史scope测试修正记录

根代理首轮全V10结果为2 failed、217 passed、5 skipped；两失败分别是`test_real_invariants_keep_env_byte_difference_and_exact_recording_ast`和`test_v2_fixed_profile_and_probe_function_ast_only`，仍假定当前runtime只含旧v2试训签名差异。新formal_handoff超过那个旧scope，生产验证正确返回False；不能扩旧AST白名单来使其通过。

集成实现仅改`test_v10_resource_a1.py`与`test_v10_resource_v2.py`：明确当前source直接旧v2不合格且批准拒绝，真实d542归档用显式source_root通过，删去历史source_diff项仍拒绝。独立已逐diff确认生产文件零新增修改，并实际追加上述30项CPU通过。这是对真实scope变化的测试契约更新，不抹去先前失败，也不宣称新source获得旧scope默认豁免。最终全套由根代理重新验证并另报。

本独审初稿绑定9ec源码（报告SHA 462e82be361b0f4fca4cc912d1aa2595df3d371ac1e73cca6e6228d16a755aef），未用于最终兼容签证；两测试定稿后，本最终版本绑定cf4d源码。除这两个测试，生产及已审接线保持相同字节。

## 精确差异文件SHA

- `scripts/run_a2_v10.py`：before `1c5a3f5f0254581d27a91bea3938ef3b94d88a542d86dc071c5cafa0bfe6dc20`；after `1aa347b4ad17aede53a616281f8e1fc061fac1c6963e0d09da3c476ed694eb83`。
- `scripts/run_a2_v10_formal.py`：before `None`；after `e96c145ea70436a7fd7399ba3e1e084b8d97df7526e57e6470a2badfeb75515a`。
- `src/tasks/velocity/v10/checkpoint.py`：before `12620a96dd69db42ad4c5769519cb89d7c65367b99dda38e581dfcb99675c737`；after `ce749c561664a9a88cbb5ce7a1c620d3c95f6af9093cf23fbe912e56f6405eb0`。
- `src/tasks/velocity/v10/handoff.py`：before `None`；after `22bd3fafb5a6597eebeb9f1bcf05c0ee13f663ca139f9b23bf3e4c5bf8422ade`。
- `src/tasks/velocity/v10/orchestration.py`：before `23011556f52da8816b7d441ab04594b146cf065cd9b1a2752586f7442b2b76e3`；after `6edc6071676d7e30d6cc5d2d9b19e2de8a5c9da21d03ab0016a7275298670a1b`。
- `src/tasks/velocity/v10/resource_a1.py`：before `0a6a7a0eab96a159f19fca451d1549efb9f3416a16c6dc6efd2b3f01886a6c22`；after `08bd6e3114c255fbec19d1a7a28111a83316f65e8b455192a70974e15e9fbd7e`。
- `src/tasks/velocity/v10/runtime.py`：before `00677d34add3c195061d4167b0eff2611ffa8007a58c07556b4d0fecbae7f043`；after `d4ae9c9292d71db40834281450e65242d536e23602b2431af4e3607a1197ff6f`。
- `src/tasks/velocity/v10/source_compatibility.py`：before `None`；after `2e13b7d554e184c90ea936433ec67d3328ab1fea9172dc9bbb2896b59b787702`。
- `tests/test_v10_formal_orchestration.py`：before `None`；after `da8664fc03723b476d83861b649860b723552041c6927cdccad27e73804fa8d2`。
- `tests/test_v10_handoff.py`：before `None`；after `2737bb7596d47d985e431e329ea3520d3fc0e086f6e6d33525fe6d46f9a80a10`。
- `tests/test_v10_manual_launcher.py`：before `None`；after `c55720bb354edd9b1dbb12db572ca710af56227784cc5062ce893caf91c452cd`。
- `tests/test_v10_resource_a1.py`：before `2dc45d40faccadec5bf9d533dc4fd939daf5851426f4d72fee4291d8e113d3d7`；after `c37f0d4bf389ea7154f7d648373e29bb0caa5efa3dcb2425d4b14bbcc4f76e3b`。
- `tests/test_v10_resource_v2.py`：before `bb5bdf1be448eee86073c6cd480e0a2c56e42a0fc191dacb19779a8cc60448c1`；after `69c89eb48d91e6ba5f9b7e6666999da7e15395d9684177f099945d6799cfe80b`。
- `tests/test_v10_source_compatibility.py`：before `None`；after `414b1a1e8c6eb56ae852bf66328e20f98457f62a9f762438076e6a13c1167a3e`。

手动shell（不属于Python源码fingerprint，但本审单独绑定）：`scripts/run_sds_manual.sh` SHA256 `79499ea35da4ade41f159e18eb23cb6a9edcc07c986fc61fa59dac14b8339d0f`。
