# Q010：旧447→c9ee证据直接复用独立兼容审核

日期：2026-11-20。**兼容审核通过，仅批准本次 S1.0 旧证据向下述唯一新源码的直接只读复用。** 旧生产者及原文件保持原样，历史费用不回退、不重复计费。此结论可作为单独身份清单的review依据；不批准资源物理执行、学习或阶段晋级。

本报告不包含尚未生成的正式清单SHA，也不反写后续真实干跑/资源结果，避免报告、清单和新结果相互引用形成哈希循环。资源engineering entry须在正式清单建立后完成真实r3零物理干跑，再另行签发。学习继续未批准。

## 1. 唯一绑定身份

- 旧producer完整SHA：`44780cd5d6f9ee4029fb38b2f9348a872268e7bedf995d4b36e2e30ec2f2163f`，322文件。
- 唯一consumer完整SHA：`c9ee8789f6051bc6c36cbfc24f076deba6121d8ffedb48ce5d689e0a6b665ea4`，323文件。
- 新freeze：`logs/a2_v09/s1_seed42_r3/source_freeze_s1_q010_final.json`；文件SHA `827b51a53463ed706abf5b2123695aae02700e31883aafd23f5ff26298d5d7f2`。
- run_id：`a2-v09-s1-seed42-r3`；resolved目录 `/home/xxs/dog/unitree_rl_mjlab_a2/logs/a2_v09/s1_seed42_r3`。
- SDS：`500ad32aaa946c927459427f3a65565b15d4b77fca4be214817121d15c742976`。
- 唯一原6400：`30ed23ba16f7e490afca001fbbecadb4a11ff4b11733a5ae7ac7d0c841f9a8b2`。
- 原预留计划：`fa439aac08c47fda060413cbae2942ce92ef39194d49bf850dd68a9f5ea7b418`。

旧端完成、结账及依赖连续性依据为保留原文的 `docs/v09_s1_0_sealed_integrity_review_20261120.md`，SHA `f2fcd9e5b99be5eb643a0adfbb8bbb8ddbb1e0dc67630498d2efeb4e1daf91d2`。本轮没有重新解析或公开真实final的行为内容；对其原文件仅stream hash。

## 2. 完整源码差异的独立核对

独立读取旧 `source_archive_44780.tar.gz`，归档SHA `f62f12d5e9513072c2d315d06fa05bb637a82a856515af7c78118a8410b00d28`，tar内文件集合及322个内容SHA与旧freeze逐项相等；没有解包覆盖工作树。新323文件亦逐项匹配当前freeze。结果为 **313不变、9个既有文件修改、1个新增、无删除**。

| 变化 | 实际内容及边界 |
| --- | --- |
| `v09_execution_rulings.md` | 只追加Q010；Q001～Q009原文不变，原SDS不变 |
| 新 `evidence_identity.py` | 单向身份、精确路径/种类/suite/manifest/内容SHA、同一账本image及fresh final读取资格 |
| `resources.py`、`probe_a2_v09_s1.py` | 20唯一gate_A/L40/0.3源病例→5轮100物理槽；旧校准和输入受控读，记录Ref |
| `a2_v09_resource_worker.py` | 自行重建同一Context；CPU预检当前checkpoint；固定r3缺任一/全部identity flags提前拒绝 |
| `s1_runner.py`、`train_a2_v09_s1.py` | 受控旧索引/校准/teacher读取；新resource外层和学习批准仍严格新SHA；正式恢复绑定同一Context；传播公共参数 |
| `evaluate_a2_v09_s1.py`、`video_a2_v09_s1.py`、`compare_a2_v09_diagnostics.py` | 输入权限/来源字段和同bytes当前checkpoint预检，显式run_id；旧checkpoint不进入学生通道 |

逐段检查了全部变化：原teacher采集循环、物理步、首次终态选择、评分表达式和预算结算算法未改；共享evaluate入口的差异为前置身份/读取及新增来源字段。`calibrate_a2_v09_s1.py`、`finalize_a2_v09_baseline.py`、cases生成、env/map/model、metrics/acceptance、geometry、distillation、checkpoint schema实现、budget及原v07物理/奖励代码保持原字节。

未改变速度、随机化、奖励/损失、课程、终止条件或阈值：控制0.02秒、物理0.005秒、48步rollout、5×4=20次Adam、微批≤512、LR=3e-5、clip=1、checkpoint调度及normalizer/std冻结合同均保留。原校准冻结值 `2.136540383972894` 保留，不替换为独立重算值。

完整diff与逐文件old/new SHA已存本机证明，见第7节。失效候选2b35与最终c9ee的源码集合差异仅为worker的固定r3缺参守卫；2b35没有获得兼容或资源批准。

## 3. 运行依赖、资产及旧原件连续性

对旧三份身份清单记录的 **34,883个常规路径**重新实际读取内容SHA，零变化、零读取错误。八关键包非缓存文件、stdlib/site-packages全部非缓存Python源、Python启动文件集合均无新增/删除；包含A2资产（17个STL）、mjlab/MuJoCo/Warp/Torch/Numpy相关已记录内容和唯一6400文件。

覆盖边界沿用旧端报告：不是live sys.modules完整转储；设备/匿名映射、排除的生成缓存及未记录晚加载资源不被夸称已全量测量。Q010新增模块仅用既有标准库及原budget/cases实现，没有新增安装依赖。已记录外部依赖和实际资产不变，结合全部源差异核对，未见改变旧基线/校准结果的输入变化。

旧catalog v3中 **941项文件**（7 frozen_inputs、12拟消费runtime、922 audit材料）逐项重新hash，无不符。七manifest均从同一份已验bytes解析，并通过当前未改变的生成算法验证；三个final的结果原文只stream hash、不解码。原producer字段及原校准/索引文件内容SHA不变。

原活账本路径仍为本run的 `budget.jsonl`。批准锚点为前 **6,866,456 bytes / 908行 / sequence907**，SHA `c526987c278867f390ae7c02dd50ccd5de898f8df734ed0d6a157f7441e4194e`，末事件SHA `503fbb81eb11acff4bc068d722c4cba65905c8a1424359c9bf3d01a539ee737f`；实际前缀再次匹配。前缀副本仅供audit hash，不替代活账本。校准内嵌sequence906为保存时历史状态，实际close907，绝不回退该费用或计数。

## 4. 批准清单的最小权限

正式清单采用schema整数1、`approval_type=q010_s1_0_evidence_identity`、唯一scope `S1.0_evidence_reuse`。顶层绑定完整旧/新SHA、固定run/resolved路径、SDS/6400、上述活账本前缀及本报告内容SHA。清单位于snapshot之外，没有通配、传递兼容或通用ignore-SHA分支。

- **frozen_inputs：** 七套原manifest单列，包括S1_DIAG输入。它们没有producer code字段，走专门read_manifest并保留purpose限制；没有借此兼容旧DIAG运行结果。
- **runtime_evidence：** 仅12项：baseline_index（suite=None）、calibration（suite=train_calibration）、五套teacher_summary/teacher_episodes。原JSON保留447；Ref另记唯一新consumer和清单身份。
- **audit_material：** 原PT/geometry/source_snapshot、原独审proof、依赖/归档/前缀副本和本次核验证据仅可hash，read_legacy_bytes/json均拒绝；其嵌套记录不授予其他路径读取权限。
- **禁止复用：** 旧resource、旧学习/执行批准、旧学生或probe checkpoint、旧DIAG结果。新resource外层、新checkpoint、新DIAG/评估和学习批准继续严格新SHA。

teacher_summary实际无SDS/source字段，按原checkpoint、manifest、code及内容SHA验证；JSONL行无code SHA，通过显式关联summary与两份原始内容SHA证明producer。解析同一份已经hash的bytes，不在验证后重开原文。正式resume和Q009训练恢复还要求保存的同一Context metadata。

final的hash-only完整性核验没有解封效果；读取final输入、teacher摘要或episodes仍须真实fresh `validate_final_unseal`、当前candidate SHA、当前套及活journal/ledger。没有以旧907的学习0/门禁0替代当前资格。子进程已close费用而父尚未finish的窗口可读，父finish后旧资格拒绝。

## 5. 缺陷闭合与验证

本轮明确闭合：

- R-RES-ENTRY-01：真实CLI混淆20源与100槽；修为共享helper验证20唯一SHA/id/key、gate_A/L40/0.3、各5005步，复制五轮。
- R-ID-TAIL01：长生命周期Context漏验非法新尾部；现在每次实际读取核完整当前链/状态。
- schema浮点身份：JSON浮点1.0拒绝，只接受整数1。
- PREFIX-SWAP01：前缀与snapshot两次open混验不同文件版本；改为LOCK_SH单次捕获整个活image，prefix/事件/原Ledger `_read/_state` 全部基于该image，不ctor/snapshot重开。budget字节未改。
- R-RES-ID-OMIT01：固定r3 worker省双flags回退普通loader；新增限定OLD_RUN_ID前置拒绝，两个mode及原字节/等价空格、缺单flag均有真实CLI负例。

验证证据分清执行者和版本：

1. 本审核员此前独立核心31项通过，原两版本替换及真实journal/ledger final窗口正反例通过；当前EID源仍为同一SHA `7afc473fad762fb3a95f742716aa76bebaf16c2da8c3f87a18fd04e680d3df0a`。
2. 对最终c9ee增量，独立真实CLI测试 **30 passed，69.11s**，一条既有Torch弃用警告；源323及测试前后SHA一致，临时ledger/输出目录拒绝路径通过，CUDA禁用、0物理。
3. root最终c9ee全部16模块 **239 passed，197.04s，exit0**；其证明、日志SHA和源/tests不变字段已核对。失效2b35的独立231项不作为最终版本准入依据。
4. fresh whole-feature独审最终 **62 passed，70.82s**，无未解决Critical/Important/Minor，报告SHA `4015c67106d52e7b9b557561a1b56bfd2134255758e481fd5c58a4861b0b5b3d`。本报告另行完成了该fresh报告明确留给后续的实际依赖、资产和catalog逐项核验。

这些是代码、CPU及文件身份验证，没有运行GPU仿真，没有验证真实资源稳定档、显存或学生行为效果。

## 6. 当前结论及后续顺序

确认所列旧S1.0证据对唯一c9ee consumer兼容，可签发精确只读直接复用清单。原文件、原code SHA和原收费保持原样；无法证明的其他旧产物不获得读取权。A/B、旧S1.0报告和签字不修改。

随后只用该清单运行真实r3资源主CLI **不带`--execute`**，核实际旧校准、20源→100槽×5005、无输出目录和活ledger字节不变；结果另存，不回写本报告或不可变清单。该干跑通过后，再签单独 **S1_RESOURCE engineering entry**。真实资源须继续四档2048→1536→1024→512、Q009完整负载及原16,030,340资源预留，不更改预算或门槛。

本报告与复用清单都没有S1.1/S1.2学习范围。正式资源实测通过后，学习批准另审；Q008新证据和行为门禁仍按SDS执行，S2/S3没有授权。

## 7. 独立材料身份

本机下列路径相对于run目录；不含真实final成绩，也不作为runtime原文白名单。正式清单将只引用本报告的内容SHA，报告不引用清单自身SHA。

| 材料 | SHA256 |
| --- | --- |
| `q010_source_comparison_c9ee.json` | `05fa6dd42faccb7fd59d648039b2d57cd47f89141cb58684f267d8670fcde3b2` |
| `q010_source_diff_old447_to_c9ee.diff` | `41ab966d441f4df633a7733d11dc2166133c49b5d796a0f43ec9b29987ac7df5` |
| `q010_runtime_comparison_c9ee.json` | `b35e7c37c1989176fca7ec2df627f832d10151d19c7ce77d778854f4242b29d7` |
| `q010_old_catalog_verification_c9ee.json` | `296fb14a73be0c14dc83e6a3c8a9c5c6a00879661f4036ffb4ba445732f60e91` |
| `q010_independent_cli_c9ee.json` | `4e982bdc491bfe4993162ab86a779d52f9b5b95a9f73369075b38e9336ec7e9c` |
| `q010_independent_cli_c9ee.log` | `cbe51d93910169941a882f8d8216495ae7a373943882221737b79abd8fc3751b` |
| `q010_full_cpu_final_20261120.json` | `074d46c48c11d3012f56b4e30680c2fa789d134c0e08bcaa29dd88ab41bccb33` |
| `q010_full_cpu_final_20261120.log` | `526bbe4caefea34459ce37124e0171cb73696fb0f594278ee38e714ade40883b` |

审核员只新增本报告及本机diff/证明/清单材料，不改作者源码、测试、Git index或真实账本，不读取final成绩。修改日志及报告提交由root汇总至 `logs/codex_changes.md`；本机大清单、轨迹和批准JSON不提交Git。
