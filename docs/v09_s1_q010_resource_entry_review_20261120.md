# Q010 S1_RESOURCE 工程执行入口独立批准

2026-11-20。**批准按下述冻结源码、同一证据清单和既定账本执行一次四档资源探测。** 这是工程探测入口批准，不是资源容量已通过、学习批准或行为晋级。

- code：`c9ee8789f6051bc6c36cbfc24f076deba6121d8ffedb48ce5d689e0a6b665ea4`。
- run：`a2-v09-s1-seed42-r3`；目录 `/home/xxs/dog/unitree_rl_mjlab_a2/logs/a2_v09/s1_seed42_r3`。
- 只读证据清单：`q010_s1_0_evidence_identity_c9ee.json`，SHA `0d158b266dde7daeefaf81bc5b270a4294a94f473203015ea8fd9d6df6380408`。
- SDS：`500ad32aaa946c927459427f3a65565b15d4b77fca4be214817121d15c742976`；唯一6400：`30ed23ba16f7e490afca001fbbecadb4a11ff4b11733a5ae7ac7d0c841f9a8b2`。

## 实际干跑证据

独立核对root的真实r3命令、验证脚本、stdout/stderr内容SHA及关键字段，命令没有`--execute`，exit0。实际旧校准原producer447、sV=`2.136540383972894`及Ref与正式清单一致；gate_A和train_calibration输入Ref一致。实际20源病例扩展100槽×5005，录像1槽×5005；5热身、50测量、48步/20次Adam及两次额外N×48恢复负载未改。

输出目录`resource_q010_r1`不存在，活ledger字节未变，SHA仍为 `c526987c278867f390ae7c02dd50ccd5de898f8df734ed0d6a157f7441e4194e`。四个probe均open、spent=reserved=0，每档59个work item，额度如下：

| N | 已预留控制样本 |
| --- | ---: |
| 2048 | 6,108,833 |
| 1536 | 4,708,001 |
| 1024 | 3,307,169 |
| 512 | 1,906,337 |
| 合计 | 16,030,340 |

stage envelope保持124,142,190，原预留计划SHA `fa439aac08c47fda060413cbae2942ce92ef39194d49bf850dd68a9f5ea7b418`，正式学习操作为零。根干跑证明`q010_real_resource_dry_proof.json` SHA `df567e11c48d8408758e3a3361ce91c9f59c5521c1a8ca92a0ebf3314b5de5c1`；独立复核证明`q010_resource_dry_independent_verification_c9ee.json` SHA `8c503eeca04a61a29339b169c645002d476ef8ed70d13371b7c8dea0ea8cba9e`。完整stdout留本机，不公开其中大ledger对象；final成绩未读取。

## 获准执行范围

root完成已审范围提交、push及远端完整提交核验后，可用干跑的同一命令仅追加`--execute`：segment=`s1-q010-resource-20261120-r1`，output=`resource_q010_r1`，`CUDA_VISIBLE_DEVICES=0`、`--device cuda:0`、`--physical-gpu-index 0`、`--monitor-interval 0.02`。开始前仍由CLI验证当前完整code及同一清单。

按2048→1536→1024→512顺序，每档5热身+50测量，完整N×48/20次Adam；保存恢复后的S1.2完整负载、卸载后独立100×5005评估和1×5005录像、恢复后N×48/20更新均按Q009执行。20×5仅为资源负载，不是100个独立行为病例。模型是一次性探测模型，不启动正式learner。

全卡保守峰值≤7168 MiB、host最低MemAvailable≥4 GiB、完整工作项/监控/恢复指纹/RNG/视频验证等条件不改。只有既定OOM/容量超限可降档；工程、监控、非有限、接触overflow或一致性错误停止，保留证据与费用。父子只结算一次，未知按既定上界付费；不超探测义务、不自动扩大或重跑，不更改阈值、预算或学习配方。

本报告不预设选中N或宣称真实资源稳定。实际资源结果须另行独审；S1.1/S1.2学习仍须独立批准，S2/S3未授权。原兼容报告和清单保持原样，不回写本次或后续结果。新增本报告/批准JSON及验证材料的日志、提交由root汇总至`logs/codex_changes.md`。
