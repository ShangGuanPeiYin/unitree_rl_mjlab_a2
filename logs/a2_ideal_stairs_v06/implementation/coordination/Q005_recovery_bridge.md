# Q005 跨源码恢复桥清单

- Q005 裁决 SHA256：`0159b03296ae078c96d6fd32851e736ce97a5119a89475b6bc4d85004464cb2d`（45 行）。
- 旧 checkpoint（trial02 update100）：`logs/a2_ideal_stairs_v06/train_seed42_trial02/checkpoints/boundary_000100.pt`
- 旧 checkpoint 完整 SHA256：`8ca0760e8c7a67da2c2143927e6ba5baf2459deba1967b0756a2aa601e58e805`（主 Codex 复算；未改写）。
- 旧 checkpoint 内 `source_closure_sha256`：`3ff5b4b5711a2774701055244031fe389e0dc495b8280379014126a3fa049f4e`
  - 对象/算法：`recovery.source_block()` 对 `spec.source_hashes()` 返回的“路径→SHA256”映射做本仓库 canonical JSON 的 SHA256。
  - 与 FREEZE 报告中的 `27c42ece…` **不是同一对象**：后者是冻结 manifest/binding 层的摘要，不参与 checkpoint 的 `source_closure_sha256` 计算；两者不应混称或互相替换。
- 新修复源码（runner/recovery 等）闭包：以 `spec.source_hashes()` 在修复后源码上重新计算，由主对话重建 manifest 时冻结；本桥只绑定下述模块 SHA：
  - runner `e3cc27ad1a72c53e4f53c611768138db5f8fe50badd78cb42d12352c0f0d386c`
  - recovery `4bd2b6761fb76a2f413f9de53c5bb0c4c82b71a9cea7bbdc03b48ec8ca7d2606`
  - tests/test_v06_recovery.py `dd70b3c03e69d21e5ac6f55587bd7474d0769ed3465ec474677e55bc343c7064`
- 实际 diff 范围：仅 Q005 授权的 F0/flat retention 跨来源一致性容差（`abs<=1e-6, rtol=0`）、端点有限性检查与证据字段、受限恢复桥断言、相应测试；未改奖励/PPO/课程/速度/随机化/终止/门禁阈值/资源限额。
- 恢复语义：`resume_contract(..., allow_bridge=True)` 只接受上述 checkpoint SHA 与旧闭包 SHA；`allow_bridge=False` 时该 checkpoint 被显式拒绝；其他来源按原通用源码校验拒绝。恢复读取 checkpoint 的模型/Adam（含 lr=`8.888888888888888e-05`）/自适应状态/normalizer/RNG/课程累计与窗口/门禁 streak/台账，`resume_mode=new_episode`；先按同一 case 表身份补 update100 validation+diagnostic 并做一次课程决策（结果去重），再进入 101；不重训/不重复计费，不覆盖 `boundary_000100.pt`。
- 待独立验收填入：Q005 修复源码冻结闭包 SHA、独立验收报告 SHA、D 第 6 版结论。
