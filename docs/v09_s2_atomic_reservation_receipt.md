# S2 阶段原子预留回执（DSH-Q021）

- 时间：2026-11-22
- 执行主对话：A2继续训练（`session-8862d921`）
- SDS：v1.2，SHA256 `09fa2d4cdd858914c46a51fd52670c20cb20f2b6f3e76e6d266054574f64ccbe`
- 台账命名空间：`a2-v09-s2-sequential-r2`；run_id `a2-v09-s2-q021-r1`
- 台账文件：`logs/a2_v09/s2/ledger.jsonl`（4363 B，不随仓库上传）
- 台账文件 SHA256：`2f520b80c9b5f5985acd4eb570480b8e1888534b209aa6268cbb1ef7d34e87bb`
- stage_reserve 事件：sequence 1，event SHA256 `240871fca58af458fb1e43f0fdac0206d2749359359920881f7571e0907f9deb`
- plan SHA256：`773b53c68e065e228841a907ff7df14b7786c7c86809522158a4d46f203174d3`
- 独立复核：`docs/v09_s2_0_reservation_independent_review_v07a.md`（结论：逐项重算一致，可执行 `--write`）

## 预留额度

| obligation | 类别 | 控制步上限 |
| --- | --- | ---: |
| resource_probe_1024 | engineering | 3,709,184 |
| resource_probe_512 | engineering | 2,332,928 |
| s2_0_matrix_two_branches | engineering | 420,390 |
| h_series_baseline_supplement | engineering | 9,432 |
| h_series_candidate_retest_3_models | evaluation | 28,296 |
| l40_milestone_cache_gate_A | evaluation | 219,220 |
| l40_milestone_cache_gate_B | evaluation | 219,220 |
| gate_admissions_up_to_40 | evaluation | 173,940,000 |
| final_sets_3_with_frozen_comparisons | evaluation | 83,009,700 |
| waiting_empty_steps_no_extra_charge | engineering | 0（已含批上界） |
| **非学习合计** | | **263,888,370（≤4 亿）** |

学习额度预留：`S2_1` 800,000,000、`S2_2` 1,200,000,000、`S2_3` 1,200,000,000，合计 3,200,000,000（≤3.6e9）。
阶段合计 3,463,888,370（≤4e9）；S2.0 子额度 429,822（≤5,000 万）；物理子步 13,855,553,480。

## 计费口径

- 预留不是消费：当前 `reserved`/`spent` 全为 0；每项义务按 `obligation_reserve` → `obligation_commit` 结算，先预留后推进。
- 每个物理推进步按 1 控制步计费，物理子步 ×4 另记不重复收费；等待空步已含在每批 `num_envs × 最长硬 timeout` 上界内。
- 不得复用 S1 已计费样本（`s1_samples_reused=false`、`double_counting=false`）。
- 该预留为一次性原子预留；重复写入拒绝（`exit 2`）。
