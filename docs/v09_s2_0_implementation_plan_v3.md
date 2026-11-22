# A2 v0.9 S2.0 实施收敛记录（v3）

**状态：** 纯 CPU；未训练、未仿真、未 Git 写
**Supersedes：** `docs/v09_s2_0_implementation_plan_v2.md`（SHA256 `f3168adf74d269983527896375da85463961372c32108c3f92121803630dc064`）
**继承：** v2 的缺口清单/文件归属/§2 对照继续有效；本文件只记录 Q026 收尾与最终数字
**裁定：** DSH-Q015～Q026
**实现子 agent：** A2-S2-IMPL

---

## 1. 冻结件（字节未动）

| 文件 | SHA256 | 状态 |
|---|---|---|
| `logs/a2_v09/s2_0/cases/S2_0.json` | `09e2fc4636ba6f0cd1d310abcf615eec751c7f8b020bdcd281e858deda6c7079` | 旧 suite `S2_0`，原样保留 |
| `logs/a2_v09/s2_0/cases/h_series_case0.json` | `1b0ad3529869a9037f93d3171ac7eeea0965c0401661df34e7c5ffc9afac1ee6` | 原样保留 |
| `logs/a2_v09/s2_0/cases/S2_DIAG.json` | `b990052c16a7f3cf1c9f09813805556e1f892120eeb13a3cb7b30cf870153b7a` | 75 例，manifest `fab7372e…` |
| `logs/a2_v09/s2_0/cases/S2_DIAG_run_plan.json` | `bc9ac8ba0748c9cb9d497a6618e0341fa167f11814990568fc104501d7e8f8c1` | 150 条，plan `3f09c009…` |
| `logs/a2_v09/s2_0/cases/history/history_index.json` | `42ce377dc501e97da637fec44ced862775becfed3b251bbf58e35d29d48f7f91` | 旧件副本/事故记录；未改 |
| `logs/a2_v09/s2_0/cases/history/history_index_v2.json` | `f3892f2d2e271b5e701d931a8872b8be35320c8debdb023faa54dee675c9f875` | 追加记录；未改 |
| `logs/a2_v09/s2_0/cases/supersede_register.jsonl` | `09abd45a7bd75c9fdf6a94fbec04e5f18cf0395deeaf604f17ef417a1653a2cb` | 新增只追加登记 |

### Q026 替代登记

- 旧 `S2_0.json`、旧 `freeze_receipt.json`、旧 `history_index.json` 字节与 suite 原样未动。
- `supersede_register.jsonl` 每行一个 JSON，字段：`old_suite`、`old_manifest_sha256`、`old_file_sha256`、`new_suite`、`new_manifest_sha256`、`new_file_sha256`、`new_receipt_path`、`new_receipt_sha256`、`reason`、`ruling`、`written_at`。
- 当前记录：旧 `S2_0`/`5efda5ab…`/`09e2fc46…` → 新 `S2_DIAG`/`fab7372e…`/`b990052c…`，回执 `freeze_receipt_S2_DIAG.json`，reason `Q022 更名、旧清单零物理`，ruling `DSH-Q026`。
- 幂等：同一 `(old_manifest_sha256,new_manifest_sha256)` 重复调用只返回 `already_registered=True`，不追加；`lookup_supersede(root, old_sha)` 支持旧 manifest 或旧 file SHA 反查。
- 登记不是物理准入条件；`429,722` 仅历史，`429,822` 为 S2.0 上界，均非实际消费。

---

## 2. 帧计数语义（Q026-1 / B-V03-1）

- 先判该帧是否被本例 Actor 真正消费，再更新计数；终止帧被消费时计入，其后回调不改变计数/首次暴露。
- 未激活或触发帧的早返回路径现在也处理 `terminal`：置 `finished=True`，后续 `consumed=True` 直接忽略。
- 截至终止尚未消费替换帧：`replacement_frames_consumed=0`、`saturation_denominator=0`、`saturation=None`、`exposed=False`。
- `reset()` 清空帧数/有效格/饱和/暴露/采样引用，跨 reset 不串账；新帧只归新病例。
- 回归用例：
  - t=30 触发、31 消费并终止、32 再调用 → `frames_consumed=2`，替换 0，饱和 NA；
  - 触发与终止同帧 → 只计该帧；
  - 31 消费、32 以 `consumed=False, terminal=True` 上报 → 30、31 计入、32/33 不计；
  - reset 后计数归零。

---

## 3. Q023 真实支与证据

- 新增 `S2RealBranchEvidence`：真实支用自己的前足真实顶面支撑事件 `t` 与严格大于 `t` 的首个正常刷新 `s` 界定窗口；不借用替换支时刻，不写“已替换”。
- 真实支 `exact_clip_fraction=None`，理由 `real_map_stores_only_clipped_height`；`replacement_applied=False`、`first_replacement_frame=None`；未到事件/未等到刷新记 `not_exposed_before_terminal`。
- 逐步校验：每个消费帧比较 policy 与同帧未替换输入的 valid/age 逐元素一致；无效格严格 0/0/1；只有有效格高度可变；违反即 `ValueError`。
- 非刷新帧同步当前帧 valid/age（不复制旧 age），避免 age 变化被误判/漏判。
- 记录 `frame_checks`、`frame_check_failures`、`sample_references`（每消费帧引用 `{refresh_index,sample_sim_time_s,produced_at_actor_step}`，存关联键不存图）、`sample_references_sha256`、首/末引用。

### 接入 `s2_metrics`

- `normalize_map_evidence` / `attach_map_evidence` / `summarize_map_evidence`。
- 字段：`valid_fraction`、`saturation`、`exact_clip_fraction` 及各自分子/分母；`event_step`、`native_sample_id`、`first_replacement_actor_step`、`s_minus_t`、`first_replacement_frame`、`replacement_applied`、`replacement_frames_consumed`、`sample_reference_count`、`sample_references_sha256`。
- 组汇总分子分母相加；零分母 `None`；未暴露、零有效格单列。
- `summarize_case(..., map_evidence=...)` 接入病例记录；`summarize_group` 输出 `map_evidence` 汇总。

### Q024 最小证据校验

- `validate_pairing` 要求：`case_sha256/stream_sha256/weight_sha256/code_sha256/physics_sha256/reward_sha256/termination_sha256/state_sha256/command/action/sample_frame`；缺项直接报错。
- `config_diff` 仅允许 `diagnostic_map_branch`、`record_id`；其他键报错。
- `validate_branch_pair` 要求真实/替换支共享 case/stream/权重/代码/物理/奖励/终止身份；返回 `shared_identity_sha256` 与允许 diff。
- `paired_proof` 每帧/首替换帧验证 valid/age 不变、无效格 0/0/1、仅有效格高度变化，并保存原图/替换图 SHA。
- 三元组保持 `{refresh_index, sample_sim_time_s, produced_at_actor_step}`，另记消费步；关联键用于同状态同采样对照。

---

## 4. 预算最终数字（Q021，草案但可复核）

| 项目 | 控制样本 |
|---|---:|
| 资源探测 1024 | 3,709,184 |
| 资源探测 512（回退） | 2,332,928 |
| S2.0 两分支（新 75 例） | 420,390 |
| 六例 H 缺字段补采（≤6 条） | 9,432 |
| 六例候选复测（3 候选×9,432） | 28,296 |
| §6.1.1 L40 里程碑 A/B 两套一次性缓存 | 438,440 |
| ≤40 次门禁（最坏 S2.3） | 173,940,000 |
| 三套终验 + 冻结对照 | 83,009,700 |
| **去重合计** | **263,888,370** |
| 4 亿剩余 | 136,111,630 |

- S2.0 子额度：420,390 + 9,432 = 429,822 ≤ 5,000 万；旧 429,722 作废。
- L40 缓存：A/B 各 3 组×20 例，按首次缓存上界各 219,220，合计 438,440；复用不重复计费，不预设跨 A/B 命中。
- 计费口径：环境控制样本；物理子步 4×；每批 `批环境数×批次最长硬 timeout`（含完成 slot 等待空步/padding）；reset 后帧不计；S1 已计费样本不计入 S2。
- 探测：整卡峰值 ≤7168 MiB、主机可用 ≥4 GiB、5 热身 + 50 测量 + 一次恢复检查（48 步 rollout + 5×4=20 Adam）、独立副本、100 环境评估。
- `q021_reservation_written=false`；正式预留前仍需对最终代码/全部义务重算并独审。

---

## 5. 测试

```bash
/home/xxs/anaconda3/envs/a2/bin/python -m pytest -q \
  tests/test_v09_s2_0_cases.py tests/test_v09_s2_0_metrics.py \
  tests/test_v09_s2_0_map_ablation.py tests/test_v09_s2_0_budget.py
```

**79 passed。**

| 文件 | 数量 | 新增重点 |
|---|---:|---|
| `test_v09_s2_0_cases.py` | 17 | Q022/supersedes、旧件保留、150 计划、登记幂等/反查 |
| `test_v09_s2_0_metrics.py` | 28 | Q025 质量窗、map evidence 接入/汇总/NA |
| `test_v09_s2_0_map_ablation.py` | 24 | B-V03-1 三种终止边界、reset、真实支窗口、逐帧 valid/age、采样引用、证据缺项/禁止 diff、分支身份 |
| `test_v09_s2_0_budget.py` | 10 | 新清单、L40 缓存、billing、Q025 敏感性 |

既有 v09 非 `resource_cli` 套件此前 273 passed；`test_v09_resource_cli.py` 21 项 legacy 失败与本任务新增文件无关，未宣称通过。

---

## 6. 仍缺字段 / 未决依赖

- H 六例原 JSON 无逐帧足位/接触：首足/四足入台归因需补采（≤6 条，已计入 9,432 预留），不得在无数据时声称“平台摔倒”。
- 原 JSON 仅有 qpos/qvel/RNG 的 SHA，无原始向量；全速率帧审计为子采样。
- 无待裁定语义项（Q015～Q026 均已落实）；剩余为实现缺口（v2 §2）。
- 未执行：训练、仿真、env.step、GPU、梯度、Git 写。
