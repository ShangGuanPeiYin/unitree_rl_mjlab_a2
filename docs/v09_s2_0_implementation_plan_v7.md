# A2 v0.9 Q028/Q029 证据字段落地（v7）

**状态：** 纯 CPU；未训练、未仿真、未写正式 S2 台账、未 Git 写
**Supersedes：** `docs/v09_s2_0_implementation_plan_v6.md`（SHA256 `a6778d16872eb77051f17e67f51062f437accfd6b0a95665d13122885e90f919`）
**裁定：** DSH-Q015～Q029

---

## 1. Q028 精确字段（`g_level_window_report`）

主窗口严格等于 §6.5 巡航掩码（`cruise_window_mask`）：
- 完整登阶：四足首次全入台 + 机身净前进 1 m 的首个有效巡航样本起；
- 平台起步：出生点净前进 1 m 起；
- 至本次巡航结束；末个正巡航步计入，DECEL/HOLD 排除；reset/padding 不计。

每例分列（禁止共用无定义 `duty`）：
- `N`：窗口内有效控制步数；
- `sum_g`：Σg；
- `average_intensity`：Σg/N；
- `enabled_frames`：帧数(g>0)；
- `enabled_fraction`：enabled_frames/N；
- `longest_disabled_seconds`：窗口内连续 g==0 最长帧数×0.02 s；
- `longest_below_full_seconds`：窗口内连续 g<1（含 0 与爬升）最长帧数×0.02 s；
- `N=0` → 以上统计 `NA`；非空窗未出现记 0；
- `window_valid`：N≥50（1 s）；不足/提前终止保留原因与前缀，不算质量通过。

归零原因（`GeometryFailureTracker`）：`feet_not_platform`、`body_before_margin`、`both`、`other`；只在“曾满足→失效”转移计事件，初次满足时 g=0/正常 ramp 不误记，重复零帧不重复计。

首米诊断用同一函数并置 `first_meter_window=True`，与正式巡航窗单列。

## 2. Q029 四事件

`events` 分列四足首次全入台、首次完整几何、首次 g>0、首次 g==1 的控制步与原生仿真时刻；`delays` 给相对四足入台的秒数，参考事件自身 delay=0；缺失为 `NA` 并给 `na_reasons`。

Q027 离散序列：t0 g=0、t0+1 才 >0、t0+25 才 =1；四足入台与完整几何可同帧但不合并。`positive_intervals`/`full_intervals` 保留中断重入起止；`first_full_interval_start` 为首次满值所在正区间起点。

## 3. 强制测试向量

`t0..t0+25` 共 26 条、g=[0,1/25,…,1]：

```text
N=26; Σg=13; average=0.5; enabled=25/26
longest g==0 = 0.02 s
longest g<1 = 0.50 s
delays: four=0, full_geometry=0, g>0=0.02, g==1=0.50
```

已实现为 `test_q028_q029_forced_vector_t0_to_t25` 并通过。另覆盖空窗 NA、巡航掩码排除 DECEL、几何原因转移、中断区间、平台出生起点。

## 4. 测试与状态

- `pytest -q tests/test_v09_s2_0_*.py` → **161 passed**。
- T4 reserve dry-run 复跑：plan `773b53c6…`，非学习 263,888,370，学习 3,200,000,000，stage 3,463,888,370，S2.0 429,822；`dry_run=true`，无 S2 台账文件。
- `case_record` 已带 `config_sha256`/`pairing_sha256`/`shared_identity_sha256`/`g_level_report`。
- Q028/Q029 不再阻塞；正式 S2 预留仍待主对话 `--write`。

未执行：训练、仿真、env.step、GPU、梯度、Git 写。
