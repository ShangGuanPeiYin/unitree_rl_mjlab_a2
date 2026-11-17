# Q007：update400 flat retention 因“保留集地形取当前 frontier”停机，请裁决修复与从 boundary_000400 恢复

- 时间：2026-11-17T04:20+08:00；提出：DSH 主对话。
- 合同：SDS `1ed4bf5f…` + Q001–Q006（含 Q006 裁决 `798a7cf4…`）；Q006 段闭包 `e1a0aa09…8c17`；Q006 段活动目录 `logs/a2_ideal_stairs_v06/train_seed42_trial02_resume_000100_q006/`。

## 事实（可复现）

1. Q006 段从 101 正常训练到 400（commit_000101..400，300 次新更新；F0 两门禁均 passed，update 200 晋级 H；H 门禁 300/400 均 passed/failed_cases=0）。
2. update 400 边界：validation + diagnostic 已完成；**flat retention** 评估抛
   `EvaluationError('F0 case length disagrees with the observed steps')`，run 停机（`exception.json` update=400）。进程已退出。
3. 现场保留：`checkpoints/boundary_000400.pt`（SHA `2a81110d588e5822eded1029f6a4180a6f78a6fa58c0231c65e360adaf4f3097`；learning.update=400、Adam 8000、控制样本 9,830,400、子步 39,321,600、lr=1.1390625e-4；stop=None、lifecycle=running；pending tasks = validation/diagnostic/**retention(flat, tile=1, phase H)**，validation/diagnostic 结果已提交）；`exception_000400.*`、gates、commits 全部保留。
4. 根因（CPU 复现）：`_due_tasks` 对 retention 任务使用 `_current_tile()`（H 阶段 = H00，tile=1）构造 case 表；而 `target_tile()` 未对 retention 类型特判：
   - `make_cases('H', None, 'flat_retention', 42, tile_id=1)` → `target_tile_id=1`、`target_lesson='H00'`、`measurement_stage='F0'`、case `tile_id=1`、`phase='H'`、步数上限 500；
   - `make_cases(..., tile_id=0)` 反而抛 `EvaluationError: H gates cannot measure the flat tile`；
   - 即：**flat 保持集被放在 H00 楼梯上、按 F0 500 步协议评估**，与“平地路线保持”定义冲突；批内出现长度/终止不一致 → 本次停机。
5. 这会同样影响 stair_retention（应固定 H00）与 H/D/L/V/R 各阶段的 flat retention（应固定 flat tile 0，F0 500 步协议）。

## 裁决请求（建议）

- **A（建议）**：批准修复并**从 `boundary_000400.pt` 回退恢复同一 trial02**（唯一新段 `logs/a2_ideal_stairs_v06/train_seed42_trial02_resume_000400_q007`）：
  1) `target_tile` 特判 retention：`flat_retention → spec.FLAT_TILE_ID(0)`、`stair_retention → tile_id_for('H00')(1)`，与当前 phase 无关；`_due_tasks` 记录 retention 的实际目标 tile（0/1），replay 用同一身份，不得拼接；
  2) 步数：flat retention 用 F0 500 步/净位移 1.5/2.5/4.0 m/横向 ≤0.20/航向 ≤10° 判据；stair retention 用 H00 楼梯超时公式与成功/失败判据；
  3) 恢复语义：`rollback_from_update=400`、`resume_from_update=400`、`lost_updates=0`（无 PPO 丢失；仅 pending 保留集评估需重放一次）、`replay_billed_updates=false`、`replay_pending_evaluations=true`；先补完 400 的 flat retention（+ 已完成的 validation/diagnostic 按 Q004 去重、不重做、不双增 streak），再做一次课程决策，然后 401；
  4) 成本：trial02 活动分支仍 30,000（已消费 400，剩 29,600）；正式累计上限仍 **30,100**（trial01 50 + trial02 原段 100 + Q005 丢失段 50 + 活动段最多 30,000）；本段无新增丢失；
  5) 受限 Q007 桥绑定 `2a81110d…`、新闭包、Q006+Q007、diff、独立验收 SHA；旧 Q006 段与 checkpoint 原字节保留；不新建 fresh、不换 seed。
- **B**：其他处置（如暂停 v0.6、先做更全面审计再恢复）。

## 验收要求（若 A）

- 修复后：完整 CPU（含 retention 专项：各 phase 下 flat retention 固定 flat/500 步、stair retention 固定 H00、表身份/去重/不拼接、8 env 批量、失败/超时语义）。
- 新闭包上重跑 smoke（8env/H00/2 updated）与 56 更新 probe（保持 50 诊断与 55/56 loaded/ready 验证）。
- 独立 D 验收同 SHA、blocking=0，并解释旧测试为何未覆盖 retention 地形选择。
- 恢复后回送：boundary_000400/loaded/ready SHA、400 保留集结果、401 首个提交、成本台账。

未 commit；R2 seed43 保持停止。
