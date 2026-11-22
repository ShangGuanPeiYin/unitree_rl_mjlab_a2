# S2 smoke 观测/动作一致性对照报告（T9）

**状态：** 代码修复与 CPU 对照工具已完成；真实 64 步双环境对照需 GPU 窗口
**绑定：** S2_DIAG case 0 (`flat_walk`, real, speed 0.3)，`s1_least.pt` `efa9caa8…`

## 1. 已定位的方向性异常根因

- 现象：`route_x` 从 −1.0859 到 −1.1662（后退 8 cm），`velocity_route` −0.29…0.07。
- 根因（代码事实）：`S2PhysicsEnv.observe()` 之前返回**硬编码零观测**（proprio 只有部分字段、history 复制、heightmap 全零、route_error 全零），没有读取真实 backend 的 `V07Env._observations()`。Actor 收到的不是实际 A2 状态，动作自然无方向意义。
- 修复：`S2PhysicsEnv` 现在每控制步从 backend 取真实 `_observations()`，缓存一次并供下一步 Actor 使用；plane 分支用 ablation policy map 替换 heightmap。S2 与 S1 现在共用同一 `proprio(45)/history(5×45)/heightmap(3×40×30)/route_error(3)` 来源。
- 预期：修复后同一 Actor 在 flat real 上应产生与 S1 基线同方向的运动；仍需 GPU 双环境实测确认。

## 2. 对照内容与工具

- `s2_parity.compare_observation`：proprio/history/heightmap/route_error 的 shape 与逐元素 max_abs。
- `s2_parity.compare_trace_tables`：逐帧 a_request/a_applied/pd_target/c_exec 及观测字段差异。
- `scripts/compare_a2_v09_s2_observation_parity.py --left s1_trace.jsonl --right s2_trace.jsonl`。
- 摩擦合同：`CONTACT_FRICTION=(0.6,0.005,0.0001)` 与 v07 相同；S2AtlasCfg 使用同一组值（用户要求不动物理）。

## 3. 两侧 trace 路径（修复后需重跑生成）

- S1 基线：建议 `logs/a2_v09/s2_0/parity/s1_flat_case0/trace.jsonl`
- S2：`logs/a2_v09/s2_0/evidence/<attempt_id>/trace.jsonl`
- 对照输出：`logs/a2_v09/s2_0/parity/parity_report.json`

## 4. summary 口径

- `platform_support_feet_mean`：四足在顶平台的支撑足数（平地应≈0，不代表无接触）。
- `contact_feet_mean`：四足 raw contact 数。
- `foot_on_platform_mean`：同上平台口径。
- `contact_pairs_mean`：每帧真实 foot↔terrain 接触对数。
- 新增 `route_x_start/end/delta`、`velocity_route_mean`。

## 5. 仍待 GPU

1. 同一物理状态下 S1 vs S2 观测逐元素对照（尤其 heightmap 通道/缩放、route_error、history）。
2. 同一 Actor 在 v07/v09 基线与 S2 各 64 步的 route_x/velocity/pitch/contact 并排。
3. 若基线也后退，需按 S1 证据说明权重行为；若仅 S2 后退，继续查地图/动作缩放/命令映射/出生高度/关节顺序。
4. 本次 S2 smoke 64 步已计费，保留不动。

未执行：GPU 双环境对照、训练、仿真、git 写。
