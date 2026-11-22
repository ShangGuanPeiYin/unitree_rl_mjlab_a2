# S2.2/S2.3 课程、门禁与终验执行（CPU 计划）

## T15 课程

- `--course S2.2|S2.3` 已接入训练 CLI；512 下 48,828 rollouts、24,576
  样本/rollout、checkpoint 204、cap 1.2e9。
- S2.2：主 40 级、post `Uniform[5,10]`、复习 10/20 固定 5 m、平台专项
  8 m/10 m。
- S2.3：`stairs_stop`/`platform_stop`，post `Uniform[3,10]`，instant/ramp
  各半，HOLD 20 s；复习 10/20 不停车。
- `s2_course.StopState` 提供 CPU 停止时序/奖励列：instant 下一 Actor、
  ramp 100 步、HOLD 1000 步、末 500 步 STAND；`I_cruise`/前进/腾空关闭，
  `stable_stand` 仅 `c_exec≈0`；奖励权重不变。
- **未完成**：真实 V07Env 训练后端的 stop 状态机/奖励列注入仍是下一项；
  现有 fake/CPU 状态机测试通过，不能宣称真实 S2.3 训练已可跑。

## T16 课程门禁

冻结件：

- S2.2：`logs/a2_v09/s2/gate_cases/v2/S2_GATE_A_V2.json`（33 组/660 例，
  2,061,980 上界）、B 同目录。
- S2.3：`logs/a2_v09/s2/gate_cases/v3/S2_GATE_A_V3.json`（51 组/1020 例，
  4,348,500 上界）、B 同目录。
- 每套独立 suite 盐；每组 20 例，阈值 18/20。

dry-run：

```bash
python scripts/run_a2_v09_s2_gate.py --course S2.2 --suite gate_A \
  --dry-run --checkpoint <schema11.pt> --output-dir <dir>
python scripts/run_a2_v09_s2_gate.py --course S2.3 --suite gate_A \
  --dry-run --checkpoint <schema11.pt> --output-dir <dir>
```

执行器已能读对应清单、计算预算和组数；真实 execute 仍受缓存/检查点
门禁约束（6400 平地缓存需按各自 course 生成，不能复用 S2.1 flat cache）。

## T17 终验

```bash
python scripts/run_a2_v09_s2_final.py --dry-run \
  --checkpoint <schema11.pt> --output-dir <dir>
```

- 27 组×100 例=2,700；阈值 95/100；
- 平地持续 3、平地走停 6、L40 停车 12、L40 不停车 6；
- 预算 `final_acceptance_cost`：每套 27,669,900；3 套 83,009,700；
- 6400 平地 9×100 按 Q034 同身份在解封时生成；
- `--execute` 当前拒绝，等待正式 final 清单冻结/解封和真实执行器；
- 报告骨架字段 `S2_WALK_STOP_ACCEPTED` 仍为 pending，不得在 dry-run 宣告通过。
