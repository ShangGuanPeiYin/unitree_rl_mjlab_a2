# A2 v0.9 S2.0/S2.1 T3 实现记录（v5）

**状态：** 纯 CPU；未训练、未仿真、未 `env.step`、未 Git 写
**Supersedes：** `docs/v09_s2_0_implementation_plan_v4.md`（SHA256 `f56956c3d7227b365eafb7ea5961a58b224fb5aeb48b542d6caab49188fc31e5`）
**裁定：** DSH-Q015～Q027
**实现子 agent：** A2-S2-IMPL

---

## 1. V4.1 修正

| 项 | 落实 |
|---|---|
| B-V41-1 | `write_history_projection` 每次写入前由 register **现场重算**投影并逐字节/规范化比对；手工改条目并重算 SHA 必拒绝 |
| register 版本化 | 新行带 `schema_version=2`、`supersedes_line`、`legacy_pre_dual_sha=false`；历史行字节不动；投影确定性取最新有效行并标注 legacy |
| 第三原因码 | `zero_valid_denominator` 与 `event_not_reached` / `no_valid_refresh` 并列；零有效格保留 NA，组汇总单列 `zero_valid_denominator_cases` / reason counts；不冲突 Q023 |

当前 register `426b3ffc…`（历史行 + Q026 双 SHA 升级行）；投影 `ab90ae15…`，只读派生。

---

## 2. Q027 g_level 正式口径

- **0.5 s 是完整 0→1 ramp**，无额外保持；`G_LEVEL_RAMP_STEPS=25` 集中一处。
- `t0` 为本次连续满足“四足 XY 同真实顶平台 + 机身严格 > 前缘 0.4 m”的首个后物理采样；`g(t0)=0`，`g(t0+k)=min(k/25,1)`；`t0+1=0.04`、`t0+24=0.96`、`t0+25=1`。
- 不预记首帧 0.02 s；条件失效同帧归零、解除 t0；下次成立重新 t0（满值后亦然）。
- 平地固定 1；楼梯 0；平台起步首个有效采样为 t0；临时腾空不清零；reset 清计时。
- 报告四时刻：四足首次入台、首次完整几何、首次 g>0、首次 g=1；延迟按实际步数算；未达事件 NA；中断/重入、占空比、最长关闭单列。
- 快照带 `version='q027_ramp_total_0p5s'`；旧口径快照 `pre_q027_hold_plus_ramp` 载入直接报错，禁止静默混用。
- 奖励两项门控同帧用同一 `g`；病例/超时/预算不变。

测试覆盖：t0/t0+1/t0+24/t0+25、满值保持、途中失效重入、满值后失效重入、平地、平台起步、reset、报告与快照拒绝。

---

## 3. T3 文件

| 文件 | SHA256 | 内容 |
|---|---|---|
| `s2_rewards.py` | `5a1c6d043bc1d0df0151399b32e6dc0bd91f386fc89d144f666068df51da772e` | §5.1/§5.3 全 20 项、两列权重、g_level 门控、停车关闭、逐足抬脚、I_cruise、dt 一次 |
| `s2_loss.py` | `305953390a0727f02e04ececc624a05370955654c71797141f65148d9700ee6c` | §5.4 精确 L_anchor、β=0.1、clean mask、微批分子累计/统一分母 |
| `s2_runner.py` | `65480cc0fa98724269f8f4bdac8863fc23461ae7b60592cc71f2c8a19416385b` | PPO 参数、KL 调 LR、std 投影、clipped loss、GAE、加权累积、dry-run 计划 |
| `s2_resources.py` | `f16999e0e2ac16f13ee4d23e2c81429927a279f65994b704464592ecaa8c38d2` | 1024→512 探测、§2.3 全项、峰值/内存/失败证据 |
| `s2_checkpoint.py` | `8017db748be143d99db6ffba9601189621116cf4d58b46fd2927c02ee9832cb5` | schema 11 S2 元数据合同 |
| `s2_ledger.py` | `c7273b681fb90310e1456eb2a70106e318ad9cf3e52620cd9d1f48821eb3720c` | S2 独立命名空间、追加哈希链 |
| `s2_acceptance.py` | `6280a089b3face69af329751ebe9a54d7af7d4595b5732ac52b8b6fc02166194` | §6.1.1 门禁矩阵、§6.5 质量与 27 组 |
| `scripts/train_a2_v09_s2.py` | `b91de8b0fcb8d5593757e718f0c15610ccdd611124dd81d7ea2f14f14b96eb35` | S2.1 训练 dry-run；非 dry 拒绝 |

`g_level` 所在 `s2_state.py` 当前 `bf436ce4…`；`s2_cases.py` `647ffb5a…`（B-V41/版本化）。

---

## 4. 测试与 dry-run

```bash
/home/xxs/anaconda3/envs/a2/bin/python -m pytest -q tests/test_v09_s2_0_*.py
```

**144 passed。** 覆盖前四轮 113 项 + Q027 序列（14 项 state）+ reward 20 项/loss/runner/resources/checkpoint/ledger/acceptance/train-CLI。

S2.0 dry-run 仍通过：75 case / 150 分支 / 75 对 / H 6；`physics_started=false`。
S2.1 train dry-run（`--num-envs 1024 --device cpu --dry-run`）：
```text
samples_per_rollout=49152
complete_rollouts={S2_1:16276,S2_2:24414,S2_3:24414}
config: rollout=48 epochs=5 minibatches=4 micro=512 gamma=.99 lambda=.95 clip=.2 lr=3e-5
training_started=false; physics_started=false
```

---

## 5. 仍无法 CPU 验证/需 GPU

1. 实际 mjwarp/mjlab 物理、接触力、延迟/噪声与真实轨迹。
2. 两分支干预前逐帧一致、首米/后段指标在真实轨迹上的运行。
3. `g_level` 真实地形投影与“临时腾空不清零”的物理验证。
4. H 六例首足/四足入台归因仍需逐帧足位/接触补采（≤6 条，已计入 9,432）。
5. PPO/std/Adam/KL 的真实训练行为、资源峰值与恢复检查。
6. Q021 正式原子预留尚未写入；预算草案仍 263,888,370。

未执行：训练、仿真、env.step、GPU、梯度、Git 写。
