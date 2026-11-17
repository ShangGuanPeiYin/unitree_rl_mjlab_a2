# W0 执行合同冻结（v0.6）

- 冻结时间：2026-11-16T14:41+08:00；冻结人：DSH 主对话。
- 合同：`doc/sds/SDS_A2_IdealHeightMap_Stairs40_v0.6.md`（SHA256 `1ed4bf5fa8f83d4276e278c7d161be3b6d9c0032cbdc1e3540dade9606e6ad60`，970 行，revision 1.0）。
- 本文件只冻结接口、owner、执行顺序与不可变数值清单；**任何数值、公式、预算、门禁以 SDS 为唯一来源**，本文件与 SDS 冲突时以 SDS 为准并立即写 Q 文件。
- 本文件不代表工程或训练通过。

## 1. 合同文件与基线 SHA

| 文件 | SHA256 | 用途 |
|---|---|---|
| `doc/sds/SDS_A2_IdealHeightMap_Stairs40_v0.6.md` | `1ed4bf5f…6ad60` | 批准规格（唯一数值源） |
| `doc/sds/reviews/SDS_v0.6_reference_audit.md` | `02fe8b6c…2f916` | 参考源码审计（上游原值 vs A2 适配） |
| `doc/sds/reviews/SDS_v0.6_independent_review.md` | `4478f97e…4b6b0` | 独立设计审核与主审处置 |
| `doc/sds/PROMPT_A2_Stairs_v0.6_DSH_Execution.md` | `99178618…bb59e` | 执行授权与交接 |

参考源码固定 commit（不得跟随上游 main）：`ashutoshpanpalia/unitree_go2_stairs@85d4145e…`、`saifahmadgit/go2-sim2real-locomotion-rl@0830f3e3…`。参考源码**不在本机**，数值一律以 SDS §3/§4/§5 的批准表为准；审计文档只用于区分“上游原值/A2 适配值”。

## 2. 不可变清单（A/B/C 都不得自行改动）

- 身份：`0.6.0` / `0.6-go2refs-r1` / schema `8` / `v06-route-state-v1` / `Unitree-A2-IdealHeightMap-Stairs-V06` / `v06-go2refs-climb-first-v1` / `logs/a2_ideal_stairs_v06/`。
- 初始化：`/home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt`，SHA `8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722`；只迁移 Actor MLP + 冻结 45 维归一化器；融合旁路零初始化；Critic/Adam/课程/训练计数全新；初始 std 恰为 `0.30`。
- 动力学/观测不变项：12 关节、动作比例 `0.25`、`dt=0.02`、物理步 `0.005`、每控制步 4 子步、Actor 输入 `proprio45 + history[5,45] + heightmap[3,48,32] + route_error[3]`、Critic 宽 `211`、privileged `35`、高程图 `10 Hz`、`route_error=[clip(local_y/0.20,-4,4), sin(e_world), cos(e_world)]`。
- 奖励数值（SDS §3）：17 个连续项权重表、`forward=0.4 元/米`、`+10/-10` 终端事件、`dt` 连续项只乘一次、失败同帧优先、超时奖励 0、不平滑台阶边缘地表、`h_target=max(0.17, h_step+0.02)`、平地净空 `0.075`、机身高度目标 `0.40`；R2 旧付款项系数一律 0（speed 分段/超速罚/倒退罚/侧速罚/正速门控/footfall/swing/stall/pose/angular_momentum/edge/soft_landing/nonfoot）。
- PPO（SDS §4）：lr `3e-4`、范围 `[1e-5,3e-4]`、gamma `0.99`、GAE `0.95`、clip `0.2`、entropy `0.003`、value `1.0`、clipped value loss 开、全模型裁剪 `1.0`、std `[0.05,1.00]` 初始 `0.30`、单 Adam、`betas=(0.9,0.999)`、`eps=1e-8`、`weight_decay=0`、KL `>0.02` 除 1.5 / `(0,0.005)` 乘 1.5、每次 Adam 后投影 std；不得加双优化器/Actor 冻结/KL 提前退出/自动回滚；梯度分组统计不得改梯度。
- 课程与预算（SDS §5）：F0 1000 / H 12000 / D 3000 / L 6000 / V 4000 / R 4000，总计 **30000**；22 个 tile 地形表 `T0..T20` + flat；512 环境分配 flat 102 / frontier 164 / near 123 / easy 123；H `level_tick∈[0,100]`、64 例窗口、EMA `0.97/0.03`、升级 `S≥0.65 & Q≥0.45 & F≤0.35` 连续 5 次、退级 `F≥0.40 或 S≤0.20` 连续 2 次且 `level_tick-=3`、cooldown 5；D 子级 1500/1500、L 子级 1500/1500/3000；V 至少 1200 更新（≥1000 后计 streak）、R 至少 600（≥400 后计 streak）；命令 F0/H/D/L `U[0.3,0.8]`、V ramp 1000 次后 `U[0.2,0.6]`、R 固定 `U[0.2,0.6]`；出生 `x[-1.10,-0.90] y±0.05 yaw±5°`，R 前 400 次线性扩到 `x[-1.20,-0.80] y±0.10 yaw±10°`。
- 门禁（SDS §6）：每 100 更新一次完整门禁；F0/H/D/L 速度 `0.3/0.5/0.8`、V/R `0.2/0.4/0.6`、每速 20 例、每档 ≥18；最终冻结测试每速 100 例、每档 ≥95；F0 每例 500 步、`max|local_y|≤0.20`、`max|yaw|≤10°`、前进 ≥`0.5*c*10 s`；楼梯超时 `n_max=ceil((2(2.2+N*sqrt(h²+d²))/c+10)/0.02)`；成功定义沿用 R2 平台接触（四足 50 步内有安全支撑、单足连续 2 帧、至少 2 足、平台起点后 ≥0.8 m、`|local_y|≤0.20`、0.042 m 内缩、连续 5 帧、无 failure）；`N*0.20` 一律改 `N*当前踏深`。
- 诊断（SDS §6.3）：每 50 更新 15 例诊断；F0/H00 通过后每 200 更新 15 例保持集（每档 5 例、固定 case），连续两次 ≥12/15 失败 → `behavior_regression` 停机；最低可退子级连续 4 个课程窗口 `F_EMA≥0.80` → 停机。
- 账本（SDS §8）：每次更新 24,576 控制样本 / 98,304 物理子步 / 20 Adam；30000 更新 = 737,280,000 / 2,949,120,000 / 600,000；评估/视频/smoke/probe 单列不计入；每 100 保存、每 50 诊断、异常/门禁/阶段转换也保存；单 GPU worker；整卡 ≤7168 MiB、进程 ≤5120 MiB、host 可用 ≥4096 MiB、磁盘 ≥10 GiB；用户停止 = `user_requested_stop`，不得归为监控/门禁失败。
- D1 防回归（SDS §8.3）：case 构造层只加一次 offset——validation `+200300`、diagnostic `+500300`、perturbed `+300300`、final 固定 `400342`；runner 传原始 seed；`statistics_unchanged` 不得写死。

## 3. 文件 owner（一个文件只有一个 owner）

| Owner | 文件 |
|---|---|
| **A（奖励）** | `src/tasks/velocity/v06/spec.py`、`rewards.py`；`scripts/diagnostics/analyze_a2_v06_rewards.py`；`tests/test_v06_spec.py`、`test_v06_rewards.py`、`test_v06_reward_contract.py`、`test_v06_reward_counterexamples.py` |
| **B（环境/课程）** | `src/tasks/velocity/v06/terrain.py`、`map.py`、`contacts.py`、`events.py`、`env_cfg.py`、`env.py`、`curriculum.py`；`tests/test_v06_terrain.py`、`test_v06_env.py`、`test_v06_curriculum.py`、`test_v06_contacts_events.py` |
| **C（训练/评估/入口）** | `src/tasks/velocity/v06/model.py`、`ppo.py`、`runner.py`、`recovery.py`、`evaluation.py`、`speed.py`、`orchestration.py`、`telemetry.py`；`scripts/a2_v06_bootstrap.py`、`train_a2_ideal_stairs_v06.py`、`run_a2_ideal_stairs_v06.py`、`evaluate_a2_ideal_stairs_v06.py`、`play_a2_ideal_stairs_v06.py`、`smoke_a2_ideal_stairs_v06.py`、`test_a2_ideal_stairs_v06.py`、`check_a2_ideal_stairs_v06_migration.py`；`tests/test_v06_model.py`、`test_v06_ppo.py`、`test_v06_runner.py`、`test_v06_recovery.py`、`test_v06_evaluation.py`、`test_v06_schema.py`、`test_v06_entrypoints.py`、`test_v06_migration.py`、`test_v06_orchestration.py` |
| **主对话（我）** | `src/tasks/velocity/v06/__init__.py`、`connection.py`；`logs/a2_ideal_stairs_v06/**`（W0、manifest、SHA 冻结、运行证据、QP 记录）；`logs/codex_changes.md`；跨模块接线与 GPU 运行 |
| **D（独立验收）** | 只写 `logs/a2_ideal_stairs_v06/implementation/V06_IMPLEMENTATION_ACCEPTANCE.md` 及其证据子目录；**不得修改任何生产文件** |

禁止事项：不得修改 `src/tasks/velocity/v05_r2/**`、`v05/**`、`v02/**`（含 `terrain.py`）、`v02_formal/**`、`v03/**`、`v04/**`、R1/R2 的 `logs/**`、`control.json`、历史报告与 SDS/审计/审核文件。v06 必须自闭合，不得靠改旧版本共享行为实现。

## 4. 冻结的 Python 接口（签名与语义）

导入方向必须无环：`spec ← rewards ← env`；`spec ← terrain ← env`；`spec ← curriculum ← runner`；`spec ← evaluation ← runner`；`model/ppo ← runner`。除 `spec.py` 外任何模块不得写死 SDS 数值。

### 4.1 `spec.py`（A）
```python
CODE_VERSION="0.6.0"; RECIPE_VERSION="0.6-go2refs-r1"; CHECKPOINT_SCHEMA=8
ARCHITECTURE="v06-route-state-v1"; TASK_ID="Unitree-A2-IdealHeightMap-Stairs-V06"
PROTOCOL_ID="v06-go2refs-climb-first-v1"; LOG_ROOT="logs/a2_ideal_stairs_v06"
CONTROL_DT=0.02; PHYSICS_DT=0.005; DECIMATION=4; ACTION_SCALE=0.25
N_JOINTS=12; PROPRIO_DIM=45; HISTORY_LEN=5; HEIGHTMAP_SHAPE=(3,48,32); ROUTE_DIM=3
CRITIC_WIDTH=211; PRIVILEGED_DIM=35; ETH_ACTOR_DIM=45
FLAT_TILE_ID=0

@dataclass(frozen=True)
class TerrainSpec:
    tile_id:int; lesson_id:str; N:int; h:float; step_depth:float
    width:float; tread_thickness:float; platform_length:float
    frame_origin_world:tuple[float,float,float]
    valid_bounds_local:tuple[float,float,float,float]

TERRAIN_TABLE: tuple[TerrainSpec,...]      # 22 项：flat + T0..T20
def terrain_spec(tile_id:int) -> TerrainSpec
def tile_id_for(lesson_id:str) -> int

@dataclass(frozen=True)
class PhaseSpec:
    phase:str; budget:int; substages:tuple[str,...]; substage_budgets:tuple[int,...]
    gate_speeds:tuple[float,...]; reward_speeds:tuple[float,float]

PHASES: tuple[PhaseSpec,...]                # F0,H,D,L,V,R
def phase_spec(phase:str) -> PhaseSpec
def phase_budget(phase:str) -> int
def total_budget() -> int                   # 30000
GATE_INTERVAL=100; DIAGNOSTIC_INTERVAL=50; SAVE_INTERVAL=100; RETENTION_INTERVAL=200
GATE_CASES=20; GATE_PASS_MIN=18; FINAL_CASES=100; FINAL_PASS_MIN=95

REWARD_TERMS: tuple[str,...]                # SDS §3.2 顺序的 17 项
REWARD_WEIGHTS: Mapping[str,float]
FORWARD_PROGRESS_PER_M=0.4; SUCCESS_REWARD=10.0; FAILURE_REWARD=-10.0
FLAT_CLEARANCE_TARGET=0.075; BASE_HEIGHT_TARGET=0.40
def clearance_target(step_h:float, *, flat:bool) -> float
PPO_CONFIG: Mapping[str,Any]                # SDS §4 全字段
CURRICULUM: Mapping[str,Any]                # SDS §5 全字段
def reward_config() -> dict
def reward_config_sha256() -> str
def recipe_config(**cli) -> dict            # 只做断言，不改值
def contract(**cli) -> dict
def identity_block() -> dict
def source_hashes(root=ROOT) -> dict
def behavior_hashes(root=ROOT) -> dict
def product_entries(root=ROOT) -> list
def pending_source_entries(root=ROOT) -> list
def sha256(path) -> str
def canonical_json(value) -> str
```

### 4.2 `rewards.py`（A）
```python
@dataclass
class RewardFrame:   # 全部为 torch tensor，批量 B
    executed_command: Tensor                       # (B,)
    local_x: Tensor; local_x_prev: Tensor          # (B,) 所属 tile 系
    body_lin_vel_b: Tensor; body_ang_vel_b: Tensor; gravity_b: Tensor
    base_height_above_surface: Tensor              # (B,)
    q: Tensor; qdot: Tensor; q_target_executed: Tensor
    action: Tensor; previous_action: Tensor
    foot_position_world: Tensor; foot_velocity_world: Tensor
    foot_force_world: Tensor; contact_raw: Tensor; contact_filtered: Tensor
    tile_ids: Tensor                               # (B,) int64
    local_y: Tensor; yaw_world: Tensor             # (B,)
    resets: Tensor                                 # (B,) bool，本步已 reset
    success: Tensor; failure: Tensor               # (B,) bool

@dataclass
class RewardResult:
    raw_terms: dict[str,Tensor]; weighted_terms: dict[str,Tensor]; total: Tensor

class AirTimeState:      # 每足 previous_raw_contact / has_contacted_once / air_time
    def reset(self, ids=None) -> None
    def update(self, contact_raw: Tensor, dt: float) -> Tensor   # Σ first_contact*(air-0.10)
    def state_dict(self) -> dict; def load_state_dict(self, value) -> None

class RewardState:       # previous_action / previous_qdot / previous_local_x + AirTimeState
    def reset(self, ids=None) -> None
    def state_dict(self) -> dict; def load_state_dict(self, value) -> None

class RewardComputer:
    def __init__(self, weights=None, dt=CONTROL_DT) -> None
    def reset(self, ids=None) -> None
    def __call__(self, frame: RewardFrame) -> RewardResult
```
必须同时暴露每个未加权分量函数（供 §9.1 精确用例直接调用）：`tracking_lin_vel_rate`、`tracking_ang_vel_rate`、`lin_vel_z_rate`、`orientation_roll_rate`、`base_height_rate`、`action_rate_rate`、`similar_to_default_rate`、`dof_acc_rate`、`dof_vel_rate`、`ang_vel_xy_rate`、`foot_slip_rate`、`foot_clearance_rate`、`joint_tracking_rate`、`center_rate`、`heading_rate`、`joint_limits_rate`、`feet_air_time_bonus`、`forward_progress_reward`、`event_rewards`（失败优先）。

### 4.3 `terrain.py` / `map.py` / `contacts.py` / `events.py` / `env_cfg.py`（B）
```python
# terrain.py
def build_terrain_specs() -> tuple[TerrainSpec,...]        # 与 spec.TERRAIN_TABLE 同源校验
def build_terrain(seed=42) -> ...                          # 真实 MuJoCo 地形 + 每 tile 几何句柄
def tile_geoms(tile_id:int) -> list[int]                   # geom id 索引
def query_surface_height(points_world: Tensor, tile_ids: Tensor) -> Tensor   # 毫秒级批量
def platform_bounds(tile_id:int) -> tuple[float,float,float,float]           # local x/y 安全区
# map.py
class HeightMap:  # 3×48×32 采样，10 Hz，统一读 TerrainSpec（禁止隐式 v02 atlas）
# contacts.py / events.py
class ContactState  # 每足 滤波接触=raw|prev；foot_force_world_z>5N
def climb_success(...) -> Tensor; def failure_masks(...) -> dict[str,Tensor]
# env_cfg.py
def env_cfg() -> ...  # 场景/地形/接触/传感器，全部读 v06 terrain
# env.py
@dataclass(frozen=True)
class ScheduleAssignment:
    phase:str; substage:str|None; frontier_tile_id:int
    roles:tuple[str,...]; tile_ids:tuple[int,...]

class V06Env(FrozenEnv):   # 与 V05R2Env 相同生命周期
    def __init__(self, num_envs=512, device='cuda:0', seed=42, phase='F0',
                 substage=None, stage_iteration=0, evaluation=False,
                 fixed_vx=None, initial_cases=None, render_mode=None,
                 parking_diagnostic=False) -> None
    def set_schedule(self, schedule:ScheduleAssignment, stage_iteration:int=0) -> None
    def step(self, action) -> tuple                 # info 含逐环境 outcome 与 reward 分项
    def diagnostics(self) -> dict
    def last_reward_terms(self) -> dict
    def route_error(self) -> Tensor
```

### 4.4 `curriculum.py`（B）
```python
@dataclass
class CurriculumState:   # SDS §7.2 全字段，JSON-safe、可序列化
class EpisodeOutcome:    # env_id, step, role, tile_id, success, failure, timeout, reason, tracking_score, command
class TransitionEvent:   # kind: advance/regress/generation/substage_advance/substage_regress/final
class CurriculumManager:
    def __init__(self, num_envs=512, seed=42, state=None) -> None
    def schedule(self) -> ScheduleAssignment
    def observe(self, outcomes: Sequence[EpisodeOutcome]) -> None
    def on_update(self, global_update:int) -> list[TransitionEvent]   # 每次更新最多一个 64 窗口
    def retention_due(self, global_update:int) -> str|None
    def state_dict(self) -> dict
    def load_state_dict(self, value:dict) -> None
```

### 4.5 训练/评估/入口（C）
- `model.py`：`class V06ActorCritic`（`route_state_dict()` 零初始化旁路、`migrate_eth(...)` 断言动作均值 `≤1e-6`）。
- `ppo.py`：`class PPOTrainer`（单 Adam、std 投影、KL 自适应、`grad_norms_by_group()` 记录 actor/critic/fusion/std 裁剪前值）。
- `runner.py`：`class V06Runner`（`run()`、`train_one_update()`、`save()`、`load()`、`final_test()`、`result()`）。
- `recovery.py`：schema8 序列化/校验/预算审计；拒绝 R1/R2/smoke/probe 作为恢复源。
- `evaluation.py`：`make_cases(phase, substage, kind, seed, round=None)`（offset 只加一次）、`summarize()`、`gate_result()`；`speed.py`：速度与保持集辅助。
- `orchestration.py`：`class V06Orchestrator`（`preflight()` 无 `--execute` 不建目录、`execute()`、暂停/停止状态、单 worker 串行）。
- `telemetry.py`：W&B 面板（训练/奖励分项/PPO 分组梯度/课程/门禁/保持集/资源）。
- 入口脚本 CLI 必须逐字支持 SDS §11.2 全部参数，并把参数作为配方一致性断言；`--resume-run` 为唯一恢复接口。

## 5. 执行顺序与门禁（不可跳步）

```text
W0 冻结（本文件 + manifest + changelog）
→ 实施（A/B/C 并行）→ 集成（主对话接线 + __init__）
→ CPU 全量测试（scripts/test_a2_ideal_stairs_v06.py，必须 0 failed）
→ ETH 迁移审计（scripts/check_a2_ideal_stairs_v06_migration.py，seed42）
→ GPU smoke（seed42/8env/H00/2 updates，768 控制样本/3072 子步/40 Adam）
→ probe（512+8 env、5 预热+50 测量+1 恢复=56 更新、1,376,256 控制样本）
→ 独立验收 D（同一冻结源码 SHA，blocking=0 且 SHA 校验通过）
→ 新建 v06 seed42 train（F0→H→D→L→V→R 连续，30000 上限）
→ 最终冻结测试（三速×100 例，各 ≥95）
→ 视频与交付
```

- 任一步失败：保存完整证据、写 Q 文件、停下等待 Codex；不自动重试、不换 seed、不加预算、不改门禁、不挑旧 checkpoint。
- 验收 D 通过后由主对话**自动**启动 seed42；不再跑两个 pilot。
- 训练失败/退化：按 SDS 停机状态记录并停止（`behavior_regression`、预算耗尽、门禁不达标、资源不足）。

## 6. 证据与协作

- 证据根：`logs/a2_ideal_stairs_v06/implementation/`（`coordination/`、`cpu_tests/`、`migration_seed42/`、`preflight/`、`smoke/`、`probe/`、`acceptance/`、`train_seed42_trial01/`）。
- 每个 owner 交回：文件清单 + 新 SHA256 + 测试命令与输出 + 未验证事项；不得自称“通过验收”。
- 独立验收 D 的规则：不审自己模块、按 SDS §9 逐项复算、检查真实地形与 terminal/reset 顺序、复核 seed/隔离/恢复、输出 blocking 数；blocking>0 不得启动训练。
- Q 协议：写入 `logs/a2_ideal_stairs_v06/implementation/coordination/Qxxx.md`，内容含问题、可复现证据、候选处置、影响面；同步投递 Codex 线程 `01a1d49c-5f5d-7890-ac95-9234b307139a`；依赖裁决的工作等待，不阻塞的离线工作可继续。
- 追加 `logs/codex_changes.md`（主对话负责）。
- 禁止 commit/push。

## 7. 旧版本与停止状态（W0 核验）

- R2 seed43：`logs/a2_ideal_stairs_v05_r2/pilot_seed42_trial02/control.json` 仍为 `action=pause`、`resume_authorized=false`；`monitoring_blocked_000495.pt` SHA `798b367b…7c71`；无训练进程；**不恢复、不改动其目录与停止文件**。
- R1/R2 配方、日志、证据不改；v06 不读取 R2 失败模型作为初始化（只用 ETH）。
