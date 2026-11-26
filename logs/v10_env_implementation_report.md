# A2 v1.0 environment implementation report

Date: 2026-11-26. Implementer: `/root/integration_inventory`.
SDS: `doc/sds/SDS_A2_v1.0_DualTeacher_DAgger_PPO_v1.1.md`.
Scope owned: new `v10/{geometry,tasks,contacts,env}.py`, `tests/test_v10_env*.py`.
No existing v09 files, Git staging/commits, shared spec, or changelog were edited by this agent. Controller owns final change log and Git actions.

## Implementation

- `geometry.py`: one finite primitive inventory produces collision boxes and highest-top measurement queries. G includes finite low ground, U removes only under-platform ground; staircase low ground remains through the platform front. Treads and platform remain 0.01 m plates; ground is 0.10 m thick.
- Top-face safety uses Euclidean erosion of real coplanar unions by 0.042 m. Internal last-tread/platform seams disappear; exposed boundaries and holes remain. Queries and erosion are batched tensor operations. Compiled MuJoCo boxes are checked one-to-one against source center/half-size; unexpected terrain geoms cause failure.
- `tasks.py`: six fixed roles via largest remainder; group combinations are from frozen v10 cases, globally rotating inside each role. D1/D2 terminate stairs at top; D3/PPO use three post-top route variants. Abrupt stop yields 1000 HOLD actions; ramp yields exactly 100 DECEL actions then 1000 HOLD actions; restart is immediate. Ordinary 48-step boundaries never reset the environment.
- `contacts.py`: raw force norm >5 N, one-control-frame OR reward contact, and independent current+previous upward top force plus eroded real surface safe-support. Actual force extraction is reused from v07 without reusing its contact filter or support debounce.
- `env.py`: independent ManagerBasedRlEnv using unchanged nominal S2 asset/configuration. No old runner/course/budget/success/termination/reward code is inherited. Four real 0.005 s substeps per action; 0.02 s control and 10 Hz native 3x40x30 map.
- Observations are constructed once per control frame; repeated `observe()` clones the snapshot without advancing history/map. Current command equals history last-frame command. Reset sets action zero, draws specified spawn/joint offsets, fills five history frames, resets contacts/clocks, and requeries map.
- Reward telemetry uses post-action physics with pre-action command/phase/g_level, actual previous action/joint velocity, displacement, measured heights, real hard limits and one-shot events. Reward implementation is external `rewards.compute_reward`.
- Normal terminal success is determined by shared `evaluation.EpisodeMetrics`; full episode quality failure receives terminal failure rather than success reward. A no-return base measurement exposes `base_surface_valid=False` with finite -1.5 m clearance sentinel, while reward height error remains specified 1.5 m.
- Drop protection uses route expected height, never measured low ground after leaving a platform. Corridor widths use actual 0.8/1.2/4.0 m width, without the old 0.4 m blanket boundary. Immediate body, persistent >10 N nonfoot (0.1 s), 70-degree attitude, route fall, bounds, timeout failure are implemented. Nonfinite/capacity problems raise engineering exceptions.
- Top requires all four inset projections and foot-bottom heights plus >=2 current platform force contacts >5 N for two consecutive control frames. Event is settled in causing transition; teacher changes next action without clearing map/history/action.
- Q2 evaluation mode: `auto_reset=False` freezes task clock, EpisodeMetrics and terminal result at first termination. Padding keeps all physical slots advancing with action exactly zero, emits no repeated episodes/events, masks reward, and returns cumulative done. Default True provides training auto-reset with preserved terminal frame/observation snapshots.
- Learning state `state_dict()` contains RNG, fixed role rotation cursors, episode IDs and total control steps. Restore intentionally creates new physical episodes, not a claim of trajectory continuation.

## Public API

`V10Env(num_envs=1024, course='D1', seed=42, device='cuda:0', cases=None, auto_reset=True)`.

- `observe()`: cloned proprio/history/heightmap/route_error/privileged.
- `metadata()`: same-frame task, teacher (`blind`/`stairs`), case SHA, episode ID and control step.
- `step(actions)`: `(observation, reward, done, info)`.
- `info.frame`: reward keys agreed with learning_stack; position/v_body/v_world/roll_deg/pitch_deg/clearance/safe_support/stable/step/phase/top_event plus foot forces/positions and padding mask.
- `info.terminal_observation`: pre-reset physical snapshot; `info.episodes`: only first terminal per case; `info.failure_masks`: physical causes.
- `state_dict/load_state_dict`, `reset`, `close`.

## Verification and exact physical cost

1. RED: CPU contract tests initially failed collection because new geometry module did not exist.
2. CPU: six tests passed, covering real G/U finite boundaries, coplanar union seam safety, route-height and wide-corridor protection, three independent contact signals, exact 1024/512 role counts and 154/226/334 combinations, exact ramp/HOLD clocks, top second-frame routing and same-frame failure precedence.
3. Initial real GPU constructor failed before any physical step because mjlab renames terrain geoms; corrected to match actual box poses and sizes. Cost 0.
4. Real CUDA 1-env constructor and 3 actual control steps passed: observation shape, hard-limit shape and finite reward. Cost 3 samples.
5. Real CUDA six-role, 20-step engineering run passed: repeated observations unchanged; proprio equals final history frame; current command consistent; finite rewards; old telemetry snapshots unchanged across later steps. Cost 120 samples.
6. Real CUDA three platform heights 2/4/8 m, placed below expected route surface and stepped once: all terminated as failures despite valid low ground below. Cost 3 samples.
7. Opt-in CUDA pytest `test_real_first_terminal_freezes_and_padding_is_zero` passed (1 passed): first terminal only, metrics/task/terminal records frozen, next action forced zero and no repeated events. Cost 2 samples.

**Total physical engineering control samples executed by this agent: 128.** All GPU workers exited. No learning, resource probe or behavioral teacher admission was performed by this agent.

Raw transient evidence: `/tmp/v10_env_smoke.log`, `/tmp/v10_env_multi.log`, `/tmp/v10_env_padding.log`, `/tmp/v10_env_cpu_final.log`. Persistent reproducible test code is in `tests/test_v10_env_contract.py` and opt-in `tests/test_v10_env_gpu.py`. The final default suite deliberately skips GPU cases to avoid unbudgeted simulation. Opt-in tests require `V10_RUN_GPU_TESTS=1`, `MUJOCO_GL=egl`, and explicit controller physical-cost accounting.

## Limitations / not verified

- 1024/512 complete-stack resource probes, all teacher A/B admission, long learning and full behavioral suites are **not verified** by these engineering checks.
- Short finite-state trajectories are not evidence of behavior success. No improvement claim is made.
- Existing action manager preserves raw policy actions and applies the nominal affine target path; no new +/-10 raw-action clamp has been added. Runtime physical reports must describe actual actuator/controller bounds rather than claim a nonexistent raw clamp. This was reported to controller and learning_stack.
- Final independent review and main-branch commit/push remain controller responsibilities.
