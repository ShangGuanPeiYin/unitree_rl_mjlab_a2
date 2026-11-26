# A2 v1.0 runtime implementation — 2026-11-26

Owned additions: `runtime.py`, `checkpoint.py`, `resources.py`, `scripts/run_a2_v10.py`, `tests/test_v10_runtime_core.py`; follow-up corrections in owned model/replay/learning tests. Parent owns project change log and Git.

## Flow and implementation

`smoke` runs real nominal physics, all six roles, same-snapshot student/teacher inference, measured contact/rewards, deterministic student actions. `worker` executes explicit D1/D2/D3/P0/P1 phases without automatic gate transitions. Every full rollout is 48 control steps and survives episode boundaries; all completed physical samples are retained, terminal GAE masks reset and only the real final nonterminal observation bootstraps. A gate boundary exits only after the current complete update and state save. PPO keeps raw sampled actions/log-probabilities separately from executed actuator actions. Stair teacher construction always reloads the SHA-identified original S2 source, including on D2/D3 resume.

Atomic checkpoint schema is `a2-v10-state-v1`, binding SDS, recipe, both teacher hashes, physical contract, fixed N, all model/optimizer/std/normalization states, replay/count/RNG, physical episode restart metadata, controller state, external costs, and frozen keep model. Resume creates new physical episodes while preserving the learning state and role rotation. Checkpoints save on crossing each 2,457,600 confirmed learning samples after a full update, rather than relying on divisibility; boundary checkpoints are separately retained.

`BudgetLedger` implements planning ruling Q1: fsync append-only pre-reservation, unique idempotent step identity, executed marking, conservative uncertain reconciliation, one in-flight step, no refund/replay, and independent completed-update sample events. Charged cost is executed+uncertain; only confirmed executed samples drive gate/save thresholds. Incomplete rollouts have no update events and never enter GAE/gradients. Process-local validated offset/identity caches read only appended journal events; fresh processes replay and verify the entire journal. Reads and mutations lock the file. The report includes charged/executed/uncertain/updated, and checkpoint recovery cannot decrease external costs.

Physical identity excludes world/terrain body inferred inertias and includes only owned robot body/joint/dof/actuator/collision data, dynamics, gravity/solver/integrator settings, and fixed control/observation contract. Terrain identity is separate. Raw actions retain the existing no-pre-clip path; no ±10 clamp was introduced.

`probe` spawns a fresh subprocess for each N/stack. It tries N=1024, then 512 on failure; each candidate must pass full DAgger and PPO stacks at the same N. Every stack fills all 196,608 CPU replay slots from actual collected snapshots, runs 5 warmup plus 20 measured full rollout/update iterations, performs full checkpoint save/restore, and records actual GPU/host resource telemetry. PPO releases unused dual teachers before its phase. No probe state enters formal learning. Initial fill is an additional explicitly billed 48-step rollout. GPU own-process telemetry must be present; missing PID data is not treated as zero usage. Thresholds are total GPU≤7680 MiB, own GPU≤7168 MiB, host available≥4 GiB, no nonfinite/capacity/OOM error. Probe result/selection bind SDS and the exact code fingerprint; nonempty reused output directories are refused.

Export requires an accepted candidate checkpoint and emits a single traced Actor with normalization/history/map/route-input contract, without teachers.

## Verification

- `python -m pytest tests/test_v10_learning_core.py tests/test_v10_runtime_core.py -q`: 23 passed, one upstream Torch JIT deprecation warning (latest run; see parent integration output for final broad suite).
- Actual CUDA RNG before/after independent Critic initialization is bitwise equal; CPU initialization uses the CPU default generator only.
- Actual six-environment, 48-control-step smoke: **288 executed engineering samples**, all six roles, finite snapshots/labels/rewards. Evidence: `logs/a2_v10/runtime_smoke_20261126/smoke.json` and its append-only cost journal.
- Real D1 and D3 one-environment assemblies at zero control steps produced the same robot physics hash `aa74d2d53526f2025b46054771ec1cf9e785b9dcee7a860e98e2c8303ee58442` with distinct terrain hashes. Evidence: `logs/a2_v10/physics_course_hashes.json` and `.log`.
- Unit cases include real MuJoCo CPU models with different terrain inferred masses, robot mass-change detection, no GPU PID admission, old-checkpoint external-cost monotonicity, reservation crash/duplicate resume/idempotency/boundaries, independent update-cost accounting, exact integer metadata, full PPO+keep microbatch gradient equivalence, and partial-rollout shifted checkpoint thresholds.

## Not verified by this agent

Complete 1024/512 resource probe has not been launched pending independent review and parent scheduling. Teacher behavior admission, formal DAgger/PPO training, export deployment playback, and final behavior acceptance remain unverified. No improvement is claimed. Two failed zero-step diagnostic invocations (local `/tmp/profile.py` import shadow and initial actuator-index attribute naming) consumed zero physics samples and were corrected. The smoke predates expanded robot-only physics hashing; its physics hash is historical, while the cross-course hash evidence uses the final filter.
