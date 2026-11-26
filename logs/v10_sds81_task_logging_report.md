# SDS8.1 training collection telemetry — 2026-11-27

## Scope and actual changes

Only `src/tasks/velocity/v10/runtime.py`, the CPU fixture in `tests/test_v10_runtime_memory.py`, and new `tests/test_v10_runtime_task_logging.py` changed. No changes to resource/probe budgets, curriculum, role proportions, loss, optimizer or any training setting.

Previously, `worker` retained only aggregate completed episode/failure counts before discarding rollout metadata. It now writes each rollout's `per_task` statistics to `updates.jsonl`:

- `collected_samples`: confirmed executed transitions for the task, including ongoing episodes, not inferred full-episode length.
- `completed_episodes`, `failures`, `failure_rate`: terminal episode delta in this rollout; rate denominator is the full completed episode count. Empty denominator is JSON null.
- `completed_episode_length_steps`: count/total/min/max/mean from actual terminal `episode.steps`, including episodes that began in an earlier rollout. An empty population has null min/max/mean.
- `unfinished_episodes_at_rollout_end`, `unfinished_episode_steps_at_rollout_end`, `unfinished_episode_length_steps`: boundary gauges from read-only final metadata for episodes with control_step>0. They are not cumulative completions and never count as failures at a rollout/gate/session boundary.
- `teacher_labels_applicable`, `teacher_labels_expected`, `teacher_labels_labeled`, `teacher_label_coverage`, `teacher_label_expected_sources`, `teacher_label_sources`: explicit task label denominators and same-observation source counts for blind/stairs. They expose routing transitions within an episode. PPO declares labels not applicable, expected/labeled=0 and null coverage.

Each persisted row has a unique `rollout_id`, a new `physical_session_id` per worker/environment construction, and `collection_statistics_scope='training_rollout'`. Session identity uses OS UUID randomness and does not enter Actor observations or training RNG. Resumed physical sessions have new identities; finished counts are deltas and unfinished counts remain snapshots. These are training collection statistics, never formal evaluation denominators.

DAgger now requires the teacher result to be a finite Tensor of shape `[N,12]` before any reservation or physical action. Invalid shape, nonfinite values or non-Tensor output raises an engineering error; no filtering, denominator reduction or behavior-failure conversion occurs.

## Evidence

New tests run the real `collect_rollout` inside the real `worker`, using CPU model/Adam and deterministic environment seams with all six production role counts for N=512. Collector is not mocked. They cover:

- 48-step rollout with no finished episodes and null empty statistics.
- A 60-step episode crossing rollout boundaries, logged once as length60 rather than48.
- Explicit failure/episode denominators, ongoing episode step snapshots and same-frame stairs→blind counts.
- Worker checkpoint restoration with a new physical session; discarded partial episodes are not counted as completions/failures.
- PPO's zero/not-applicable teacher denominators.
- Invalid teacher labels rejected before step or ledger cost.

Initial RED: **5 failed** before implementation; raw output `logs/a2_v10/sds81_task_logging_red.log`.

New tests GREEN: **5 passed**, one upstream Torch JIT deprecation warning, 33.18s; output `logs/a2_v10/sds81_task_logging_green.log`.

Combined regression with existing CPU save/restore/lifecycle tests: **14 passed, 1 skipped, 1 warning**, 40.33s; output `logs/a2_v10/sds81_runtime_cpu_regression.log`. The skipped full3GB check was not rerun; this logging-only change does not alter storage lifecycle.

No real GPU context, new physics samples, probe, teacher admission or formal learning was run. Virtual CPU seam accounting is isolated in pytest temporary directories and is not evidence of physical behavior. Performance or success-rate improvement is unverified. Parent owns the shared change log, independent review and Git commit/push.
