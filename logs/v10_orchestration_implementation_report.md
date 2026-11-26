# A2 v1.0 orchestration implementation

Date: 2026-11-26. Agent: `/root/integration_inventory`.

Owned new files: `src/tasks/velocity/v10/orchestration.py`, `src/tasks/velocity/v10/identity.py`, `scripts/train_a2_v10.py`, `tests/test_v10_orchestration.py`.

No GPU probe, teacher admission or learning was launched by this implementation task. No Git operations or shared modification-log edits were performed.

## Implemented protocol

The parent is CPU bookkeeping only. Probe, learning worker, evaluation worker and export are awaited subprocesses; the parent never constructs an environment or model. Every child has a durable operation key, command/log and immutable result receipt. Interrupted resource operations require an explicit engineering reason; failed final sets cannot be silently reopened. Parent-controller locking and independently owned shared GPU-child locking prevent concurrent scheduling/orphan overlap.

The protocol requires exact-code/SDS independent engineering approval and a real matching review report. It freezes the 71-run evaluation intention, actual A/B/final case manifests and SHA values, maximum evaluation cost **1,685,390,840**, training flow, SDS and approved-rulings SHA. Existing frozen artifacts cannot change. All actual Python source dependencies under `src`, v10 entry scripts and v10 tests enter `code_fingerprint`.

Resource selection either uses an already qualified exact-code selection or launches the approved 1024-first/512-fallback two-stack probe. Both selected-N stacks must report the same physics, 5 warmup plus 20 complete measurement rounds, full reservoir, restored learning state and actual resource thresholds. Selection provenance is pinned, not relabeled with the current code hash.

Only after all six D1/D2/D3 teacher A/B suites qualify can learning begin. Admission requests the evaluator's first-bad-group stop, preserves the partial failure evidence and treats it as capability failure, never as a smaller passing suite. Ordinary evaluation additionally requires every frozen group and immutable resource-admitted receipt.

DAgger runs independent D1/D2/D3 sample budgets at the fixed 40/80/100% boundaries, evaluates A then B on the same checkpoint bytes with paired teacher references, and promotes only on both passing. No unused budget transfers. D3 then opens exactly one independent 100-per-group terminal suite. A successful paired final seals the unchanged checkpoint through `V10_DAGGER_ACCEPTED` before PPO.

P0 runs exactly 4,915,200 confirmed samples (100 rounds at 1024 or 200 at 512), keeping its resettable full state. P1 runs on the common PPO external sample clock and checks at 25/50/75/100% against both teachers and frozen accepted DAgger. Two consecutive relative-retention failures restore the latest passing PPO state or P0, without touching the external budget journal. At most two rollback events are allowed. A passed gate immediately proceeds to the one independent PPO final, including after a controller crash; it cannot resume learning at the next gate instead. PPO failure leaves accepted DAgger intact.

On full PPO terminal success, only the accepted single student is exported; the export's checkpoint and actor SHA are verified against real files.

## Persistent identity and Q3 sidecars

- Candidate checkpoint bytes are never rewritten to add their own later verdict.
- `controller_events.jsonl` is a hash-chained, append-only, uniquely keyed protocol journal. Opening a final is journaled **before** executing any final candidate/reference.
- The paired verdict is persisted, then an acceptance event binding checkpoint/SDS/frozen rulings/suite/physics/results is fsynced, then the certificate is written, and only then the controller snapshot is updated.
- Recovery validates every referenced file SHA and replays missing certificate/snapshot writes idempotently. Old controller snapshots/checkpoint extras cannot refund rollback/failure counters, erase processed PPO gates, change candidate identity or reopen final sets.
- PPO gate handling commits retention counters, rollback history, resume source and processed-gate marker together through one uniquely identified event. A crash cannot count the same failure or rollback twice.
- Subsequent worker checkpoints embed the latest controller state through `--controller-state`; those new files have new SHA identities rather than inheriting an older candidate's certification.

## Q1 partial-rollout recovery

The controller accepts the first completed-update boundary satisfying `0 <= consumed - target < N*48`, while verifying charged samples stay within the phase cap. It does not require exact divisibility after previously confirmed partial control steps. A worker which cannot reach a target within the nonrefundable cap stops; it is never reported as a completed gate.

## Verification

RED: the initial orchestration tests failed import because the module had not yet been implemented.

Final task-only CPU run:

```text
/home/xxs/anaconda3/envs/a2/bin/python -m pytest -q tests/test_v10_orchestration.py
20 passed, 1 upstream Torch JIT deprecation warning in 31.16s
```

Coverage includes fixed schedules/budgets, explicit engineering rejection, code identity scope, teacher failure before any learning, stage exhaustion without PPO, immutable A/B candidate, DAgger-final failure lock, full passing controller path, nonrefundable PPO rollback, Q3 wrong-identity/hash-chain/tamper rejection, old-snapshot replay, unchanged candidate bytes, crash after durable acceptance before certificate, idempotent replay, partial-confirmed threshold overshoot, resource-receipt rejection, and both PPO passed-gate crash recovery positions.

The full protocol tests use CPU-only child receipts to exercise controller decisions. They are explicitly **not physical evidence** and cannot satisfy resource/teacher/behavior gates. `scripts/train_a2_v10.py --help` and Python compilation succeeded. Formal probes and behavioral training remain unverified until run by the controller under the independent admission signed by the main agent.

Final identity integration check: controller evaluation receipts now also require exact frozen `code_sha256` and `rulings_sha256`, matching the evaluator cache identity. Re-ran the same 20-test task suite after this last code edit: **20 passed, 1 upstream warning in 26.42s**. No GPU/physics was advanced. Owned source files are ready for the controller's final independent review and freeze.

## Q4 appendendum — streaming identity and global probe accounting (2026-11-26)

This task performed CPU implementation/auditing only. No GPU, new resource probe, teacher admission or learning was started. The requested additional probe budget is still awaiting explicit user approval; no production authorization file was created.

### Memory change and limits

`orchestration.file_sha` previously allocated the entire checkpoint with `read_bytes()`. It now hashes in **8 MiB** blocks, with byte-identical SHA256. The learner's checkpoint helper is also now 8 MiB streaming. This is not a claim that checkpoint hashing caused the observed resource failure: the previously deployed checkpoint helper was already 1 MiB streaming, and the prior logs did not capture internal stage RSS. No resource-effect improvement is claimed without a newly authorized real probe.

### Historical facts migrated without refund

`probe_budget.py` adds a persistent global ledger at:

`logs/a2_v10/global_probe_costs.jsonl`

Every original legacy JSONL row is parsed and checked for sequence, reservation/finalization identity, probe-only cost, completed-update bounds and source SHA. Each source snapshot is imported once. Repeated imports and byte-identical copies do not charge twice; changed or overlapping nonidentical sources are rejected rather than silently merged.

Actual audited legacy sources:

| Legacy source | Rows | Executed / charged | Completed updates | Uncertain |
|---|---:|---:|---:|---:|
| `formal_20261126_approved/resource_probe/n1024_dagger/costs.jsonl` | 193 | 98,304 | 49,152 | 0 |
| `formal_20261126_approved/resource_probe/n512_dagger/costs.jsonl` | 2,521 | 638,976 | 614,400 | 0 |
| **Total** | **2,714** | **737,280** | **663,552** | **0** |

The source-file SHA values and matching totals are saved in `logs/a2_v10/global_probe_migration_report.json`. A second real migration call verified that the global journal file SHA and all totals remained unchanged. Migration advances **zero** physical steps and grants **zero** future budget. Formal learning remains zero.

### Explicit future budget, not an inferred cap

The old execution plan did not freeze a global probe cap. `3,833,856 = 26×48×(2×1024+2×512)` appears only as `FULL_SEQUENCE_DEMAND`, a conservative workload demand for one complete sequence, not as a historical or default authorization.

A future probe requires a separately supplied, explicitly user-approved authorization document. The journal checks:

- schema `a2-v10-global-probe-budget-authorization-v1`;
- `approved=true`, unique `authorization_id`, `scope=probe_only`;
- SDS and approved-rulings SHA;
- exact absolute global ledger path;
- explicit cumulative `max_probe_samples` and audited `prior_charged_samples`;
- `approval_source.kind=user` and its reference.

No code fills in a production cap. Missing, denied, wrong-scope, wrong-ledger, changed-ID or wrong-baseline approval is rejected before physical launch. Learning/evaluation allowances cannot fund probe samples.

A durable single-active-sequence lease checks the entire requested workload against remaining cumulative budget before children start. Child stacks use the same global ledger as their **only** physical-cost ledger, with separate scope views and the unchanged per-stack `26×N×48` maximum. Every future control step checks both scope and global cap. Q1 single-inflight reserve/execute/uncertain and idempotent completed-update facts survive crashes; no uncertain debit is refunded or replayed. New run IDs and new output directories do not clear global costs.

The train/controller and explicit phase CLI now forward `--global-probe-ledger`, `--probe-budget-authorization` and `--probe-sequence-id`. Resource selection receipts bind those identities and the authorization SHA. `Orchestrator.prepare` consumes the real selector receipt through `admit_resource_selection`, rejecting legacy receipts without global-cost identity.

### CPU evidence

Initial new tests failed import before `probe_budget.py` existed. After implementation, tests exposed a legacy malformed-JSON error escaping the typed audit exception; this was corrected without accepting malformed records.

Final focused run:

```text
python -m pytest -q tests/test_v10_probe_budget.py tests/test_v10_orchestration.py
32 passed, 1 upstream warning in 36.62s
```

The 12 new budget tests cover historical migration/repeated imports/copy de-duplication, source modification rejection, no authorization, explicit denial, wrong scope/ledger/baseline, cross-run and cross-stack cap exhaustion, one inflight step, crash reconciliation, no uncertain refund, update idempotence and evidence bounds, post-new-cost repeated migration, and bounded streaming SHA.

Integration tests call the **real** `resources.run_probes` selector and feed its actual serialized `selection.json` directly to the **real** controller resource validator, covering 1024 success, 1024-to-512 failure, and missing-authorization refusal before any child. Only the GPU child is a CPU receipt stub. All synthetic approvals and accounting are confined to pytest temporary paths, are explicitly not physical/resource evidence, and cannot certify a real probe. Actual project costs remain the historical **737,280** only.

Independent review, shared modification-log append, commit/push and any future budget-approved real execution remain the main controller's responsibility.
