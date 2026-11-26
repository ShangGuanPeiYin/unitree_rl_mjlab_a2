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
