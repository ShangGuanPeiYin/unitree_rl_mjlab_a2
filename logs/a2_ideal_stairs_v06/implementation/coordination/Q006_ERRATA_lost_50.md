# Q006 问题勘误：实际丢失 101-150 共 50 次更新

```json
{
 "schema": "a2-v06-q006-errata-v1",
 "created": "2026-11-17T02:10:33+08:00",
 "question": {
  "path": "logs/a2_ideal_stairs_v06/implementation/coordination/Q006_v06_resume_metadata_loss.md",
  "sha256": "10d2d8adfa189df8cc3cf1497211834a1df6acee75498b127bd645e11d4056c1"
 },
 "decision": {
  "path": "logs/a2_ideal_stairs_v06/implementation/coordination/Q006_v06_decision.md",
  "sha256": "798a7cf4603fc50abbe6a83c30e20680fb884824c7cccddb78679bc698d5bd09"
 },
 "correction": {
  "wrong_claim": "lost 101-149 = 49 updates",
  "correct_claim": "lost 101-150 = 50 updates",
  "evidence": {
   "commit_files_count": 50,
   "first": "commit_000101.json",
   "last": "commit_000150.json",
   "commit_150": {
    "update": 150,
    "committed": true,
    "phase": "F0",
    "spent": {
     "D": 0,
     "F0": 150,
     "H": 0,
     "L": 0,
     "R": 0,
     "V": 0
    },
    "sha256": "9e378c82f38cbb72afb04c57609530c8c1a6213854f8ceb210aa7338dcd351da"
   },
   "first_sha256": "c4b1c6847dc8c74c45fa0ec0f45c92664342440f02b1e84a6d466eadf9efa181",
   "explanation": "PPO update 150 completed and commit_learning_update wrote commit_000150 (committed=true, spent.F0=150); the exception happened in finalize_boundary at the pre-diagnostic save(boundary). telemetry is written after the boundary, so a missing update-150 telemetry does not prove PPO did not run."
  },
  "cost": {
   "updates": 50,
   "control_samples": 1228800,
   "physics_substeps": 4915200,
   "adam_updates": 1000
  }
 },
 "note": "append-only errata; the original Q006 question file and its SHA are preserved unchanged (Q006 section 2)."
}
```
