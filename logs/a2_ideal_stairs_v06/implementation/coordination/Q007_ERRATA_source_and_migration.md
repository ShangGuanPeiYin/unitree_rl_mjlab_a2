# Q007 问题勘误：恢复源改为 exception_000400 与受限任务迁移

```json
{
 "schema": "a2-v06-q007-errata-v1",
 "created": "2026-11-17T04:19:55+08:00",
 "question": {
  "path": "logs/a2_ideal_stairs_v06/implementation/coordination/Q007_v06_retention_tile_bug.md",
  "sha256": "04f1d6fe211627b5c2cb027fa5ecfc97c52c080493f1397fce52eab22cbc0782"
 },
 "decision": {
  "path": "logs/a2_ideal_stairs_v06/implementation/coordination/Q007_v06_decision.md",
  "sha256": "567df511509ee7414f113432be3b55f34e47655a4da53e7e16ffd2e72887a48f"
 },
 "correction": {
  "wrong_claim": "resume source = boundary_000400.pt",
  "correct_claim": "resume source = exception_000400.pt (SHA 50e939963e7986477f7f9472daa9e1bf527f11b08d11c67d957a1af7092492b5)",
  "evidence": {
   "boundary_000400.pt": {
    "sha256": "2a81110d588e5822eded1029f6a4180a6f78a6fa58c0231c65e360adaf4f3097",
    "pending_results": [],
    "gate_records": [
     100,
     200,
     300
    ],
    "note": "no 400 validation/diagnostic completion inside the container"
   },
   "exception_000400.pt": {
    "sha256": "50e939963e7986477f7f9472daa9e1bf527f11b08d11c67d957a1af7092492b5",
    "pending_results": [
     "400:validation:None:H:0ccaa01f...",
     "400:diagnostic:None:H:d7468056..."
    ],
    "gate_records": [
     100,
     200,
     300,
     400
    ],
    "stop_state": null,
    "lifecycle": "running",
    "pending_update": null,
    "incomplete_update": null,
    "note": "filename says exception but the container is a valid, resumable running state; boundary disk JSON cannot be used to claim completion inside boundary"
   }
  },
  "also": "the unfinished flat retention task in the source still points at tile 1 / old table afb64ce6...; Q007 authorises one restricted migration to tile 0 / new fixed-protocol table SHA, with an append-only old->new mapping",
  "cost_formula": "trial01 lost 50 + Q005 lost 50 + active trial02 branch up to 30000; the original trial02 first 100 updates are already inside the active branch and must not be added again; current actual cumulative 500"
 },
 "note": "append-only errata; original Q007 question and its SHA are preserved."
}
```
