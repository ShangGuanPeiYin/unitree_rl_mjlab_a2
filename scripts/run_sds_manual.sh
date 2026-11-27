#!/usr/bin/env bash
# Resume the existing A2 run with its original ledgers and admitted resources.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON=/home/xxs/anaconda3/envs/a2/bin/python
MODE="${1:---check}"
case "$MODE" in
  --check|--teachers|--all) ;;
  *) printf 'Usage: bash scripts/run_sds_manual.sh [--check|--teachers|--all]\n' >&2; exit 2 ;;
esac
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
exec "$PYTHON" - "$MODE" <<'PY'
import fcntl
import json
import sys
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root))
base = root / 'logs/a2_v10'
run = base / 'formal_A1_20261127_approved'
selection = run / 'resource_probe_v2_sds_resume_20261127_183400/selection.json'
mode = sys.argv[1]
from src.tasks.velocity.v10.orchestration import Orchestrator
s = json.loads(selection.read_text())
diagnostic = json.loads((run / 'diagnostic_a1_112_sds_resume/result.json').read_text())
if not (diagnostic.get('passed') and diagnostic.get('complete') and diagnostic.get('tested_case_count') == 112):
    raise SystemExit('112-case diagnostic must be complete and passed')
ctl = Orchestrator(
    run, base / 'engineering_admission_A1_v2_trial_20261127.json', selection,
    device='cuda:0', resume_reason='User requested Ctrl+C; manual recovery of interrupted teacher admission',
    global_probe_ledger=base / 'global_probe_costs.jsonl',
    probe_budget_authorization=base / 'probe_authorization_A1_v2_inherited_20261127.json',
    probe_sequence_id=s['probe_sequence_id'],
    evaluation_ledger=base / 'formal_A_20261127_approved/evaluation_costs.jsonl',
)
ctl._assert_identity()
print('Run:', run, flush=True)
print('Resource selection:', selection, flush=True)
print('Environment count: 1024; rollout: 48; device: cuda:0', flush=True)
if mode == '--check':
    print('Teacher admission can resume with --teachers.')
    print('Full training requires the approved trial-to-formal handoff implementation.')
    raise SystemExit(0)
if mode == '--all':
    raise SystemExit('STOP: trial-to-formal handoff is not implemented. The 50-round checkpoint and spent budget must be retained. --all cannot yet run safely.')
with (run / 'controller.lock').open('a+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        ctl.prepare()
        ctl.state['phase'] = 'teacher_admission'
        ctl.persist()
        print('Teacher admission running. Per-operation logs:', run / 'operations', flush=True)
        ctl.teacher_admission()
        ctl.state.update(status='waiting_trial_handoff', phase='teacher_admission_complete')
        ctl.persist()
        print('Teacher admission passed. Formal D1 requires the approved handoff.', flush=True)
    except BaseException as error:
        ctl.state.update(status='stopped', stop_reason=repr(error))
        ctl.persist()
        raise
PY
