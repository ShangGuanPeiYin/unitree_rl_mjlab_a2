"""T11 resumable serial S2.0 matrix runner (single environment per process).

Each selected case is executed by ``run_a2_v09_s2_0.py`` in its own process so
the ledger reserve/commit/release path stays exactly the audited one.  The
runner owns ordering, per-case evidence directories, resume validation, failure
continuation and progress reporting; it never writes the ledger itself.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from a2_v07_bootstrap import activate
activate(ROOT)
from src.tasks.velocity.v09 import s2_ledger, s2_execute, s2_spec

RUNNER = ROOT / 'scripts/run_a2_v09_s2_0.py'


def load_entries():
    plan = json.loads((ROOT / 'logs/a2_v09/s2_0/cases/S2_DIAG_run_plan.json').read_text())
    h = json.loads((ROOT / 'logs/a2_v09/s2_0/cases/h_series_case0.json').read_text())
    entries = []
    for index, row in enumerate(plan['entries']):
        entries.append(dict(kind='s2', index=index, suite='S2_DIAG',
                            branch=row['map_branch'], case_sha256=row['case_sha256'],
                            task=row['task'], steps=int(row['hard_timeout_steps']),
                            obligation='s2_0_matrix_two_branches'))
    for index, record in enumerate(h['records']):
        case = record['case']
        entries.append(dict(kind='h', index=index, suite='H_SERIES',
                            branch='real', case_sha256=case['case_sha256'],
                            task=case['task'], steps=int(case['hard_timeout_steps']),
                            obligation='h_series_baseline_supplement'))
    return entries


def _sha(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def valid_completed(summary_path, entry):
    try:
        summary = json.loads(Path(summary_path).read_text())
    except Exception:
        return False, 'summary_unreadable'
    if summary.get('status') != 'completed':
        return False, f"status={summary.get('status')}"
    if summary.get('case_sha256') != entry['case_sha256']:
        return False, 'case_sha256_mismatch'
    for item in summary.get('evidence_files', []):
        path = Path(item['path'])
        if not path.exists() or _sha(path) != item['sha256']:
            return False, f"evidence_hash_mismatch:{path.name}"
    if not summary.get('evidence_files'):
        return False, 'evidence_files_missing'
    return True, 'ok'


def find_completed(entry_dir, entry):
    if not entry_dir.exists():
        return None
    for summary in sorted(entry_dir.glob('*/smoke_summary.json')):
        ok, _reason = valid_completed(summary, entry)
        if ok:
            return summary.parent
    return None


def next_attempt(entry_dir):
    if not entry_dir.exists():
        return 'r1'
    used = {path.name for path in entry_dir.iterdir() if path.is_dir()}
    index = 1
    while f'r{index}' in used:
        index += 1
    return f'r{index}'


def command(entry, attempt, output, args):
    argv = [sys.executable, str(RUNNER), '--execute', '--smoke',
            '--suite', entry['suite'], '--case-sha', entry['case_sha256'],
            '--branch', entry['branch'], '--steps', str(entry['steps']),
            '--device', args.device, '--attempt-id', attempt,
            '--obligation-id', entry['obligation'], '--ledger', str(args.ledger),
            '--output-dir', str(output)]
    if args.save_inputs and entry['branch'] == 'real':
        argv.append('--save-inputs')
    return argv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--only-task')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--save-inputs', action='store_true',
                        help='persist full Actor inputs for each real-branch case')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--ledger', type=Path, default=ROOT / s2_spec.LEDGER_HINT)
    parser.add_argument('--output-root', type=Path,
                        default=ROOT / 'logs/a2_v09/s2_0/matrix')
    parser.add_argument('--run-tag',
                        default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    args = parser.parse_args(argv)
    entries = load_entries()
    if args.only_task:
        entries = [entry for entry in entries if entry['task'] == args.only_task]
    if args.limit is not None:
        if args.limit < 0:
            raise SystemExit('--limit must be nonnegative')
        entries = entries[:args.limit]
    total_steps = sum(entry['steps'] for entry in entries)
    run_id = s2_ledger.read_run_id(args.ledger) if args.ledger.exists() else None
    if args.dry_run:
        print(json.dumps(dict(dry_run=True, entries=len(entries),
                              total_hard_timeout_steps=total_steps,
                              run_id=run_id, selected=entries),
                         ensure_ascii=False, indent=2))
        return 0
    if not args.ledger.exists():
        raise SystemExit('ledger missing')
    progress_path = args.output_root / f'progress-{args.run_tag}.jsonl'
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    completed_ids, failed_ids = [], []
    total_actual = 0
    started_all = time.time()
    for ordinal, entry in enumerate(entries, 1):
        entry_id = (f"s2-{entry['index']:03d}-{entry['branch']}-{entry['case_sha256'][:12]}"
                    if entry['kind'] == 's2' else
                    f"h-{entry['index']:02d}-{entry['case_sha256'][:12]}")
        entry_dir = args.output_root / entry_id
        if args.resume:
            previous = find_completed(entry_dir, entry)
            if previous is not None:
                line = dict(ordinal=ordinal, entry_id=entry_id, branch=entry['branch'],
                            case_sha256=entry['case_sha256'], task=entry['task'],
                            status='skipped_completed', evidence_dir=str(previous),
                            elapsed_s=0.0)
                print(json.dumps(line, ensure_ascii=False), flush=True)
                with progress_path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(line, ensure_ascii=False) + '\n')
                completed_ids.append(entry_id)
                continue
        attempt = f'matrix-{args.run_tag}-{entry_id}-{next_attempt(entry_dir)}'
        output = entry_dir / attempt.rsplit('-', 1)[-1]
        output.mkdir(parents=True, exist_ok=True)
        log_path = output / 'runner.log'
        started = time.time()
        status = 'failed'
        error = None
        summary = None
        try:
            with log_path.open('w', encoding='utf-8') as log:
                proc = subprocess.run(command(entry, attempt, output, args),
                                      cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                      text=True)
            summary_path = output / 'smoke_summary.json'
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                ok, reason = valid_completed(summary_path, entry)
                status = 'completed' if ok else 'failed'
                error = None if ok else reason
                if ok:
                    total_actual += int(summary.get('settled', {}).get(
                        'actual_samples', summary.get('control_steps', 0)))
            else:
                error = f'child_exit={proc.returncode}:summary_missing'
        except Exception as exc:
            error = repr(exc)
        elapsed = time.time() - started
        line = dict(ordinal=ordinal, entry_id=entry_id, branch=entry['branch'],
                    case_sha256=entry['case_sha256'], task=entry['task'],
                    status=status, error=error, elapsed_s=elapsed,
                    control_steps=(summary or {}).get('control_steps'),
                    evidence_dir=str(output))
        print(json.dumps(line, ensure_ascii=False), flush=True)
        with progress_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(line, ensure_ascii=False) + '\n')
        (completed_ids if status == 'completed' else failed_ids).append(entry_id)
    ledger_state = s2_ledger.S2Ledger(args.ledger, run_id=run_id)
    balances = {}
    for oid in ('s2_0_matrix_two_branches', 'h_series_baseline_supplement'):
        try:
            balances[oid] = s2_execute.reservation_preflight(ledger_state,
                                                              obligation_id=oid)
        except Exception as exc:
            balances[oid] = {'error': repr(exc)}
    report = dict(run_tag=args.run_tag, selected=len(entries),
                  total_hard_timeout_steps=total_steps,
                  completed=len(completed_ids), failed=len(failed_ids),
                  failed_ids=failed_ids,
                  total_actual_committed=total_actual,
                  elapsed_s=time.time() - started_all,
                  obligations=balances,
                  progress_log=str(progress_path),
                  serial_single_env=True)
    report_path = args.output_root / f'matrix-summary-{args.run_tag}.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failed_ids else 1


if __name__ == '__main__':
    raise SystemExit(main())
