"""Read-only C1 evidence observer. Writes only to a separate audit directory.

Retention interpretation is summary-only, never an independent case-level gate.
No torch, simulation, control writes, training restart or policy selection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

SOURCE_SHA = '108e03dc64effa94d2824bbf36842aee8f9c5fd9c2122be391b337706f4a51b9'
SPEEDS = (.3, .5, .6)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True) + '\n')


def interpret(gate, retention, streak_before, speeds):
    summary = gate['summary']; raw = gate['gate']; rows = summary['per_speed']
    if len(rows) != 3 or tuple(float(x['c']) for x in rows) != tuple(speeds):
        raise ValueError('retention speeds or number of rows differ')
    if summary['gate_eligible'] is not False:
        raise ValueError('retention must not be promotion eligible')
    for row in rows:
        passed = row['case_pass_count']
        if row['total'] != 5 or type(passed) is not int or not 0 <= passed <= 5:
            raise ValueError('incomplete or invalid retention count')
    failed = sum(5 - row['case_pass_count'] for row in rows)
    table = summary['case_table_sha256']
    if raw['case_table_sha256'] != table or retention['case_table_sha256'] != table:
        raise ValueError('retention evidence table identities differ')
    if raw['failed_cases'] != failed or retention['failed_cases'] != failed:
        raise ValueError('retention failure counts disagree')
    after = streak_before + 1 if failed >= 12 else 0
    if retention['streak'] != after or bool(retention['stop']) != (after >= 2):
        raise ValueError('retention independent streak or stop disagrees')
    return dict(evidence_level='summary_only', cases=15, failed_cases=failed,
                per_speed=[dict(vx=x['c'], passed=x['case_pass_count'], total=5) for x in rows],
                case_table_sha256=table, raw_passed=raw['passed'],
                raw_required=[x.get('case_pass_required') for x in rows],
                retention_alarm=failed >= 12, retention_threshold_satisfied=failed < 12,
                streak_before=streak_before, streak_after=after, stop_expected=after >= 2,
                stop_observed=bool(retention['stop']), matched=True,
                interpretation='Below 12 failures is below alarm threshold, not necessarily 15 successes.')


def observe(run, output):
    run = Path(run).resolve(); output = Path(output).resolve()
    if output == run or output.is_relative_to(run):
        raise ValueError('audit output must be outside the production run')
    contract = read(run / 'contract.json')
    if contract.get('continuation_id') != 'v06-c1-speed030060-budget36000' or contract.get('source_sha256') != SOURCE_SHA:
        raise ValueError('observer only accepts the approved C1 continuation identity/source')
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / 'observer_state.json'
    state = read(state_path) if state_path.exists() else dict(
        last_update=4224, d20_open=None, retention_seen={}, streaks=dict(flat=0, h00=0))
    def persist():
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, sort_keys=True, indent=2)+'\n')
        temporary.replace(state_path)
    previous_update = state['last_update']
    for path in sorted(run.glob('commit_*.json')):
        update = int(path.stem.split('_')[-1])
        if update <= state['last_update']: continue
        if update != state['last_update'] + 1: raise ValueError('committed update gap')
        commit = read(path)
        if commit['update'] != update or commit['committed'] is not True or sum(commit['spent'].values()) != update:
            raise ValueError('learning commit is not complete or balanced')
        span = state['d20_open']
        if commit['substage'] == 'D20':
            if span is None: span = dict(first_billed_update=update, last_billed_update=update, billed_updates=0)
            span['last_billed_update'] = update; span['billed_updates'] += 1
            state['d20_open'] = span
        elif span is not None:
            append(output / 'd20_spans.jsonl', dict(span, closed_at_update=update, closed=True))
            state['d20_open'] = None
        state['last_update'] = update; state['latest_commit'] = commit
        persist()
    if state['last_update'] != previous_update:
        update = state['last_update']
        append(output / 'progress.jsonl', dict(
            recorded_at=time.time(), latest_commit=state['latest_commit'], d20_open=state['d20_open'],
            calculated_from_committed_updates=True, active_branch_cap=36000,
            active_remaining=36000-update, formal_physical_updates=update+100,
            formal_physical_cap=36100, active_control_samples=update*24576,
            active_physics_substeps=update*98304, active_adam_updates=update*20,
            full_evaluation_rng_isolation='not_established_Q008'))
    for path in sorted((run / 'gates').glob('*retention_*.json')):
        stem, number = path.stem.rsplit('_', 1); update = int(number)
        kind = 'flat' if stem == 'flat_retention' else 'h00' if stem == 'stair_retention' else None
        if kind is None: continue
        key = path.name; digest = sha(path)
        retention_path = run / 'retention' / f'{kind}_{update:06}.json'
        if key in state['retention_seen']:
            previous = state['retention_seen'][key]
            if previous != dict(gate_sha256=digest, retention_sha256=sha(retention_path)):
                raise ValueError('historical gate or retention evidence changed')
            continue
        if not retention_path.exists(): continue
        row = interpret(read(path), read(retention_path), state['streaks'][kind], SPEEDS)
        row.update(update=update, retention_kind=kind, run_dir=str(run),
                   source_path=str(path), source_sha256=digest,
                   retention_path=str(retention_path), retention_sha256=sha(retention_path),
                   original_checkpoint_sha256=SOURCE_SHA, recorded_at=time.time())
        append(output / 'retention_interpretation_q008.jsonl', row)
        state['retention_seen'][key] = dict(gate_sha256=digest, retention_sha256=sha(retention_path)); state['streaks'][kind] = row['streak_after']
        persist()
    persist()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--poll-seconds', type=float, default=30.0)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.poll_seconds <= 0: parser.error('poll-seconds must be positive')
    while True:
        try:
            state = observe(args.run_dir, args.output_dir)
            print(json.dumps(dict(update=state['last_update'], audit_dir=str(args.output_dir))), flush=True)
        except (ValueError, KeyError, OSError) as error:
            print(json.dumps(dict(audit_error=repr(error))), flush=True)
            if args.once: raise
        if args.once or (args.run_dir/'stop.json').exists() or (args.run_dir/'result.json').exists(): break
        time.sleep(args.poll_seconds)


if __name__ == '__main__': main()
