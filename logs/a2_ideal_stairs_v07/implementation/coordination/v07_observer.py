"""Read-only audit of the designated v0.7 continuation; no training control.

Writes only outside the producer run. Retention interpretation is summary-only.
Native later segments require an explicit new audit contract; this observer refuses them.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

SOURCE_SHA = 'a41963a8bf5e6f1d4810b0a0c912ac65747d61bd481ea3432ef638a99e076e64'
RECIPE = '0.7-d22-eval-regression-r1'
SPEEDS = (.3, .5, .6)
BUDGETS = dict(F0=1000, H=12000, D=21000, L=6000, V=4000, R=4000)


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
    contract_path = run / 'contract.json'
    receipt_path = run / 'recovery_v07/receipt.json'
    contract = read(contract_path); receipt = read(receipt_path)
    if (contract.get('recipe') != RECIPE or contract.get('source_sha256') != SOURCE_SHA
            or contract.get('phase_budgets') != BUDGETS):
        raise ValueError('observer contract differs from approved v07 source/recipe/budgets')
    if (receipt.get('resume_kind') != 'designated_4817'
            or receipt.get('resume_from_update') != 4817
            or receipt.get('generation') != 73 or receipt.get('generation_updates') != 0):
        raise ValueError('observer accepts only the designated initial v07 segment')
    learning = receipt['actual_learning']
    if (learning['update'], learning['control_samples'], learning['physics_substeps']) != (4817, 118382592, 473530368):
        raise ValueError('initial learning accounting differs')
    adam = learning.get('adam_steps', learning.get('optimizer_step'))
    if adam != 96340:
        raise ValueError('initial Adam accounting differs')
    identity = dict(run_dir=str(run), contract_sha256=sha(contract_path), receipt_sha256=sha(receipt_path))
    state_path = output / 'observer_state.json'
    state = read(state_path) if state_path.exists() else dict(
        identity=identity, last_update=4817, stage_counts_segment={}, generation_counts={'73': 0},
        open_spans={}, retention_seen={}, streaks=dict(flat=0, h00=0), commit_hashes={})
    if state['identity'] != identity:
        raise ValueError('audit identity changed or reused for another run')
    # Validate the whole new batch before writing audit output, including gaps.
    pending = []; cursor = state['last_update']
    for path in sorted(run.glob('commit_*.json')):
        update = int(path.stem.split('_')[-1])
        digest = sha(path)
        if update <= cursor:
            prior = state['commit_hashes'].get(str(update))
            if prior is not None and prior != digest:
                raise ValueError('historical learning commit changed')
            continue
        if update != cursor + 1:
            raise ValueError('committed update gap')
        commit = read(path)
        if (commit['update'] != update or commit['committed'] is not True
                or set(commit['spent']) != set(BUDGETS)
                or any(type(n) is not int or n < 0 for n in commit['spent'].values())
                or sum(commit['spent'].values()) != update):
            raise ValueError('learning commit is not complete or balanced')
        pending.append((update, commit, digest)); cursor = update
    output.mkdir(parents=True, exist_ok=True)
    def persist():
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, sort_keys=True, indent=2) + '\n')
        temporary.replace(state_path)
    def close_span(stage, at_update):
        span = state['open_spans'].pop(stage)
        append(output / (stage.lower() + '_spans.jsonl'), dict(span, closed=True, closed_at_update=at_update))
    for update, commit, digest in pending:
        stage = commit['substage'] if commit['substage'] is not None else commit['phase']
        generation = str(commit['generation'])
        for opened, span in list(state['open_spans'].items()):
            if stage != opened or generation != str(span['generation']):
                close_span(opened, update)
        if stage in ('D22', 'D20'):
            span = state['open_spans'].setdefault(stage, dict(generation=commit['generation'],
                first_billed_update=update, last_billed_update=update, billed_updates=0))
            span['last_billed_update'] = update; span['billed_updates'] += 1
        state['stage_counts_segment'][stage] = state['stage_counts_segment'].get(stage, 0) + 1
        state['generation_counts'][generation] = state['generation_counts'].get(generation, 0) + 1
        state['last_update'] = update; state['latest_commit'] = commit
        state['commit_hashes'][str(update)] = digest
    if pending:
        update = state['last_update']; commit = state['latest_commit']; delta = update - 4817
        phases = tuple(BUDGETS); index = phases.index(commit['phase'])
        feasible = sum(max(0, BUDGETS[p] - commit['spent'][p]) for p in phases[index:])
        append(output / 'progress.jsonl', dict(recorded_at=time.time(), latest_commit=commit,
            stage_counts_segment=state['stage_counts_segment'], generation_counts=state['generation_counts'],
            open_spans=state['open_spans'], calculated_from_committed_updates=True,
            active_branch_cap=48000, active_remaining=48000-update,
            phase_cap_upper_bound_remaining=feasible, formal_billed_updates=update+100,
            formal_billed_cap=48100, active_control_samples=learning['control_samples']+delta*24576,
            active_physics_substeps=learning['physics_substeps']+delta*98304,
            active_adam_updates=adam+delta*20, full_evaluation_rng_isolation='not_established_Q008'))
        persist()
    for path in sorted((run / 'gates').glob('*retention_*.json')):
        stem, number = path.stem.rsplit('_', 1); update = int(number)
        kind = 'flat' if stem == 'flat_retention' else 'h00' if stem == 'stair_retention' else None
        if kind is None: continue
        key = path.name; digest = sha(path)
        retention_path = run / 'retention' / f'{kind}_{update:06}.json'
        if not retention_path.exists(): continue
        hashes = dict(gate_sha256=digest, retention_sha256=sha(retention_path))
        if key in state['retention_seen']:
            if state['retention_seen'][key] != hashes:
                raise ValueError('historical gate or retention evidence changed')
            continue
        row = interpret(read(path), read(retention_path), state['streaks'][kind], SPEEDS)
        row.update(update=update, retention_kind=kind, run_dir=str(run), source_path=str(path),
            source_sha256=digest, retention_path=str(retention_path), retention_sha256=hashes['retention_sha256'],
            original_checkpoint_sha256=SOURCE_SHA, recorded_at=time.time())
        append(output / 'retention_interpretation_q008.jsonl', row)
        state['retention_seen'][key] = hashes; state['streaks'][kind] = row['streak_after']
        persist()
    terminal = next((name for name in ('stop.json', 'result.json', 'error.json') if (run / name).exists()), None)
    if terminal:
        for stage in list(state['open_spans']): close_span(stage, state['last_update'])
        state['terminal_evidence'] = dict(path=terminal, sha256=sha(run / terminal))
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
        state = observe(args.run_dir, args.output_dir)
        print(json.dumps(dict(update=state['last_update'], audit_dir=str(args.output_dir))), flush=True)
        if args.once or state.get('terminal_evidence'): break
        time.sleep(args.poll_seconds)


if __name__ == '__main__': main()
