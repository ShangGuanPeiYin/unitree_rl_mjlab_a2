"""S2.0 diagnostic executor: frozen-input validation and dry-run reporting.

T2 delivers the CPU dry-run path.  Physics execution is deliberately gated:
a non-dry-run invocation stops with an explicit message until the execution
main conversation registers the Q021 budget and provides a GPU window.  This
script never imports or constructs a simulator in dry-run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from a2_v07_bootstrap import activate
activate(ROOT)
from src.tasks.velocity.v09 import s2_cases, s2_env, s2_state, s2_terrain
from src.tasks.velocity.v09.cases import canonical_json, sha256

BRANCHES = {'real': ('real',), 'plane_ablation': ('plane_diagnostic',),
            'both': ('real', 'plane_diagnostic')}


def _load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_inputs(root=ROOT):
    root = Path(root)
    cases_dir = root / 'logs/a2_v09/s2_0/cases'
    matrix = s2_cases.load_frozen_manifest(cases_dir / 'S2_DIAG.json', root)
    h_series = s2_cases.load_frozen_manifest(cases_dir / 'h_series_case0.json', root)
    run_plan = _load_json(cases_dir / 'S2_DIAG_run_plan.json')
    s2_cases.validate_s2_diag_run_plan(run_plan, matrix)
    if run_plan['manifest_sha256'] != matrix['manifest_sha256']:
        raise ValueError('run plan does not belong to the frozen S2_DIAG manifest')
    return matrix, h_series, run_plan


def selected_entries(run_plan, branches):
    selected = [row for row in run_plan['entries'] if row['map_branch'] in branches]
    if not selected:
        raise ValueError('no run-plan entries for the selected branch')
    return selected


def dry_run(*, root=ROOT, map_branch='both', max_envs=20, device='cpu',
            output_dir=None):
    if not 1 <= max_envs <= 20:
        raise ValueError('--max-envs must be 1..20 (frozen batch upper bound 20)')
    matrix, h_series, run_plan = load_inputs(root)
    branches = BRANCHES[map_branch]
    entries = selected_entries(run_plan, branches)
    adapters = {}
    for case in matrix['cases']:
        adapter = s2_env.S2TaskAdapter.from_case(case)
        adapter.state_machine()
        adapter.level_gate()
        adapters[case['case_sha256']] = adapter
    h_adapters = {}
    for record in h_series['records']:
        adapter = s2_env.S2TaskAdapter.from_case(record['case'])
        adapter.state_machine()
        h_adapters[record['case']['case_sha256']] = adapter
    branch_pairs = s2_env.branch_execution_plan(matrix, run_plan)
    contract = s2_env.evidence_contract()
    if not contract['required_fields']:
        raise ValueError('evidence contract lacks required fields')
    by_task = {}
    for row in entries:
        by_task[row['task']] = by_task.get(row['task'], 0) + 1
    batch_table = []
    for branch in branches:
        branch_rows = [row for row in run_plan['entries']
                       if row['map_branch'] == branch]
        groups = {}
        for row in branch_rows:
            groups.setdefault(row['group_role_sha256'], []).append(row)
        for group_sha, rows in sorted(groups.items()):
            batch_table.append(dict(
                branch=branch, group_role_sha256=group_sha,
                task=rows[0]['task'], speed=rows[0]['speed'],
                platform_width=rows[0]['platform_width'],
                case_ids=sorted(row['case_id'] for row in rows),
                batch_size=len(rows),
                max_hard_timeout_steps=max(row['hard_timeout_steps'] for row in rows),
                control_samples=len(rows) * max(
                    row['hard_timeout_steps'] for row in rows)))
    for row in entries:
        if row['case_sha256'] not in adapters:
            raise ValueError('run-plan entry has no frozen case adapter')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory = Path(output_dir) if output_dir else (
        Path(root) / 'logs/a2_v09/s2_0/eval_dry_run' / stamp)
    directory.mkdir(parents=True, exist_ok=True)
    summary = dict(
        schema_version=11, stage='S2', course='S2.0', suite=matrix['suite'],
        dry_run=True, physics_started=False, device=device,
        sds_sha256=matrix['sds_sha256'],
        manifest_sha256=matrix['manifest_sha256'],
        run_plan_sha256=run_plan['plan_sha256'],
        h_manifest_sha256=h_series['manifest_sha256'],
        frozen_cases=len(matrix['cases']),
        h_series_cases=len(h_series['records']),
        selected_branch=list(branches),
        selected_branch_trajectories=len(entries),
        total_branch_trajectories=len(run_plan['entries']),
        branch_pairs=branch_pairs['pair_count'],
        branch_shared_prefix=branch_pairs['shared_prefix'],
        entries_by_task=by_task,
        max_envs=max_envs,
        env_batches=math.ceil(len(entries) / max_envs),
        run_plan_batches=len(batch_table),
        batch_table=batch_table,
        terrain_keys=sorted({row.terrain.terrain_key for row in adapters.values()}),
        h_terrain_keys=sorted({row.terrain.terrain_key for row in h_adapters.values()}),
        evidence_contract_sha256=sha256(contract),
        evidence_required_fields=contract['required_fields'],
        note='dry-run validates frozen inputs, adapters, terrain and evidence contract only',
    )
    output = directory / 'dry_run_summary.json'
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    output.write_text(canonical_json(summary) + '\n', encoding='utf-8')
    return summary, output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map-branch', choices=sorted(BRANCHES), default='both')
    parser.add_argument('--max-envs', type=int, default=20,
                        help='1..20; frozen S2.0 group batch upper bound is 20')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.max_envs <= 20:
        parser.error('--max-envs must be between 1 and 20')
    if not args.dry_run:
        raise SystemExit(
            'S2.0 physics execution is gated: the Q021 reservation must be '
            'registered and a GPU execution window assigned by the main '
            'conversation; use --dry-run for the CPU input/evidence check')
    summary, output = dry_run(map_branch=args.map_branch, max_envs=args.max_envs,
                              device=args.device, output_dir=args.output_dir)
    print(json.dumps(dict(summary=summary, output=str(output)),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
