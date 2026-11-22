"""CPU report generator for the S2.0 matrix and six H reproductions.

Reads only completed evidence; never rewrites trace/npz/windows.  Produces a
versioned report and evidence index after the matrix finishes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from a2_v07_bootstrap import activate
activate(ROOT)
from src.tasks.velocity.v09 import s2_cases, s2_metrics
import run_a2_v09_s2_0 as run_cli


def _sha(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _load_json(path):
    return json.loads(Path(path).read_text())


def _rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.strip()]
    for index, row in enumerate(rows):
        row['_raw_index'] = index
        row['post_reset'] = bool(row.get('post_reset', False)) or (
            index == len(rows) - 1 and bool(row.get('terminal')))
    return rows


def _trace_for_metrics(rows, summary):
    valid = [dict(row) for row in rows if not row['post_reset']]
    if not valid:
        return valid
    if summary.get('failure') or summary.get('timeout') or summary.get('status') == 'completed':
        last = dict(valid[-1])
        last['step'] = int(last['step']) + 1
        last['terminal'] = True
        last['failure'] = bool(summary.get('failure'))
        last['post_reset'] = False
        valid.append(last)
    return valid


def _platform(summary):
    geometry = summary.get('geometry') or (summary.get('case_meta') or {}).get('geometry')
    if not geometry or not geometry.get('platform_present'):
        return None
    return dict(front_x=float(geometry['platform_front_local_x']),
                length=float(geometry['platform_length']),
                y_half_width=float(geometry['platform_width']) / 2.0)


def _termination(summary):
    if summary.get('status') != 'completed':
        return 'runner_failed'
    if summary.get('failure'):
        return 'physical_failure'
    if summary.get('timeout'):
        return 'timeout'
    return 'completed'


def _quality(summary, metrics):
    if metrics is None:
        return 'not_evaluated'
    quality = metrics.get('quality_window') or {}
    if not quality.get('entered'):
        return 'not_entered'
    if not quality.get('completed'):
        return 'failed'
    return 'passed'


def _summarize_evidence(summary_path):
    summary = _load_json(summary_path)
    directory = Path(summary_path).parent
    trace_path = Path(summary.get('trace_path') or (directory / 'trace.jsonl'))
    if not trace_path.exists():
        return dict(case_sha256=summary.get('case_sha256'),
                    branch=summary.get('branch'), attempt_id=summary.get('attempt_id'),
                    status='missing_trace', error=str(trace_path))
    rows = _rows(trace_path)
    events = [event for event in (summary.get('key_events') or {}).values() if event]
    case = {'case_sha256': summary.get('case_sha256'),
            'case_id': summary.get('case_id'), 'task': summary.get('task'),
            'stair_count': summary.get('stair_count'), 'speed': summary.get('speed'),
            'geometry': summary.get('geometry')}
    platform = _platform(summary)
    windows = run_cli._four_windows(
        rows, platform_front=(platform.get('front_x') if platform else None),
        case=case, events=events)
    try:
        metrics = (s2_metrics.summarize_case(
            _trace_for_metrics(rows, summary), platform=platform,
            zero_anchor='four_feet_platform') if platform else None)
    except Exception as error:
        metrics = None
        metrics_error = repr(error)
    else:
        metrics_error = None
    if metrics:
        first = metrics['segments']['first_meter']
        post = metrics['segments']['post_first_meter']
        failure_context = metrics.get('failure_context') or {}
    else:
        first = post = failure_context = {}
    route = np.asarray([float(row['route_x']) for row in rows if not row['post_reset']])
    route_delta = float(route[-1] - route[0]) if route.size else None
    route_max = float(route.max()) if route.size else None
    return dict(
        case_sha256=summary.get('case_sha256'), case_id=summary.get('case_id'),
        suite=summary.get('suite'), task=summary.get('task'),
        branch=summary.get('branch'), speed=summary.get('speed'),
        stair_count=summary.get('stair_count'),
        platform_width=(summary.get('geometry') or {}).get('platform_width'),
        attempt_id=summary.get('attempt_id'), status=summary.get('status'),
        termination=_termination(summary), failure=bool(summary.get('failure')),
        timeout=bool(summary.get('timeout')), top_arrived=bool(summary.get('top_arrived')),
        termination_reason=summary.get('termination_reason'),
        route_delta=route_delta, route_max=route_max,
        first_meter=dict(entered=first.get('entered'), completed=first.get('completed'),
                         failed=first.get('failed'), not_reached=first.get('not_reached'),
                         end_reason=first.get('end_reason')),
        post_first_meter=dict(entered=post.get('entered'), completed=post.get('completed'),
                             failed=post.get('failed'), not_reached=post.get('not_reached'),
                             end_reason=post.get('end_reason')),
        failure_context=failure_context,
        failure_class=(failure_context.get('classification')
                       if failure_context.get('present') else None),
        quality=_quality(summary, metrics), metrics_error=metrics_error,
        success=bool((summary.get('state_summary') or {}).get('success')),
        events=summary.get('key_events') or {},
        d_post_terminal_m=(metrics.get('d_post_terminal_m') if metrics else None),
        d_post_zero_index=(metrics.get('d_post_zero_index') if metrics else None),
        windows=windows,
        trace_sha256=_sha(trace_path),
        npz_sha256=summary.get('npz_sha256'),
        evidence_dir=str(directory), trace_path=str(trace_path))


def _select_latest(candidates):
    """One completed evidence per (case_sha, branch); prefer completed, latest."""
    chosen = {}
    for row in candidates:
        key = (row.get('case_sha256'), row.get('branch'))
        previous = chosen.get(key)
        rank = (row.get('status') == 'completed', row.get('attempt_id') or '')
        if previous is None or rank > previous[0]:
            chosen[key] = (rank, row)
    return [value[1] for value in chosen.values()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path,
                        default=ROOT / 'logs/a2_v09/s2_0/matrix')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    candidates = []
    for summary_path in sorted(Path(args.evidence_root).glob('**/smoke_summary.json')):
        try:
            row = _summarize_evidence(summary_path)
        except Exception as error:
            row = dict(status='report_error', error=repr(error),
                       evidence_dir=str(summary_path.parent))
        candidates.append(row)
    selected = _select_latest(candidates)
    s2 = [row for row in selected if row.get('suite') == 'S2_DIAG']
    h = [row for row in selected if row.get('suite') == 'S1_DIAG']
    run_plan = _load_json(ROOT / 'logs/a2_v09/s2_0/cases/S2_DIAG_run_plan.json')
    h_manifest = _load_json(ROOT / 'logs/a2_v09/s2_0/cases/h_series_case0.json')
    expected_s2 = len(run_plan['entries'])
    expected_h = len(h_manifest['records'])
    h_by_sha = {record['case']['case_sha256']: record for record in h_manifest['records']}
    for row in h:
        original = h_by_sha.get(row['case_sha256'], {}).get('original_failure_position')
        row['original_failure_position'] = original
        top = row.get('windows', {}).get('top_arrived') or {}
        row['top_forward_lean'] = (None if top.get('status') != 'ok'
                                   else bool(top.get('nose_down_fraction', 0) > 0.5))
        if row.get('failure_context') and original:
            row['failure_coverage'] = s2_metrics.original_failure_coverage(
                dict(max_route_x=row['route_max'], failure_context=row['failure_context']),
                original)
        else:
            row['failure_coverage'] = dict(status='not_verified',
                                           note='original failure position or failure context missing')
    def t10_reference():
        refs = {}
        base = ROOT / 'logs/a2_v09/s2_0/evidence'
        for key, name in (('real', 't10-stairs-real-r4'),
                          ('plane', 't10-stairs-plane-r1')):
            directory = base / name
            windows_path = directory / 'windows.json'
            summary_path = directory / 'smoke_summary.json'
            if not windows_path.exists() or not summary_path.exists():
                refs[key] = dict(status='pending', reason='reference evidence missing')
                continue
            summary = _load_json(summary_path)
            refs[key] = dict(status='available', attempt_id=summary.get('attempt_id'),
                             task=summary.get('task'), speed=summary.get('speed'),
                             top_arrived=summary.get('top_arrived'),
                             failure=summary.get('failure'),
                             termination_reason=summary.get('termination_reason'),
                             windows=_load_json(windows_path),
                             trace_sha256=_sha(directory / 'trace.jsonl'))
        return dict(real_vs_plane=refs,
                    statement='real is the 40-level 0.3 m/s observation; plane is an '
                              'intervention control. This single case does not prove '
                              'simulation/field opposition or that map replacement caused the fall.')

    user_observation = t10_reference()
    report = dict(
        schema='a2_v09_s2_0_report_v1',
        generated_at_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        coverage=dict(expected_s2=expected_s2, observed_s2=len(s2),
                      expected_h=expected_h, observed_h=len(h)),
        s2_cases=s2, h_cases=h,
        failure_counts={
            'runner_failed': sum(row['termination'] == 'runner_failed' for row in s2),
            'physical_failure': sum(row['termination'] == 'physical_failure' for row in s2),
            'timeout': sum(row['termination'] == 'timeout' for row in s2),
            'completed': sum(row['termination'] == 'completed' for row in s2),
            'not_top': sum(row['termination'] == 'completed' and not row['top_arrived']
                           for row in s2 if row.get('task', '').startswith(('stairs_', 'platform_'))),
        },
        evidence_index=dict(
            schema='a2_v09_s2_0_evidence_index_v1',
            cases=[dict(case_sha256=row['case_sha256'],
                        branch=row['branch'], attempt_id=row['attempt_id'],
                        status=row['status'], evidence_dir=row['evidence_dir'],
                        trace_sha256=row['trace_sha256'], npz_sha256=row['npz_sha256'])
                   for row in selected]),
        user_observation=user_observation,
        note='all windows exclude post_reset rows and retain terminal events; '
             'first-meter uses four-feet zero per s2_metrics; old artifacts unchanged')
    if args.dry_run:
        print(json.dumps(dict(coverage=report['coverage'],
                              failure_counts=report['failure_counts'],
                              sample_s2=report['s2_cases'][:2],
                              sample_h=report['h_cases'][:1]),
                         ensure_ascii=False, indent=2))
        return 0
    directory = args.output_dir or (ROOT / 'logs/a2_v09/s2_0/reports' /
                                    datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'report.json'
    index = directory / 'evidence_index.json'
    if path.exists() or index.exists():
        raise FileExistsError(f'refusing to overwrite {path} / {index}')
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    index.write_text(json.dumps(report['evidence_index'], ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(report=str(path), evidence_index=str(index),
                          coverage=report['coverage'],
                          failure_counts=report['failure_counts']),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
