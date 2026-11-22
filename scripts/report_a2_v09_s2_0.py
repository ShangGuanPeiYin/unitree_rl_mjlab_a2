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


def _trace_dict(rows):
    if not rows:
        return {}
    keys = set(rows[0])
    return {key: np.asarray([row[key] for row in rows])
            for key in keys if key not in ('contact_pairs', 'native_sample_id')}


def _platform(summary):
    geometry = summary.get('geometry') or (summary.get('case_meta') or {}).get('geometry')
    if not geometry:
        return None
    if geometry.get('platform_present') is False:
        return None
    if geometry.get('platform_front_local_x') is not None:
        front = float(geometry['platform_front_local_x'])
        width = float(geometry.get('platform_width') or geometry.get('width'))
        length = float(geometry['platform_length'])
    else:
        # H legacy geometry omits the S2 platform convenience fields.
        if not all(key in geometry for key in ('N', 'step_depth', 'width',
                                               'platform_length')):
            return None
        front = float(geometry.get('frame_origin_world', [0.0])[0]) +             float(geometry['N']) * float(geometry['step_depth'])
        width = float(geometry['width'])
        length = float(geometry['platform_length'])
    if length <= 0 or width <= 0:
        return None
    return dict(front_x=front, length=length, y_half_width=width / 2.0)


def _termination(summary):
    if summary.get('status') != 'completed':
        error = str(summary.get('error') or '')
        return 'q023_frame_proof' if 'Q023' in error else 'runner_failed'
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
            _trace_dict(_trace_for_metrics(rows, summary)), platform=platform,
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
        error=summary.get('error'),
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


def _original_failure(record):
    """Read the original H episode reason; it is not stored in the H manifest."""
    source = (record or {}).get('source') or {}
    path = source.get('original_path')
    if not path:
        return {}
    path = ROOT / path
    if not path.is_file():
        return {}
    episode = (_load_json(path) or {}).get('episode') or {}
    return dict(reason=episode.get('reason'),
                success=episode.get('success'),
                failure=episode.get('failure'),
                masks=episode.get('failure_masks'),
                path=str(path))


def _h_reasons(row, original):
    """Q033: every H case remains not_verified; preserve the diagnostic reason."""
    reasons = []
    if row.get('status') != 'completed':
        reasons.append('physical_result_missing')
    if row.get('success') or not row.get('failure'):
        reasons.append('success_no_failure')
    original_reason = (original or {}).get('reason')
    current_reason = row.get('termination_reason') if row.get('failure') else None
    if original_reason and current_reason and original_reason != current_reason:
        reasons.append('category_mismatch')
    diagnostic = row.get('coverage_diagnostic') or {}
    if diagnostic.get('covered') is not True:
        reasons.append('position_not_covered')
    if not reasons:
        # Matching category/position is only a diagnostic candidate, never a
        # formal Q033 verification.  Old matrix evidence was not a Q033 rerun.
        if str(row.get('attempt_id') or '').startswith('h-q033-r1'):
            reasons.append('candidate_only')
        else:
            reasons.append('diagnostic_match_not_rerun')
    return reasons


def _window_summary(rows):
    """Four fixed Q032 windows, aggregated by group; top is explicitly counted."""
    names = ('front_edge', 'four_feet', 'top_arrived', 's6_5_cruise')
    groups = {}
    for row in rows:
        key = (row.get('task'), row.get('branch'), row.get('platform_width'))
        groups.setdefault(key, []).append(row)
    result = {}
    for (task, branch, width), group in sorted(groups.items(),
                                                key=lambda item: str(item[0])):
        item = dict(cases=len(group),
                    top_arrived_cases=sum(bool(row.get('top_arrived'))
                                          for row in group),
                    windows={})
        for name in names:
            windows = [(row.get('windows') or {}).get(name) or {} for row in group]
            ok = [window for window in windows if window.get('status') == 'ok']
            na = [window for window in windows if window.get('status') != 'ok']
            frames = [int(window['frames']) for window in ok
                      if window.get('frames') is not None]
            item['windows'][name] = dict(
                cases=len(group), ok=len(ok), na=len(na),
                frames_min=min(frames) if frames else None,
                frames_max=max(frames) if frames else None,
                nose_down_all=(all(float(window.get('nose_down_fraction', 0)) == 1.0
                                   for window in ok) if ok else None),
                reasons=sorted({str(window.get('reason')) for window in na
                                if window.get('reason')}))
        result[f'{task}/{branch}/w{width}'] = item
    return result


def _q031_inputs():
    """Representative Q031 real Actor inputs: valid channel and invalid counts."""
    base = ROOT / 'logs/a2_v09/s2_0/evidence'
    result = dict(
        schema='a2_v09_s2_0_q031_input_validity_v1',
        valid_channel='heightmap[1] (1=valid, 0=invalid)',
        coverage_limitation=(
            '两个代表性 Q031 输入的所有帧 invalid_cells 均为 0；因此这些样本'
            '不能区分“台外低面有效回波”和“无回波无效格”，也不能推断某个'
            '边缘格是摔倒的因果格。'),
        cases={})
    for key, name in (('platform', 'q031-platform-inputs-r1'),
                      ('stairs', 'q031-stairs-inputs-r1')):
        directory = base / name
        inputs_path = directory / 'inputs.npz'
        metadata_path = directory / 'inputs_metadata.json'
        result['dirs'] = result.get('dirs', {})
        result['dirs'][key] = str(inputs_path)
        if not inputs_path.is_file():
            result['cases'][key] = dict(status='missing', path=str(inputs_path))
            continue
        payload = np.load(inputs_path, allow_pickle=False)
        heightmaps = payload['heightmap']
        valid = heightmaps[:, 1] > 0.5
        invalid = ~valid
        result['cases'][key] = dict(
            status='ok', path=str(inputs_path),
            inputs_sha256=_sha(inputs_path),
            metadata_sha256=_sha(metadata_path),
            frames=int(heightmaps.shape[0]),
            cells_per_frame=int(heightmaps.shape[2] * heightmaps.shape[3]),
            invalid_cells=int(invalid.sum()),
            frames_with_invalid=int(invalid.any(axis=(1, 2)).sum()),
            all_valid=bool(valid.all()))
    return result


def _group_stats(rows, task, branch, width):
    group = [row for row in rows
             if row.get('task') == task and row.get('branch') == branch
             and row.get('platform_width') == width]
    routes = [float(row['route_max']) for row in group
              if row.get('route_max') is not None]
    return dict(cases=len(group),
                failures=sum(bool(row.get('failure')) for row in group),
                top_arrived=sum(bool(row.get('top_arrived')) for row in group),
                route_min=min(routes) if routes else None,
                route_max=max(routes) if routes else None)


def _width_confound(rows):
    """The 0.8/4.0 comparison is not a paired control and cannot be causal."""
    return dict(
        status='confounded_not_causal',
        case_paired=False,
        spawn_paired=False,
        stair_collision_boxes_identical=True,
        only_platform_half_width_differs=True,
        observed=dict(
            stairs_real=dict(w0_8=_group_stats(rows, 'stairs_walk', 'real', 0.8),
                             w4_0=_group_stats(rows, 'stairs_walk', 'real', 4.0)),
            stairs_plane=dict(w0_8=_group_stats(rows, 'stairs_walk',
                                               'plane_diagnostic', 0.8),
                              w4_0=_group_stats(rows, 'stairs_walk',
                                                'plane_diagnostic', 4.0))),
        statement=(
            '病例/出生未配对，宽度非已证因果。楼梯局部碰撞几何相同（仅顶平台'
            '半宽不同），但不同宽度组使用不同病例、tile/origin 与出生 x/y，'
            '因此 6.95–8.16 与 9.17–9.31 m 的差异不能归因于平台宽度。'),
        valid_control=(
            '只有同一病例、同一速度的 real/plane 分支配对才是有效对照；'
            '当前 0.8/4.0 宽度组不是该配对。'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path,
                        default=ROOT / 'logs/a2_v09/s2_0/matrix')
    parser.add_argument('--h-evidence-root', type=Path,
                        default=ROOT / 'logs/a2_v09/s2_0/evidence')
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
    # H reruns live under the standalone evidence root; include only H there.
    h_candidates = [row for row in candidates if row.get('suite') == 'H_SERIES']
    if args.h_evidence_root.exists():
        for summary_path in sorted(Path(args.h_evidence_root).glob('**/smoke_summary.json')):
            try:
                row = _summarize_evidence(summary_path)
            except Exception as error:
                row = dict(status='report_error', error=repr(error),
                           evidence_dir=str(summary_path.parent))
            if row.get('suite') == 'H_SERIES':
                h_candidates.append(row)
    selected = _select_latest([row for row in candidates if row.get('suite') != 'H_SERIES'] + h_candidates)
    s2 = [row for row in selected if row.get('suite') == 'S2_DIAG']
    h = [row for row in selected if row.get('suite') == 'H_SERIES'
         or (row.get('task') == 'stairs'
             and (row.get('case_id') is not None))]
    run_plan = _load_json(ROOT / 'logs/a2_v09/s2_0/cases/S2_DIAG_run_plan.json')
    h_manifest = _load_json(ROOT / 'logs/a2_v09/s2_0/cases/h_series_case0.json')
    expected_s2 = len(run_plan['entries'])
    expected_h = len(h_manifest['records'])
    h_by_sha = {record['case']['case_sha256']: record for record in h_manifest['records']}
    for row in h:
        record = h_by_sha.get(row['case_sha256'], {})
        original = record.get('original_failure_position')
        row['lesson'] = record.get('lesson')
        row['original_failure_position'] = original
        original_failure = _original_failure(record)
        row['original_failure_reason'] = original_failure.get('reason')
        if original and 'route_x' not in original and 'terminal_route_x' in original:
            original = dict(original, route_x=original['terminal_route_x'])
        top = row.get('windows', {}).get('top_arrived') or {}
        row['top_forward_lean'] = (None if top.get('status') != 'ok'
                                   else bool(top.get('nose_down_fraction', 0) > 0.5))
        if row.get('failure_context') and original:
            row['coverage_diagnostic'] = s2_metrics.original_failure_coverage(
                dict(max_route_x=row['route_max'], failure_context=row['failure_context']),
                original)
        else:
            row['coverage_diagnostic'] = dict(
                status='not_verified',
                note='original failure position or failure context missing')
        reasons = _h_reasons(row, original_failure)
        row['failure_coverage'] = dict(
            status='not_verified',
            q033='all six H cases remain not_verified; do not read diagnostic '
                 'position/surface matches as verified reproductions',
            reasons=reasons,
            reason=('; '.join(reasons) if reasons else 'not_verified'),
            diagnostic=row['coverage_diagnostic'])
        row['h_verification'] = dict(status='not_verified', reasons=reasons,
                                     original_reason=original_failure.get('reason'),
                                     current_reason=(row.get('termination_reason')
                                                     if row.get('failure') else None))
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
    four_windows = _window_summary(s2)
    q031_inputs = _q031_inputs()
    edge_path = (ROOT / 'logs/a2_v09/s2_0/reports' /
                 'platform_edge_analysis-v1.json')
    edge = _load_json(edge_path) if edge_path.is_file() else None
    edge_diagnostic = None
    if edge:
        edge_agg = edge.get('width_0_8') or {}
        edge_diagnostic = dict(
            source=str(edge_path), source_sha256=_sha(edge_path),
            case_sha256=edge.get('case_sha256'),
            input_frames=edge.get('input_frames'),
            outside_valid=edge_agg.get('outside_valid'),
            outside_invalid=edge_agg.get('outside_invalid'),
            outside_clip_minus1=edge_agg.get('outside_clip_minus1'),
            outside_clip_fraction=edge_agg.get('outside_clip_fraction'),
            inside_valid_height_mean=edge_agg.get('inside_valid_height_mean'),
            outside_valid_height_mean=edge_agg.get('outside_valid_height_mean'),
            caveat=edge.get('caveat') or
                   'whole-image plane replacement cannot identify an edge cell as causal')
    report = dict(
        schema='a2_v09_s2_0_report_v2',
        generated_at_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        coverage=dict(expected_s2=expected_s2, observed_s2=len(s2),
                      expected_h=expected_h, observed_h=len(h)),
        s2_cases=s2, h_cases=h,
        h_coverage=dict(
            status='not_verified', cases=len(h),
            counts={'not_verified': sum(
                (row.get('failure_coverage') or {}).get('status') == 'not_verified'
                for row in h)},
            rule='Q033: all six H cases remain not_verified; diagnostic '
                 'position/surface matches are not formal verification'),
        four_window_summary=four_windows,
        top_window_checks=dict(
            stairs_real_w0_8=dict(
                n=four_windows.get('stairs_walk/real/w0.8', {}).get('cases'),
                top_arrived=four_windows.get('stairs_walk/real/w0.8', {}).get(
                    'top_arrived_cases'),
                frames_min=four_windows.get('stairs_walk/real/w0.8', {}).get(
                    'windows', {}).get('top_arrived', {}).get('frames_min'),
                frames_max=four_windows.get('stairs_walk/real/w0.8', {}).get(
                    'windows', {}).get('top_arrived', {}).get('frames_max')),
            stairs_plane=dict(
                n=sum(four_windows.get(f'stairs_walk/plane_diagnostic/w{width}', {}).get(
                    'cases', 0) for width in (0.8, 4.0)),
                top_arrived=sum(four_windows.get(f'stairs_walk/plane_diagnostic/w{width}', {}).get(
                    'top_arrived_cases', 0) for width in (0.8, 4.0)),
                all_not_top=all(
                    four_windows.get(f'stairs_walk/plane_diagnostic/w{width}', {}).get(
                        'top_arrived_cases') == 0 for width in (0.8, 4.0)))),
        width_confound=_width_confound(s2),
        q031_representative_inputs=q031_inputs,
        edge_diagnostic=edge_diagnostic,
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
             'first-meter uses four-feet zero per s2_metrics; old artifacts unchanged; '
             'H coverage is not_verified for all six cases; invalid_cells=0 in both '
             'representative Q031 inputs')
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
