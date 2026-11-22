"""CPU grouped attribution from the completed S2.0 report and raw traces."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def last_nonreset(trace_path):
    rows = load_rows(trace_path)
    valid = [r for r in rows if not (r.get('post_reset') or (r is rows[-1] and r.get('terminal')))]
    if not valid:
        return None
    row = valid[-1]
    map_valid = np.asarray(row.get('map_valid'), dtype=bool)
    return dict(route_x=float(row['route_x']), pitch=float(row['pitch']) * 180 / math.pi,
                feet_z=[float(v) for v in row.get('feet_z', [])],
                feet_y_min=(min(float(v[1]) for v in row.get('feet_xy', []))
                            if row.get('feet_xy') else None),
                feet_y_max=(max(float(v[1]) for v in row.get('feet_xy', []))
                            if row.get('feet_xy') else None),
                foot_contact=sum(bool(v) for v in row.get('foot_contact', [])),
                foot_on_platform=sum(bool(v) for v in row.get('foot_on_platform', [])),
                map_valid_fraction=float(map_valid.mean()) if map_valid.size else None)


def group_key(row):
    return (row.get('task'), row.get('branch'), row.get('platform_width'))


def summarize_group(rows):
    failures = [r for r in rows if r.get('failure')]
    routes = [r['route_max'] for r in rows if r.get('route_max') is not None]
    steps = [r.get('control_steps') for r in rows if r.get('control_steps') is not None]
    failure_locations = []
    for row in failures:
        context = row.get('failure_context') or {}
        route = context.get('route_x')
        if route is not None:
            failure_locations.append(dict(route_x=float(route),
                                          level=int(math.floor(max(0.0, float(route)) / 0.2)),
                                          before_front_m=8.0 - float(route)))
    last = []
    for row in rows[:3]:
        if row.get('trace_path') and Path(row['trace_path']).exists():
            item = last_nonreset(row['trace_path'])
            if item:
                last.append(item)
    by_speed = {}
    for speed in sorted({r.get('speed') for r in rows}):
        subset = [r for r in rows if r.get('speed') == speed]
        routes_s = [r['route_max'] for r in subset if r.get('route_max') is not None]
        by_speed[str(speed)] = dict(
            n=len(subset), failures=sum(bool(r.get('failure')) for r in subset),
            route_min=min(routes_s) if routes_s else None,
            route_max=max(routes_s) if routes_s else None,
            reasons={str(k): sum(1 for r in subset if r.get('termination_reason') == k)
                     for k in {r.get('termination_reason') for r in subset}})
    return dict(n=len(rows), by_speed=by_speed, top_confirmed=sum(bool(r.get('top_arrived')) for r in rows),
                failure_classes={str(name): sum(1 for r in failures if (r.get('failure_context') or {}).get('classification') == name)
                                 for name in {(r.get('failure_context') or {}).get('classification') for r in failures}},
                failures=len(failures), success=len(rows) - len(failures),
                route_max_min=min(routes) if routes else None,
                route_max_max=max(routes) if routes else None,
                route_max_mean=float(np.mean(routes)) if routes else None,
                failure_route_min=min((x['route_x'] for x in failure_locations), default=None),
                failure_route_max=max((x['route_x'] for x in failure_locations), default=None),
                failure_locations=failure_locations,
                last_samples=last)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path,
                        default=ROOT / 'logs/a2_v09/s2_0/reports/matrix-final-v3/report.json')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'logs/a2_v09/s2_0/reports/group_attribution-v1.json')
    args = parser.parse_args(argv)
    report = json.loads(Path(args.report).read_text())
    groups = defaultdict(list)
    for row in report['s2_cases']:
        groups[group_key(row)].append(row)
    result = {}
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        label = f'{key[0]}/{key[1]}/w{key[2]}'
        result[label] = summarize_group(rows)
    result['H'] = {h['case_sha256'][:12]: dict(
        status=h['status'], termination=h['termination'],
        route_max=h['route_max'], failure_context=h['failure_context'],
        top_arrived=h['top_arrived'], top_forward_lean=h.get('top_forward_lean'),
        windows=h['windows'], failure_coverage=h.get('failure_coverage'),
        events=h.get('events')) for h in report['h_cases']}
    result['evidence_gaps'] = dict(
        heightmap='trace/npz store only map_valid; no heightmap channels or inputs.npz',
        platform_heightmap_comparison='not available; needs Q031 --save-inputs',
        h_q023='3 H cases stopped on Q023 real-map frame proof at 284/349/284/349; no physical failure event')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
