"""CPU-only comparison of completed platform_walk real/plane evidence."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def summarize(summary_path):
    summary = json.loads(Path(summary_path).read_text())
    directory = Path(summary_path).parent
    trace_path = directory / 'trace.jsonl'
    rows = load(trace_path) if trace_path.exists() else []
    valid = [r for r in rows if not (r.get('post_reset') or
                                     (r is rows[-1] and r.get('terminal')))] if rows else []
    feet = [sum(bool(v) for v in r.get('foot_on_platform', [])) for r in valid]
    contacts = [sum(bool(v) for v in r.get('foot_contact', [])) for r in valid]
    g = [float(r.get('g_level', 0.0)) for r in valid]
    return dict(
        case_sha256=summary.get('case_sha256'), branch=summary.get('branch'),
        attempt_id=summary.get('attempt_id'), speed=summary.get('speed'),
        platform_width=(summary.get('geometry') or {}).get('platform_width'),
        status=summary.get('status'), steps=summary.get('control_steps'),
        top_arrived=bool(summary.get('top_arrived')),
        failure=bool(summary.get('failure')), timeout=bool(summary.get('timeout')),
        termination_reason=summary.get('termination_reason'),
        route_max=(max((float(r['route_x']) for r in valid)) if valid else None),
        feet_on_platform_frames=sum(v == 4 for v in feet),
        foot_contact_frames=sum(v > 0 for v in contacts),
        max_feet_on_platform=max(feet) if feet else None,
        g_level_max=max(g) if g else None, g_level_positive=sum(v > 0 for v in g),
        feet_on_platform_counts=feet,
        g_level_series=g,
        map_valid_shape=(list(np.asarray(valid[0]['map_valid']).shape) if valid else None),
        map_valid_all=(bool(np.asarray(valid[0]['map_valid']).all()) if valid else None),
        trace_sha256=summary.get('trace_sha256'),
        npz_sha256=summary.get('npz_sha256'),
        map_evidence=summary.get('map_evidence'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'logs/a2_v09/s2_0/matrix')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'logs/a2_v09/s2_0/reports/platform_split.json')
    args = parser.parse_args(argv)
    rows = []
    for path in sorted(Path(args.root).glob('**/smoke_summary.json')):
        summary = json.loads(path.read_text())
        if summary.get('task') == 'platform_walk' and summary.get('status') == 'completed':
            rows.append(summarize(path))
    by_case = {}
    for row in rows:
        by_case.setdefault(row['case_sha256'], {})[row['branch']] = row
    pairs = []
    for sha, branches in sorted(by_case.items()):
        if 'real' not in branches or 'plane_diagnostic' not in branches:
            continue
        real, plane = branches['real'], branches['plane_diagnostic']
        pairs.append(dict(case_sha256=sha, real=real, plane=plane,
                          real_plane_route_max_delta=(None if None in (real['route_max'], plane['route_max'])
                                                      else plane['route_max'] - real['route_max']),
                          real_plane_steps_delta=(None if None in (real['steps'], plane['steps'])
                                                  else plane['steps'] - real['steps'])))
    report = dict(
        schema='a2_v09_s2_platform_split_v1', pairs=pairs,
        heightmap_data=dict(available=False,
                            reason='trace/npz store map_valid but not heightmap channels; no inputs.npz'),
        map_evidence_note=('plane map_evidence carries paired_proof real/policy map SHA at birth; '
                           'real platform summary carries no heightmap hash; heightmap real-vs-plane '
                           'difference cannot be quantified from existing evidence'),
        required_followup='collect --save-inputs for one representative platform real case, then run Q031')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
