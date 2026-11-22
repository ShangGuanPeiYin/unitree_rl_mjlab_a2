"""Q033-3: platform-edge map geometry from the saved real Actor input."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GRID_X = (-0.8, 1.2)
GRID_Y = (-0.75, 0.75)
GRID_STEP = 0.05
HEIGHT_SCALE = 1.5


def load_json(path):
    return json.loads(Path(path).read_text())


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path,
                        default=ROOT / 'logs/a2_v09/s2_0/evidence/q031-platform-inputs-r1')
    parser.add_argument('--case-manifest', type=Path,
                        default=ROOT / 'logs/a2_v09/s2_0/cases/S2_DIAG.json')
    parser.add_argument('--report', type=Path,
                        default=ROOT / 'logs/a2_v09/s2_0/reports/matrix-final-v4/report.json')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'logs/a2_v09/s2_0/reports/platform_edge_analysis-v1.json')
    args = parser.parse_args(argv)
    metadata = load_json(args.input_dir / 'inputs_metadata.json')
    payload = np.load(args.input_dir / 'inputs.npz', allow_pickle=False)
    heightmaps = payload['heightmap']
    steps = payload['step']
    base_z = payload['base_sample_z']
    route_error = payload['route_error']
    trace = {int(row['step']): row for row in load_rows(args.input_dir / 'trace.jsonl')}
    manifest = load_json(args.case_manifest)
    case = next(row for row in manifest['cases']
                if row['case_sha256'] == metadata['case_sha256'])
    geometry = case['geometry']
    front = float(geometry['platform_front_local_x'])
    width = float(geometry['platform_width'])
    origin_y = float(geometry['frame_origin_world'][1])
    gx = GRID_X[0] + (np.arange(heightmaps.shape[2]) + 0.5) * GRID_STEP
    gy = GRID_Y[0] + (np.arange(heightmaps.shape[3]) + 0.5) * GRID_STEP
    gx, gy = np.meshgrid(gx, gy, indexing='ij')
    frames = []
    for index, step in enumerate(steps):
        row = trace.get(int(step))
        if row is None:
            continue
        base = np.asarray(row['base_xy'], dtype=float)
        re = np.asarray(route_error[index], dtype=float)
        yaw = math.atan2(float(re[1]), float(re[2]))
        wx = base[0] + math.cos(yaw) * gx - math.sin(yaw) * gy
        wy = origin_y + base[1] + math.sin(yaw) * gx + math.cos(yaw) * gy
        inside = ((wx >= front) & (wx <= front + float(geometry['platform_length']))
                  & (np.abs(wy - origin_y) <= width / 2.0))
        h = np.asarray(heightmaps[index], dtype=float)
        valid = h[1].astype(bool)
        age = h[2]
        out_valid = (~inside) & valid
        out_invalid = (~inside) & (~valid)
        in_valid = inside & valid
        frames.append(dict(
            step=int(step), base_xy=base.tolist(), yaw=yaw,
            inside_cells=int(inside.sum()), outside_cells=int((~inside).sum()),
            inside_valid=int(in_valid.sum()), outside_valid=int(out_valid.sum()),
            outside_invalid=int(out_invalid.sum()),
            outside_valid_height_min=float(h[0][out_valid].min()) if out_valid.any() else None,
            outside_valid_height_max=float(h[0][out_valid].max()) if out_valid.any() else None,
            outside_valid_height_mean=float(h[0][out_valid].mean()) if out_valid.any() else None,
            outside_clip_minus1=int(np.sum(np.isclose(h[0][out_valid], -1.0, atol=1e-6))),
            outside_age_mean=float(age[out_valid].mean()) if out_valid.any() else None,
            inside_valid_height_mean=float(h[0][in_valid].mean()) if in_valid.any() else None,
            inside_age_mean=float(age[in_valid].mean()) if in_valid.any() else None,
            feet_on_platform=int(sum(bool(v) for v in row.get('foot_on_platform', []))),
            foot_contact=int(sum(bool(v) for v in row.get('foot_contact', []))),
            feet_outside_platform=int(sum(
                abs(float(f[1]) - origin_y) > width / 2.0
                for f in row.get('feet_xy', []))),
            base_z=float(base_z[index])))
    def aggregate(rows):
        if not rows:
            return dict(frames=0)
        def nums(key): return [x[key] for x in rows if x[key] is not None]
        return dict(
            frames=len(rows), inside_cells=sum(x['inside_cells'] for x in rows),
            outside_valid=sum(x['outside_valid'] for x in rows),
            outside_invalid=sum(x['outside_invalid'] for x in rows),
            outside_clip_minus1=sum(x['outside_clip_minus1'] for x in rows),
            outside_valid_height_min=min(nums('outside_valid_height_min'), default=None),
            outside_valid_height_max=max(nums('outside_valid_height_max'), default=None),
            outside_valid_height_mean=(float(np.mean(nums('outside_valid_height_mean')))
                                       if nums('outside_valid_height_mean') else None),
            inside_valid=sum(x['inside_valid'] for x in rows),
            inside_valid_height_mean=(float(np.mean(nums('inside_valid_height_mean')))
                                       if nums('inside_valid_height_mean') else None),
            inside_age_mean=(float(np.mean(nums('inside_age_mean')))
                             if nums('inside_age_mean') else None),
            outside_age_mean=(float(np.mean(nums('outside_age_mean')))
                              if nums('outside_age_mean') else None),
            outside_valid_fraction=(sum(x['outside_valid'] for x in rows)
                                    / max(1, sum(x['outside_valid'] + x['outside_invalid']
                                                for x in rows))),
            outside_clip_fraction=(sum(x['outside_clip_minus1'] for x in rows)
                                   / max(1, sum(x['outside_valid'] for x in rows))),
            feet_outside_platform=sum(x['feet_outside_platform'] for x in rows),
            frames_with_feet_outside=sum(x['feet_outside_platform'] > 0 for x in rows),
            frames_with_zero_contact=sum(x['foot_contact'] == 0 for x in rows),
            last=rows[-1])
    cf_path = args.input_dir / 'counterfactual_report.json'
    counterfactual = load_json(cf_path) if cf_path.exists() else None
    report = load_json(args.report)
    current = next((row for row in report['s2_cases']
                    if row.get('case_sha256') == metadata['case_sha256']
                    and row.get('branch') == 'real'), None)
    width4 = [row for row in report['s2_cases']
              if row.get('task') == 'platform_walk' and row.get('branch') == 'real'
              and row.get('platform_width') == 4.0]
    width4_row = width4[0] if width4 else None
    output = dict(
        schema='a2_v09_s2_platform_edge_v1',
        case_sha256=metadata['case_sha256'], platform_width=width,
        geometry=geometry, input_frames=len(frames),
        width_0_8=dict(aggregate(frames),
                       platform_5m_success=sum(not row['failure'] for row in report['s2_cases']
                                               if row.get('task') == 'platform_walk'
                                               and row.get('branch') == 'real'
                                               and row.get('platform_width') == 0.8),
                       windows=(current or {}).get('windows'),
                       first_meter=(current or {}).get('first_meter'),
                       failure_context=(current or {}).get('failure_context'),
                       counterfactual=counterfactual,
                       classification='outside_valid_low_echo_and_invalid separately counted; '
                                      'inside/outside via true platform rectangle'),
        width_4_0=dict(heightmap='missing: no inputs.npz for width 4.0',
                       windows=(width4_row or {}).get('windows'),
                       first_meter=(width4_row or {}).get('first_meter'),
                       trace_cases=len(width4),
                       successes=sum(not row['failure'] for row in width4),
                       platform_5m_success=sum(not row['failure'] for row in width4),
                       route_max=max((row['route_max'] for row in width4), default=None)),
        caveat='A whole-image plane replacement cannot identify an edge cell as the causal '
               'cell; same-frame action differences are map-sensitivity evidence only.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
