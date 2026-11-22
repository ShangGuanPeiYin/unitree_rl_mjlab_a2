"""Recompute Q030 pitch windows from raw traces, excluding auto-reset rows."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from src.tasks.velocity.v09 import s2_env


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    # Existing pre-Q030 traces predate post_reset.  v07 resets on the terminal
    # step, so only a terminal final row is an auto-reset row.
    for index, row in enumerate(rows):
        row['_row'] = index
        row['_post_reset'] = bool(row.get('post_reset', False)) or (
            index == len(rows) - 1 and bool(row.get('terminal')))
    return rows


def stats(rows):
    pitch = [float(row['pitch']) * 180.0 / math.pi for row in rows]
    return dict(
        frames=len(rows), first_step=int(rows[0]['step']), last_step=int(rows[-1]['step']),
        first_row=int(rows[0]['_row']), last_row=int(rows[-1]['_row']),
        first_route_x=float(rows[0]['route_x']), last_route_x=float(rows[-1]['route_x']),
        mean_deg=sum(pitch) / len(pitch), min_deg=min(pitch), max_deg=max(pitch),
        nose_down_fraction=sum(v < 0 for v in pitch) / len(pitch),
        nose_up_fraction=sum(v > 0 for v in pitch) / len(pitch),
        level_fraction=sum(v == 0 for v in pitch) / len(pitch),
        pitch_sign_convention=dict(s2_env.PITCH_SIGN_CONVENTION),
        step_row_pairs=[[int(row['step']), int(row['_row'])] for row in rows])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-trace', type=Path,
                        default=ROOT / 'logs/a2_v09/s2_0/evidence/t10-stairs-real-r4/trace.jsonl')
    parser.add_argument('--plane-trace', type=Path,
                        default=ROOT / 'logs/a2_v09/s2_0/evidence/t10-stairs-plane-r1/trace.jsonl')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'logs/a2_v09/s2_0/evidence/t10-stairs-plane-r1/q030_pitch_recompute.json')
    args = parser.parse_args(argv)
    real = [row for row in load(args.real_trace)
            if int(row['step']) >= 1563 and not row['_post_reset']]
    plane = [row for row in load(args.plane_trace)
             if float(row['route_x']) >= 8.0 and not row['_post_reset']]
    report = dict(
        q030='pitch retained raw; positive=nose_up, negative=nose_down; zero included in denominator',
        rule='exclude auto-reset birth rows, retain terminal event; windows are raw trace rows',
        real_r4=dict(selector='step>=1563 to last non-reset row', **stats(real)),
        plane_r1=dict(selector='first body route_x>=8.0 to last non-reset row', **stats(plane)),
        old_artifacts='existing smoke summaries retain old positive_forward_lean_fraction and are superseded')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
