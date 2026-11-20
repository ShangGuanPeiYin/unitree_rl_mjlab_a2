"""Index complete frozen 6400 caches without exposing final results to training."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from a2_v07_bootstrap import activate
activate(ROOT)
from src.tasks.velocity.v09 import spec
from src.tasks.velocity.v09.cases import GATE_SUITES, FINAL_SUITES, load_manifest
from src.tasks.velocity.v09.budget import AppendOnlyLedger


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--teacher-segment-prefix', required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--code-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec.validate_sources()
    if spec.code_snapshot()['code_sha256'] != args.code_sha256:
        raise ValueError('code differs from independently approved snapshot')
    ledger = AppendOnlyLedger(args.run_dir / 'budget.jsonl', run_id=args.run_id)
    state = ledger.snapshot()['stages']['S1']
    teacher_cache = {}
    for suite in GATE_SUITES + FINAL_SUITES:
        manifest = load_manifest(args.run_dir / 'cases' / f'{suite}.json', purpose='freeze_audit')
        folder = args.run_dir / 'evidence' / f'{args.teacher_segment_prefix}{suite}'
        if (folder / 'engineering_error.json').exists():
            raise ValueError(f'{suite} has an engineering error')
        if not state['obligations']['teacher_' + suite]['closed']:
            raise ValueError(f'{suite} obligation is incomplete')
        summary_path, episodes_path = folder / 'summary.json', folder / 'episodes.jsonl'
        summary = json.loads(summary_path.read_text())
        expected = dict(checkpoint_sha256=spec.SOURCE_SHA256,
                        manifest_sha256=manifest['manifest_sha256'], code_sha256=args.code_sha256)
        if any(summary.get(key) != value for key, value in expected.items()):
            raise ValueError(f'{suite} source/manifest/code identity mismatch')
        rows = [json.loads(line) for line in episodes_path.read_text().splitlines()]
        identities = [row['case_sha256'] for row in rows]
        if len(identities) != len(set(identities)) or set(identities) != {
                case['case_sha256'] for case in manifest['cases']}:
            raise ValueError(f'{suite} denominator or case identity incomplete')
        if any(row['checkpoint_sha256'] != spec.SOURCE_SHA256 for row in rows):
            raise ValueError('teacher cache contains a different policy')
        teacher_cache[suite] = dict(episodes_path=str(episodes_path.resolve()),
            summary_path=str(summary_path.resolve()), episodes_file_sha256=digest(episodes_path),
            summary_file_sha256=digest(summary_path), case_count=len(rows), **expected)
    calibration = json.loads(args.calibration.read_text())
    if (calibration['sds_sha256'] != spec.SDS_SHA256
            or calibration['source_sha256'] != spec.SOURCE_SHA256
            or calibration['code_sha256'] != args.code_sha256
            or not state['obligations']['calibration']['closed']):
        raise ValueError('calibration identity or completion mismatch')
    result = dict(schema=11, sds_sha256=spec.SDS_SHA256,
        source_sha256=spec.SOURCE_SHA256, code_sha256=args.code_sha256,
        calibration_path=str(args.calibration.resolve()), calibration_sha256=digest(args.calibration),
        calibration_manifest_sha256=calibration['manifest_sha256'],
        value_scale=calibration['value_scale'], teacher_cache=teacher_cache,
        ledger=ledger.checkpoint_metadata(), baseline_complete=True,
        resource_admitted=False, behavior_restored=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(path=str(args.output), sha256=digest(args.output))))


if __name__ == '__main__':
    main()
