"""Rebuild the read-only history projection from the Q026 register.

The register (`supersede_register.jsonl`) is the only authority.  This script
is the only writer of `history/history_index_v2.json`; hand edits are refused
by `write_history_projection`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from a2_v07_bootstrap import activate
activate(ROOT)
from src.tasks.velocity.v09 import s2_cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    digest = s2_cases.rebuild_history_projection(args.root)
    print(digest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
