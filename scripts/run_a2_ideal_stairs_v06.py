"""v0.6 supervisor entry point: preflight (default) or the single seed42 run.

This is a thin wrapper around ``src/tasks/velocity/v06/orchestration.py``, which
owns the CLI contract, the ``--execute`` gate, the single-GPU-worker lock, the
four resource limits, the ``--resume-run`` contract validation and the refusal to
overwrite an existing ``--output-dir``.

Guarantees (SDS 11.2):

* without ``--execute`` only a preflight record under
  ``logs/a2_ideal_stairs_v06/implementation/preflight/`` is written: no run
  directory, no CUDA context, no W&B session, no training;
* an existing ``--output-dir`` is refused (never overwritten);
* every numeric flag is an assertion against the frozen recipe: a different value
  is refused instead of silently changing the SDS;
* ``--resume-run PATH`` is the only resume interface and never re-initializes
  from ETH.

The full SDS 11.2 command line is documented in
``scripts/train_a2_ideal_stairs_v06.py``.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / 'scripts'):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import a2_v06_bootstrap  # noqa: E402  (stdlib-only, before any src.tasks import)

a2_v06_bootstrap.activate(str(ROOT))


def main(argv=None):
    # Imported after bootstrap activation so the legacy task scanner never runs.
    from src.tasks.velocity.v06.orchestration import main as supervisor_main
    from train_a2_ideal_stairs_v06 import build_probe_worker, build_worker

    # ``--mode train`` uses the training worker, ``--mode probe`` the SDS 9.3
    # resource/recovery probe worker; admission stays in orchestration.py.
    return supervisor_main(argv, worker_factory=build_worker,
                           probe_factory=build_probe_worker)


if __name__ == '__main__':
    raise SystemExit(main())
