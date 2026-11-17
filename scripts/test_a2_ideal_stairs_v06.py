"""Run only the v0.6 CPU test set with pytest after activating the isolated namespace.

SDS v0.6 §10.2 requires::

    CUDA_VISIBLE_DEVICES= /home/xxs/anaconda3/envs/a2/bin/python \
      scripts/test_a2_ideal_stairs_v06.py --pattern 'test_v06_*.py'

The legacy ``src/tasks/__init__.py`` scans every task package, so this launcher
activates ``scripts/a2_v06_bootstrap.py`` *before* pytest imports any matching
``test_v06_*.py`` module.  Tests are never modified, generated or skipped by
this runner: the ``--pattern`` glob only selects files that already exist under
``tests/``.

The run is CPU-only: ``CUDA_VISIBLE_DEVICES`` is defaulted to empty (a value the
caller already set is preserved) and ``MUJOCO_GL`` is defaulted to ``egl`` for
headless physics-only scene construction.  No simulator rollout, GPU context,
W&B session or network transfer is started by this launcher itself.

Output: one line per test (``PASSED``/``FAILED``/``SKIPPED``/``ERROR``), a
summary line with passed/failed/skipped/error counts and the wall-clock
duration, and the process exit code (0 only when every collected test passed).
``--log PATH`` additionally writes the same transcript to ``PATH`` (the log path
must not exist yet, so evidence is never overwritten).

This is a byte-level derivative of ``scripts/test_a2_ideal_stairs_v05.py``: only
the v0.6 bootstrap, the default ``test_v06_*.py`` pattern and the version
labels differ (SDS §8.2).  The r1 launcher is never modified.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

# This launcher never needs a GPU.  setdefault keeps an explicit caller value.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('MUJOCO_GL', 'egl')

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / 'scripts', ROOT / 'tests'):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import a2_v06_bootstrap  # noqa: E402  (stdlib-only, before any src.tasks import)

a2_v06_bootstrap.activate(str(ROOT))

# Imported after bootstrap activation: pytest itself is version-stable, but the
# bootstrap must be the first task-related import in this process.
import pytest  # noqa: E402


DEFAULT_PATTERN = 'test_v06_*.py'


def parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pattern', default=DEFAULT_PATTERN,
                   help=f'glob for tests/ files (default: {DEFAULT_PATTERN!r})')
    p.add_argument('--verbose', '-v', action='store_true',
                   help='also print each test id and its duration')
    p.add_argument('--log', default=None,
                   help='optional new file that receives the same transcript')
    p.add_argument('--keyword', '-k', default=None,
                   help='optional pytest -k expression inside the selected files')
    p.add_argument('--maxfail', type=int, default=0,
                   help='stop after N failures (0 = run everything)')
    return p


class _Collector:
    """Minimal pytest plugin that counts outcomes and measures wall time."""

    def __init__(self):
        self.outcomes = {}
        self.reports = []
        self.started = time.monotonic()
        self.finished = None
        self.exitstatus = 0

    def pytest_runtest_logreport(self, report):
        if report.when == 'call' or (report.when in ('setup', 'teardown')
                                     and report.outcome != 'passed'):
            self.outcomes[report.nodeid] = report.outcome
            self.reports.append((report.nodeid, report.when, report.outcome,
                                 float(getattr(report, 'duration', 0.0))))

    def pytest_collectreport(self, report):
        if report.failed:
            self.outcomes[report.nodeid] = 'error'
            self.reports.append((report.nodeid, 'collect', 'error', 0.0))

    def pytest_sessionfinish(self, session, exitstatus):
        self.finished = time.monotonic()
        self.exitstatus = int(exitstatus)

    def summary(self):
        counts = {'passed': 0, 'failed': 0, 'skipped': 0, 'error': 0}
        for outcome in self.outcomes.values():
            if outcome in counts:
                counts[outcome] += 1
        counts['collected'] = len(self.outcomes)
        duration = ((self.finished if self.finished is not None else time.monotonic())
                    - self.started)
        return counts, duration


class _Tee:
    """Duplicate text to the real streams and an optional evidence log.

    pytest's terminal reporter inspects the stream (``isatty``/``encoding``/
    ``fileno``), so every unknown attribute is delegated to the real stream.
    """

    def __init__(self, stream, log=None):
        self.stream = stream
        self.log = log

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def isatty(self):
        return self.stream.isatty()

    def write(self, text):
        self.stream.write(text)
        if self.log is not None:
            self.log.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        if self.log is not None:
            self.log.flush()


def _select_files(pattern):
    tests_dir = ROOT / 'tests'
    files = sorted(path for path in tests_dir.glob(pattern) if path.is_file())
    if not files:
        raise SystemExit(f'no test files under {tests_dir} match {pattern!r}')
    return files


def main(argv=None):
    args = parser().parse_args(argv)
    files = _select_files(args.pattern)
    log = None
    if args.log:
        log_path = Path(args.log).resolve()
        if log_path.exists():
            raise SystemExit(f'--log must be a new file: {log_path}')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, 'w', encoding='utf-8')
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(stdout, log), _Tee(stderr, log)
    collector = _Collector()
    print('v0.6 CPU test run')
    print(f'project   : {ROOT}')
    print(f'version   : code 0.6.0 / recipe 0.6-go2refs-r1 / schema 8 / v06-route-state-v1')
    print(f'pattern   : {args.pattern}')
    print(f'files     : {len(files)}')
    for path in files:
        print(f'  - {path.relative_to(ROOT)}')
    pytest_args = ['-p', 'no:cacheprovider', '-o', 'addopts=', '-q',
                   '--rootdir', str(ROOT)]
    if args.verbose:
        pytest_args.append('-v')
    if args.keyword:
        pytest_args += ['-k', args.keyword]
    if args.maxfail:
        pytest_args += ['--maxfail', str(args.maxfail)]
    pytest_args += [str(path) for path in files]
    started = time.monotonic()
    code = int(pytest.main(pytest_args, plugins=[collector]))
    counts, duration = collector.summary()
    if args.verbose:
        for nodeid, when, outcome, seconds in collector.reports:
            print(f'{outcome.upper():8s} {seconds:8.3f}s {when:8s} {nodeid}')
    print('-' * 72)
    print(f"summary   : collected={counts['collected']} passed={counts['passed']} "
          f"failed={counts['failed']} skipped={counts['skipped']} "
          f"errors={counts['error']}")
    print(f'duration  : {duration:.3f}s (pytest {time.monotonic() - started:.3f}s)')
    print(f'exit code : {code}')
    if counts['collected'] == 0:
        print('NOTE: pytest collected no test in the selected files')
    sys.stdout, sys.stderr = stdout, stderr
    if log is not None:
        print(f'log       : {Path(args.log).resolve()}')
        log.close()
    return 0 if code == 0 and counts['collected'] > 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
