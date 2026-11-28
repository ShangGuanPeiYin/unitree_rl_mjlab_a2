#!/usr/bin/env bash
# Foreground execution: Ctrl+C reaches controller and worker in the same terminal group.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
PYTHON="${A2_PYTHON:-python}"
# Keep older shorthand usable; all other flags are explicit Python CLI arguments.
case "${1:-}" in
  --check) shift; set -- --mode check "$@" ;;
  --all) shift; set -- --mode all "$@" ;;
  --teachers) shift; set -- --mode teachers "$@" ;;
esac
# Stream newly appended child output and progress while retaining original files.
# Keep this display-only wrapper outside the scientific Python source fingerprint.
exec "$PYTHON" - "$@" <<'PYLIVE'
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

run = Path('logs/a2_v10/formal_A1_20261127_approved')
positions = {}
progress = {}
partials = {}
bar = ''
tty = sys.stdout.isatty()
ROUND_TOTALS = {'D1': 500, 'D2': 1000, 'D3': 1500, 'P0': 100, 'P1': 1000}

def show_bar(label, current, total):
    global bar
    if not total:
        return
    fraction = min(1., max(0., current / total))
    filled = int(24 * fraction)
    bar = f"{label} [{'#' * filled}{'-' * (24-filled)}] {current}/{total} {fraction:6.1%}"
    if tty:
        sys.stdout.write('\r\033[2K' + bar)
        sys.stdout.flush()
    else:
        print('[progress] ' + bar, flush=True)

def output(text):
    if tty and bar:
        sys.stdout.write('\r\033[2K')
    print(text, end='', flush=True)
    if tty and bar:
        sys.stdout.write('\r\033[2K' + bar)
        sys.stdout.flush()

def evaluation_total(path):
    # Use the actual running command and frozen suite, never guess denominators.
    for parent in path.parents:
        state_path = parent / 'controller_state.json'
        if not state_path.is_file():
            continue
        state = json.loads(state_path.read_text())
        for key, operation in state.get('operations', {}).items():
            if key != path.parent.name or operation.get('status') != 'running':
                continue
            command = operation.get('command', [])
            course = command[command.index('--course') + 1]
            suite = command[command.index('--suite') + 1]
            manifest = json.loads((parent/'frozen'/f'{course}_{suite}.json').read_text())
            return len({row['group_id'] for row in manifest['cases']})
        return None
    return None

def files():
    for base in [run, *run.glob('dynamic_N*')]:
        for folder, pattern in [('operations', 'attempt_*.log'), ('learning', 'updates.jsonl'), ('evaluation', 'progress.json')]:
            yield from (base / folder).rglob(pattern)

# Existing output is history; print only activity from this invocation.
for path in files():
    if path.name == 'progress.json':
        progress[path] = path.stat().st_mtime_ns
    else:
        positions[path] = path.stat().st_size

stop = threading.Event()
def display():
    while True:
        for path in files():
            try:
                if path.name == 'progress.json':
                    stamp = path.stat().st_mtime_ns
                    if progress.get(path) != stamp:
                        row = json.loads(path.read_text())
                        output('[progress] ' + str(path.parent.relative_to(run)) + ' ' + json.dumps(row, ensure_ascii=False) + '\n')
                        show_bar(path.parent.name, row['completed_groups'], evaluation_total(path))
                        progress[path] = stamp
                    continue
                size = path.stat().st_size
                offset = positions.get(path, 0)
                if size < offset:
                    offset = 0
                if size > offset:
                    with path.open('rb') as stream:
                        stream.seek(offset)
                        data = stream.read()
                        positions[path] = stream.tell()
                    text = data.decode('utf-8', errors='replace')
                    output(text)
                    if path.name == 'updates.jsonl':
                        text = partials.get(path, '') + text
                        lines = text.split('\n')
                        partials[path] = lines.pop()
                        for line in lines:
                            if not line.strip():
                                continue
                            row = json.loads(line)
                            phase = row.get('phase', '')
                            show_bar(phase + ' 训练轮次', row['round'], ROUND_TOTALS.get(phase))
            except (OSError, ValueError):
                continue
        if stop.wait(0.3):
            break

thread = threading.Thread(target=display, daemon=True)
thread.start()
child = subprocess.Popen([sys.executable, 'scripts/run_a2_v10_formal.py', *sys.argv[1:]])
try:
    code = child.wait()
except KeyboardInterrupt:
    # Terminal Ctrl+C already reaches the whole foreground process group.
    try:
        code = child.wait(timeout=15)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        code = 130
finally:
    time.sleep(0.4)
    stop.set()
    thread.join(timeout=2)
    if tty and bar:
        print(flush=True)
sys.exit(code if code >= 0 else 128 - code)
PYLIVE
