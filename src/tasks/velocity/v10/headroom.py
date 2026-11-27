"""Standalone stdlib-only admission, loaded by CLI before importing src.tasks.

No process is killed. A dip or missing measurement refuses this startup rather
than resetting the timer and silently waiting for another attempt.
"""
import json
import os
import time
import sys
from pathlib import Path

STARTUP_BYTES=12*1024**3
STARTUP_SECONDS=60
PREFILL_BYTES=19*1024**3//2


def available_memory_bytes():
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    raise RuntimeError('MemAvailable unavailable')


def wait_for_headroom(read_available=available_memory_bytes,monotonic=time.monotonic,sleep=time.sleep):
    start=monotonic();rows=[];passed=False;error=None
    while True:
        now=monotonic()
        try:available=int(read_available())
        except Exception as exc:error=repr(exc);break
        rows.append(dict(elapsed_seconds=now-start,available_bytes=available))
        if available<STARTUP_BYTES:break
        if now-start>=STARTUP_SECONDS:passed=True;break
        sleep(min(1.,STARTUP_SECONDS-(now-start)))
    return dict(schema='a2-v10-v2-startup-headroom-v1',passed=passed,required_bytes=STARTUP_BYTES,
        required_seconds=STARTUP_SECONDS,observed_seconds=monotonic()-start,min_available_bytes=min((r['available_bytes'] for r in rows),default=0),
        owner_pid=os.getpid(),created_unix=time.time(),pre_torch_import=('torch' not in sys.modules),samples=rows,error=error)


def prefill_headroom(read_available=available_memory_bytes):
    available=int(read_available())
    return dict(passed=available>=PREFILL_BYTES,required_bytes=PREFILL_BYTES,available_bytes=available)


def validate_startup_report(path,current_pid=False,fresh=False,live=False):
    if not path:raise ValueError('startup headroom report required')
    row=json.loads(Path(path).read_text())
    if (row.get('schema')!='a2-v10-v2-startup-headroom-v1' or row.get('passed') is not True
        or row.get('required_bytes')!=STARTUP_BYTES or row.get('required_seconds')!=STARTUP_SECONDS
        or row.get('observed_seconds',0)<STARTUP_SECONDS or row.get('min_available_bytes',0)<STARTUP_BYTES
        or row.get('pre_torch_import') is not True or not row.get('samples') or any(r['available_bytes']<STARTUP_BYTES for r in row['samples'])
        or row['samples'][-1]['elapsed_seconds']<STARTUP_SECONDS):raise ValueError('valid continuous startup headroom report required')
    if current_pid and row.get('owner_pid')!=os.getpid():raise ValueError('startup report must belong to current parent process')
    if fresh and not 0<=time.time()-row['created_unix']<=300:raise ValueError('startup headroom report not fresh')
    if live:os.kill(row['owner_pid'],0)
    return row
