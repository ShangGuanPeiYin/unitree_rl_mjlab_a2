"""One project-wide GPU execution lease, held by the actual child process.

A controller crash cannot release its still-running child's lease. Waiting does
not terminate other GPU work. Probe controllers do not hold this child lease.
"""
from contextlib import contextmanager
import fcntl
import hashlib
from pathlib import Path
import tempfile

DEFAULT_LOCK=Path(tempfile.gettempdir())/('a2_v10_gpu_'+hashlib.sha256(str(Path(__file__).resolve().parents[4]).encode()).hexdigest()[:16]+'.lock')

@contextmanager
def gpu_execution_lock(path=None,*,blocking=True):
    lock=Path(path) if path is not None else DEFAULT_LOCK
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)
