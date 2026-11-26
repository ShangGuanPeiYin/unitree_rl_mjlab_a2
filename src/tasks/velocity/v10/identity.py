"""Stable source identity shared by engineering, probes and the controller."""
import hashlib
from pathlib import Path
from . import spec


def code_fingerprint(root=spec.ROOT):
    root=Path(root)
    # Include actual reused asset/config/map/contact/model code, including existing dirty bytes.
    files=set((root/'src').rglob('*.py'))
    files.update((root/'scripts').glob('*a2_v10*.py'))
    files.update((root/'tests').glob('test_v10*.py'))
    digest=hashlib.sha256()
    for path in sorted(files,key=lambda p:p.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b'\0');digest.update(path.read_bytes());digest.update(b'\0')
    return digest.hexdigest()
