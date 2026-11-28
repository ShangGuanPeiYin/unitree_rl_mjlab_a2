#!/usr/bin/env python3
"""Restore the immutable server snapshot without overwriting an active run."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def restore(root):
    bundle = ROOT / 'server_bundle'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    # Check all release assets and destinations before writing anything.
    for row in manifest['files']:
        target = (root / row['target']).resolve()
        if not target.is_relative_to(root.resolve()):
            raise ValueError('destination outside repository')
        if target.exists() and sha256(target) != row['sha256']:
            raise ValueError(f'existing file differs; refusing overwrite: {target}')
        for part in row['parts']:
            path = bundle / part['file']
            if sha256(path) != part['sha256']:
                raise ValueError(f'release asset missing or damaged: {path}; download the release assets')
    for row in manifest['files']:
        target = root / row['target']
        if target.exists():
            print(f'Already verified: {target}')
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as out:
                temporary = Path(out.name)
                for part in row['parts']:
                    opener = gzip.open if row['compression'] == 'gzip' else open
                    with opener(bundle / part['file'], 'rb') as source:
                        for block in iter(lambda: source.read(8 * 1024**2), b''):
                            out.write(block)
                out.flush()
                os.fsync(out.fileno())
            if temporary.stat().st_size != row['size'] or sha256(temporary) != row['sha256']:
                raise ValueError(f'reconstructed file failed checksum: {target}')
            # Exclusive publication: never replace a file another process created.
            os.link(temporary, target)
            print(f'Restored and verified: {target}')
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    restore(p.parse_args().root.resolve())
