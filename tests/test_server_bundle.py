import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def test_restore_split_compressed_idempotent_and_refuses_conflict(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / 'scripts/restore_server_bundle.py'
    spec = importlib.util.spec_from_file_location('restore_bundle_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    bundle = tmp_path / 'server_bundle'
    bundle.mkdir()
    files = []
    for name, data, compressed in [('checkpoint.pt', b'complete-state', False), ('cost.jsonl', b'cost-history', True)]:
        chunks = [gzip.compress(data)] if compressed else [data[:4], data[4:]]
        parts = []
        for i, chunk in enumerate(chunks):
            part = f'{name}.{i}'
            (bundle / part).write_bytes(chunk)
            parts.append(dict(file=part, sha256=hashlib.sha256(chunk).hexdigest()))
        files.append(dict(target='logs/'+name, size=len(data), sha256=hashlib.sha256(data).hexdigest(),
                          compression='gzip' if compressed else 'none', parts=parts))
    (bundle / 'manifest.json').write_text(json.dumps(dict(files=files)))
    target = tmp_path / 'target'
    module.restore(target)
    module.restore(target)
    assert (target/'logs/checkpoint.pt').read_bytes() == b'complete-state'
    assert (target/'logs/cost.jsonl').read_bytes() == b'cost-history'
    (target/'logs/cost.jsonl').write_bytes(b'new-costs')
    with pytest.raises(ValueError, match='refusing overwrite'):
        module.restore(target)
    assert (target/'logs/cost.jsonl').read_bytes() == b'new-costs'
    (bundle/files[0]['parts'][0]['file']).write_bytes(b'bad-lfs')
    with pytest.raises(ValueError, match='missing or damaged'):
        module.restore(tmp_path/'fresh')
    assert not (tmp_path/'fresh').exists()
