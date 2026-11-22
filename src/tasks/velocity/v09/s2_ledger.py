"""S2 append-only ledger with its own namespace (never the S1 header)."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path

from . import s2_spec
from .cases import canonical_json, sha256

NAMESPACE = s2_spec.LEDGER_NAMESPACE
CATEGORIES = ('learning', 'engineering', 'evaluation')


class S2Ledger:
    def __init__(self, path, *, run_id):
        if not isinstance(run_id, str) or not run_id:
            raise ValueError('run_id required')
        self.path = Path(path)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked() as handle:
            events = self._read(handle)
            if not events:
                self._append(handle, [], dict(
                    kind='header', stage='S2', run_id=run_id, namespace=NAMESPACE,
                    schema_version=s2_spec.CHECKPOINT_SCHEMA,
                    sds_sha256=s2_spec.SDS_SHA256, recipe=s2_spec.RECIPE,
                    caps=dict(
                        total=s2_spec.S2_TOTAL_CAP,
                        learning=s2_spec.S2_LEARNING_CAP,
                        nonlearning=s2_spec.S2_NONLEARNING_CAP)))
            else:
                self._state(events)

    @contextmanager
    def _locked(self):
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        with os.fdopen(fd, 'r+', encoding='utf-8') as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield handle
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read(self, handle):
        handle.seek(0)
        events = []
        previous = None
        for index, line in enumerate(handle.readlines()):
            if not line.endswith('\n'):
                raise ValueError('incomplete S2 ledger tail')
            row = json.loads(line)
            checksum = row.pop('event_sha256')
            if row.get('sequence') != index or row.get('previous_sha256') != previous:
                raise ValueError('S2 ledger sequence/hash chain broken')
            if checksum != sha256(row):
                raise ValueError('S2 ledger event SHA mismatch')
            row['event_sha256'] = checksum
            previous = checksum
            events.append(row)
        return events

    def _state(self, events):
        if not events or events[0].get('kind') != 'header':
            raise ValueError('S2 ledger header missing')
        header = events[0]
        if (header.get('namespace') != NAMESPACE
                or header.get('run_id') != self.run_id
                or header.get('stage') != 'S2'
                or header.get('sds_sha256') != s2_spec.SDS_SHA256):
            raise ValueError('S2 ledger namespace/identity mismatch')
        state = dict(envelope=0, spent={key: 0 for key in CATEGORIES},
                     reserved={key: 0 for key in CATEGORIES}, items=[])
        for row in events[1:]:
            if row['kind'] == 'stage_reserve':
                if state['envelope']:
                    raise ValueError('S2 stage already reserved')
                state['envelope'] = int(row['samples'])
            elif row['kind'] == 'reserve':
                category = row['category']
                if category not in CATEGORIES:
                    raise ValueError('unknown category')
                state['reserved'][category] += int(row['samples'])
            elif row['kind'] == 'commit':
                category = row['category']
                state['reserved'][category] -= int(row['samples'])
                state['spent'][category] += int(row['samples'])
            elif row['kind'] == 'event':
                state['items'].append(dict(row))
            else:
                raise ValueError(f'unknown S2 ledger event {row["kind"]}')
        return state

    def _append(self, handle, events, event):
        event = dict(event, sequence=len(events),
                     previous_sha256=(events[-1]['event_sha256'] if events else None))
        event['event_sha256'] = sha256(event)
        self._state(events + [event])
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json(event) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
        return event

    def append(self, kind, **payload):
        with self._locked() as handle:
            events = self._read(handle)
            return self._append(handle, events, dict(kind=kind, stage='S2', **payload))

    def state(self):
        with self._locked() as handle:
            return self._state(self._read(handle))
