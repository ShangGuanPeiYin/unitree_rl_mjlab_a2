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


def _file_sha(path):
    return __import__('hashlib').sha256(Path(path).read_bytes()).hexdigest()


PLAN_ITEMS_CATEGORIES = ('engineering', 'evaluation')


def validate_plan(plan):
    if not isinstance(plan, dict):
        raise ValueError('plan must be a mapping')
    required = {'stage', 'sds_sha256', 'items', 'learning_items',
                'total_control_samples', 'learning_total_control_samples',
                'stage_total_control_samples', 'physics_substeps', 'billing',
                'admitted', 'plan_sha256'}
    missing = required - set(plan)
    if missing:
        raise ValueError(f'plan missing fields: {sorted(missing)}')
    if plan['stage'] != 'S2' or plan['sds_sha256'] != s2_spec.SDS_SHA256:
        raise ValueError('plan stage/SDS identity mismatch')
    if plan.get('admitted') is not True:
        raise ValueError('plan must be admitted')
    payload = {key: value for key, value in plan.items() if key != 'plan_sha256'}
    if plan['plan_sha256'] != sha256(payload):
        raise ValueError('plan SHA mismatch')
    items, learning = plan['items'], plan['learning_items']
    if not isinstance(items, list) or not items:
        raise ValueError('plan items required')
    if not isinstance(learning, list):
        raise ValueError('learning_items must be a list')
    seen_ids, seen_names = set(), set()
    for item in items:
        for key in ('obligation_id', 'name', 'category', 'control_samples', 'basis'):
            if key not in item:
                raise ValueError(f'plan item missing {key}')
        if item['category'] not in PLAN_ITEMS_CATEGORIES:
            raise ValueError('non-learning item category must be engineering/evaluation')
        if item['obligation_id'] in seen_ids or item['name'] in seen_names:
            raise ValueError('duplicate plan obligation id/name')
        seen_ids.add(item['obligation_id']); seen_names.add(item['name'])
        if isinstance(item['control_samples'], bool) or not isinstance(
                item['control_samples'], int) or item['control_samples'] < 0:
            raise ValueError('item control_samples must be a nonnegative integer')
        if not item['basis']:
            raise ValueError('every obligation needs a basis clause')
    for item in learning:
        for key in ('obligation_id', 'name', 'category', 'control_samples', 'basis'):
            if key not in item:
                raise ValueError(f'learning item missing {key}')
        if item['category'] != 'learning':
            raise ValueError('learning item category must be learning')
        if item['obligation_id'] in seen_ids or item['name'] in seen_names:
            raise ValueError('duplicate plan obligation id/name')
        seen_ids.add(item['obligation_id']); seen_names.add(item['name'])
        if isinstance(item['control_samples'], bool) or not isinstance(
                item['control_samples'], int) or item['control_samples'] < 0:
            raise ValueError('learning control_samples must be a nonnegative integer')
    total = sum(item['control_samples'] for item in items)
    learning_total = sum(item['control_samples'] for item in learning)
    if total != int(plan['total_control_samples']):
        raise ValueError('items total differs from total_control_samples')
    if learning_total != int(plan['learning_total_control_samples']):
        raise ValueError('learning items differ from learning total')
    if total + learning_total != int(plan['stage_total_control_samples']):
        raise ValueError('items + learning differs from stage total')
    if int(plan['physics_substeps']) != 4 * (total + learning_total):
        raise ValueError('physics_substeps must be 4x stage total')
    if total > s2_spec.S2_NONLEARNING_CAP:
        raise ValueError('nonlearning envelope exceeds S2 cap')
    if learning_total > s2_spec.S2_LEARNING_CAP:
        raise ValueError('learning allocation exceeds S2 cap')
    if total + learning_total > s2_spec.S2_TOTAL_CAP:
        raise ValueError('stage total exceeds S2 cap')
    subcap = sum(item['control_samples'] for item in items
                 if item.get('subcap') == 'S2_0')
    if subcap > s2_spec.S2_0_NONLEARNING_SUBCAP:
        raise ValueError('S2.0 sub-cap exceeded')
    if plan.get('s2_0_subcap_used', subcap) != subcap:
        raise ValueError('declared S2.0 sub-cap usage differs')
    billing = plan['billing']
    for key in ('padding_included', 'waiting_empty_steps_billed',
                'final_reset_frames_excluded', 's1_samples_reused',
                'double_counting'):
        if key not in billing:
            raise ValueError(f'billing missing {key}')
    if billing['s1_samples_reused'] is not False or billing['double_counting'] is not False:
        raise ValueError('S2 plan cannot reuse/count S1 samples')
    return True


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
        state = dict(envelope=0, learning_total=0,
                     spent={key: 0 for key in CATEGORIES},
                     reserved={key: 0 for key in CATEGORIES},
                     obligations={}, learning_obligations={}, items=[],
                     stage_plan_sha256=None, operations={})
        for row in events[1:]:
            kind = row['kind']
            if kind == 'stage_reserve':
                if state['envelope'] or state['obligations']:
                    raise ValueError('S2 stage already reserved')
                plan = row.get('plan')
                validate_plan(plan)
                state['envelope'] = int(plan['total_control_samples'])
                state['learning_total'] = int(plan['learning_total_control_samples'])
                state['stage_plan_sha256'] = plan['plan_sha256']
                for item in plan['items']:
                    state['obligations'][item['obligation_id']] = dict(
                        name=item['name'], category=item['category'],
                        limit=int(item['control_samples']), spent=0, reserved=0,
                        basis=item['basis'], closed=False)
                for item in plan['learning_items']:
                    state['learning_obligations'][item['obligation_id']] = dict(
                        name=item['name'], category='learning',
                        limit=int(item['control_samples']), spent=0, reserved=0,
                        basis=item['basis'], closed=False)
            elif kind == 'obligation_reserve':
                operation_id = row.get('operation_id')
                if not operation_id or operation_id in state['operations']:
                    raise ValueError('operation_id must be unique / nonempty')
                obligation = self._obligation(state, row['obligation_id'])
                state['operations'][operation_id] = dict(kind='reserve',
                                                         obligation_id=row['obligation_id'])
                samples = int(row['samples'])
                if samples <= 0:
                    raise ValueError('reserve samples must be positive')
                if obligation['spent'] + obligation['reserved'] + samples > obligation['limit']:
                    raise ValueError(f"obligation {row['obligation_id']} exceeds its limit")
                obligation['reserved'] += samples
                state['reserved'][obligation['category']] += samples
            elif kind == 'obligation_commit':
                operation_id = row.get('operation_id')
                if not operation_id or operation_id in state['operations']:
                    raise ValueError('operation_id must be unique / nonempty')
                obligation = self._obligation(state, row['obligation_id'])
                state['operations'][operation_id] = dict(kind='commit',
                                                         obligation_id=row['obligation_id'])
                samples = int(row['samples'])
                if samples <= 0 or obligation['reserved'] < samples:
                    raise ValueError('commit exceeds reserved samples')
                obligation['reserved'] -= samples
                obligation['spent'] += samples
                state['reserved'][obligation['category']] -= samples
                state['spent'][obligation['category']] += samples
            elif kind == 'event':
                state['items'].append(dict(row))
            else:
                raise ValueError(f'unknown S2 ledger event {kind}')
        return state

    @staticmethod
    def _obligation(state, obligation_id):
        if obligation_id in state['obligations']:
            return state['obligations'][obligation_id]
        if obligation_id in state['learning_obligations']:
            return state['learning_obligations'][obligation_id]
        raise ValueError(f'unknown obligation {obligation_id}')

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

    def reserve_stage(self, plan):
        """One atomic S2 nonlearning reservation; refuse a second one."""
        validate_plan(plan)
        with self._locked() as handle:
            events = self._read(handle)
            state = self._state(events)
            if state['envelope'] or state['obligations']:
                raise ValueError('S2 stage already reserved; refusing duplicate')
            event = self._append(handle, events, dict(
                kind='stage_reserve', stage='S2', plan=plan,
                plan_sha256=plan['plan_sha256']))
            return dict(sequence=event['sequence'],
                        event_sha256=event['event_sha256'],
                        ledger_file_sha256=_file_sha(self.path),
                        plan_sha256=plan['plan_sha256'],
                        envelope=int(plan['total_control_samples']))

    def reserve(self, obligation_id, samples, *, operation_id, reason):
        if not operation_id or not reason:
            raise ValueError('reserve requires operation_id and reason')
        with self._locked() as handle:
            events = self._read(handle)
            self._state(events)
            event = self._append(handle, events, dict(
                kind='obligation_reserve', stage='S2',
                obligation_id=obligation_id, samples=int(samples),
                operation_id=operation_id, reason=reason))
            return dict(sequence=event['sequence'], event_sha256=event['event_sha256'])

    def commit(self, obligation_id, samples, *, operation_id, reason):
        if not operation_id or not reason:
            raise ValueError('commit requires operation_id and reason')
        with self._locked() as handle:
            events = self._read(handle)
            self._state(events)
            event = self._append(handle, events, dict(
                kind='obligation_commit', stage='S2',
                obligation_id=obligation_id, samples=int(samples),
                operation_id=operation_id, reason=reason))
            return dict(sequence=event['sequence'], event_sha256=event['event_sha256'])

    def state(self):
        with self._locked() as handle:
            return self._state(self._read(handle))
