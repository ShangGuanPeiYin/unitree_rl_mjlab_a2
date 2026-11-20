"""Physical control-sample accounting and append-only external schema-11 state.

Reservations happen before physics advances. Checkpoints may describe a past
model, but cannot replace this ledger. Uncertain executions consume their entire
reserved bound; only operations with known actual counts release unused cost.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path

from . import spec
from .cases import (CALIBRATION_SUITE, DIAGNOSTIC_SUITE, FINAL_SUITES, GATE_SUITES, canonical_json, sha256,
                    validate_manifest)

STAGE_CAPS = {
    'S1': dict(total=1_000_000_000, learning=850_000_000, nonlearning=150_000_000),
    'S2': dict(total=4_000_000_000, learning=3_600_000_000, nonlearning=400_000_000),
    'S3': dict(total=5_000_000_000, learning=4_400_000_000, nonlearning=600_000_000),
}
GATE_LIMITS = {'S1': 11, 'S2': 40, 'S3': 61}
CATEGORIES = ('learning', 'engineering', 'evaluation')
S1_LEARNING_ALLOCATIONS = {'S1.1': 20_000_000, 'S1.2_main': 630_000_000,
                           'S1.2_retry_1': 100_000_000, 'S1.2_retry_2': 100_000_000}


def _int(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def batch_worst_cost(cases, *, batch_size):
    """Equal group batches only; every completed slot's waiting steps count.

    Smaller batches must equally divide a group; no extra empty environments.
    Batch membership/order is explicit and includes max-timeout padding.
    """
    _int(batch_size, 'batch_size', 1)
    grouped = {}
    for case in cases:
        group = tuple(canonical_json(case.get(key)) for key in (
            'stage', 'suite', 'task', 'stair_count', 'geometry', 'speed',
            'stop_mode', 'force_family'))
        grouped.setdefault(group, []).append(case)
    batches = []
    for rows in grouped.values():
        if len(rows) % batch_size:
            raise ValueError('batch_size must equally divide each group; no empty slots')
        for offset in range(0, len(rows), batch_size):
            members = rows[offset:offset + batch_size]
            steps = max(_int(row['hard_timeout_steps'], 'hard_timeout_steps', 1)
                        for row in members)
            batches.append(dict(num_envs=batch_size, control_steps=steps,
                control_samples=batch_size * steps,
                case_sha256s=[row['case_sha256'] for row in members]))
    samples = sum(batch['control_samples'] for batch in batches)
    return dict(batches=batches, case_count=len(cases), control_samples=samples,
                physics_substeps=4 * samples)


def s1_reservation_plan(manifests, *, calibration_manifest=None, diagnostic_manifest=None, probe_extra_costs, engineering_extra_samples,
                        calibration_samples=1_000_000):
    """Recompute the S1 cap from real manifests and explicit probe operations.

    probe_extra_costs must contain each resource candidate and explicit lists of
    (num_envs, control_steps, reason) for evaluation, save/resume and any extra
    advances. Empty lists are rejected: an unmeasured recovery is not free.
    The caller must justify these concrete probe limits before stage admission.
    """
    suites = GATE_SUITES + FINAL_SUITES
    if set(manifests) != set(suites):
        raise ValueError('all two gates and three final manifests required')
    for suite, manifest in manifests.items():
        validate_manifest(manifest)
        if manifest['suite'] != suite:
            raise ValueError('manifest suite mismatch')
    if calibration_manifest is None:
        raise ValueError('explicit frozen calibration_manifest required')
    validate_manifest(calibration_manifest)
    if calibration_manifest['suite'] != CALIBRATION_SUITE:
        raise ValueError('calibration must use independent training calibration suite')
    if diagnostic_manifest is None:
        raise ValueError('Q008 explicit frozen diagnostic_manifest required')
    validate_manifest(diagnostic_manifest)
    if diagnostic_manifest['suite'] != DIAGNOSTIC_SUITE:
        raise ValueError('Q008 requires the independent S1_DIAG suite')
    diagnostic_cost = batch_worst_cost(diagnostic_manifest['cases'], batch_size=20)['control_samples']
    video_cases = [row for row in diagnostic_manifest['cases'] if row['case_id'] == 0]
    video_cost = batch_worst_cost(video_cases, batch_size=1)['control_samples']
    calibration_target = batch_worst_cost(calibration_manifest['cases'],
        batch_size=calibration_manifest['cases_per_group'])['control_samples']
    _int(calibration_samples, 'calibration_samples')
    if calibration_target > calibration_samples:
        raise ValueError('frozen calibration worst cost exceeds reserved allowance')
    if calibration_samples > 1_000_000:
        raise ValueError('S1 calibration exceeds one million control samples')
    _int(engineering_extra_samples, 'engineering_extra_samples')
    if set(probe_extra_costs) != set(spec.ENV_CANDIDATES):
        raise ValueError('explicit probe extras for all 2048/1536/1024/512 candidates required')
    costs = {suite: batch_worst_cost(manifests[suite]['cases'],
             batch_size=20 if suite in GATE_SUITES else 100)['control_samples']
             for suite in suites}
    # Alternation may begin with either A/B after an aborted prior attempt.
    # Freeze the upper bound rather than assume the cheaper suite runs six times.
    candidate_gates = max(6 * costs['gate_A'] + 5 * costs['gate_B'],
                          5 * costs['gate_A'] + 6 * costs['gate_B'])
    baseline = sum(costs.values())
    candidate_finals = sum(costs[suite] for suite in FINAL_SUITES)
    probes = []
    for count in spec.ENV_CANDIDATES:
        extras = probe_extra_costs[count]
        if not extras:
            raise ValueError('probe evaluation/save/resume extra advances must be explicit')
        layout = [(count, 48), (100, 5005), (1, 5005), (count, 48)]
        if [(extra.get('num_envs'), extra.get('control_steps')) for extra in extras] != layout:
            raise ValueError('Q009 requires S1.2 rollout,100-env evaluation,1-env video,and restored rollout')
        extra_total = 0
        for extra in extras:
            if not extra.get('reason'):
                raise ValueError('probe extra needs a reason')
            extra_total += _int(extra['num_envs'], 'num_envs', 1) * _int(
                extra['control_steps'], 'control_steps', 1)
        probes.append(dict(num_envs=count, warmups=5, measurements=50,
            rollout_steps=48, adam_updates_per_rollout=20,
            rollout_samples=count * 55 * 48, extra_operations=extras,
            control_samples=count * 55 * 48 + extra_total))
    baseline_subtotal = baseline + calibration_samples
    total = (baseline_subtotal + candidate_gates + candidate_finals
             + sum(row['control_samples'] for row in probes) + engineering_extra_samples
             + 2 * diagnostic_cost + 6 * video_cost)
    catalogue = {}
    for suite in suites + (CALIBRATION_SUITE, DIAGNOSTIC_SUITE, 'S1_VIDEO'):
        rows = (calibration_manifest['cases'] if suite == CALIBRATION_SUITE else
                diagnostic_manifest['cases'] if suite == DIAGNOSTIC_SUITE else
                video_cases if suite == 'S1_VIDEO' else manifests[suite]['cases'])
        catalogue[suite] = {row['case_sha256']: dict(
            timeout=row['hard_timeout_steps'],
            group_sha256=sha256({key: row[key] for key in ('stage', 'task',
                'geometry', 'speed', 'stop_mode', 'force_family')}))
            for row in rows}
    obligations = {}
    for suite in suites:
        obligations['teacher_' + suite] = dict(kind='teacher_baseline',
            control_samples=costs[suite], suite=suite,
            manifest_sha256=manifests[suite]['manifest_sha256'])
    for index in range(11):
        suite = GATE_SUITES[index % 2]
        obligations[f'candidate_gate_{index + 1:02d}'] = dict(kind='candidate_gates',
            control_samples=costs[suite], suite=suite,
            manifest_sha256=manifests[suite]['manifest_sha256'])
    for suite in FINAL_SUITES:
        obligations['candidate_' + suite] = dict(kind='candidate_finals',
            control_samples=costs[suite], suite=suite,
            manifest_sha256=manifests[suite]['manifest_sha256'])
    for probe in probes:
        obligations[f"probe_{probe['num_envs']}"] = dict(kind='probe',
            control_samples=probe['control_samples'],
            work_items={**{f'rollout_{index + 1:03d}': dict(num_envs=probe['num_envs'],
                control_steps=48, require_full_steps=True) for index in range(55)},
                **{f'extra_{index + 1:03d}': dict(num_envs=extra['num_envs'],
                    control_steps=extra['control_steps'], require_full_steps=extra.get('require_full_steps',
                        extra['num_envs'] == probe['num_envs'] and extra['control_steps'] == 48))
                   for index, extra in enumerate(probe['extra_operations'])}})
    obligations['calibration'] = dict(kind='calibration', control_samples=calibration_samples,
        target_control_samples=calibration_target, suite=CALIBRATION_SUITE,
        manifest_sha256=calibration_manifest['manifest_sha256'])
    for phase in ('before', 'after'):
        obligations['diag_' + phase] = dict(kind='diagnostic', control_samples=diagnostic_cost,
            suite=DIAGNOSTIC_SUITE, manifest_sha256=diagnostic_manifest['manifest_sha256'])
    for model in ('6400', 'init', 'adapt', 'final1', 'final2', 'final3'):
        obligations['video_' + model] = dict(kind='video', control_samples=video_cost,
            suite='S1_VIDEO', manifest_sha256=diagnostic_manifest['manifest_sha256'])
    obligations['engineering'] = dict(kind='engineering', control_samples=engineering_extra_samples)
    result = dict(schema_version=11, sds_sha256=spec.SDS_SHA256, recipe=spec.RECIPE,
        stage='S1', obligations=obligations, case_catalogue=catalogue, manifest_sha256s={s: manifests[s]['manifest_sha256'] for s in suites},
        suite_control_samples=costs, original_6400_baseline=baseline,
        calibration_samples=calibration_samples, baseline_subtotal=baseline_subtotal,
        candidate_gates=candidate_gates, candidate_gate_count=11,
        candidate_finals=candidate_finals, candidate_final_count=3,
        probes=probes, engineering_extra_samples=engineering_extra_samples,
        total_control_samples=total, physics_substeps=4 * total,
        nonlearning_cap=spec.S1_ENGINEERING_CAP,
        remaining_nonlearning=spec.S1_ENGINEERING_CAP - total,
        baseline_cap=spec.S1_BASELINE_CAP, diagnostic_samples=2 * diagnostic_cost,
        video_samples=6 * video_cost, diagnostic_manifest_sha256=diagnostic_manifest['manifest_sha256'],
        admitted=(total <= spec.S1_ENGINEERING_CAP
                  and baseline_subtotal <= spec.S1_BASELINE_CAP))
    result['plan_sha256'] = sha256(result)
    return result


class AppendOnlyLedger:
    """Linux flock + fsynced hash-chain JSONL, one shared ledger across rollback.

    stage admission reserves a plan envelope. Individual physical operations
    reserve inside that envelope, so parent and child are never double charged.
    Learning operations are bounded separately. The ledger alone owns monotonous
    gate/final/recovery/episode counters; never load these from a checkpoint.
    """

    def __init__(self, path, *, run_id):
        self.path = Path(path)
        if not isinstance(run_id, str) or not run_id:
            raise ValueError('run_id required')
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked() as handle:
            events = self._read(handle)
            if not events:
                self._append(handle, events, dict(kind='header', run_id=run_id,
                    schema_version=11, sds_sha256=spec.SDS_SHA256, recipe=spec.RECIPE,
                    stage_caps=STAGE_CAPS))
            else:
                self._state(events)

    @contextmanager
    def _locked(self):
        # Lock the data inode itself; it is never renamed or replaced.
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        with os.fdopen(fd, 'r+', encoding='utf-8') as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield handle
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read(self, handle):
        handle.seek(0)
        lines = handle.readlines()
        events = []
        previous = None
        for index, line in enumerate(lines):
            if not line.endswith('\n'):
                raise ValueError('incomplete ledger tail: stop and reconcile, never overwrite')
            row = json.loads(line)
            checksum = row.pop('event_sha256')
            if row.get('sequence') != index or row.get('previous_sha256') != previous:
                raise ValueError('ledger sequence/hash chain broken')
            if checksum != sha256(row):
                raise ValueError('ledger event SHA mismatch')
            row['event_sha256'] = checksum
            previous = checksum
            events.append(row)
        return events

    def _append(self, handle, events, event):
        event = dict(event, sequence=len(events), previous_sha256=(
            events[-1]['event_sha256'] if events else None))
        event['event_sha256'] = sha256(event)
        # Validate a proposed event under the same lock before making it durable.
        self._state(events + [event])
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json(event) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
        directory_fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return event

    def _state(self, events):
        if not events or events[0].get('kind') != 'header':
            raise ValueError('ledger header missing')
        header = events[0]
        if (header.get('run_id') != self.run_id or header.get('schema_version') != 11
                or header.get('sds_sha256') != spec.SDS_SHA256
                or header.get('recipe') != spec.RECIPE or header.get('stage_caps') != STAGE_CAPS):
            raise ValueError('external ledger identity/caps mismatch')
        stages = {stage: dict(envelope=0, spent={category: 0 for category in CATEGORIES},
                             reserved={category: 0 for category in CATEGORIES},
                             progress={}, allocations={}, evidence=[], obligations={}) for stage in STAGE_CAPS}
        operations = {}
        for row in events[1:]:
            kind = row['kind']
            stage = row['stage']
            if stage not in stages:
                raise ValueError('unknown stage')
            state = stages[stage]
            if kind == 'stage_reserve':
                if state['envelope']:
                    raise ValueError('stage already admitted')
                plan = row['plan']
                if not plan.get('admitted') or plan.get('stage') != stage:
                    raise ValueError('stage plan not admitted')
                plan_payload = {k: v for k, v in plan.items() if k != 'plan_sha256'}
                if (plan.get('plan_sha256') != sha256(plan_payload)
                        or plan.get('sds_sha256') != spec.SDS_SHA256):
                    raise ValueError('stage plan SHA/contract mismatch')
                state['envelope'] = _int(plan['total_control_samples'], 'stage reservation', 1)
                if sum(duty['control_samples'] for duty in plan['obligations'].values()) != state['envelope']:
                    raise ValueError('obligations must cover the entire stage reservation')
                for name, duty in plan['obligations'].items():
                    amount = _int(duty['control_samples'], 'obligation cost')
                    state['obligations'][name] = dict(duty, limit=amount, target=duty.get('target_control_samples', amount),
                        spent=0, reserved=0, completed_upper=0, closed=False,
                        expected_cases=plan['case_catalogue'].get(duty.get('suite'), {}),
                        completed_cases={}, claimed_cases={}, completed_items={}, claimed_items={},
                        bound_checkpoint_sha256=None)
                if 'engineering' not in state['obligations']:
                    raise ValueError('explicit engineering obligation required')
            elif kind == 'stage_extend':
                if not state['envelope']:
                    raise ValueError('stage not admitted')
                amount = _int(row['samples'], 'extra reservation', 1)
                state['envelope'] += amount
                state['obligations']['engineering']['limit'] += amount
            elif kind == 'probe_fail':
                name = row['obligation_id']
                duty = state['obligations'].get(name)
                attempted = any(op['stage'] == stage and op.get('obligation_id') == name
                                for op in operations.values())
                if not duty or duty['kind'] != 'probe' or duty['closed'] or not attempted or duty['reserved']:
                    raise ValueError('failed probe must be attempted and all pending costs settled first')
                if not row.get('reason') or not row.get('evidence_sha256'):
                    raise ValueError('explicit resource failure report required')
                released = duty['limit'] - duty['spent']
                duty['limit'] = duty['spent']
                duty['closed'] = True
                duty['failed'] = True
                duty['failure_reason'] = row['reason']
                duty['failure_evidence_sha256'] = row['evidence_sha256']
                state['obligations']['engineering']['limit'] += released
            elif kind == 'probe_skip':
                name = row['obligation_id']
                duty = state['obligations'][name]
                attempted = any(op['stage'] == stage and op.get('obligation_id') == name
                                for op in operations.values())
                if duty['kind'] != 'probe' or duty['closed'] or attempted:
                    raise ValueError('only a never-attempted probe can be skipped')
                if row.get('status') != 'not_attempted' or not row.get('reason'):
                    raise ValueError('explicit not_attempted probe record required')
                released = duty['limit']
                duty['limit'] = 0
                duty['closed'] = True
                duty['skipped'] = True
                duty['skip_reason'] = row['reason']
                duty['skip_evidence_sha256'] = row.get('evidence_sha256')
                state['obligations']['engineering']['limit'] += released
            elif kind == 'obligation_retry':
                amount = _int(row['samples'], 'retry cost', 1)
                duty = state['obligations'][row['obligation_id']]
                engineering = state['obligations']['engineering']
                if duty['closed'] or row['obligation_id'] == 'engineering':
                    raise ValueError('only an open existing obligation can receive a retry allowance')
                if engineering['limit'] - engineering['spent'] - engineering['reserved'] < amount:
                    raise ValueError('insufficient engineering remainder for obligation retry')
                engineering['limit'] -= amount
                duty['limit'] += amount
            elif kind == 'obligation_invalidate':
                duty = state['obligations'][row['obligation_id']]
                if duty['closed'] or duty['reserved']:
                    raise ValueError('in-flight or closed obligation cannot be invalidated')
                duty['completed_cases'].clear()
                duty['completed_items'].clear()
                duty['completed_upper'] = 0
                duty['bound_checkpoint_sha256'] = None
            elif kind == 'obligation_close':
                name = row['obligation_id']
                duty = state['obligations'][name]
                if (name == 'engineering' or duty['closed'] or duty['reserved']
                        or (not duty.get('work_items') and duty['completed_upper'] < duty['target'])):
                    raise ValueError('cannot release an unexecuted, incomplete or in-flight obligation')
                if duty['expected_cases'] and set(duty['completed_cases']) != set(duty['expected_cases']):
                    raise ValueError('complete fixed-manifest case coverage required')
                if duty.get('work_items') and set(duty['completed_items']) != set(duty['work_items']):
                    raise ValueError('all planned physical probe operations must complete')
                released = duty['limit'] - duty['spent']
                duty['limit'] = duty['spent']
                duty['closed'] = True
                state['obligations']['engineering']['limit'] += released
            elif kind == 'reserve':
                key = row['operation_key']
                if key in operations:
                    raise ValueError('operation already exists; never execute a duplicate for free')
                category = row['category']
                if not state['envelope']:
                    raise ValueError('reserve the full stage plan before any physical operation')
                if category not in CATEGORIES:
                    raise ValueError('invalid operation category')
                n = _int(row['num_envs'], 'num_envs', 1)
                steps = _int(row['max_control_steps'], 'max_control_steps', 1)
                operation = dict(row, reserved_samples=n * steps, settled=False)
                operations[key] = operation
                state['reserved'][category] += operation['reserved_samples']
                if category != 'learning':
                    name = row.get('obligation_id')
                    if name not in state['obligations']:
                        raise ValueError('explicit frozen nonlearning obligation required')
                    duty = state['obligations'][name]
                    if duty['closed']:
                        raise ValueError('obligation already completed')
                    expected_category = ('evaluation' if duty['kind'] in (
                        'teacher_baseline', 'candidate_gates', 'candidate_finals') else 'engineering')
                    if category != expected_category:
                        raise ValueError('obligation has a different accounting category')
                    if duty.get('manifest_sha256') and row.get('case_sha256') != duty['manifest_sha256']:
                        raise ValueError('operation must bind the obligation manifest SHA')
                    if duty['kind'] in ('teacher_baseline', 'calibration') and row['checkpoint_sha256'] != spec.SOURCE_SHA256:
                        raise ValueError('teacher baseline must use the unique original 6400 SHA')
                    if duty['kind'] in ('diagnostic', 'video') and row['course'] != 'S1_EVIDENCE':
                        raise ValueError('Q008 diagnostic/video operations use S1_EVIDENCE course')
                    if name == 'video_6400' and row['checkpoint_sha256'] != spec.SOURCE_SHA256:
                        raise ValueError('6400 video must retain the unique original source SHA')
                    if duty['kind'] == 'probe' and row['course'] != 'S1_RESOURCE':
                        raise ValueError('Q005: resource probes use independent S1_RESOURCE course')
                    if duty['expected_cases']:
                        if duty['bound_checkpoint_sha256'] is None:
                            duty['bound_checkpoint_sha256'] = row['checkpoint_sha256']
                        elif duty['bound_checkpoint_sha256'] != row['checkpoint_sha256']:
                            raise ValueError('one evaluation obligation requires one frozen checkpoint SHA')
                        ids = row.get('case_sha256s')
                        if not isinstance(ids, list) or len(ids) != n or len(set(ids)) != n:
                            raise ValueError('evaluation requires exactly N unique case SHA identities')
                        if any(case not in duty['expected_cases'] for case in ids):
                            raise ValueError('evaluation case not in frozen obligation')
                        if any(case in duty['completed_cases'] or case in duty['claimed_cases'] for case in ids):
                            raise ValueError('case already completed or in flight for this obligation')
                        expected = [duty['expected_cases'][case] for case in ids]
                        if len({entry['group_sha256'] for entry in expected}) != 1:
                            raise ValueError('a physical batch cannot mix task/geometry/speed/stop groups')
                        group_count = sum(entry['group_sha256'] == expected[0]['group_sha256']
                                          for entry in duty['expected_cases'].values())
                        if group_count % n:
                            raise ValueError('evaluation sub-batches must equally divide the fixed group')
                        if steps != max(entry['timeout'] for entry in expected):
                            raise ValueError('evaluation must reserve its real full hard timeout')
                        for case in ids:
                            duty['claimed_cases'][case] = key
                    if duty.get('work_items'):
                        item = row.get('work_item_id')
                        if item not in duty['work_items'] or item in duty['completed_items'] or item in duty['claimed_items']:
                            raise ValueError('unknown, completed or in-flight probe work item')
                        expected = duty['work_items'][item]
                        if n != expected['num_envs'] or steps != expected['control_steps']:
                            raise ValueError('probe work must reserve exact planned physical upper bound')
                        duty['claimed_items'][item] = key
                    duty['reserved'] += operation['reserved_samples']
                    if duty['spent'] + duty['reserved'] > duty['limit']:
                        raise ValueError('obligation exhausted; future gates/finals remain protected')
                if category == 'learning' and stage == 'S1':
                    allocation = row.get('allocation_id')
                    if allocation not in S1_LEARNING_ALLOCATIONS:
                        raise ValueError('explicit S1 learning allocation required; no automatic retry borrowing')
                    state['allocations'][allocation] = state['allocations'].get(allocation, 0) + operation['reserved_samples']
                    if state['allocations'][allocation] > S1_LEARNING_ALLOCATIONS[allocation]:
                        raise ValueError('learning allocation exhausted; unused courses cannot be borrowed')
            elif kind == 'commit':
                key = row['operation_key']
                if key not in operations or operations[key]['stage'] != stage:
                    raise ValueError('unknown operation')
                operation = operations[key]
                if operation['settled']:
                    raise ValueError('operation already settled')
                steps = _int(row['actual_control_steps'], 'actual_control_steps')
                actual = steps * operation['num_envs']
                if actual > operation['reserved_samples']:
                    raise ValueError('actual exceeds reserved upper bound')
                if row['status'] not in ('complete', 'interrupted', 'unknown'):
                    raise ValueError('unknown settlement status')
                if row['status'] == 'unknown' and actual != operation['reserved_samples']:
                    raise ValueError('unknown operation must consume its entire bound')
                category = operation['category']
                if (category != 'learning' and row['status'] == 'complete'
                        and state['obligations'][operation['obligation_id']]['kind'] == 'probe'
                        and state['obligations'][operation['obligation_id']]['work_items'][operation['work_item_id']].get('require_full_steps')
                        and steps != operation['max_control_steps']):
                    raise ValueError('a completed probe work item must advance its full planned rollout/assessment')
                state['reserved'][category] -= operation['reserved_samples']
                state['spent'][category] += actual
                if category != 'learning':
                    duty = state['obligations'][operation['obligation_id']]
                    duty['reserved'] -= operation['reserved_samples']
                    duty['spent'] += actual
                    for case in operation.get('case_sha256s') or []:
                        del duty['claimed_cases'][case]
                        if row['status'] == 'complete':
                            duty['completed_cases'][case] = True
                    item = operation.get('work_item_id')
                    if item is not None and duty.get('work_items'):
                        del duty['claimed_items'][item]
                        if row['status'] == 'complete':
                            duty['completed_items'][item] = True
                    if row['status'] == 'complete':
                        duty['completed_upper'] += operation['reserved_samples']
                if category == 'learning' and stage == 'S1':
                    state['allocations'][operation['allocation_id']] -= operation['reserved_samples'] - actual
                operation.update(settled=True, actual_control_samples=actual,
                                 status=row['status'], physics_substeps=actual * 4)
            elif kind == 'evidence':
                if row.get('evidence_kind') not in ('gate', 'final', 'rollback', 'engineering_rerun'):
                    raise ValueError('unknown evidence kind')
                for field in ('checkpoint_sha256', 'case_sha256', 'result_sha256', 'reason'):
                    if not isinstance(row.get(field), str) or not row[field]:
                        raise ValueError('evidence checkpoint/case/result/reason required')
                state['evidence'].append(row)
            elif kind == 'progress':
                for key, value in row['counters'].items():
                    _int(value, key)
                    if value < state['progress'].get(key, 0):
                        raise ValueError('external progress cannot rewind with model rollback')
                    state['progress'][key] = value
                counters = state['progress']
                if counters.get('gate_attempts', 0) > GATE_LIMITS[stage]:
                    raise ValueError('gate attempt limit exceeded')
                if counters.get('final_suites_used', 0) > 3:
                    raise ValueError('only three pre-frozen final suites exist')
                if counters.get('final_recoveries', 0) > 2:
                    raise ValueError('final recovery limit exceeded')
            else:
                raise ValueError('unknown ledger event')
            if stage == 'S1':
                baseline_kinds = ('teacher_baseline', 'calibration')
                baseline_commitment = sum(duty['limit'] for duty in state['obligations'].values()
                    if duty['kind'] in baseline_kinds)
                for operation in operations.values():
                    if operation['stage'] != 'S1' or operation['course'] != 'S1.0':
                        continue
                    duty = state['obligations'].get(operation.get('obligation_id'))
                    if duty and duty['kind'] in baseline_kinds:
                        continue
                    baseline_commitment += (operation['actual_control_samples'] if operation['settled']
                                            else operation['reserved_samples'])
                if baseline_commitment > spec.S1_BASELINE_CAP:
                    raise ValueError('S1.0 baseline/diagnostic sub-cap exhausted; request planning decision')
                state['baseline_commitment'] = baseline_commitment
            cap = STAGE_CAPS[stage]
            used = {c: state['spent'][c] + state['reserved'][c] for c in CATEGORIES}
            nonlearning = used['engineering'] + used['evaluation']
            if (state['envelope'] > cap['nonlearning']
                    or nonlearning > state['envelope'] or used['learning'] > cap['learning']
                    or sum(used.values()) > cap['total']):
                raise ValueError('stage reservation/budget exhausted; request planning decision')
        for stage, state in stages.items():
            state['next_gate_suite'] = 'gate_A' if state['progress'].get('gate_attempts', 0) % 2 == 0 else 'gate_B'
        return dict(schema_version=11, sds_sha256=spec.SDS_SHA256, run_id=self.run_id,
                    sequence=len(events) - 1,
                    ledger_sha256=events[-1]['event_sha256'], stages=stages,
                    operations=operations,
                    total_control_samples=sum(sum(s['spent'].values()) for s in stages.values()))

    def _record(self, event):
        with self._locked() as handle:
            events = self._read(handle)
            self._state(events)
            return self._append(handle, events, event)

    def reserve_stage(self, plan):
        return self._record(dict(kind='stage_reserve', stage=plan['stage'], plan=plan))

    def extend_stage(self, stage, samples, *, reason):
        if not reason:
            raise ValueError('reason required')
        return self._record(dict(kind='stage_extend', stage=stage, samples=samples,
                                 reason=reason))

    def fail_probe_obligation(self, stage, obligation_id, *, reason, evidence_sha256):
        """Close resource failure with spent cost retained and no pass claim."""
        if not reason or not evidence_sha256:
            raise ValueError('resource failure report and reason required')
        return self._record(dict(kind='probe_fail', stage=stage, obligation_id=obligation_id,
            reason=reason, evidence_sha256=evidence_sha256))

    def skip_unattempted_probe(self, stage, obligation_id, *, reason, evidence_sha256=None):
        """Q005: release an unused resource candidate without claiming a pass."""
        if not reason:
            raise ValueError('explicit not-attempted explanation required')
        return self._record(dict(kind='probe_skip', stage=stage, obligation_id=obligation_id,
            status='not_attempted', reason=reason, evidence_sha256=evidence_sha256))

    def fund_obligation_retry(self, stage, obligation_id, samples, *, reason):
        if not reason:
            raise ValueError('explicit engineering rerun reason required')
        return self._record(dict(kind='obligation_retry', stage=stage,
            obligation_id=obligation_id, samples=samples, reason=reason))

    def invalidate_obligation(self, stage, obligation_id, *, reason):
        if not reason:
            raise ValueError('engineering invalidation reason required')
        return self._record(dict(kind='obligation_invalidate', stage=stage,
            obligation_id=obligation_id, reason=reason))

    def close_obligation(self, stage, obligation_id, *, reason):
        if not reason:
            raise ValueError('obligation completion evidence/reason required')
        return self._record(dict(kind='obligation_close', stage=stage,
            obligation_id=obligation_id, reason=reason))

    def reserve(self, *, segment_id, operation_id, stage, course, category, num_envs,
                max_control_steps, checkpoint_sha256, reason, case_sha256=None, allocation_id=None, obligation_id=None, case_sha256s=None, work_item_id=None):
        if not all(isinstance(v, str) and v for v in
                   (segment_id, operation_id, course, checkpoint_sha256, reason)):
            raise ValueError('segment/operation/course/checkpoint/reason required')
        if category == 'engineering' and obligation_id is None:
            obligation_id = 'engineering'
        if category == 'learning' and allocation_id is None and course == 'S1.1':
            allocation_id = 'S1.1'
        key = canonical_json([self.run_id, segment_id, operation_id])
        self._record(dict(kind='reserve', operation_key=key, run_id=self.run_id,
            segment_id=segment_id, operation_id=operation_id, stage=stage, course=course,
            category=category, num_envs=num_envs, max_control_steps=max_control_steps,
            checkpoint_sha256=checkpoint_sha256, case_sha256=case_sha256, reason=reason,
            allocation_id=allocation_id, obligation_id=obligation_id,
            case_sha256s=case_sha256s, work_item_id=work_item_id))
        return key

    def commit(self, operation_key, *, actual_control_steps=None, status='complete'):
        with self._locked() as handle:
            events = self._read(handle)
            state = self._state(events)
            if operation_key not in state['operations']:
                raise ValueError('unknown operation')
            operation = state['operations'][operation_key]
            if status == 'unknown':
                if actual_control_steps is not None:
                    raise ValueError('unknown execution cannot supply a known step count')
                actual_control_steps = operation['max_control_steps']
            return self._append(handle, events, dict(kind='commit',
                stage=operation['stage'], operation_key=operation_key,
                actual_control_steps=actual_control_steps, status=status))

    def record_evidence(self, *, stage, evidence_kind, checkpoint_sha256,
                        case_sha256, result_sha256, reason, **details):
        return self._record(dict(kind='evidence', stage=stage, evidence_kind=evidence_kind,
            checkpoint_sha256=checkpoint_sha256, case_sha256=case_sha256,
            result_sha256=result_sha256, reason=reason, details=details))

    def settle_pending_unknown(self, *, segment_id, reason):
        """Explicit crash recovery: conservatively settle an abandoned segment."""
        if not segment_id or not reason:
            raise ValueError('abandoned segment and recovery reason required')
        with self._locked() as handle:
            events = self._read(handle)
            state = self._state(events)
            results = []
            for key, operation in state['operations'].items():
                if operation['segment_id'] != segment_id or operation['settled']:
                    continue
                row = self._append(handle, events, dict(kind='commit',
                    stage=operation['stage'], operation_key=key,
                    actual_control_steps=operation['max_control_steps'], status='unknown',
                    reason=reason))
                events.append(row)
                results.append(key)
            return results

    def advance_state(self, stage, **counters):
        if not counters:
            raise ValueError('progress counters required')
        return self._record(dict(kind='progress', stage=stage, counters=counters))

    def snapshot(self):
        with self._locked() as handle:
            return self._state(self._read(handle))

    def checkpoint_metadata(self):
        state = self.snapshot()
        return dict(schema_version=11, sds_sha256=spec.SDS_SHA256,
                    external_ledger_path=str(self.path.resolve()),
                    external_ledger_sequence=state['sequence'],
                    external_ledger_sha256=state['ledger_sha256'],
                    cumulative_control_samples=state['total_control_samples'],
                    stages=state['stages'])

    def validate_checkpoint(self, metadata):
        """Old checkpoints are allowed; authoritative counters stay external."""
        with self._locked() as handle:
            events = self._read(handle)
            state = self._state(events)
            sequence = _int(metadata['external_ledger_sequence'], 'ledger sequence')
            if (metadata.get('schema_version') != 11
                    or metadata.get('sds_sha256') != spec.SDS_SHA256
                    or metadata.get('external_ledger_path') != str(self.path.resolve())
                    or sequence >= len(events)
                    or events[sequence]['event_sha256'] != metadata['external_ledger_sha256']
                    or metadata['cumulative_control_samples'] > state['total_control_samples']):
                raise ValueError('checkpoint/external ledger contract mismatch')
            return state
