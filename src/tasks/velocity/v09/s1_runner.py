"""S1-only sequential distillation and fail-closed behavior scheduling.

Pure scheduling is CPU testable. Real physics is constructed only after all
independent source, baseline, calibration, resource and code admissions pass.
The external journal and ledger remain authoritative across model rollback.
"""
from __future__ import annotations

from copy import deepcopy
import gc
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import uuid

from . import spec
from .cases import (FINAL_SUITES, GATE_SUITES, TRAINING_ROLES, allocate_training_roles,
                    canonical_json, load_manifest, sha256, training_case)


class PlanningRequired(RuntimeError):
    """A prescribed cap/evidence boundary cannot be repaired by changing recipe."""


SEGMENTS = ('adapt', 'main', 'retry_1', 'retry_2')
CAPS = {'adapt': 20_000_000, 'main': 630_000_000,
        'retry_1': 100_000_000, 'retry_2': 100_000_000}
ALLOCATIONS = {'adapt': 'S1.1', 'main': 'S1.2_main',
               'retry_1': 'S1.2_retry_1', 'retry_2': 'S1.2_retry_2'}
SLOTS = tuple([(i + 1, 'main', amount) for i, amount in enumerate(
    (100_000_000, 200_000_000, 300_000_000, 400_000_000, 500_000_000, 600_000_000, 630_000_000))]
    + [(8, 'retry_1', 50_000_000), (9, 'retry_1', 100_000_000),
       (10, 'retry_2', 50_000_000), (11, 'retry_2', 100_000_000)])


@dataclass
class S1Schedule:
    num_envs: int
    segment: str = 'adapt'
    spent: dict = field(default_factory=lambda: {name: 0 for name in SEGMENTS})
    complete_rollouts: int = 0
    used_slots: list = field(default_factory=list)
    cancelled_slots: list = field(default_factory=list)
    gates: list = field(default_factory=list)
    pending_confirmation: dict | None = None
    final_pending: bool = False
    final_suites_used: int = 0
    final_outcomes: list = field(default_factory=list)
    recoveries: int = 0
    accepted: bool = False
    blocked: str | None = None
    engineering_adapt_passed: bool = False

    def __post_init__(self):
        if self.num_envs not in spec.ENV_CANDIDATES or self.segment not in SEGMENTS:
            raise ValueError('invalid S1 schedule environment count/segment')
        if set(self.spent) != set(SEGMENTS) or any(not isinstance(v, int) or v < 0 for v in self.spent.values()):
            raise ValueError('invalid physical learning sample counters')
        if any(self.spent[k] > CAPS[k] for k in SEGMENTS):
            raise ValueError('course physical samples exceed their separate caps')
        if len(set(self.used_slots)) != len(self.used_slots) or set(self.used_slots) & set(self.cancelled_slots):
            raise ValueError('repeated/cancelled gate slot')
        if len(self.gates) > 11 or not 0 <= self.final_suites_used <= 3 or not 0 <= self.recoveries <= 2:
            raise ValueError('gate/final/recovery cap exceeded')

    @property
    def course(self):
        return 'S1.1' if self.segment == 'adapt' else 'S1.2'

    @property
    def learning_samples(self):
        return sum(self.spent.values())

    @property
    def rollout_samples(self):
        return self.num_envs * 48

    @property
    def effective_cap(self):
        return CAPS[self.segment] // self.rollout_samples * self.rollout_samples

    @property
    def next_suite(self):
        return GATE_SUITES[len(self.gates) % 2]

    def available_slots(self):
        return [row for row in SLOTS if row[0] not in self.used_slots + self.cancelled_slots]

    def due_slot(self):
        if self.segment == 'adapt' or self.accepted or self.final_pending:
            return None
        if self.pending_confirmation:
            following = [row for row in self.available_slots()
                         if row[0] > self.pending_confirmation['original_slot']]
            if not following:
                raise PlanningRequired('no remaining scheduled gate quota for same-SHA A/B confirmation')
            return following[0][0]
        available = [row for row in self.available_slots() if row[1] == self.segment]
        # A lone first pass could never acquire its required same-SHA partner.
        if len(self.available_slots()) < 2:
            raise PlanningRequired('only one gate quota remains for a new candidate')
        current = self.spent[self.segment]
        for slot, _, threshold in available:
            boundary = min(math.ceil(threshold / self.rollout_samples) * self.rollout_samples,
                           self.effective_cap)
            if current >= boundary or not self.can_collect(check_gate=False):
                return slot
        return None

    def can_collect(self, *, check_gate=True):
        if self.accepted or self.blocked or self.final_pending or self.pending_confirmation:
            return False
        if self.spent[self.segment] + self.rollout_samples > CAPS[self.segment]:
            return False
        return not check_gate or self.due_slot() is None

    def consume_rollout(self, control_steps=48, *, complete=True):
        if self.pending_confirmation or self.final_pending or self.accepted:
            raise ValueError('learning is frozen during gate confirmation/final acceptance')
        if not isinstance(control_steps, int) or not 0 <= control_steps <= 48:
            raise ValueError('actual physical rollout must lie within its reserved 48 intervals')
        amount = self.num_envs * control_steps
        if self.spent[self.segment] + amount > CAPS[self.segment]:
            raise PlanningRequired('course cap cannot borrow unused main/retry allocations')
        if complete and control_steps != 48:
            raise ValueError('incomplete rollout cannot enter the distillation optimizer')
        self.spent[self.segment] += amount
        self.complete_rollouts += int(complete)
        if not complete:
            self.blocked = 'interrupted rollout billed; resume an explicitly selected checkpoint with new episodes'

    def finish_adapt(self, *, finite, migration_valid, contacts_valid, comparison_report_sha256):
        if self.segment != 'adapt' or self.can_collect(check_gate=False):
            raise ValueError('S1.1 requires its maximum affordable complete-rollout count')
        if not all((finite, migration_valid, contacts_valid, comparison_report_sha256)):
            raise PlanningRequired('S1.1 engineering/numeric and before-after evidence incomplete')
        self.engineering_adapt_passed = True
        self.segment = 'main'

    def start_gate(self, candidate_sha256):
        slot = self.due_slot()
        if slot is None:
            raise ValueError('no prescribed gate is due')
        if self.pending_confirmation and self.pending_confirmation['candidate_sha256'] != candidate_sha256:
            raise ValueError('A/B confirmation must retain the exact frozen candidate SHA')
        if len(self.gates) >= 11:
            raise PlanningRequired('S1 maximum eleven full gates exhausted')
        row = dict(attempt=len(self.gates) + 1, original_slot=slot,
            original_segment=SLOTS[slot - 1][1], original_threshold=SLOTS[slot - 1][2],
            suite=self.next_suite, candidate_sha256=candidate_sha256,
            purpose='confirmation' if self.pending_confirmation else 'scheduled', status='running')
        self.used_slots.append(slot)
        self.gates.append(row)
        return deepcopy(row)

    def finish_gate(self, *, candidate_sha256, passed, engineering_valid, report_sha256):
        if not self.gates or self.gates[-1]['status'] != 'running':
            raise ValueError('there is no running gate')
        row = self.gates[-1]
        if row['candidate_sha256'] != candidate_sha256 or not report_sha256:
            raise ValueError('gate evidence identity mismatch')
        row.update(report_sha256=report_sha256, engineering_valid=bool(engineering_valid),
                   passed=bool(passed), status='complete' if engineering_valid else 'engineering_invalid')
        if not engineering_valid:
            self.blocked = 'whole gate engineering invalid; preserve fee and request a whole-suite rerun decision'
            self.pending_confirmation = None
            return
        if passed and self.pending_confirmation:
            previous = self.pending_confirmation
            if previous['candidate_sha256'] != candidate_sha256 or previous['suite'] == row['suite']:
                raise ValueError('two adjacent A/B passes must use one candidate SHA')
            self.pending_confirmation = None
            self.final_pending = True
            # Q007: close the accepted learning segment and cancel all of its
            # remaining unused thresholds. Confirmation can consume a future
            # retry slot without enabling that retry's learning allocation.
            self.cancelled_slots.extend(row[0] for row in self.available_slots()
                                        if row[1] == self.segment)
        elif passed:
            self.pending_confirmation = deepcopy(row)
        else:
            self.pending_confirmation = None
            if not self.can_collect(check_gate=False):
                self.blocked = 'gate failed at learning cap; planning decision required'

    def begin_final(self):
        if not self.final_pending or self.final_suites_used >= 3:
            raise PlanningRequired('a next frozen final suite requires same-SHA A/B and available final quota')
        return FINAL_SUITES[self.final_suites_used]

    def finish_final(self, *, candidate_sha256, passed, engineering_valid, report_sha256,
                     planning_required=False):
        suite = self.begin_final()
        if self.gates[-1]['candidate_sha256'] != candidate_sha256 or not report_sha256:
            raise ValueError('final candidate/report identity differs from confirmed gates')
        self.final_outcomes.append(dict(suite=suite, candidate_sha256=candidate_sha256,
            report_sha256=report_sha256, passed=bool(passed),
            engineering_valid=bool(engineering_valid), planning_required=bool(planning_required)))
        if not engineering_valid:
            self.blocked = 'engineering-invalid final suite must be rerun in full without consuming behavior quota'
            return
        self.final_suites_used += 1
        self.final_pending = False
        if planning_required:
            self.blocked = 'final reference metric is unjudgeable (Q001); opened suite retained, planning decision required'
            return
        if passed:
            self.accepted = True
            return
        if self.recoveries >= 2:
            self.blocked = 'all three final suites exhausted; planning decision required'
            return
        # Early completion leaves old learning allocations unused; never borrow.
        future_segment = f'retry_{self.recoveries + 1}'
        unused_old = [row[0] for row in self.available_slots()
                      if SEGMENTS.index(row[1]) < SEGMENTS.index(future_segment)]
        if unused_old:
            raise ValueError('Q007 ended-segment gate thresholds were not cancelled')
        self.recoveries += 1
        self.segment = future_segment
        self.pending_confirmation = None

    def state_dict(self):
        return asdict(self)


class ScheduleJournal:
    """Append-only complete scheduling receipts; caller holds the run lock."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self):
        rows, previous = [], None
        if not self.path.exists():
            return rows
        for line in self.path.read_text().splitlines(keepends=True):
            if not line.endswith('\n'):
                raise PlanningRequired('incomplete schedule journal tail; reconcile before resume')
            row = json.loads(line)
            if row.get('schema_version') != 11 or row.get('sds_sha256') != spec.SDS_SHA256:
                raise ValueError('schedule journal schema/SDS identity mismatch')
            digest = row.pop('event_sha256')
            if row.get('sequence') != len(rows) or row.get('previous_sha256') != previous or sha256(row) != digest:
                raise ValueError('schedule journal integrity mismatch')
            row['event_sha256'] = digest
            rows.append(row)
            previous = digest
        return rows

    def append(self, schedule, *, event, **details):
        rows = self.read()
        row = dict(schema_version=11, sds_sha256=spec.SDS_SHA256, sequence=len(rows),
            previous_sha256=rows[-1]['event_sha256'] if rows else None,
            event=event, schedule=schedule.state_dict(), details=details)
        row['event_sha256'] = sha256(row)
        with self.path.open('a') as stream:
            stream.write(canonical_json(row) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        return row


def _read_json(path):
    return json.loads(Path(path).read_text())


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def validate_learning_approval(approval, *, run_dir, run_id, code_sha256):
    """An S1.0 engineering entry cannot authorize the automatic two-course learner."""
    if not isinstance(approval, dict):
        raise PlanningRequired('explicit independent learning approval JSON required')
    if (approval.get('approval_type') == 'independent_engineering_entry_only'
            or approval.get('course') == 'S1.0'):
        raise PlanningRequired('S1.0 engineering-only approval does not authorize S1.1/S1.2 learning')
    scope = approval.get('scope')
    required = {'S1.1_learning', 'S1.2_learning'}
    if (not isinstance(scope, list) or not all(isinstance(item, str) for item in scope)
            or not required <= set(scope)):
        raise PlanningRequired('automatic S1 driver requires explicit S1.1_learning and S1.2_learning scope')
    if (approval.get('approved') is not True or approval.get('stage') != 'S1'
            or approval.get('sds_sha256') != spec.SDS_SHA256
            or approval.get('code_sha256') != code_sha256
            or approval.get('source_sha256') != spec.SOURCE_SHA256):
        raise PlanningRequired('learning approval source/SDS/code identity mismatch')
    approved_path = approval.get('run_dir')
    if (not isinstance(approved_path, str) or not approved_path
            or approval.get('run_id') != run_id
            or Path(approved_path).resolve() != Path(run_dir).resolve()):
        raise PlanningRequired('learning approval belongs to a different run_id/run_dir')
    return True


def validate_admissions(*, run_dir, resource_path, baseline_path, calibration_path, code_sha256, ledger):
    """Validate only final-cache index metadata until its formal acceptance turn."""
    baseline, resource, calibration = map(_read_json, (baseline_path, resource_path, calibration_path))
    for name, record in (('baseline', baseline), ('resource', resource), ('calibration', calibration)):
        if record.get('sds_sha256') != spec.SDS_SHA256 or record.get('source_sha256') != spec.SOURCE_SHA256:
            raise ValueError(f'{name} source/SDS admission mismatch')
        if record.get('code_sha256') != code_sha256:
            raise ValueError(f'{name} independently admitted code SHA differs')
    if baseline.get('schema') != 11 or baseline.get('baseline_complete') is not True:
        raise PlanningRequired('all original baseline caches must be admitted before learning')
    if (baseline.get('calibration_path') != str(Path(calibration_path).resolve())
            or baseline.get('value_scale') != calibration.get('value_scale')
            or baseline.get('calibration_manifest_sha256') != calibration.get('manifest_sha256')
            or baseline.get('calibration_sha256') != file_sha256(calibration_path)):
        raise ValueError('baseline/calibration contract mismatch')
    if (not isinstance(calibration.get('value_scale'), (int, float))
            or not math.isfinite(calibration['value_scale']) or calibration['value_scale'] < 1
            or calibration.get('method') != 'population_std_max_1'
            or not 0 < calibration.get('physical_control_samples', 0) <= 1_000_000):
        raise ValueError('independent frozen value calibration is incomplete')
    caches = baseline.get('teacher_cache', {})
    if set(caches) != set(GATE_SUITES + FINAL_SUITES):
        raise PlanningRequired('all A/B and three sealed final teacher cache indexes required')
    obligations = ledger.snapshot()['stages']['S1']['obligations']
    for suite, entry in caches.items():
        if entry.get('checkpoint_sha256') != spec.SOURCE_SHA256 or len(entry.get('manifest_sha256', '')) != 64:
            raise ValueError('teacher cache identity mismatch')
        if not obligations.get('teacher_' + suite, {}).get('closed'):
            raise PlanningRequired('original teacher suite not externally completed')
        if not Path(entry['episodes_path']).is_file() or not Path(entry['summary_path']).is_file():
            raise PlanningRequired('teacher cache files missing')
    if not obligations.get('calibration', {}).get('closed'):
        raise PlanningRequired('calibration obligation must be fully completed')
    if resource.get('report_sha256') != sha256({k: v for k, v in resource.items() if k != 'report_sha256'}):
        raise ValueError('resource admission report SHA mismatch')
    if (not resource.get('admitted') or resource.get('selected_num_envs') not in spec.ENV_CANDIDATES
            or resource.get('calibration', {}).get('value_scale') != calibration['value_scale']
            or resource.get('calibration', {}).get('calibration_file_sha256') != file_sha256(calibration_path)):
        raise PlanningRequired('stable maximum resource candidate and matching calibration required')
    n = resource['selected_num_envs']
    duty = obligations.get(f'probe_{n}', {})
    if not duty.get('closed') or duty.get('skipped') or duty.get('failed'):
        raise PlanningRequired('selected resource probe did not physically complete')
    # Full resource qualification remains the independently tested module's job.
    from .resources import qualifies
    selected = [row for row in resource['candidates'] if row['num_envs'] == n]
    if len(selected) != 1 or not qualifies(selected[0]):
        raise PlanningRequired('resource peak memory/workload/restore conditions failed')
    return dict(num_envs=n, value_scale=calibration['value_scale'], baseline=baseline,
                calibration=calibration, resource=resource)


def compare_suite(manifest, teacher_records, candidate_records):
    """Reconstruct real group identity from immutable case SHA, never case_id alone."""
    from .acceptance import compare_flat_group, compare_stair_group
    expected = {row['case_sha256']: row for row in manifest['cases']}
    by_policy = []
    for records in (teacher_records, candidate_records):
        mapping = {row['case_sha256']: row for row in records}
        if len(mapping) != len(records) or set(mapping) != set(expected):
            raise ValueError('suite requires every fixed case exactly once')
        for identity, row in mapping.items():
            if row.get('manifest_sha256') != manifest['manifest_sha256']:
                raise ValueError('evaluation record manifest identity mismatch')
        by_policy.append(mapping)
    groups = {}
    for identity, case in expected.items():
        group_key = canonical_json([case[key] for key in ('task', 'stair_count', 'geometry',
            'speed', 'stop_mode', 'force_family')])
        groups.setdefault(group_key, []).append(identity)
    if len(groups) != 75:
        raise ValueError('S1 behavior evidence must contain all 75 groups')
    results = []
    for key, identities in groups.items():
        rows = [[dict(policy[identity], case_id=expected[identity]['case_id'])
                 for identity in identities] for policy in by_policy]
        case = expected[identities[0]]
        result = (compare_stair_group(*rows) if case['task'] == 'stairs' else
                  compare_flat_group(*rows, stop=case['task'] == 'flat_stop'))
        results.append(dict(group_identity=json.loads(key), case_sha256s=identities, **result))
    return dict(suite=manifest['suite'], manifest_sha256=manifest['manifest_sha256'],
        groups=results, fixed_denominator=len(expected), group_count=75,
        engineering_valid=not any(row['engineering_invalid'] for row in results),
        planning_required=any(row.get('planning_required') for row in results),
        passed=all(row['passed'] for row in results))


class SubprocessEvaluator:
    def __init__(self, *, run_dir, run_id, device, code_sha256, python, teacher_cache):
        self.run_dir, self.run_id, self.device = Path(run_dir), run_id, device
        self.code_sha256, self.python, self.teacher_cache = code_sha256, python, teacher_cache

    def __call__(self, *, suite, candidate_path, candidate_sha256, attempt, segment_id):
        final = suite in FINAL_SUITES
        obligation = 'candidate_' + suite if final else f'candidate_gate_{attempt:02d}'
        command = [self.python, str(spec.PROJECT_ROOT / 'scripts/evaluate_a2_v09_s1.py'),
            '--run-dir', str(self.run_dir), '--run-id', self.run_id, '--suite', suite,
            '--policy', 'student', '--checkpoint', str(candidate_path),
            '--checkpoint-sha256', candidate_sha256, '--code-sha256', self.code_sha256,
            '--obligation-id', obligation, '--segment-id', segment_id,
            '--batch-size', '100' if final else '20', '--device', self.device, '--execute']
        subprocess.run(command, cwd=spec.PROJECT_ROOT, check=True)
        # This is the only final-cache unseal point, after actual same-SHA A/B.
        manifest = load_manifest(self.run_dir / 'cases' / f'{suite}.json',
                                 purpose='acceptance' if final else 'gate')
        entry = self.teacher_cache[suite]
        if file_sha256(entry['episodes_path']) != entry['episodes_file_sha256']:
            raise ValueError('sealed teacher behavior cache SHA mismatch')
        teacher = [json.loads(line) for line in Path(entry['episodes_path']).read_text().splitlines()]
        output = self.run_dir / 'evidence' / segment_id
        candidate = [json.loads(line) for line in (output / 'episodes.jsonl').read_text().splitlines()]
        if any(row.get('checkpoint_sha256') != spec.SOURCE_SHA256 for row in teacher):
            raise ValueError('sealed teacher record source mismatch')
        if any(row.get('checkpoint_sha256') != candidate_sha256 for row in candidate):
            raise ValueError('candidate suite mixed checkpoint identities')
        result = compare_suite(manifest, teacher, candidate)
        result.update(candidate_sha256=candidate_sha256, code_sha256=self.code_sha256,
                      teacher_checkpoint_sha256=spec.SOURCE_SHA256)
        result['report_sha256'] = sha256(result)
        with (output / 'paired_report.json').open('x') as stream:
            stream.write(canonical_json(result) + '\n')
        return result


class LedgerEpisodeIds:
    """Reserve monotonic role ID blocks externally before giving IDs to physics.

    Crash recovery skips unused claimed IDs. It never replays an old episode
    because a checkpoint or the number of slots changed. A run lock prevents
    concurrent formal learners from competing for these role streams.
    """
    def __init__(self, ledger, *, block_size=1024):
        self.ledger, self.block_size = ledger, block_size
        self.next_by_role, self.ends = {}, {}

    def reserve(self, role):
        if role not in TRAINING_ROLES:
            raise ValueError('unknown training role')
        if self.next_by_role.get(role, 0) >= self.ends.get(role, 0):
            counter = 'episode_' + role
            current = self.ledger.snapshot()['stages']['S1']['progress'].get(counter, 0)
            self.ledger.advance_state('S1', **{counter: current + self.block_size})
            self.next_by_role[role], self.ends[role] = current, current + self.block_size
        result = self.next_by_role[role]
        self.next_by_role[role] += 1
        return result

    def state_dict(self):
        return dict(next_by_role=dict(self.next_by_role), claimed_ends=dict(self.ends))


def grouped_distillation_losses(rollout, student, *, value_scale):
    """Same physical pre-action labels, reported per task/lesson and speed."""
    import torch
    from ..v07 import spec as original
    metadata = rollout.get('group_metadata')
    if not metadata or set(metadata) != {'tile_ids', 'flat_stop', 'cruise_speed'}:
        raise ValueError('same-state task/geometry/speed audit metadata required')
    observations = rollout['observations']
    values = {key: value.flatten(0, 1) for key, value in observations.items()}
    teacher_mean = rollout['teacher_mean'].flatten(0, 1)
    teacher_value = rollout['teacher_value'].flatten(0, 1)
    count = teacher_mean.shape[0]
    errors = []
    with torch.no_grad():
        for start in range(0, count, 512):
            obs = {key: value[start:start + 512] for key, value in values.items()}
            errors.append(torch.stack(((student(obs) - teacher_mean[start:start + 512]).square().mean(-1),
                ((student.evaluate(obs) - teacher_value[start:start + 512]) / value_scale).square().flatten()), -1))
    errors = torch.cat(errors).cpu()
    tile = metadata['tile_ids'].flatten().cpu().tolist()
    stop = metadata['flat_stop'].flatten().cpu().tolist()
    speed = metadata['cruise_speed'].flatten().cpu().tolist()
    groups = {}
    for index, (tile_id, stopping, command) in enumerate(zip(tile, stop, speed)):
        exact_speed = min(spec.SPEEDS, key=lambda value: abs(value - command))
        if abs(exact_speed - command) > 1e-5:
            raise ValueError('rollout loss audit has an unsupported cruise speed')
        task = ('flat_stop' if stopping else 'flat_walk') if tile_id == 0 else 'stairs'
        lesson = original.terrain_spec(int(tile_id)).lesson_id
        key = canonical_json([task, lesson, exact_speed])
        row = groups.setdefault(key, dict(task=task, lesson=lesson, speed=exact_speed,
                                         samples=0, action_sum=0., value_sum=0.))
        row['samples'] += 1
        row['action_sum'] += float(errors[index, 0])
        row['value_sum'] += float(errors[index, 1])
    return [dict(task=row['task'], lesson=row['lesson'], speed=row['speed'], samples=row['samples'],
        action_loss=row['action_sum'] / row['samples'],
        value_loss=row['value_sum'] / row['samples']) for row in groups.values()]


def validate_schedule_ledger(schedule, state):
    """Unknown/in-flight charges cannot disappear behind an older journal."""
    stage = state['stages']['S1']
    if any(op['category'] == 'learning' and not op['settled']
           for op in state['operations'].values()):
        raise PlanningRequired('unsettled learning operation: conservatively settle/reconcile before resume')
    for segment, allocation in ALLOCATIONS.items():
        if stage['allocations'].get(allocation, 0) != schedule.spent[segment]:
            raise PlanningRequired('external allocation consumption differs from journal; reconcile without rewind')
    if stage['spent']['learning'] != schedule.learning_samples:
        raise PlanningRequired('external physical learning total differs from schedule')
    expected = {'gate_attempts': len(schedule.gates), 'final_suites_used': schedule.final_suites_used,
                'final_recoveries': schedule.recoveries}
    if any(stage['progress'].get(key, 0) != value for key, value in expected.items()):
        raise PlanningRequired('external gate/final/recovery counters differ from journal; reconcile')
    return True


def validate_final_unseal(*, run_dir, ledger, suite, candidate_sha256):
    """Read-only authority check BEFORE a student process opens final inputs."""
    if suite not in FINAL_SUITES:
        raise ValueError('a frozen final suite is required')
    rows = ScheduleJournal(Path(run_dir) / 's1_schedule.jsonl').read()
    if not rows:
        raise PlanningRequired('final suite remains sealed without authoritative same-SHA A/B history')
    schedule = S1Schedule(**rows[-1]['schedule'])
    validate_schedule_ledger(schedule, ledger.snapshot())
    if (schedule.blocked or schedule.accepted or not schedule.final_pending
            or schedule.pending_confirmation is not None or len(schedule.gates) < 2
            or schedule.begin_final() != suite):
        raise PlanningRequired('final suite remains sealed until this candidate qualifies for this next set')
    pair = schedule.gates[-2:]
    if (pair[1]['attempt'] != pair[0]['attempt'] + 1
            or {row['suite'] for row in pair} != set(GATE_SUITES)
            or any(row.get('status') != 'complete' or row.get('passed') is not True
                   or row.get('engineering_valid') is not True
                   or row.get('candidate_sha256') != candidate_sha256
                   or not row.get('report_sha256') for row in pair)):
        raise PlanningRequired('final suite remains sealed: missing adjacent complete same-SHA A/B evidence')
    return dict(suite=suite, candidate_sha256=candidate_sha256,
        gate_attempts=[row['attempt'] for row in pair],
        gate_report_sha256s=[row['report_sha256'] for row in pair],
        journal_event_sha256=rows[-1]['event_sha256'])


def training_state_fingerprint(model, optimizer):
    """Value digest independent of CPU/CUDA location, including every Adam state."""
    import torch
    digest = hashlib.sha256()
    def visit(value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(canonical_json([str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(value, dict):
            digest.update(b'{')
            for key in sorted(value, key=str):
                digest.update(str(key).encode())
                visit(value[key])
            digest.update(b'}')
        elif isinstance(value, (list, tuple)):
            digest.update(b'[')
            for item in value:
                visit(item)
            digest.update(b']')
        else:
            digest.update(canonical_json(value).encode())
    visit(model.state_dict())
    visit(optimizer.state_dict())
    return digest.hexdigest()


def move_optimizer_state(optimizer, *, device=None, restore_devices=None):
    """Offload Adam tensors without replacing its object or parameter mapping."""
    import torch
    original_devices = {}
    def move(value, path):
        if isinstance(value, torch.Tensor):
            original_devices[path] = str(value.device)
            target = restore_devices[path] if restore_devices is not None else device
            return value.to(target)
        if isinstance(value, dict):
            return {key: move(item, path + (str(key),)) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item, path + (str(i),)) for i, item in enumerate(value)]
        if isinstance(value, tuple):
            return tuple(move(item, path + (str(i),)) for i, item in enumerate(value))
        return value
    for index, (parameter, state) in enumerate(list(optimizer.state.items())):
        optimizer.state[parameter] = move(state, (str(index),))
    return original_devices


class RealS1Trainer:
    """The formal model is always a fresh 6400 transfer or explicit v09 resume."""
    def __init__(self, *, run_dir, run_id, segment_id, device, code_sha256, admissions,
                 ledger, journal, evidence_hooks, resume=None, python=sys.executable):
        import torch
        from .model import Frozen6400Teacher, V09ActorCritic, state_sha256
        from .checkpoint import load_checkpoint
        self.run_dir, self.run_id, self.segment_id = Path(run_dir), run_id, segment_id
        self.device, self.code_sha256, self.ledger, self.journal = device, code_sha256, ledger, journal
        self.admissions = admissions
        if evidence_hooks is None:
            raise PlanningRequired('Q008 physical before/after and video evidence hooks required')
        self.evidence_hooks = evidence_hooks
        self.before_diagnostic = None
        self.evidence_session_state = None
        self._iteration_in_progress = False
        self.pending_environment_rng = None
        rows = journal.read()
        if rows and resume is None:
            raise PlanningRequired('existing formal schedule requires an explicit resume checkpoint')
        self.schedule = S1Schedule(**rows[-1]['schedule']) if rows else S1Schedule(admissions['num_envs'])
        if self.schedule.num_envs != admissions['num_envs']:
            raise PlanningRequired('N change requires a separately evidenced resource transition')
        validate_schedule_ledger(self.schedule, ledger.snapshot())
        self.last_saved_checkpoint_sha = file_sha256(resume) if resume is not None else None
        if resume is None:
            import numpy as np
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
        self.teacher = Frozen6400Teacher.from_6400_checkpoint(device=device)
        self.student = V09ActorCritic.migrate_6400_checkpoint(device=device)
        self.optimizer = torch.optim.Adam(self.student.optimizer_parameters(), lr=3e-5,
                                         betas=(.9, .999), eps=1e-8, weight_decay=0.)
        self.generator = torch.Generator(device=device).manual_seed(42)
        self.candidate_path = None
        self.candidate_sha256 = None
        if resume is not None:
            payload = load_checkpoint(resume, device='cpu')
            metadata = payload['metadata']
            if metadata.get('code_sha256') != code_sha256:
                raise ValueError('resume code SHA mismatch')
            self.ledger.validate_checkpoint(metadata['external_ledger'])
            if metadata['value_scale'] != admissions['value_scale']:
                raise ValueError('frozen calibration scale differs on resume')
            self.student.load_state_dict(payload['model'], strict=True)
            self.optimizer.load_state_dict(payload['optimizer'])
            self.generator.set_state(metadata['shuffle_generator'])
            random.setstate(metadata['random_state']['python'])
            import numpy as np
            np.random.set_state(metadata['random_state']['numpy'])
            self.pending_environment_rng = metadata['random_state'].get('environment')
            torch.set_rng_state(payload['torch_rng'])
            if payload['cuda_rng'] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(payload['cuda_rng'])
            # The schedule/fees never come from the older model checkpoint.
            if self.schedule.blocked or any(g['status'] == 'running' for g in self.schedule.gates):
                raise PlanningRequired('interrupted gate/rollout requires an explicit reconciliation decision')
            if self.schedule.pending_confirmation or self.schedule.final_pending:
                if file_sha256(resume) != self.schedule.gates[-1]['candidate_sha256']:
                    raise ValueError('frozen candidate must resume the exact gate checkpoint file')
                self.candidate_path, self.candidate_sha256 = Path(resume), file_sha256(resume)
        elif rows:
            raise ValueError('cannot reset model with an existing external schedule')
        self.student.set_s1_course(self.schedule.course)
        self.frozen = state_sha256({k: v for k, v in self.student.state_dict().items()
                                   if 'normalizer' in k or k == 'std'})
        self.ids = LedgerEpisodeIds(ledger)
        counts = allocate_training_roles(self.schedule.num_envs)
        self.roles = [role for role in TRAINING_ROLES for _ in range(counts[role])]
        self.env = None
        self.buffers = [None, None]
        self.loss_reports = []
        self.latest = sorted((self.run_dir / 'checkpoints').glob('*_latest_*.pt'))
        self.evaluator = SubprocessEvaluator(run_dir=run_dir, run_id=run_id, device=device,
            code_sha256=code_sha256, python=python, teacher_cache=admissions['baseline']['teacher_cache'])
        self._record('initialized_new_physical_episodes', resume=str(resume) if resume else None)

    def _record(self, event, **details):
        self.journal.append(self.schedule, event=event, **details)
        counters = dict(gate_attempts=len(self.schedule.gates), final_suites_used=self.schedule.final_suites_used,
                        final_recoveries=self.schedule.recoveries,
                        learning_samples=self.schedule.learning_samples)
        self.ledger.advance_state('S1', **counters)

    def _new_env(self):
        from .env import V09S1Env
        progress = self.ledger.snapshot()['stages']['S1']['progress']
        initial = [training_case(role, progress.get('episode_' + role, 0))
                   for role in self.roles]
        env = V09S1Env(initial, device=self.device, seed=42, roles=self.roles,
                       case_provider=training_case, episode_ids=self.ids, course=self.schedule.course)
        if self.pending_environment_rng is not None:
            env.rng.set_state(self.pending_environment_rng)
        return env

    def _save(self, label, *, keep_latest=False):
        import numpy as np
        from .checkpoint import save_checkpoint
        schedule = self.schedule
        journal = getattr(self, 'journal', None)
        sequence = len(journal.read()) if journal is not None else 0
        save_id = f'{sequence:08d}_{uuid.uuid4().hex}'
        path = self.run_dir / 'checkpoints' / f'{self.segment_id}_{label}_{schedule.learning_samples:012d}_{save_id}.pt'
        if path.exists():
            raise FileExistsError('refuse to overwrite any SHA-referenced checkpoint')
        metadata = dict(checkpoint_save_id=save_id, stage='S1', course=schedule.course, value_scale=self.admissions['value_scale'],
            manifest_sha256=self.admissions['calibration']['manifest_sha256'], ledger_run_id=self.run_id,
            learning_samples=schedule.learning_samples, code_sha256=self.code_sha256,
            gate_state=dict(next_suite=schedule.next_suite, attempts=len(schedule.gates),
                final_suites_used=schedule.final_suites_used, recoveries=schedule.recoveries,
                next_gate_samples=min([row[2] for row in schedule.available_slots()
                    if row[1] == schedule.segment] or [0]), passes=[row for row in schedule.gates if row.get('passed')]),
            role_state=dict(num_envs=schedule.num_envs, episode_counters=self.ids.state_dict(), assignments=self.roles),
            random_state=dict(python=random.getstate(), numpy=np.random.get_state(),
                environment=self._environment_rng_state()),
            external_ledger=self.ledger.checkpoint_metadata(), schedule=schedule.state_dict(),
            shuffle_generator=self.generator.get_state(), source_transfer=self.student.transfer_metadata if hasattr(self.student, 'transfer_metadata') else None,
            resume_mode='new_physical_episodes', loss_reports=self.loss_reports[-1:])
        digest = save_checkpoint(path, self.student, self.optimizer, metadata)
        self.last_saved_checkpoint_sha = digest
        self._record('checkpoint', path=str(path), checkpoint_sha256=digest, label=label, save_id=save_id)
        if keep_latest:
            self.latest.append(path)
            for expired in self.latest[:-5]:
                expired.unlink(missing_ok=True)
            self.latest = self.latest[-5:]
        return path, digest

    def _collect_and_train(self):
        from .runtime import collect_rollout
        from .distillation import train_iteration
        from .model import state_sha256
        self._iteration_in_progress = True
        slot = self.schedule.complete_rollouts % 2
        self.buffers[slot] = None
        operation = self.ledger.reserve(segment_id=self.segment_id,
            operation_id=f'rollout_{self.schedule.learning_samples:012d}', stage='S1',
            course=self.schedule.course, category='learning', num_envs=self.schedule.num_envs,
            max_control_steps=48, checkpoint_sha256=self.last_saved_checkpoint_sha,
            allocation_id=ALLOCATIONS[self.schedule.segment], reason='pure_S1_action_value_distillation')
        rollout = None
        settled = False
        try:
            if self.env is None:
                self.env = self._new_env()
            rollout = collect_rollout(self.env, self.teacher, self.student)
            if rollout['control_steps'] != 48 or rollout['physical_control_samples'] != self.schedule.rollout_samples:
                raise RuntimeError('formal optimizer requires exactly 48 complete physical intervals')
            self.ledger.commit(operation, actual_control_steps=48)
            settled = True
            self.schedule.consume_rollout()
            self._record('physical_rollout_committed', operation_key=operation)
            for episode in rollout['episodes']:
                if any(episode.get('failure_masks', {}).get(key) for key in ('nonfinite', 'contact_overflow')):
                    raise FloatingPointError('engineering anomaly in a training physical trajectory')
            before = grouped_distillation_losses(rollout, self.student, value_scale=self.admissions['value_scale'])
            result = train_iteration(self.student, self.optimizer, rollout['observations'],
                rollout['teacher_mean'], rollout['teacher_value'], value_scale=self.admissions['value_scale'],
                generator=self.generator, microbatch=512)
            if (result['adam_steps'] != 20 or result['samples'] != self.schedule.rollout_samples
                    or any(not math.isfinite(result[k]) for k in ('loss', 'action_loss', 'value_loss', 'gradient_norm'))):
                raise FloatingPointError('nonfinite/incomplete 5x4-Adam distillation iteration')
            frozen = state_sha256({k: v for k, v in self.student.state_dict().items()
                                   if 'normalizer' in k or k == 'std'})
            if frozen != self.frozen:
                raise RuntimeError('formal S1 changed frozen std/normalizer state')
            saturation = self.student.normalizer_saturation({k: value[-1] for k, value in rollout['observations'].items()})
            audit = dict(learning_samples=self.schedule.learning_samples, segment=self.schedule.segment,
                statistics=result, losses_by_task_speed_before=before,
                losses_by_task_speed_after=grouped_distillation_losses(rollout, self.student,
                    value_scale=self.admissions['value_scale']), normalizer_saturation=saturation)
            self.loss_reports.append(audit)
            with (self.run_dir / 'training_iterations.jsonl').open('a') as stream:
                stream.write(canonical_json(audit) + '\n')
            self.buffers[slot] = rollout
            self._record('optimizer_complete', adam_steps=20, loss_report_sha256=sha256(audit))
            self._iteration_in_progress = False
        except BaseException as error:
            if not settled:
                self.ledger.commit(operation, status='unknown')
                self.schedule.consume_rollout(48, complete=False)
            self.schedule.blocked = f'{type(error).__name__}: explicit recovery decision required'
            self._record('learning_interrupted', reason=str(error), conservatively_charged=not settled)
            raise

    def _environment_rng_state(self):
        env = getattr(self, 'env', None)
        if env is not None and hasattr(env, 'rng'):
            return env.rng.get_state()
        session = getattr(self, 'evidence_session_state', None)
        return session['rng'].get('environment') if session else getattr(self, 'pending_environment_rng', None)

    def _capture_training_rng(self):
        import numpy as np
        import torch
        return dict(python=random.getstate(), numpy=np.random.get_state(),
            torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            shuffle=self.generator.get_state(), environment=self._environment_rng_state())

    def _restore_training_rng(self, state):
        import numpy as np
        import torch
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])
        if state['cuda'] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['cuda'])
        self.generator.set_state(state['shuffle'])
        self.pending_environment_rng = state.get('environment')
        if self.env is not None and hasattr(self.env, 'rng') and state.get('environment') is not None:
            self.env.rng.set_state(state['environment'])

    def _begin_evidence_session(self, label):
        """Q009: durably save at committed update boundaries, then free physics."""
        import torch
        if self.evidence_session_state is not None:
            return  # adjacent same-SHA A/B, final and video share one session
        if self._iteration_in_progress:
            raise PlanningRequired('external evidence requires a fully committed48-step/20-Adam update')
        validate_schedule_ledger(self.schedule, self.ledger.snapshot())
        checkpoint, checkpoint_sha = self._save('evidence_' + label)
        rng = self._capture_training_rng()
        fingerprint = training_state_fingerprint(self.student, self.optimizer)
        parameter_ids = [id(parameter) for group in self.optimizer.param_groups for parameter in group['params']]
        truncated = []
        had_environment = self.env is not None
        if had_environment:
            lengths = self.env.episode_length_buf.detach().cpu().tolist()
            for slot, steps in enumerate(lengths):
                if steps:
                    truncated.append(dict(role=self.roles[slot], episode_id=self.env.case_episode_ids[slot],
                        case_sha256=self.env.v09_cases[slot]['case_sha256'], physical_control_steps=int(steps),
                        physical_control_samples=int(steps), status='external_truncation',
                        success=None, failure=None, timeout=None, reward_added=0.,
                        course=self.schedule.course, control_steps_already_billed=True))
        self._record('evidence_external_truncations', label=label, episodes=truncated,
                     checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
                     no_new_control_samples=True)
        self.evidence_session_state = dict(label=label, checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha, rng=rng, fingerprint=fingerprint,
            parameter_ids=parameter_ids, had_environment=had_environment, offloaded=False,
            ids_object=self.ids, optimizer_object=self.optimizer, model_object=self.student)
        if self.env is not None:
            self.env.close()
            self.env = None
        self.buffers = [None, None]
        self.student.to('cpu')
        self.teacher.to('cpu')
        self.evidence_session_state['optimizer_devices'] = move_optimizer_state(self.optimizer, device='cpu')
        self.evidence_session_state['offloaded'] = True
        if [id(p) for group in self.optimizer.param_groups for p in group['params']] != parameter_ids:
            raise RuntimeError('Q009 offload changed Adam parameter mapping')
        if {id(p) for p in self.student.parameters()} != set(parameter_ids):
            raise RuntimeError('Q009 model offload replaced parameters bound to Adam')
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._record('evidence_session_started', label=label,
                     checkpoint_sha256=checkpoint_sha, env_released=True, buffers_released=True,
                     fingerprint=fingerprint, physical_resume='new_monotonic_episode_ids')

    def _end_evidence_session(self, *, reopen_environment=True):
        """Verify parent state and saved checkpoint; no Adam or physical replay."""
        state = self.evidence_session_state
        if state is None:
            return
        from .checkpoint import load_checkpoint
        if (file_sha256(state['checkpoint']) != state['checkpoint_sha256']
                or self.optimizer is not state['optimizer_object']
                or self.student is not state['model_object'] or self.ids is not state['ids_object']
                or training_state_fingerprint(self.student, self.optimizer) != state['fingerprint']):
            raise PlanningRequired('external evidence changed immutable training checkpoint/model/Adam/role allocator')
        payload = load_checkpoint(state['checkpoint'], device='cpu')
        # Compare a lightweight state-view; no new model/optimizer is built.
        class ModelView:
            def state_dict(self):
                return payload['model']
        class OptimizerView:
            def state_dict(self):
                return payload['optimizer']
        if training_state_fingerprint(ModelView(), OptimizerView()) != state['fingerprint']:
            raise ValueError('Q009 saved checkpoint differs from retained model/Adam')
        self.ledger.validate_checkpoint(payload['metadata']['external_ledger'])
        self.student.to(self.device)
        self.teacher.to(self.device)
        if state['offloaded']:
            move_optimizer_state(self.optimizer, restore_devices=state['optimizer_devices'])
        if {id(p) for p in self.student.parameters()} != set(state['parameter_ids']):
            raise RuntimeError('Q009 restore changed Adam parameter mapping')
        self._restore_training_rng(state['rng'])
        if (reopen_environment and state['had_environment'] and not self.schedule.accepted
                and not self.schedule.blocked):
            self.env = self._new_env()  # same current course/ratio, fresh externally reserved IDs
            # Environment construction must not perturb learner/shuffle streams.
            self._restore_training_rng(state['rng'])
        self.evidence_session_state = None
        self._record('evidence_session_finished', label=state['label'],
            checkpoint_sha256=state['checkpoint_sha256'], optimizer_replayed=False,
            learning_samples=self.schedule.learning_samples, new_physical_episodes=self.env is not None)

    def _switch_to_s1_2(self):
        """Keep the same Adam/momentum and ID allocator, start 10% teacher episodes."""
        optimizer, ids = self.optimizer, self.ids
        self.student.set_s1_course('S1.2')
        if self.env is not None:
            self.env.set_course('S1.2')
            self.env.reset()
        if self.optimizer is not optimizer or self.ids is not ids:
            raise RuntimeError('S1.2 transition replaced optimizer or monotonic episode allocator')

    def _reuse_before_evidence(self):
        """Resume a completed zero-learning diagnostic instead of replaying it."""
        prior = [row for row in self.journal.read() if row['event'] == 'Q008_before_evidence']
        obligations = self.ledger.snapshot()['stages']['S1']['obligations']
        names = ('diag_before', 'video_6400', 'video_init')
        if not prior:
            if any(obligations.get(name, {}).get('closed') for name in names):
                raise PlanningRequired('completed initial evidence lacks its authoritative journal receipt; reconcile')
            return False
        report = prior[-1]['details']['report']
        if (report.get('sds_sha256') != spec.SDS_SHA256
                or report.get('code_sha256') != self.code_sha256
                or report.get('report_sha256') != sha256({k: v for k, v in report.items() if k != 'report_sha256'})):
            raise ValueError('frozen initial diagnostic report identity mismatch')
        if (not all(obligations.get(name, {}).get('closed') for name in names)
                or obligations['diag_before'].get('bound_checkpoint_sha256') != report.get('checkpoint_sha256')
                or obligations['video_init'].get('bound_checkpoint_sha256') != report.get('checkpoint_sha256')
                or obligations['video_6400'].get('bound_checkpoint_sha256') != spec.SOURCE_SHA256):
            raise PlanningRequired('initial diagnostic/video obligations are incomplete or bind different models')
        checkpoint, folder = Path(report['checkpoint']), Path(report['folder'])
        summary_path = folder / 'summary.json'
        if (not checkpoint.is_file() or not summary_path.is_file()
                or file_sha256(checkpoint) != report['checkpoint_sha256']
                or file_sha256(summary_path) != report.get('summary_file_sha256')
                or _read_json(summary_path) != report.get('summary')):
            raise ValueError('frozen initial checkpoint/diagnostic summary changed')
        summary = report['summary']
        if (summary.get('suite') != 'S1_DIAG' or summary.get('total') != 1500
                or summary.get('checkpoint_sha256') != report['checkpoint_sha256']
                or summary.get('code_sha256') != self.code_sha256
                or summary.get('manifest_sha256') != obligations['diag_before'].get('manifest_sha256')):
            raise ValueError('frozen initial diagnostic denominator/manifest mismatch')
        self.before_diagnostic = deepcopy(report)
        self._record('Q008_before_evidence_reused', checkpoint=str(checkpoint),
            checkpoint_sha256=report['checkpoint_sha256'], report_sha256=report['report_sha256'],
            evidence_replayed=False, additional_control_samples=0)
        return True

    def run(self):
        try:
            if self.schedule.accepted:
                marker_path = self.run_dir / 'S1_RESTORED.json'
                if not marker_path.is_file():
                    raise PlanningRequired('accepted schedule lacks its durable final marker; reconcile evidence')
                marker = _read_json(marker_path)
                if marker.get('candidate_sha256') != self.schedule.gates[-1]['candidate_sha256']:
                    raise ValueError('accepted marker candidate identity differs from external schedule')
                return marker
            reused_before = self._reuse_before_evidence() if self.schedule.segment == 'adapt' else False
            if self.schedule.segment == 'adapt' and self.schedule.spent['adapt'] and not reused_before:
                raise PlanningRequired('S1.1 resume lacks frozen initial before-evidence')
            entry_path, entry_sha = self._save('course_entry')
            if self.schedule.segment == 'adapt' and not self.schedule.spent['adapt'] and not reused_before:
                self._begin_evidence_session('before')
                self.before_diagnostic = self.evidence_hooks.diagnostic(phase='before',
                    checkpoint=entry_path, checkpoint_sha256=entry_sha)
                self.evidence_hooks.video(model_id='6400', checkpoint=spec.SOURCE_PATH,
                                          checkpoint_sha256=spec.SOURCE_SHA256)
                self.evidence_hooks.video(model_id='init', checkpoint=entry_path, checkpoint_sha256=entry_sha)
                self._record('Q008_before_evidence', report=self.before_diagnostic)
                self._end_evidence_session()
            while not self.schedule.accepted:
                if self.schedule.blocked:
                    raise PlanningRequired(self.schedule.blocked)
                if self.schedule.final_pending:
                    suite = self.schedule.begin_final()
                    self._begin_evidence_session(suite)
                    self.evidence_hooks.video(model_id=suite.replace('_', ''),
                        checkpoint=self.candidate_path, checkpoint_sha256=self.candidate_sha256)
                    self._record('final_started', suite=suite, candidate_sha256=self.candidate_sha256)
                    result = self.evaluator(suite=suite, candidate_path=self.candidate_path,
                        candidate_sha256=self.candidate_sha256, attempt=len(self.schedule.gates),
                        segment_id=f'{self.segment_id}_{suite}')
                    if result.get('group_count') != 75 or result.get('fixed_denominator') != 7500:
                        raise ValueError('S1_RESTORED requires every 100-case group, full7500 denominator')
                    self.schedule.finish_final(candidate_sha256=self.candidate_sha256,
                        passed=result['passed'], engineering_valid=result['engineering_valid'],
                        report_sha256=result['report_sha256'],
                        planning_required=result.get('planning_required', False))
                    self._record('final_finished', suite=suite, report=result)
                    self._save('stage_accepted' if self.schedule.accepted else
                               'final_planning_blocked' if self.schedule.blocked else 'final_failure_recovery_entry')
                    if self.schedule.accepted:
                        marker = dict(state='S1_RESTORED', sds_sha256=spec.SDS_SHA256,
                            candidate_sha256=self.candidate_sha256, final_suite=suite,
                            final_report_sha256=result['report_sha256'], fixed_denominator=7500,
                            groups=75, code_sha256=self.code_sha256, ledger=self.ledger.checkpoint_metadata())
                        with (self.run_dir / 'S1_RESTORED.json').open('x') as stream:
                            stream.write(canonical_json(marker) + '\n')
                        return marker
                    self.candidate_path = self.candidate_sha256 = None
                    continue
                gate_slot = self.schedule.due_slot()
                if gate_slot is not None:
                    if not self.schedule.pending_confirmation:
                        self.candidate_path, self.candidate_sha256 = self._save('frozen_candidate')
                    self._begin_evidence_session(f'gate_{len(self.schedule.gates) + 1:02d}')
                    gate = self.schedule.start_gate(self.candidate_sha256)
                    self._record('gate_started', gate=gate)
                    result = self.evaluator(suite=gate['suite'], candidate_path=self.candidate_path,
                        candidate_sha256=self.candidate_sha256, attempt=gate['attempt'],
                        segment_id=f"{self.segment_id}_gate_{gate['attempt']:02d}")
                    self.schedule.finish_gate(candidate_sha256=self.candidate_sha256,
                        passed=result['passed'], engineering_valid=result['engineering_valid'],
                        report_sha256=result['report_sha256'])
                    if result['planning_required']:
                        self.schedule.blocked = 'paired metric has no valid teacher window; planning required'
                    self._record('gate_finished', report=result)
                    if self.schedule.final_pending:
                        self._save('course_qualified')
                    continue
                if not self.schedule.can_collect():
                    if self.schedule.segment != 'adapt':
                        raise PlanningRequired('course cap reached without two same-SHA A/B passes')
                    if not self.loss_reports:
                        raise PlanningRequired('S1.1 before/after engineering evidence unavailable')
                    adapted_path, adapted_sha = self._save('S1.1_adapted')
                    self._begin_evidence_session('after')
                    after = self.evidence_hooks.diagnostic(phase='after',
                        checkpoint=adapted_path, checkpoint_sha256=adapted_sha)
                    comparison = self.evidence_hooks.compare(before=self.before_diagnostic, after=after)
                    self.evidence_hooks.video(model_id='adapt', checkpoint=adapted_path,
                                              checkpoint_sha256=adapted_sha)
                    if not comparison.get('engineering_valid'):
                        raise PlanningRequired('Q008 before/after engineering evidence invalid')
                    self.schedule.finish_adapt(finite=comparison.get('finite'),
                        migration_valid=comparison.get('migration_valid'),
                        contacts_valid=comparison.get('contacts_valid'),
                        comparison_report_sha256=comparison.get('report_sha256'))
                    self._switch_to_s1_2()
                    self._record('S1.1_engineering_accepted', comparison=comparison)
                    self._save('S1.2_entry')
                    continue
                self._end_evidence_session()
                previous = self.schedule.learning_samples
                self._collect_and_train()
                current = self.schedule.learning_samples
                if current // 5_000_000 > previous // 5_000_000:
                    self._save('latest', keep_latest=True)
                if current // 100_000_000 > previous // 100_000_000:
                    self._save('archive')
                if not self.schedule.can_collect(check_gate=False):
                    self._save('course_cap')
            raise AssertionError('accepted loop must emit its audited marker')
        finally:
            self._end_evidence_session(reopen_environment=False)
            if self.env is not None:
                self.env.close()
                self.env = None


class SubprocessEvidenceHooks:
    """Q008 uses explicit diagnostic, comparison and independent replay CLIs."""
    def __init__(self, *, python, run_dir, run_id, segment_id, device, code_sha256,
                 calibration_path, script=None):
        self.python = python
        self.eval_script = Path(script) if script else spec.PROJECT_ROOT / 'scripts/evaluate_a2_v09_s1.py'
        self.compare_script = spec.PROJECT_ROOT / 'scripts/compare_a2_v09_diagnostics.py'
        self.video_script = spec.PROJECT_ROOT / 'scripts/video_a2_v09_s1.py'
        if not all(path.is_file() for path in (self.eval_script, self.compare_script, self.video_script)):
            raise PlanningRequired('Q008 evidence driver is not implemented/approved')
        self.run_dir, self.run_id, self.segment_id = Path(run_dir), run_id, segment_id
        self.device, self.code_sha256 = device, code_sha256
        self.calibration_path = Path(calibration_path)

    def diagnostic(self, *, phase, checkpoint, checkpoint_sha256):
        if phase not in ('before', 'after'):
            raise ValueError('diagnostic phase must be before/after')
        segment = f'{self.segment_id}_diag_{phase}'
        command = [self.python, str(self.eval_script), '--run-dir', str(self.run_dir),
            '--run-id', self.run_id, '--segment-id', segment, '--suite', 'S1_DIAG',
            '--policy', 'student', '--checkpoint', str(checkpoint),
            '--checkpoint-sha256', checkpoint_sha256, '--code-sha256', self.code_sha256,
            '--obligation-id', 'diag_' + phase, '--batch-size', '20',
            '--device', self.device, '--execute']
        subprocess.run(command, cwd=spec.PROJECT_ROOT, check=True)
        folder = self.run_dir / 'evidence' / segment
        summary = _read_json(folder / 'summary.json')
        manifest = load_manifest(self.run_dir / 'cases/S1_DIAG.json', purpose='diagnostic')
        if (summary.get('suite') != 'S1_DIAG' or summary.get('total') != 1500
                or summary.get('checkpoint_sha256') != checkpoint_sha256
                or summary.get('manifest_sha256') != manifest['manifest_sha256']
                or summary.get('code_sha256') != self.code_sha256):
            raise ValueError('Q008 diagnostic fixed denominator/source mismatch')
        result = dict(sds_sha256=spec.SDS_SHA256, code_sha256=self.code_sha256,
            folder=str(folder.resolve()), checkpoint=str(Path(checkpoint).resolve()),
            checkpoint_sha256=checkpoint_sha256, summary=summary,
            summary_file_sha256=file_sha256(folder / 'summary.json'))
        result['report_sha256'] = sha256(result)
        return result

    def compare(self, *, before, after):
        output = self.run_dir / 'evidence' / f'{self.segment_id}_adapt_comparison.json'
        command = [self.python, str(self.compare_script), '--run-dir', str(self.run_dir),
            '--before-dir', before['folder'], '--after-dir', after['folder'],
            '--initial-checkpoint', before['checkpoint'],
            '--initial-checkpoint-sha256', before['checkpoint_sha256'],
            '--adapted-checkpoint', after['checkpoint'],
            '--adapted-checkpoint-sha256', after['checkpoint_sha256'],
            '--calibration', str(self.calibration_path), '--code-sha256', self.code_sha256,
            '--device', self.device, '--output', str(output)]
        subprocess.run(command, cwd=spec.PROJECT_ROOT, check=True)
        report = _read_json(output)
        if report.get('sds_sha256') != spec.SDS_SHA256 or report.get('code_sha256') != self.code_sha256:
            raise ValueError('before/after comparison source identity mismatch')
        required = ('engineering_valid', 'finite', 'migration_valid', 'contacts_valid')
        if any(report.get(key) is not True for key in required):
            raise PlanningRequired('before/after helper has not supplied complete verified engineering flags')
        report['report_sha256'] = file_sha256(output)
        report['report_path'] = str(output.resolve())
        return report

    def video(self, *, model_id, checkpoint, checkpoint_sha256):
        segment = f'{self.segment_id}_video_{model_id}'
        output = self.run_dir / 'videos' / segment
        command = [self.python, str(self.video_script), '--run-dir', str(self.run_dir),
            '--run-id', self.run_id, '--segment-id', segment, '--code-sha256', self.code_sha256,
            '--policy', 'teacher' if model_id == '6400' else 'student',
            '--checkpoint', str(checkpoint), '--checkpoint-sha256', checkpoint_sha256,
            '--obligation-id', 'video_' + model_id, '--output-dir', str(output),
            '--device', self.device, '--width', '640', '--height', '480', '--fps', '25', '--execute']
        subprocess.run(command, cwd=spec.PROJECT_ROOT, check=True)
        report = _read_json(output / 'video_summary.json')
        if (report.get('schema_version') != 11 or report.get('sds_sha256') != spec.SDS_SHA256
                or report.get('source_sha256') != spec.SOURCE_SHA256
                or report.get('code_sha256') != self.code_sha256
                or report.get('checkpoint_sha256') != checkpoint_sha256
                or report.get('engineering_valid') is not True
                or report.get('full75') is not True or report.get('actual_cases') != 75
                or len(report.get('video_paths', [])) != 75
                or report.get('report_sha256') != sha256({k: v for k, v in report.items() if k != 'report_sha256'})):
            raise PlanningRequired('all75 fixed replay videos, including failures, must be valid and retained')
        from .budget import AppendOnlyLedger
        duty = AppendOnlyLedger(self.run_dir / 'budget.jsonl', run_id=self.run_id).snapshot()['stages']['S1']['obligations']['video_' + model_id]
        if not duty.get('closed') or duty.get('bound_checkpoint_sha256') != checkpoint_sha256:
            raise PlanningRequired('video replay obligation has not externally completed for this exact model')
        return report
