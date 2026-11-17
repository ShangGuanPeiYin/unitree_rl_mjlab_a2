"""Q006 formal-cost ledger: the 30100 cumulative ceiling and the four cost views.

Q006 section 3 explicitly raises the *cumulative formal* ceiling from 30050 to
30100 while the active branch keeps the frozen 30000-update cap:

* trial01 lost exactly 50 billed updates (Q004);
* the active trial02 branch may still spend at most 30000 updates (29900 remain
  after the update-100 rollback);
* the lost trial02 Q005 segment 101-150 billed another 50 updates.

The ledger therefore keeps four separate views instead of collapsing them:

``effective_learning_update``
    the update value of the surviving model branch (100 now, 30000 at full cap);
``physical_training_updates``
    every update really executed (200 now, 30100 at full cap);
``lost_branch_updates``
    updates that were physically executed but whose model state was lost
    (trial01 50 + Q005 101-150 50 = 100);
``engineering_test_cost``
    smoke/probe/zero-learning evaluation cost, listed separately and never
    billed to the formal budget.

Entries are keyed by ``(run_id, segment_id, update)``.  The same update number
(101-150) exists on two different segments (the lost Q005 segment and the Q006
replay segment); both are real executions and both entries are kept.  Only a
repeated execution of the same ``(run_id, segment_id, update)`` triple is a
duplicate.
"""
from __future__ import annotations

from . import ppo as ppo_module
from . import spec

#: Q006-approved formal ceiling (Q004's 30050 plus the lost 101-150 interval).
FORMAL_CUMULATIVE_CAP = spec.TOTAL_BUDGET + 100
#: The active trial02 branch keeps its frozen single-run cap.
ACTIVE_BRANCH_CAP = spec.TOTAL_BUDGET
#: Q004 lost trial01 interval.
TRIAL01_LOST_UPDATES = 50
#: Q006 lost trial02 Q005 interval (101-150, Q006_ERRATA).
LOST_Q005_UPDATES = 50
#: Full-cap cost of the 30100 formally billed updates (512 envs x 48 rollout).
FULL_CAP_COST = dict(control_samples=FORMAL_CUMULATIVE_CAP * 24576,
                     physics_substeps=FORMAL_CUMULATIVE_CAP * 98304,
                     adam_updates=FORMAL_CUMULATIVE_CAP * 20)

Q006_DECISION_SHA256 = '798a7cf4603fc50abbe6a83c30e20680fb884824c7cccddb78679bc698d5bd09'
Q006_ERRATA_SHA256 = 'f3c64d2e81e06d1c0d0a8c5c71476f08e50849e82c98b791a542b8f9339e5c3c'
Q006_ERRATA_PATH = ('logs/a2_ideal_stairs_v06/implementation/coordination/'
                    'Q006_ERRATA_lost_50.md')
Q006_BRIDGE_MANIFEST_PATH = ('logs/a2_ideal_stairs_v06/implementation/coordination/'
                             'Q006_recovery_bridge.md')
Q006_LEDGER_PATH = 'logs/a2_ideal_stairs_v06/implementation/Q006_cost_ledger.json'
#: The independent verifier (D) fills the real acceptance SHA; until then the
#: bridge records an explicit placeholder instead of inventing a digest.
Q006_ACCEPTANCE_SHA256 = 'PENDING_INDEPENDENT_ACCEPTANCE_SHA256'

#: The unique new segment authorised by Q006 section 1.
Q006_RUN_ID = 'trial02'
Q006_SEGMENT_ID = 'trial02_q006_resume_000100'
Q006_SEGMENT_DIR = ('logs/a2_ideal_stairs_v06/'
                    'train_seed42_trial02_resume_000100_q006')
Q006_SEGMENT_SUFFIX = '_q006'
Q006_ROLLBACK_FROM_UPDATE = 150
Q006_RESUME_FROM_UPDATE = 100
Q006_LOST_UPDATES = 50
Q006_NUM_ENVS = 512
Q006_ROLLOUT = int(ppo_module.ROLLOUT)

# ---------------------------------------------------------------- Q007 (400)
Q007_DECISION_SHA256 = '567df511509ee7414f113432be3b55f34e47655a4da53e7e16ffd2e72887a48f'
Q007_ERRATA_SHA256 = 'db1952c6e2c5d4dfc2946c0eade78ea818708003834e80f6c7df209adb3a9bde'
Q007_ERRATA_PATH = ('logs/a2_ideal_stairs_v06/implementation/coordination/'
                    'Q007_ERRATA_source_and_migration.md')
Q007_BRIDGE_MANIFEST_PATH = ('logs/a2_ideal_stairs_v06/implementation/coordination/'
                             'Q007_recovery_bridge.md')
Q007_TASK_MIGRATION_PATH = ('logs/a2_ideal_stairs_v06/implementation/'
                            'Q007_task_migration.json')
Q007_LEDGER_PATH = 'logs/a2_ideal_stairs_v06/implementation/Q007_cost_ledger.json'
Q007_ACCEPTANCE_SHA256 = 'PENDING_INDEPENDENT_ACCEPTANCE_SHA256'
#: The designated resume source: the Q006 exception container that completed the
#: PPO update and the 400 validation/diagnostic records but not the flat set.
Q007_SOURCE_CHECKPOINT_PATH = ('logs/a2_ideal_stairs_v06/'
                               'train_seed42_trial02_resume_000100_q006/'
                               'checkpoints/exception_000400.pt')
Q007_SOURCE_CHECKPOINT_SHA256 = (
    '50e939963e7986477f7f9472daa9e1bf527f11b08d11c67d957a1af7092492b5')
#: The pre-evaluation control is never a resume source (Q007 section 1).
Q007_FORBIDDEN_CHECKPOINT_PATH = ('logs/a2_ideal_stairs_v06/'
                                  'train_seed42_trial02_resume_000100_q006/'
                                  'checkpoints/boundary_000400.pt')
Q007_FORBIDDEN_CHECKPOINT_SHA256 = (
    '2a81110d588e5822eded1029f6a4180a6f78a6fa58c0231c65e360adaf4f3097')
#: The Q006 segment closure the source container was written under.
Q007_OLD_SOURCE_CLOSURE_SHA256 = (
    '68e71f27292f223dc9332300f52a638358507291c9c0e0a3e8632e884c7e0e56')
Q007_OLD_FLAT_TASK_TILE = 1
Q007_OLD_FLAT_TASK_SHA256 = (
    'afb64ce64395297bd73ea3a842b1c8a9ecc1ae7fce97d0fe888cbf60cfecc5eb')
Q007_VALIDATION_RESULT_SHA256 = (
    '0ccaa01f4161fa1006d49e41cb8bec298e68bd6c76a7f3d9bb195de603e45fd7')
Q007_DIAGNOSTIC_RESULT_SHA256 = (
    'd7468056efc6ef8911fa26736a430f16a2d68b4717c22640a150e146e1b6caf8')
Q007_COMPLETED_RESULT_IDS = (
    '400:validation:None:H:' + Q007_VALIDATION_RESULT_SHA256,
    '400:diagnostic:None:H:' + Q007_DIAGNOSTIC_RESULT_SHA256,
)
Q007_RUN_ID = 'trial02'
Q007_SEGMENT_ID = 'trial02_q007_resume_000400'
Q007_SEGMENT_DIR = ('logs/a2_ideal_stairs_v06/'
                    'train_seed42_trial02_resume_000400_q007')
Q007_SEGMENT_SUFFIX = '_q007'
Q007_ROLLBACK_FROM_UPDATE = 400
Q007_RESUME_FROM_UPDATE = 400
Q007_LOST_UPDATES = 0
#: Active-branch updates already consumed by the time of the Q007 rollback.
Q007_ACTIVE_CONSUMED = 400
Q007_ACTIVE_REMAINING = ACTIVE_BRANCH_CAP - Q007_ACTIVE_CONSUMED
#: Current physical total: active 400 + trial01 lost 50 + Q005 lost 50.
Q007_PHYSICAL_TOTAL = 500
Q007_LOST_TOTAL = TRIAL01_LOST_UPDATES + LOST_Q005_UPDATES

SEGMENTS = (
    dict(run_id='trial01', segment_id='trial01',
         state='implementation_crash_nonresumable', billed_updates=50,
         lost=True, effective=False),
    dict(run_id='trial02', segment_id='trial02_original',
         state='running_original_branch', billed_updates=100,
         lost=False, effective=True),
    dict(run_id='trial02', segment_id='trial02_q005_resume_000100',
         state='implementation_crash_nonresumable', billed_updates=50,
         lost=True, effective=False),
    dict(run_id='trial02', segment_id=Q006_SEGMENT_ID,
         state='update400_retention_tile_crash_pending_migration',
         billed_updates=300, lost=False, effective=True),
)

ENGINEERING_COST = dict(
    note='smoke/probe/zero-learning evaluation cost is engineering only and is '
         'never billed to the formal 30100 ceiling',
    counts_against_training=False,
    smoke='logs/a2_ideal_stairs_v06/implementation/smoke_seed42_trial14',
    probe='logs/a2_ideal_stairs_v06/implementation/probe/probe_seed42',
)


def training_cost(updates, *, num_envs=Q006_NUM_ENVS, rollout=Q006_ROLLOUT):
    """Control samples / physics substeps / Adam steps of ``updates`` updates."""
    updates = int(updates)
    num_envs = int(num_envs)
    if updates < 0 or num_envs < 1:
        raise ValueError('updates must be >= 0 and num_envs >= 1')
    samples = updates * num_envs * int(rollout)
    return dict(control_samples=samples,
                physics_substeps=ppo_module.physics_substeps(samples),
                adam_updates=updates * int(ppo_module.ADAM_STEPS_PER_UPDATE))


def segment_metadata(*, rollback_from_update=Q006_ROLLBACK_FROM_UPDATE,
                     resume_from_update=Q006_RESUME_FROM_UPDATE,
                     lost_updates=Q006_LOST_UPDATES):
    """The Q006 segment identity recorded in the contract and recovery receipt."""
    if int(resume_from_update) != Q006_RESUME_FROM_UPDATE:
        raise ValueError('the Q006 segment resumes from update 100')
    if int(rollback_from_update) != Q006_ROLLBACK_FROM_UPDATE:
        raise ValueError('the Q006 segment rolls back from update 150')
    if int(lost_updates) != Q006_LOST_UPDATES:
        raise ValueError('the Q006 segment lost updates 101-150 (50)')
    return dict(run_id=Q006_RUN_ID, segment_id=Q006_SEGMENT_ID,
                directory=Q006_SEGMENT_DIR,
                rollback_from_update=int(rollback_from_update),
                resume_from_update=int(resume_from_update),
                lost_updates=int(lost_updates),
                replay_billed_updates=True,
                q006_decision_sha256=Q006_DECISION_SHA256,
                q006_errata_sha256=Q006_ERRATA_SHA256)


def entry(run_id, segment_id, update, *, phase='F0', effective=True, lost=False,
          engineering=False, num_envs=Q006_NUM_ENVS, rollout=Q006_ROLLOUT,
          **extra):
    """One physical execution row keyed by ``(run_id, segment_id, update)``."""
    update = int(update)
    if update < 0:
        raise ValueError('update must be >= 0')
    cost = training_cost(0 if engineering else 1, num_envs=num_envs,
                         rollout=rollout)
    row = dict(run_id=str(run_id), segment_id=str(segment_id), update=update,
               phase=str(phase),
               effective_learning_update=bool(effective),
               physical_training_updates=0 if engineering else 1,
               lost_branch_updates=1 if lost else 0,
               engineering_test_cost=1 if engineering else 0,
               counts_against_training=not engineering)
    row.update(cost)
    row.update(extra)
    return row


def validate_entries(entries):
    """No duplicate ``(run_id, segment_id, update)`` triples; numbers are coherent."""
    seen = set()
    for row in entries:
        key = (str(row['run_id']), str(row['segment_id']), int(row['update']))
        if key in seen:
            raise ValueError(f'ledger repeats a real execution entry: {key}')
        seen.add(key)
        if int(row['update']) < 1:
            raise ValueError('ledger entries start at update 1')
        if bool(row['effective_learning_update']) and int(row['lost_branch_updates']):
            raise ValueError('an update cannot be both effective and lost')
        if bool(row['effective_learning_update']) and row['engineering_test_cost']:
            raise ValueError('engineering tests are not effective learning updates')
    return True


def build_ledger(*, entries=None, effective_learning_update=100,
                 physical_training_updates=200, lost_branch_updates=100,
                 engineering_test_cost=None):
    """The appendable Q006 ledger document (current 200-update state by default)."""
    entries = list(entries or ())
    validate_entries(entries)
    physical = int(physical_training_updates)
    effective = int(effective_learning_update)
    lost = int(lost_branch_updates)
    if effective + lost != physical:
        raise ValueError('effective + lost updates must equal the physical total')
    if physical > FORMAL_CUMULATIVE_CAP:
        raise ValueError('physical training updates exceed the Q006 30100 ceiling')
    if effective > ACTIVE_BRANCH_CAP:
        raise ValueError('the active branch exceeds its frozen 30000 cap')
    engineering = dict(ENGINEERING_COST if engineering_test_cost is None
                       else engineering_test_cost)
    return dict(
        schema='a2-v06-q006-ledger-v1',
        q006_decision_sha256=Q006_DECISION_SHA256,
        q006_errata=dict(path=Q006_ERRATA_PATH, sha256=Q006_ERRATA_SHA256,
                         lost_updates=LOST_Q005_UPDATES,
                         control_samples=LOST_Q005_UPDATES * Q006_NUM_ENVS
                         * Q006_ROLLOUT,
                         physics_substeps=ppo_module.physics_substeps(
                             LOST_Q005_UPDATES * Q006_NUM_ENVS * Q006_ROLLOUT),
                         adam_updates=LOST_Q005_UPDATES
                         * int(ppo_module.ADAM_STEPS_PER_UPDATE)),
        budget=dict(per_run_cap=ACTIVE_BRANCH_CAP,
                    combined_formal_cap=FORMAL_CUMULATIVE_CAP,
                    breakdown=dict(trial01_lost=TRIAL01_LOST_UPDATES,
                                   trial02_active_branch_max=ACTIVE_BRANCH_CAP,
                                   trial02_lost_branch_101_150=LOST_Q005_UPDATES),
                    full_cap_cost=dict(FULL_CAP_COST),
                    remaining_active_branch_after_rollback=(
                        ACTIVE_BRANCH_CAP - Q006_RESUME_FROM_UPDATE)),
        totals=dict(effective_learning_update=effective,
                    physical_training_updates=physical,
                    lost_branch_updates=lost,
                    engineering_test_cost=engineering),
        segments=[dict(item) for item in SEGMENTS],
        entries=entries,
    )


def validate_ledger(ledger):
    """Validate the four views, the ceiling and the full-cap cost arithmetic."""
    if not isinstance(ledger, dict):
        raise ValueError('ledger must be a mapping')
    totals = ledger.get('totals') or {}
    effective = int(totals.get('effective_learning_update', -1))
    physical = int(totals.get('physical_training_updates', -1))
    lost = int(totals.get('lost_branch_updates', -1))
    if effective < 0 or physical < 0 or lost < 0:
        raise ValueError('ledger totals are missing or negative')
    if effective + lost != physical:
        raise ValueError('effective + lost updates must equal the physical total')
    if physical > FORMAL_CUMULATIVE_CAP:
        raise ValueError('physical training updates exceed the Q006 30100 ceiling')
    if effective > ACTIVE_BRANCH_CAP:
        raise ValueError('the effective branch exceeds its 30000 cap')
    validate_entries(ledger.get('entries') or [])
    full = dict(training_cost(FORMAL_CUMULATIVE_CAP))
    planned = (ledger.get('budget') or {}).get('full_cap_cost') or {}
    if {key: int(planned.get(key, -1)) for key in full} != full:
        raise ValueError('the full-cap cost does not match 30100 x 512 x 48 x 20')
    return True


def build_current_ledger():
    """The 200-update state recorded right after the Q005 loss (Q006 section 3)."""
    rows = []
    for index in range(1, TRIAL01_LOST_UPDATES + 1):
        rows.append(entry('trial01', 'trial01', index, phase='F0', effective=False,
                          lost=True))
    for index in range(1, Q006_RESUME_FROM_UPDATE + 1):
        rows.append(entry('trial02', 'trial02_original', index, phase='F0'))
    for index in range(Q006_RESUME_FROM_UPDATE + 1,
                       Q006_RESUME_FROM_UPDATE + LOST_Q005_UPDATES + 1):
        rows.append(entry('trial02', 'trial02_q005_resume_000100', index,
                          phase='F0', effective=False, lost=True))
    return build_ledger(entries=rows)


def q007_segment_metadata(*, rollback_from_update=Q007_ROLLBACK_FROM_UPDATE,
                          resume_from_update=Q007_RESUME_FROM_UPDATE,
                          lost_updates=Q007_LOST_UPDATES):
    """The Q007 segment identity: same run, same update, only the pending task."""
    if int(resume_from_update) != Q007_RESUME_FROM_UPDATE:
        raise ValueError('the Q007 segment resumes from update 400')
    if int(rollback_from_update) != Q007_ROLLBACK_FROM_UPDATE:
        raise ValueError('the Q007 segment rolls back from update 400')
    if int(lost_updates) != Q007_LOST_UPDATES:
        raise ValueError('the Q007 segment loses no learning update')
    return dict(run_id=Q007_RUN_ID, segment_id=Q007_SEGMENT_ID,
                directory=Q007_SEGMENT_DIR,
                rollback_from_update=int(rollback_from_update),
                resume_from_update=int(resume_from_update),
                lost_updates=int(lost_updates),
                replay_billed_updates=False,
                replay_pending_evaluations=True,
                q006_decision_sha256=Q006_DECISION_SHA256,
                q007_decision_sha256=Q007_DECISION_SHA256,
                q007_errata_sha256=Q007_ERRATA_SHA256,
                source_checkpoint_sha256=Q007_SOURCE_CHECKPOINT_SHA256)


def build_q007_ledger():
    """The 500-physical/400-effective state after the Q006 update-400 crash."""
    rows = []
    for index in range(1, TRIAL01_LOST_UPDATES + 1):
        rows.append(entry('trial01', 'trial01', index, phase='F0',
                          effective=False, lost=True))
    for index in range(1, Q006_RESUME_FROM_UPDATE + 1):
        rows.append(entry('trial02', 'trial02_original', index, phase='F0'))
    for index in range(Q006_RESUME_FROM_UPDATE + 1,
                       Q006_RESUME_FROM_UPDATE + LOST_Q005_UPDATES + 1):
        rows.append(entry('trial02', 'trial02_q005_resume_000100', index,
                          phase='F0', effective=False, lost=True))
    for index in range(101, Q007_RESUME_FROM_UPDATE + 1):
        phase = 'F0' if index <= 200 else 'H'
        rows.append(entry('trial02', Q006_SEGMENT_ID, index, phase=phase))
    ledger = build_ledger(entries=rows,
                          effective_learning_update=Q007_RESUME_FROM_UPDATE,
                          physical_training_updates=Q007_PHYSICAL_TOTAL,
                          lost_branch_updates=Q007_LOST_TOTAL)
    ledger['schema'] = 'a2-v06-q007-ledger-v1'
    ledger['q007_decision_sha256'] = Q007_DECISION_SHA256
    ledger['q007_errata'] = dict(path=Q007_ERRATA_PATH,
                                 sha256=Q007_ERRATA_SHA256,
                                 correction='resume source is exception_000400, not '
                                            'boundary_000400')
    ledger['budget']['remaining_active_branch'] = Q007_ACTIVE_REMAINING
    ledger['budget']['formula'] = ('trial01 lost 50 + Q005 lost 50 + active trial02 '
                                   'branch max 30000; the original trial02 first 100 '
                                   'updates are inside the active branch and are not '
                                   'added again')
    ledger['budget']['current_physical_total'] = Q007_PHYSICAL_TOTAL
    ledger['segments'] = [dict(item) for item in SEGMENTS]
    validate_ledger(ledger)
    return ledger


__all__ = [
    'ACTIVE_BRANCH_CAP', 'ENGINEERING_COST', 'FORMAL_CUMULATIVE_CAP', 'FULL_CAP_COST',
    'LOST_Q005_UPDATES', 'Q006_ACCEPTANCE_SHA256', 'Q006_BRIDGE_MANIFEST_PATH',
    'Q006_DECISION_SHA256', 'Q006_ERRATA_PATH', 'Q006_ERRATA_SHA256',
    'Q006_LEDGER_PATH', 'Q006_LOST_UPDATES', 'Q006_NUM_ENVS', 'Q006_RESUME_FROM_UPDATE',
    'Q006_ROLLBACK_FROM_UPDATE', 'Q006_ROLLOUT', 'Q006_RUN_ID', 'Q006_SEGMENT_DIR',
    'Q006_SEGMENT_ID', 'Q006_SEGMENT_SUFFIX', 'SEGMENTS', 'TRIAL01_LOST_UPDATES',
    'Q007_ACCEPTANCE_SHA256', 'Q007_ACTIVE_CONSUMED', 'Q007_ACTIVE_REMAINING',
    'Q007_BRIDGE_MANIFEST_PATH', 'Q007_COMPLETED_RESULT_IDS', 'Q007_DECISION_SHA256',
    'Q007_DIAGNOSTIC_RESULT_SHA256', 'Q007_ERRATA_PATH', 'Q007_ERRATA_SHA256',
    'Q007_FORBIDDEN_CHECKPOINT_PATH', 'Q007_FORBIDDEN_CHECKPOINT_SHA256',
    'Q007_LEDGER_PATH', 'Q007_LOST_TOTAL', 'Q007_LOST_UPDATES',
    'Q007_OLD_FLAT_TASK_SHA256', 'Q007_OLD_FLAT_TASK_TILE',
    'Q007_OLD_SOURCE_CLOSURE_SHA256', 'Q007_PHYSICAL_TOTAL', 'Q007_RESUME_FROM_UPDATE',
    'Q007_ROLLBACK_FROM_UPDATE', 'Q007_RUN_ID', 'Q007_SEGMENT_DIR',
    'Q007_SEGMENT_ID', 'Q007_SEGMENT_SUFFIX', 'Q007_SOURCE_CHECKPOINT_PATH',
    'Q007_SOURCE_CHECKPOINT_SHA256', 'Q007_TASK_MIGRATION_PATH',
    'Q007_VALIDATION_RESULT_SHA256', 'build_current_ledger', 'build_ledger',
    'build_q007_ledger', 'entry', 'q007_segment_metadata', 'segment_metadata',
    'training_cost', 'validate_entries', 'validate_ledger',
]
