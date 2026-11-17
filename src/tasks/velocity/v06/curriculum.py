"""v0.6 curriculum: level ticks, 64-episode FIFO, EMA, generations and budgets.

SDS v0.6 sections 5.3, 5.5 and 5.6 are implemented here, and nowhere else:

* H uses an integer ``level_tick in [0, 100]``; ``frontier_row =
  floor(15 * level_tick / 100)`` selects ``T0..T15``.  Upgrades need
  ``S_EMA >= 0.65``, ``Q_EMA >= 0.45``, ``F_EMA <= 0.35`` for 5 consecutive
  64-episode checks and ``cooldown == 0``; downgrades need ``F_EMA >= 0.40`` or
  ``S_EMA <= 0.20`` for 2 consecutive checks and win over an upgrade.  A
  downgrade subtracts 3 ticks and is never blocked by the cooldown.
* Only **frontier** episodes enter the metrics.  The FIFO is ordered by episode
  end control step then env id, one window is consumed per update boundary and a
  not-yet-full window is retained.
* EMA is initialised with the first batch and then ``0.97 / 0.03``.
* A terrain change (H row, D/L substage, phase) starts a new generation: EMA,
  FIFO, ready/hard/gate streaks are cleared, the cooldown is set to 5 and the
  old generation's outstanding results are log-only.
* D/L substage budgets are cumulative per substage: regression keeps the spent
  budget, an old pass is never reused, and returning to an already exhausted
  substage stops with ``budget_exhausted``.
* V/R gate streaks only start counting after 1000 / 400 updates and promotion
  additionally needs 1200 / 600 updates; both are decoupled from the command /
  spawn ramp.

Every number comes from :mod:`src.tasks.velocity.v06.spec`; no SDS constant is
re-invented in this file.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Iterable, Mapping, Sequence

import numpy as np

from . import spec

# --------------------------------------------------------------- spec binding
WINDOW = int(spec.CURRICULUM_WINDOW)
WINDOWS_PER_UPDATE = int(spec.CURRICULUM_WINDOWS_PER_UPDATE)
EMA_DECAY = float(spec.EMA_DECAY)
EMA_NEW = float(spec.EMA_NEW)
ADVANCE = dict(s=float(spec.ADVANCE_S_MIN), q=float(spec.ADVANCE_Q_MIN),
               f=float(spec.ADVANCE_F_MAX), streak=int(spec.ADVANCE_STREAK))
REGRESS = dict(f=float(spec.REGRESS_F_MIN), s=float(spec.REGRESS_S_MAX),
               streak=int(spec.REGRESS_STREAK), ticks=int(spec.REGRESS_TICKS))
COOLDOWN = int(spec.COOLDOWN_CHECKS)
LEVEL_TICK_MIN = int(spec.LEVEL_TICK_MIN)
LEVEL_TICK_MAX = int(spec.LEVEL_TICK_MAX)
LEVEL_SCALE = int(spec.LEVEL_TICK_SCALE)
FRONTIER_ROWS = int(spec.FRONTIER_ROW_COUNT)
GATE_STREAK_REQUIRED = int(spec.GATE_STREAK_REQUIRED)
GATE_INTERVAL = int(spec.GATE_INTERVAL)
RETENTION_INTERVAL = int(spec.RETENTION_INTERVAL)
SUBSTAGE_ADVANCE = int(spec.SUBSTAGE_ADVANCE_STREAK)
SUBSTAGE_REGRESS = int(spec.SUBSTAGE_REGRESS_STREAK)
ROLE_NAMES = tuple(spec.ROLE_NAMES)
ROLE_TIE_ORDER = tuple(spec.ROLE_TIE_ORDER)
V_R_FRONTIER_TILE_ID = int(spec.V_R_FRONTIER_TILE_ID)
V_MIN_UPDATES = int(spec.V_MIN_UPDATES)
V_START = int(spec.V_GATE_STREAK_START)
R_MIN_UPDATES = int(spec.R_MIN_UPDATES)
R_START = int(spec.R_GATE_STREAK_START)
MIN_LEVEL_FAILURE_STREAK = int(spec.MIN_LEVEL_FAILURE_STREAK)
RETENTION_FAIL_STREAK = int(spec.RETENTION_FAIL_STREAK)
RETENTION_FAIL_MIN = int(spec.RETENTION_FAIL_MIN)
RETENTION_CASES = int(spec.RETENTION_CASES)
#: Fixed order of the two independent retention sets (SDS 6.3, Q001-7).
RETENTION_KINDS = ('flat', 'h00')
#: Gate-kind names understood by the evaluation layer (C's runner).
RETENTION_GATE_KINDS = {'flat': 'flat_retention', 'h00': 'stair_retention'}
_RETENTION_KIND_ALIASES = {
    'flat': 'flat', 'flat_retention': 'flat',
    'h00': 'h00', 'stair': 'h00', 'stair_retention': 'h00',
}
MIN_LEVEL_F = float(spec.MIN_LEVEL_FAILURE_F_MIN)
MIN_LEVEL_UPDATES = int(spec.MIN_LEVEL_PHASE_UPDATES)
NEAR_OFFSETS = tuple(int(v) for v in spec.NEAR_OFFSETS)
EASY_BACKOFF = int(spec.EASY_BACKOFF)
PHASE_ORDER = tuple(spec.PHASE_ORDER)
D_SUBSTAGES = tuple(spec.D_SUBSTAGES)
L_SUBSTAGES = tuple(spec.L_SUBSTAGES)
SUBSTAGE_BUDGETS = dict(zip(D_SUBSTAGES, spec.D_SUBSTAGE_BUDGETS))
SUBSTAGE_BUDGETS.update(dict(zip(L_SUBSTAGES, spec.L_SUBSTAGE_BUDGETS)))
FLAT_TILE_ID = int(spec.FLAT_TILE_ID)
ROLE_COUNTS = dict(spec.ROLE_COUNTS)
ROLE_TOTAL = int(spec.ROLE_COUNT_TOTAL)

#: One of the frozen SDS transition-event kinds.
EVENT_KINDS = ('advance', 'regress', 'generation', 'substage_advance',
               'substage_regress', 'final')


# ---------------------------------------------------------------- allocation
def allocate_roles(num_envs: int, phase: str) -> tuple[str, ...]:
    """Role tuple of one environment batch (frontier/episode order).

    F0 is all-flat.  Every other phase uses the SDS counts ``102 / 164 / 123 /
    123`` on 512 envs; any other batch size (engineering tests only) uses the
    integer largest-remainder method with the fixed tie order ``flat, frontier,
    near, easy``.
    """
    count = int(num_envs)
    if count < 1:
        raise ValueError('num_envs must be positive')
    if str(phase).upper() == 'F0':
        return ('flat',) * count
    if count == ROLE_TOTAL:
        counts = dict(ROLE_COUNTS)
    else:
        raw = {name: count * ROLE_COUNTS[name] / ROLE_TOTAL for name in ROLE_NAMES}
        counts = {name: int(math.floor(raw[name])) for name in ROLE_NAMES}
        remaining = count - sum(counts.values())
        order = sorted(ROLE_NAMES,
                       key=lambda name: (-(raw[name] - counts[name]),
                                         ROLE_TIE_ORDER.index(name)))
        for name in order[:remaining]:
            counts[name] += 1
    if sum(counts.values()) != count:
        raise RuntimeError('role allocation does not sum to num_envs')
    return tuple(name for name in ROLE_NAMES for _ in range(counts[name]))


def frontier_row(level_tick: int) -> int:
    """SDS 5.5 ``frontier_row = floor(15 * level_tick / 100)`` in ``[0, 15]``."""
    tick = int(level_tick)
    if not LEVEL_TICK_MIN <= tick <= LEVEL_TICK_MAX:
        raise ValueError(f'level_tick {tick} outside [{LEVEL_TICK_MIN}, {LEVEL_TICK_MAX}]')
    return min(FRONTIER_ROWS, int(math.floor(FRONTIER_ROWS * tick / LEVEL_SCALE)))


def terrain_index(tile_id: int) -> int:
    """``T`` index of a tile (``T[k] <-> tile_id = k + 1``)."""
    tile = int(tile_id)
    if tile <= FLAT_TILE_ID:
        raise ValueError('the flat tile has no T index')
    return tile - 1


def frontier_tile(phase: str, substage) -> int:
    """Frontier tile id of the current phase/substage/level tick."""
    phase = str(phase).upper()
    if phase == 'F0':
        return FLAT_TILE_ID
    if phase == 'H':
        raise ValueError('H frontier depends on level_tick; use frontier_tile_for_state')
    if phase in ('D', 'L'):
        if substage is None:
            raise ValueError(f'{phase} needs a substage')
        return int(spec.tile_id_for(substage))
    if phase in ('V', 'R'):
        return V_R_FRONTIER_TILE_ID
    raise ValueError(f'unknown v0.6 phase: {phase!r}')


def frontier_tile_for_state(phase: str, substage, level_tick: int) -> int:
    phase = str(phase).upper()
    if phase == 'H':
        return FLAT_TILE_ID + 1 + frontier_row(level_tick)
    return frontier_tile(phase, substage)


def role_tiles(roles: Sequence[str], frontier_tile_id: int, sampler) -> tuple[int, ...]:
    """Sample one tile per role from the current frontier (episode reset only).

    ``frontier: T[k]``; ``near: T[k-2]/T[k-1]`` for ``k >= 2`` else ``T[k]``;
    ``easy: T[0]..T[max(k-3, 0)]``, all equiprobable.  The flat role always
    uses tile 0.
    """
    roles = tuple(roles)
    frontier = int(frontier_tile_id)
    if frontier == FLAT_TILE_ID:
        return tuple(FLAT_TILE_ID for _ in roles)
    k = terrain_index(frontier)
    near_ids = tuple(frontier + offset for offset in NEAR_OFFSETS) if k >= 2 \
        else (frontier,)
    easy_last = max(k - EASY_BACKOFF, 0)
    easy_ids = tuple(FLAT_TILE_ID + 1 + j for j in range(easy_last + 1))
    out = []
    for role in roles:
        if role == 'flat':
            out.append(FLAT_TILE_ID)
        elif role == 'frontier':
            out.append(frontier)
        elif role == 'near':
            out.append(near_ids[int(sampler.integers(len(near_ids)))])
        elif role == 'easy':
            out.append(easy_ids[int(sampler.integers(len(easy_ids)))])
        else:
            raise ValueError(f'unknown v0.6 role: {role!r}')
    return tuple(out)


# ------------------------------------------------------------- state records
@dataclass(frozen=True)
class ScheduleAssignment:
    """One frozen schedule: phase, frontier and per-environment roles/tiles."""

    phase: str
    substage: object
    frontier_tile_id: int
    roles: tuple[str, ...]
    tile_ids: tuple[int, ...]
    generation: int = 0

    def as_dict(self):
        return dict(phase=self.phase, substage=self.substage,
                    frontier_tile_id=int(self.frontier_tile_id),
                    roles=list(self.roles), tile_ids=[int(v) for v in self.tile_ids],
                    generation=int(self.generation))


@dataclass(frozen=True)
class EpisodeOutcome:
    """One finished episode, in the shape the curriculum consumes."""

    env_id: int
    step: int
    role: str
    tile_id: int
    success: bool
    failure: bool
    timeout: bool
    reason: str
    tracking_score: float
    command: float
    generation: object = None

    @classmethod
    def from_record(cls, record: Mapping):
        if isinstance(record, EpisodeOutcome):
            return record
        if not isinstance(record, Mapping):
            raise TypeError('curriculum outcome must be a mapping or EpisodeOutcome')
        step = record.get('end_control_step',
                           record.get('terminal_step',
                                      record.get('step', record.get('control_step'))))
        if step is None:
            raise ValueError('curriculum outcome has no end control step')
        return cls(
            env_id=int(record['env_id']),
            step=int(step),
            role=str(record.get('role', 'frontier')),
            tile_id=int(record.get('tile_id', FLAT_TILE_ID)),
            success=bool(record.get('success', False)),
            failure=bool(record.get('failure', False)),
            timeout=bool(record.get('timeout', record.get('truncated', False))),
            reason=str(record.get('reason', 'running')),
            tracking_score=float(record.get('tracking_score', 0.0)),
            command=float(record.get('command', record.get('vx', 0.0))),
            generation=record.get('generation'),
        )

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TransitionEvent:
    """One curriculum transition (never more than the SDS allows per update)."""

    kind: str
    phase: str
    substage: object
    level_tick: int
    generation: int
    update: int
    status: str
    detail: dict = field(default_factory=dict)

    def as_dict(self):
        return dict(kind=self.kind, phase=self.phase, substage=self.substage,
                    level_tick=int(self.level_tick), generation=int(self.generation),
                    update=int(self.update), status=self.status,
                    detail=dict(self.detail))


@dataclass
class CurriculumState:
    """SDS 7.2 curriculum state, fully JSON-safe."""

    phase: str = 'F0'
    substage: object = None
    level_tick: int = 0
    generation: int = 0
    generation_updates: int = 0
    spent_by_phase: dict = field(default_factory=lambda: {name: 0 for name in PHASE_ORDER})
    spent_by_substage: dict = field(default_factory=lambda: {name: 0 for name in SUBSTAGE_BUDGETS})
    episode_fifo: list = field(default_factory=list)
    ema: dict = field(default_factory=lambda: dict(s=None, f=None, q=None, batches=0))
    ready_streak: int = 0
    hard_streak: int = 0
    cooldown: int = 0
    gate_streak: int = 0
    retention_streaks: dict = field(default_factory=lambda: dict(flat=0, h00=0))
    minimum_level_failure_streak: int = 0
    sampler_rng_state: dict = field(default_factory=dict)
    # --- explicit, serializable extras (kept minimal and documented) ---------
    frontier_tile_id: int = FLAT_TILE_ID
    retention_enabled: dict = field(default_factory=lambda: dict(flat=False, h00=False))
    #: Q002-6 per-set schedule: first qualified update q, strict next 200 multiple
    #: and the last completed check update.  Flat/H00 are fully independent.
    retention_qualified_at: dict = field(
        default_factory=lambda: dict(flat=None, h00=None))
    retention_next_due: dict = field(
        default_factory=lambda: dict(flat=None, h00=None))
    retention_last_completed: dict = field(
        default_factory=lambda: dict(flat=None, h00=None))
    last_update: int = 0
    gate_log: list = field(default_factory=list)
    finished: bool = False
    stop_status: object = None
    window_metrics: dict = field(default_factory=dict)


REQUIRED_STATE_FIELDS = tuple(f.name for f in fields(CurriculumState))


# ------------------------------------------------------------------ manager
class CurriculumManager:
    """Authoritative v0.6 curriculum (H ticks, D/L substages, V/R gates)."""

    def __init__(self, num_envs: int = 512, seed: int = 42, state=None):
        self.num_envs = int(num_envs)
        self.seed = int(seed)
        if state is None:
            sampler = np.random.default_rng(self.seed)
            self.state = CurriculumState(
                spent_by_phase={name: 0 for name in PHASE_ORDER},
                spent_by_substage={name: 0 for name in SUBSTAGE_BUDGETS},
                sampler_rng_state=dict(sampler.bit_generator.state),
                frontier_tile_id=FLAT_TILE_ID,
            )
            self._sampler = sampler
            self._roles = allocate_roles(self.num_envs, self.state.phase)
        else:
            self.load_state_dict(state)
        self._events = []
        self._validate()

    # ------------------------------------------------------------ state view
    def state_dict(self) -> dict:
        value = asdict(self.state)
        # keep the sampler RNG in sync with the live generator
        value['sampler_rng_state'] = dict(self._sampler.bit_generator.state)
        self.state.sampler_rng_state = dict(value['sampler_rng_state'])
        return _json_safe(value)

    def load_state_dict(self, value: dict) -> None:
        if not isinstance(value, Mapping):
            raise TypeError('curriculum state must be a mapping')
        missing = [name for name in REQUIRED_STATE_FIELDS if name not in value]
        if missing:
            raise ValueError(f'curriculum state is missing fields: {missing}')
        extra = sorted(set(value) - set(REQUIRED_STATE_FIELDS))
        if extra:
            raise ValueError(f'curriculum state has unknown fields: {extra}')
        known = {f.name for f in fields(CurriculumState)}
        payload = {name: value[name] for name in known if name in value}
        self.state = CurriculumState(**payload)
        sampler = np.random.default_rng(self.seed)
        if self.state.sampler_rng_state:
            sampler.bit_generator.state = dict(self.state.sampler_rng_state)
        self._sampler = sampler
        self._roles = allocate_roles(self.num_envs, self.state.phase)
        self._validate()

    def _validate(self):
        st = self.state
        if st.phase not in PHASE_ORDER:
            raise ValueError(f'unknown curriculum phase {st.phase!r}')
        if not 0 <= int(st.level_tick) <= LEVEL_TICK_MAX:
            raise ValueError('level_tick outside [0, 100]')
        for name in PHASE_ORDER:
            st.spent_by_phase.setdefault(name, 0)
        for name in SUBSTAGE_BUDGETS:
            st.spent_by_substage.setdefault(name, 0)
        for name in RETENTION_KINDS:
            st.retention_streaks.setdefault(name, 0)
            st.retention_enabled.setdefault(name, False)
            st.retention_qualified_at.setdefault(name, None)
            st.retention_next_due.setdefault(name, None)
            st.retention_last_completed.setdefault(name, None)
        st.ema.setdefault('s', None)
        st.ema.setdefault('f', None)
        st.ema.setdefault('q', None)
        st.ema.setdefault('batches', 0)
        for entry in st.episode_fifo:
            if not isinstance(entry, dict):
                raise TypeError('episode_fifo entries must be mappings')
        self._roles = allocate_roles(self.num_envs, st.phase)

    # -------------------------------------------------------------- schedule
    def schedule(self) -> ScheduleAssignment:
        phase = self.state.phase
        frontier = int(frontier_tile_for_state(phase, self.state.substage,
                                               self.state.level_tick))
        roles = allocate_roles(self.num_envs, phase)
        self._roles = roles
        tiles = role_tiles(roles, frontier, self._sampler)
        return ScheduleAssignment(phase=phase, substage=self.state.substage,
                                  frontier_tile_id=frontier, roles=roles,
                                  tile_ids=tiles,
                                  generation=int(self.state.generation))

    def resample_tiles(self, roles: Sequence[str], frontier_tile_id: int):
        """Sample tiles for one reset subset from the manager's sampler RNG."""
        return role_tiles(roles, frontier_tile_id, self._sampler)

    # --------------------------------------------------------------- observe
    def observe(self, outcomes) -> None:
        if self.state.finished:
            return
        if isinstance(outcomes, (EpisodeOutcome, Mapping)):
            outcomes = [outcomes]
        for raw in outcomes:
            outcome = EpisodeOutcome.from_record(raw)
            generation = outcome.generation
            if generation is not None and int(generation) != int(self.state.generation):
                # Old-generation results are log-only (SDS 5.5): never counted.
                continue
            if outcome.role != 'frontier':
                continue
            self._insert(outcome)

    def _insert(self, outcome: EpisodeOutcome) -> None:
        fifo = self.state.episode_fifo
        key = (int(outcome.step), int(outcome.env_id))
        entry = outcome.as_dict()
        position = len(fifo)
        for index in range(len(fifo)):
            if (int(fifo[index]['step']), int(fifo[index]['env_id'])) > key:
                position = index
                break
        fifo.insert(position, entry)

    # -------------------------------------------------------------- on_update
    def on_update(self, global_update: int, gates=None) -> list:
        """One complete update boundary; consumes at most one 64-episode window."""
        if self.state.finished:
            return []
        update = int(global_update)
        delta = update - int(self.state.last_update)
        if delta <= 0:
            return []
        st = self.state
        st.last_update = update
        st.spent_by_phase[st.phase] = int(st.spent_by_phase.get(st.phase, 0)) + delta
        if st.substage is not None:
            st.spent_by_substage[st.substage] = (
                int(st.spent_by_substage.get(st.substage, 0)) + delta)
        st.generation_updates += delta
        self._events = []
        for gate in gates or ():
            if isinstance(gate, TransitionEvent):
                continue
            self._ingest_gate(gate)
        self._process_window()
        self._gate_transitions()
        self._budget_check()
        return self._events

    # ------------------------------------------------------------------ gates
    def record_gate(self, *, passed, update, kind='gate') -> list:
        """Record one complete-gate result (called at its update boundary).

        Transitions are latched here but committed on the *next* ``on_update``
        boundary, so the environment schedule only changes at a full update
        boundary (SDS 5.5).
        """
        self._ingest_gate(dict(passed=passed, update=update, kind=kind))
        return []

    def _ingest_gate(self, gate) -> None:
        st = self.state
        if st.finished:
            return
        kind = str(gate.get('kind', 'gate'))
        update = int(gate.get('update', st.last_update))
        passed = bool(gate.get('passed', False))
        if kind in ('flat_retention', 'stair_retention'):
            return []
        phase = str(gate.get('phase', st.phase)).upper()
        substage = gate.get('substage', st.substage)
        record = dict(update=update, kind=kind, passed=passed, phase=phase,
                      substage=substage, generation=int(st.generation),
                      generation_updates=int(st.generation_updates),
                      counted=False, reason='')
        if any(entry.get('update') == update and entry.get('kind') == kind
               for entry in st.gate_log):
            return
        in_generation = st.generation_updates >= GATE_INTERVAL
        if update != int(st.last_update):
            record['reason'] = 'stale_gate_from_earlier_boundary'
        elif not in_generation:
            record['reason'] = 'generation_younger_than_100_updates'
        elif phase != st.phase or substage != st.substage:
            record['reason'] = 'gate_from_other_phase'
        elif st.phase in ('V', 'R'):
            start = V_START if st.phase == 'V' else R_START
            if int(st.spent_by_phase.get(st.phase, 0)) < start:
                record['reason'] = f'{st.phase.lower()}_gate_streak_starts_at_{start}'
            else:
                record['counted'] = True
        else:
            record['counted'] = True
        if record['counted']:
            st.gate_streak = st.gate_streak + 1 if passed else 0
            if (passed and st.phase == 'H' and st.level_tick == 0
                    and st.gate_streak >= GATE_STREAK_REQUIRED):
                self._qualify_retention('h00', update)
        record['gate_streak'] = int(st.gate_streak)
        st.gate_log.append(record)
        st.gate_log[:] = st.gate_log[-64:]

    # ---------------------------------------------------------------- window
    def _process_window(self) -> None:
        st = self.state
        if len(st.episode_fifo) < WINDOW:
            return
        for _ in range(WINDOWS_PER_UPDATE):
            if len(st.episode_fifo) < WINDOW:
                break
            batch = [st.episode_fifo.pop(0) for _ in range(WINDOW)]
            self._consume(batch)
            break  # one window per update boundary (SDS 5.5)

    def _consume(self, batch):
        st = self.state
        success = failure = timeout = 0
        scores = []
        for entry in batch:
            fail = bool(entry['failure'])
            succ = bool(entry['success']) and not fail
            tout = bool(entry['timeout']) and not fail and not succ
            if fail:
                failure += 1
            elif succ:
                success += 1
            elif tout:
                timeout += 1
            else:
                raise ValueError('curriculum window episode is neither success, '
                                 'failure nor timeout (numeric anomaly)')
            score = float(entry['tracking_score'])
            if not math.isfinite(score):
                raise ValueError('curriculum window tracking score is not finite')
            scores.append(score)
        total = len(batch)
        batch_metrics = dict(s=success / total, f=failure / total, t=timeout / total,
                             q=float(sum(scores) / total))
        st.ema, first = self._ema(batch_metrics)
        st.window_metrics = dict(batch=batch_metrics, ema=dict(st.ema),
                                 size=total, first_batch=bool(first))
        if st.cooldown > 0:
            st.cooldown = max(0, st.cooldown - 1)
        ema = st.ema
        advance_ok = (ema['s'] is not None and ema['s'] >= ADVANCE['s']
                      and ema['q'] is not None and ema['q'] >= ADVANCE['q']
                      and ema['f'] is not None and ema['f'] <= ADVANCE['f'])
        regress_ok = (ema['f'] is not None and ema['f'] >= REGRESS['f']) or (
            ema['s'] is not None and ema['s'] <= REGRESS['s'])
        st.ready_streak = st.ready_streak + 1 if advance_ok else 0
        st.hard_streak = st.hard_streak + 1 if regress_ok else 0
        if st.phase == 'H':
            self._h_window_decision()
        elif st.phase in ('D', 'L'):
            self._substage_regression()
        self._minimum_level_window()

    def _ema(self, batch_metrics):
        st = self.state
        first = int(st.ema.get('batches', 0)) == 0
        if first:
            st.ema = dict(s=batch_metrics['s'], f=batch_metrics['f'],
                          q=batch_metrics['q'], batches=1)
        else:
            st.ema = dict(
                s=EMA_DECAY * float(st.ema['s']) + EMA_NEW * batch_metrics['s'],
                f=EMA_DECAY * float(st.ema['f']) + EMA_NEW * batch_metrics['f'],
                q=EMA_DECAY * float(st.ema['q']) + EMA_NEW * batch_metrics['q'],
                batches=int(st.ema['batches']) + 1)
        return st.ema, first

    # --------------------------------------------------------------- H rules
    def _h_window_decision(self):
        st = self.state
        # A downgrade always wins over an upgrade (SDS 5.5); the two EMA
        # conditions are disjoint in practice but the ordering is explicit.
        if st.hard_streak >= REGRESS['streak']:
            self._level_change(-REGRESS['ticks'], 'regress')
        elif (st.ready_streak >= ADVANCE['streak'] and st.cooldown == 0
              and self._adaptive_allowed()):
            self._level_change(+1, 'advance')


    def _adaptive_allowed(self) -> bool:
        st = self.state
        if st.level_tick > 0:
            return True
        return st.gate_streak >= GATE_STREAK_REQUIRED

    def _level_change(self, delta, kind) -> list:
        st = self.state
        before = int(st.level_tick)
        after = min(LEVEL_TICK_MAX, max(LEVEL_TICK_MIN, before + int(delta)))
        if after == before:
            st.cooldown = COOLDOWN
            st.hard_streak = 0
            st.ready_streak = 0
            return
        st.level_tick = after
        st.ready_streak = 0
        st.hard_streak = 0
        st.cooldown = COOLDOWN
        new_frontier = frontier_tile_for_state(st.phase, st.substage, st.level_tick)
        events = [TransitionEvent(kind=kind, phase=st.phase, substage=st.substage,
                                  level_tick=after, generation=st.generation,
                                  update=int(st.last_update), status='committed',
                                  detail=dict(previous_level_tick=before,
                                              level_tick=after,
                                              frontier_tile_id=int(new_frontier)))]
        changed_row = int(new_frontier) != int(st.frontier_tile_id)
        st.frontier_tile_id = int(new_frontier)
        if changed_row:
            self._events.extend(self._new_generation('frontier_row'))
        self._events.extend(events)

    def _minimum_level_window(self):
        st = self.state
        if not self._at_lowest_level():
            st.minimum_level_failure_streak = 0
            return
        if st.ema.get('f') is not None and float(st.ema['f']) >= MIN_LEVEL_F:
            st.minimum_level_failure_streak += 1
        else:
            st.minimum_level_failure_streak = 0
        if (st.minimum_level_failure_streak >= MIN_LEVEL_FAILURE_STREAK
                and int(st.spent_by_phase.get(st.phase, 0)) >= MIN_LEVEL_UPDATES):
            st.finished = True
            st.stop_status = 'minimum_level_failure'
            self._events.append(TransitionEvent(
                kind='final', phase=st.phase, substage=st.substage,
                level_tick=st.level_tick, generation=st.generation,
                update=int(st.last_update), status='minimum_level_failure',
                detail=dict(streak=int(st.minimum_level_failure_streak),
                            f_ema=float(st.ema['f']))))

    def _at_lowest_level(self) -> bool:
        st = self.state
        if st.phase == 'H':
            return int(st.level_tick) <= LEVEL_TICK_MIN
        if st.phase == 'D':
            return st.substage == D_SUBSTAGES[0]
        if st.phase == 'L':
            return st.substage == L_SUBSTAGES[0]
        return False

    # ---------------------------------------------------------- D/L substages
    def _substage_regression(self):
        st = self.state
        if st.hard_streak < SUBSTAGE_REGRESS:
            return
        order = D_SUBSTAGES if st.phase == 'D' else L_SUBSTAGES
        if st.substage not in order:
            return
        index = order.index(st.substage)
        st.hard_streak = 0
        st.ready_streak = 0
        if index == 0:
            # SDS 5.6: never regress across the macro phase; stay at the lowest.
            st.cooldown = COOLDOWN
            self._events.append(TransitionEvent(
                kind='regress', phase=st.phase, substage=st.substage,
                level_tick=st.level_tick, generation=st.generation,
                update=int(st.last_update), status='lowest_substage_held',
                detail=dict(from_substage=st.substage, to_substage=st.substage)))
            return
        target = order[index - 1]
        self._events.append(self._commit_substage(
            target, kind='substage_regress', status='committed'))

    # ------------------------------------------------------------- transfer
    def _new_generation(self, reason) -> list:
        st = self.state
        st.generation += 1
        st.generation_updates = 0
        st.ema = dict(s=None, f=None, q=None, batches=0)
        st.episode_fifo = []
        st.ready_streak = 0
        st.hard_streak = 0
        st.gate_streak = 0
        st.cooldown = COOLDOWN
        st.minimum_level_failure_streak = 0
        st.window_metrics = {}
        return [TransitionEvent(kind='generation', phase=st.phase,
                                substage=st.substage, level_tick=st.level_tick,
                                generation=st.generation,
                                update=int(st.last_update), status='committed',
                                detail=dict(reason=reason))]

    def _commit_substage(self, substage, *, kind, status) -> TransitionEvent:
        st = self.state
        previous = st.substage
        st.substage = substage
        st.frontier_tile_id = int(spec.tile_id_for(substage))
        st.ready_streak = 0
        st.hard_streak = 0
        self._events.extend(self._new_generation('substage'))
        return TransitionEvent(kind=kind, phase=st.phase, substage=substage,
                               level_tick=st.level_tick, generation=st.generation,
                               update=int(st.last_update), status=status,
                               detail=dict(from_substage=previous, to_substage=substage,
                                           spent=int(st.spent_by_substage.get(substage, 0)),
                                           budget=int(SUBSTAGE_BUDGETS[substage])))

    def _commit_phase(self, phase, substage, *, kind, status='committed',
                      detail=None) -> TransitionEvent:
        st = self.state
        previous_phase, previous_substage = st.phase, st.substage
        st.phase = phase
        st.substage = substage
        st.frontier_tile_id = int(frontier_tile_for_state(phase, substage, st.level_tick))
        st.ready_streak = 0
        st.hard_streak = 0
        st.gate_streak = 0
        self._events.extend(self._new_generation('phase'))
        return TransitionEvent(kind=kind, phase=phase, substage=substage,
                               level_tick=st.level_tick, generation=st.generation,
                               update=int(st.last_update), status=status,
                               detail=dict(previous_phase=previous_phase,
                                           previous_substage=previous_substage,
                                           **(detail or {})))

    # ------------------------------------------------------ gate transitions
    def _gate_transitions(self) -> None:
        st = self.state
        events = []
        phase = st.phase
        if st.gate_streak < GATE_STREAK_REQUIRED:
            # No promotion without two complete gates; the D/L substage-budget
            # check below still runs (returning to an exhausted substage stops).
            self._substage_budget_check(events)
            self._events.extend(events)
            return
        if phase == 'F0':
            # Q002-1: F0->H is two complete gates; then flat retention qualifies.
            self._qualify_retention('flat', int(st.last_update))
            events.append(self._commit_phase('H', None, kind='advance'))
            st.level_tick = LEVEL_TICK_MIN
            st.frontier_tile_id = FLAT_TILE_ID + 1 + frontier_row(st.level_tick)
        elif phase == 'H':
            if int(st.level_tick) >= LEVEL_TICK_MAX:
                events.append(self._commit_phase('D', D_SUBSTAGES[0],
                                                 kind='advance'))
        elif phase in ('D', 'L'):
            order = D_SUBSTAGES if phase == 'D' else L_SUBSTAGES
            index = order.index(st.substage)
            if index + 1 < len(order):
                events.append(self._commit_substage(order[index + 1],
                                                    kind='substage_advance',
                                                    status='committed'))
            else:
                next_phase = 'L' if phase == 'D' else 'V'
                substage = L_SUBSTAGES[0] if next_phase == 'L' else None
                events.append(self._commit_phase(next_phase, substage,
                                                 kind='advance'))
        elif phase == 'V':
            if int(st.spent_by_phase.get('V', 0)) >= V_MIN_UPDATES:
                events.append(self._commit_phase('R', None, kind='advance'))
        elif phase == 'R':
            if int(st.spent_by_phase.get('R', 0)) >= R_MIN_UPDATES:
                st.finished = True
                st.stop_status = 'complete'
                events.append(TransitionEvent(
                    kind='final', phase='R', substage=None,
                    level_tick=st.level_tick, generation=st.generation,
                    update=int(st.last_update), status='complete',
                    detail=dict(gate_streak=int(st.gate_streak),
                                spent=int(st.spent_by_phase.get('R', 0)))))
        self._substage_budget_check(events)
        self._events.extend(events)

    def _substage_budget_check(self, events):
        st = self.state
        if st.phase not in ('D', 'L') or st.substage is None or st.finished:
            return
        budget = int(SUBSTAGE_BUDGETS.get(st.substage, 0))
        spent = int(st.spent_by_substage.get(st.substage, 0))
        if spent >= budget and not events:
            st.finished = True
            st.stop_status = 'budget_exhausted'
            events.append(TransitionEvent(
                kind='final', phase=st.phase, substage=st.substage,
                level_tick=st.level_tick, generation=st.generation,
                update=int(st.last_update), status='budget_exhausted',
                detail=dict(substage=st.substage, spent=spent, budget=budget)))

    # ----------------------------------------------------------- budget stop
    def _budget_check(self) -> None:
        st = self.state
        if st.finished:
            return
        budget = int(spec.phase_budget(st.phase))
        spent = int(st.spent_by_phase.get(st.phase, 0))
        if spent < budget:
            return
        st.finished = True
        st.stop_status = 'budget_exhausted'
        self._events.append(TransitionEvent(
            kind='final', phase=st.phase, substage=st.substage,
            level_tick=st.level_tick, generation=st.generation,
            update=int(st.last_update), status='budget_exhausted',
            detail=dict(phase=st.phase, spent=spent, budget=budget)))

    # ------------------------------------------------------------- retention
    def _qualify_retention(self, kind, update):
        """Enable one retention set and freeze its first strict 200 multiple.

        Q002-6: ``first_due = 200 * (floor(q / 200) + 1)`` where ``q`` is the
        global update at which the set first became qualified; the same update
        never runs the set, and later regressions/generation changes never
        postpone it (the schedule fields are not cleared by ``_new_generation``).
        """
        st = self.state
        name = _RETENTION_KIND_ALIASES[str(kind)]
        st.retention_enabled[name] = True
        if st.retention_qualified_at.get(name) is None:
            qualified = int(update)
            st.retention_qualified_at[name] = qualified
            st.retention_next_due[name] = (
                RETENTION_INTERVAL * (qualified // RETENTION_INTERVAL + 1))
        return dict(kind=name, qualified_at=st.retention_qualified_at[name],
                    next_due=st.retention_next_due[name])

    def retention_due(self, global_update: int):
        """All retention sets due at this boundary, in fixed ``('flat','h00')`` order.

        SDS 6.3 keeps two *independent* sets (flat after F0 passes, H00 after the
        H00 gates pass) with independent schedules (Q002-6): each set has its
        own ``retention_qualified_at``/``retention_next_due``/
        ``retention_last_completed``.  A set is due when ``global_update >=
        next_due``; the same update that qualified it never runs it
        (``next_due`` is the strict next 200 multiple), and both sets are
        returned together when both are due.  ``None`` means nothing is due;
        otherwise a tuple of short kinds (``'flat'``/``'h00'``) in fixed order,
        which the evaluation layer maps through :data:`RETENTION_GATE_KINDS`.
        """
        st = self.state
        update = int(global_update)
        if update <= 0:
            return None
        due = []
        for kind in RETENTION_KINDS:
            if not bool(st.retention_enabled.get(kind, False)):
                continue
            next_due = st.retention_next_due.get(kind)
            if next_due is None:
                # Not qualified through the Q002-6 schedule yet: never due.
                continue
            if update >= int(next_due):
                due.append(kind)
        return tuple(due) or None

    def record_retention(self, *, kind, failed_cases, passed=None, update=None):
        """Record one retention-set result; each kind keeps its own schedule/streak.

        ``kind`` accepts ``'flat'``/``'h00'`` and the gate names
        ``'flat_retention'``/``'stair_retention'``.  Only the addressed set is
        updated, so one set's result can never accumulate or clear the other's
        counter (SDS 6.3 / Q001-7).  The completion advances that set's
        ``next_due`` by exactly one 200-step period (Q002-6), sets
        ``last_completed`` and is idempotent per update so a recovered pending
        evaluation is completed once, never counted twice.  Returns the per-kind
        decision for the runner (``stop`` after ``RETENTION_FAIL_STREAK``
        consecutive ``>=12/15``).
        """
        name = _RETENTION_KIND_ALIASES.get(str(kind))
        if name is None:
            raise ValueError(f'unknown retention kind: {kind!r}')
        st = self.state
        failed = int(failed_cases)
        if failed < 0 or failed > RETENTION_CASES:
            raise ValueError(f'retention failed_cases {failed} outside [0, {RETENTION_CASES}]')
        completed_at = int(st.last_update if update is None else update)
        previous = st.retention_last_completed.get(name)
        if previous is not None and int(previous) == completed_at:
            return dict(kind=name, gate_kind=RETENTION_GATE_KINDS[name],
                        failed_cases=failed, cases=RETENTION_CASES,
                        streak=int(st.retention_streaks.get(name, 0)),
                        required=RETENTION_FAIL_STREAK,
                        stop=int(st.retention_streaks.get(name, 0)) >= RETENTION_FAIL_STREAK,
                        qualified_at=st.retention_qualified_at.get(name),
                        next_due=st.retention_next_due.get(name),
                        last_completed=int(previous), duplicate=True)
        if st.retention_qualified_at.get(name) is None:
            self._qualify_retention(name, completed_at)
        streak = (int(st.retention_streaks.get(name, 0)) + 1
                  if failed >= RETENTION_FAIL_MIN else 0)
        st.retention_streaks[name] = streak
        st.retention_last_completed[name] = completed_at
        next_due = st.retention_next_due.get(name)
        if next_due is not None:
            next_due = int(next_due)
            # Exactly one cadence step per completed check; a late (resumed)
            # completion lands on the first strict 200 multiple after it.
            while next_due <= completed_at:
                next_due += RETENTION_INTERVAL
            st.retention_next_due[name] = next_due
        return dict(kind=name, gate_kind=RETENTION_GATE_KINDS[name],
                    failed_cases=failed, cases=RETENTION_CASES, streak=streak,
                    required=RETENTION_FAIL_STREAK,
                    stop=streak >= RETENTION_FAIL_STREAK,
                    qualified_at=st.retention_qualified_at.get(name),
                    next_due=st.retention_next_due.get(name),
                    last_completed=completed_at, duplicate=False)

    # --------------------------------------------------------------- helpers
    @property
    def phase(self):
        return self.state.phase

    @property
    def substage(self):
        return self.state.substage

    @property
    def level_tick(self):
        return int(self.state.level_tick)

    @property
    def generation(self):
        return int(self.state.generation)

    @property
    def finished(self):
        return bool(self.state.finished)

    @property
    def stop_status(self):
        return self.state.stop_status

    def window_ready(self) -> bool:
        return len(self.state.episode_fifo) >= WINDOW

    def ema(self):
        return dict(self.state.ema)

    def state_json(self) -> dict:
        return self.state_dict()


def retention_gate_kind(kind):
    """Map a short retention kind to the evaluation gate kind (C's runner)."""
    try:
        return RETENTION_GATE_KINDS[str(kind)]
    except KeyError:
        raise ValueError(f'unknown retention kind: {kind!r}') from None


def _json_safe(value):
    """Recursively convert numpy scalars/containers to plain JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    return value


#: Compatibility alias (W0 4.4 names the dataclass in env.py; env re-exports it).
CurriculumAssignment = ScheduleAssignment
