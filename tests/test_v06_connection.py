"""v0.6 connection: bounded pipeline, verified ACKs, 120 s barrier (CPU only)."""
import json

import pytest

from src.tasks.velocity.v06.connection import (DELIVERY_TIMEOUT, MAX_IN_FLIGHT, GROUP,
                                               MonitoringBlocked, ReliableConnection)


class FakeClock:
    def __init__(self):
        self.value = 1000.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class FakeBackend:
    enabled = True
    project = 'a2-test'
    entity = 'tester'

    def __init__(self):
        self.logged = []
        self.started = None
        self.finished = None

    def start(self, run_id, config):
        self.started = dict(run_id=run_id, config=config)
        return dict(run_id=run_id)

    def log(self, record, step=None):
        self.logged.append(dict(record, _step=step))
        return True

    def finish(self, summary):
        self.finished = dict(summary)
        return True


class FakeReader:
    """Remote view that only exposes events already published by the backend."""

    def __init__(self, backend, *, visible=True, identity=None):
        self.backend = backend
        self.visible = visible
        self.identity = identity or dict(name='run-1', group=GROUP, config={})

    def authenticated(self):
        return True

    def run_identity(self):
        return dict(self.identity)

    def rows(self, lo, hi):
        if not self.visible:
            return []
        return [dict(event_id=record['event_id'], _step=record.get('_step') or 0)
                for record in self.backend.logged if record.get('event_id')]

    def close(self):
        pass


def _connection(tmp_path, clock, backend, reader):
    conn = ReliableConnection(tmp_path, backend=backend, clock=clock,
                              reader_factory=lambda *args, **kwargs: reader)
    conn.reader = reader
    return conn


def test_start_verifies_remote_identity_and_acks(tmp_path):
    clock, backend = FakeClock(), FakeBackend()
    conn = _connection(tmp_path, clock, backend, FakeReader(backend))
    info = conn.start('run-1', {'recipe': '0.6-go2refs-r1'})
    assert info['remote_verified'] is True
    assert conn.state['online_status'] == 'remote_verified'
    assert conn.pending() == []
    assert any(event['kind'] == 'initialize' for event in conn.state['events'])


def test_identity_mismatch_blocks(tmp_path):
    clock, backend = FakeClock(), FakeBackend()
    reader = FakeReader(backend, identity=dict(name='run-1', group='v05_r2', config={}))
    conn = _connection(tmp_path, clock, backend, reader)
    with pytest.raises(MonitoringBlocked):
        conn.start('run-1', {})


def test_pipeline_is_bounded_and_ordered(tmp_path):
    clock, backend = FakeClock(), FakeBackend()
    reader = FakeReader(backend, visible=False)
    conn = _connection(tmp_path, clock, backend, reader)
    conn.state['run_verified'] = True
    conn.state['online_status'] = 'remote_verified'
    for index in range(MAX_IN_FLIGHT + 10):
        conn.state['events'].append(dict(
            id=f'id{index:03d}', kind='update', payload={'i': index}, step=index,
            created=clock()))
    assert conn.flush() is False
    in_flight = [key for key, value in conn.state['transport'].items()
                 if value['state'] == 'enqueued']
    assert len(in_flight) == MAX_IN_FLIGHT
    steps = [conn.state['transport'][key]['step'] for key in in_flight]
    assert steps == list(range(MAX_IN_FLIGHT))


def test_barrier_marks_monitoring_blocked_after_timeout(tmp_path):
    clock, backend = FakeClock(), FakeBackend()
    reader = FakeReader(backend, visible=False)
    conn = _connection(tmp_path, clock, backend, reader)
    conn.state['run_verified'] = True
    conn.state['online_status'] = 'remote_verified'
    conn.emit('update', {'train/update': 1})
    assert conn.state['monitoring_blocked'] is False
    clock.advance(DELIVERY_TIMEOUT + 1)
    conn.flush()
    assert conn.state['monitoring_blocked'] is True


def test_visible_remote_rows_ack_and_clear_barrier(tmp_path):
    clock, backend = FakeClock(), FakeBackend()
    reader = FakeReader(backend, visible=True)
    conn = _connection(tmp_path, clock, backend, reader)
    conn.state['run_verified'] = True
    conn.state['online_status'] = 'remote_verified'
    conn.emit('update', {'train/update': 1})
    assert conn.flush() is True
    assert conn.pending() == []
    assert conn.state['monitoring_blocked'] is False
    assert conn.state['acked']


def test_disabled_backend_is_local_only_not_ack(tmp_path):
    clock = FakeClock()

    class Disabled(FakeBackend):
        enabled = False

    conn = ReliableConnection(tmp_path, backend=Disabled(), clock=clock,
                              reader_factory=lambda *args, **kwargs: None)
    info = conn.start('run-1', {})
    assert info['online_status'] == 'disabled'
    conn.emit('update', {'train/update': 1})
    assert conn.state['online_status'] == 'disabled'
    assert conn.pending(), 'disabled transport must never fake an ACK'


def test_journal_is_written_atomically(tmp_path):
    clock, backend = FakeClock(), FakeBackend()
    conn = _connection(tmp_path, clock, backend, FakeReader(backend))
    conn.start('run-1', {})
    conn.emit('update', {'train/update': 7})
    payload = json.loads((tmp_path / 'connection.json').read_text())
    assert payload['session']['run_id'] == 'run-1'
    assert payload['events']
    assert not (tmp_path / 'connection.json.tmp').exists()


def test_finish_records_delivery_receipt(tmp_path):
    clock, backend = FakeClock(), FakeBackend()
    conn = _connection(tmp_path, clock, backend, FakeReader(backend))
    conn.start('run-1', {})
    receipt = conn.finish(dict(state='completed', update=3, phase='F0', budget={}))
    assert receipt['delivery']['status'] == 'remote_verified'
    assert receipt['delivery']['monitoring_blocked'] is False
    assert backend.finished is not None
