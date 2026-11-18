"""v0.8 reliable W&B delivery (owner: main dialogue, W0 section 3).

``v08/telemetry.py`` produces panels and local JSONL evidence; this module adds
the bounded, verified transport that the previous A2 lines already used, so the
"existing W&B monitoring timeout policy" of SDS 8.5 is preserved rather than
silently weakened:

* at most 32 ordered SDK enqueues in flight;
* a 2 s delivery retry clock;
* an independent 30 s heartbeat;
* a 120 s no-progress / oldest-pending barrier that marks
  ``monitoring_blocked``;
* every ACK requires a real remote readback of the stable ``event_id`` (a local
  ``log`` return is an enqueue, never an ACK).

The transport is optional at the seam: when the backend is disabled the class
still writes the same local evidence and records ``online_status='disabled'``,
which is *not* an ACK and never counts as remote verification.  Training code
must treat a disabled transport as "local evidence only".

Only this class may talk to Forge; it never edits desktop routing, credentials
or another run.
"""
from __future__ import annotations

import hashlib
import json
import netrc
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

import requests

from .telemetry import V08Telemetry

BASE_URL = 'https://api.forge.coreweave.com'
PROXY = 'http://127.0.0.1:7897'
GROUP = 'v08'
HEARTBEAT_KIND = 'v08_heartbeat'
POLL_SECONDS = 2
HEARTBEAT_SECONDS = 30
DELIVERY_TIMEOUT = 120
MAX_IN_FLIGHT = 32


def configure_network():
    """Set this process only; never edit desktop routing or global credentials."""
    if os.environ.get('WANDB_BASE_URL', BASE_URL).rstrip('/') != BASE_URL:
        raise ValueError('Refusing to change the configured W&B service')
    proxy = os.environ.get('A2_WANDB_PROXY', PROXY)
    parsed = urlsplit(proxy)
    if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost')
            or not parsed.port or parsed.username):
        raise ValueError('A2_WANDB_PROXY must be a credential-free local HTTP proxy')
    os.environ['WANDB_BASE_URL'] = BASE_URL
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
        os.environ[key] = proxy
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = 'localhost,127.0.0.1,::1'
    return proxy


class RemoteReadError(ConnectionError):
    """Safe error class: never include responses, credentials or request headers."""


class MonitoringBlocked(RuntimeError):
    """Delivery could not be verified inside the bounded window."""


class ForgeReader:
    def __init__(self, entity, project, run_id, *, proxy=None):
        self.entity, self.project, self.run_id = entity, project, run_id
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.proxies = {'https': proxy or configure_network()}
        key = os.environ.get('WANDB_API_KEY')
        if key:
            self.session.auth = ('api', key)
        else:
            auth = netrc.netrc().authenticators(urlsplit(BASE_URL).hostname)
            if not auth:
                raise ValueError('No existing Forge credential found')
            self.session.auth = (auth[0], auth[2])

    def query(self, query, variables=None):
        try:
            response = self.session.post(BASE_URL + '/graphql',
                                         json={'query': query,
                                               'variables': variables or {}},
                                         timeout=(5, 10), allow_redirects=False)
            if response.status_code != 200:
                raise RemoteReadError('HTTP_' + str(response.status_code))
            data = response.json()
            if data.get('errors') or 'data' not in data:
                raise RemoteReadError('GraphQL_error')
            return data['data']
        except requests.RequestException as error:
            raise RemoteReadError(type(error).__name__) from None

    def authenticated(self):
        return bool(self.query('query { viewer { id } }').get('viewer'))

    def run_query(self, fields, extra=None, declarations=''):
        variables = dict(e=self.entity, p=self.project, r=self.run_id, **(extra or {}))
        query = ('query($e:String!,$p:String!,$r:String!' + declarations
                 + '){project(name:$p,entityName:$e){run(name:$r){' + fields + '}}}')
        data = self.query(query, variables)
        project = data.get('project')
        if not project or not project.get('run'):
            raise RemoteReadError('Run_not_found')
        return project['run']

    def rows(self, lo, hi):
        raw = self.run_query('history(minStep:$lo,maxStep:$hi,samples:10000,maxKeyLimit:100000)',
                             {'lo': lo, 'hi': hi}, ',$lo:Int64!,$hi:Int64!')['history']
        rows = [json.loads(row) for row in raw]
        if len(rows) >= 10000 or any(not lo <= row['_step'] < hi for row in rows):
            raise RemoteReadError('History_page_boundary_or_truncation')
        identifiers = [row['event_id'] for row in rows if row.get('event_id')]
        if len(identifiers) != len(set(identifiers)):
            raise RemoteReadError('Duplicate_event_ids')
        return rows

    def run_identity(self):
        metadata = self.run_query('name group config')
        values = metadata.get('config')
        if isinstance(values, str):
            values = json.loads(values)
        values = {key: value.get('value') if isinstance(value, dict) and 'value' in value
                  else value for key, value in (values or {}).items()}
        return dict(name=metadata['name'], group=metadata.get('group'), config=values)

    def close(self):
        self.session.close()


class ReliableConnection(V08Telemetry):
    """V08Telemetry plus verified, bounded delivery.

    The deterministic bookkeeping is injectable (``backend``/``reader_factory``/
    ``clock``) so the pipeline can be unit-tested without a GPU or network.
    """

    def __init__(self, directory=None, backend=None, *, proxy=None,
                 reader_factory=ForgeReader, clock=time.time,
                 delivery_timeout=DELIVERY_TIMEOUT, max_in_flight=MAX_IN_FLIGHT):
        super().__init__(directory, backend=backend, clock=clock)
        self.proxy = proxy
        self.reader_factory = reader_factory
        self.delivery_timeout = int(delivery_timeout)
        self.max_in_flight = int(max_in_flight)
        self.reader = None
        self.remote_floor = 0
        self.state = dict(session=None, events=[], acked=[], transport={},
                          next_transport_step=0, online_status='unstarted',
                          pending_count=0, monitoring_blocked=False,
                          last_error_type=None, run_verified=False)
        self.failed_since = None
        self.last_heartbeat = float(clock())

    # ------------------------------------------------------------ persistence
    @property
    def journal_path(self):
        if self.directory is None:
            return None
        return Path(self.directory) / 'connection.json'

    def _save(self):
        path = self.journal_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        os.replace(tmp, path)

    # ---------------------------------------------------------------- identity
    def _event_id(self, kind, payload):
        body = json.dumps([kind, payload], sort_keys=True, allow_nan=False)
        return hashlib.sha256(body.encode()).hexdigest()

    def pending(self):
        acked = set(self.state['acked'])
        return [event for event in self.state['events'] if event['id'] not in acked]

    # ------------------------------------------------------------------- start
    def start(self, run_id, config):
        session = super().start(run_id, config)
        self.state['session'] = dict(run_id=str(run_id))
        backend = self.backend
        enabled = bool(getattr(backend, 'enabled', False)) and self.directory is not None
        if not enabled:
            self.state['online_status'] = 'disabled'
            self.state['run_verified'] = False
            self._save()
            return dict(session, online_status='disabled', remote_verified=False)
        proxy = self.proxy or configure_network()
        entity = getattr(backend, 'entity', None) or getattr(backend, 'project', None)
        project = getattr(backend, 'project', None)
        self.reader = self.reader_factory(entity, project, run_id, proxy=proxy)
        if not self.reader.authenticated():
            self.state['online_status'] = 'auth_failed'
            self._save()
            raise MonitoringBlocked('Forge authentication preflight failed')
        identity = self.reader.run_identity()
        if identity['name'] != run_id or identity['group'] != GROUP:
            self.state['online_status'] = 'identity_mismatch'
            self._save()
            raise MonitoringBlocked('Remote run identity/group verification failed')
        self.state['run_verified'] = True
        self.state['online_status'] = 'remote_verified'
        self._save()
        if not any(event['kind'] == 'initialize' for event in self.state['events']):
            self.emit('initialize', {'global_iteration': 0,
                                     'system/online_initialization': 1})
        if not self.drain():
            raise MonitoringBlocked('Remote delivery unconfirmed after the startup deadline')
        return dict(session, online_status='remote_verified', remote_verified=True,
                    panels=list(self.panels()['panels']))

    # -------------------------------------------------------------------- emit
    def _emit(self, record):
        flat = super()._emit(record)
        kind = record.get('kind', 'record')
        payload = {key: value for key, value in flat.items()}
        self.emit(kind, payload, step=record.get('train', {}).get('update'))
        return flat

    def emit(self, kind, metrics, *, step=None):
        payload = dict(metrics)
        event = dict(id=self._event_id(kind, [payload, time.time()]), kind=kind,
                     payload=payload, step=step, created=float(self.clock()))
        self.state['events'].append(event)
        self.flush()
        return event

    # ------------------------------------------------------------------- flush
    def _remote_ids(self):
        pending = self.pending()
        if not pending:
            return set()
        intents = [self.state['transport'][event['id']]['step'] for event in pending
                   if event['id'] in self.state['transport']]
        self.remote_floor = min(intents) if intents else max(0, self.state['next_transport_step'] - 1)
        seen = set()
        stop = max(self.state['next_transport_step'], self.remote_floor + 1)
        for lo in range(self.remote_floor, stop, 1000):
            for row in self.reader.rows(lo, min(lo + 1000, stop)):
                if row.get('event_id'):
                    seen.add(row['event_id'])
        return seen

    def flush(self):
        pending = self.pending()
        if not pending:
            self.failed_since = None
            self.state.update(online_status='remote_verified' if self.state['run_verified']
                              else self.state['online_status'],
                              pending_count=0, monitoring_blocked=False)
            self._save()
            return True
        if self.state['online_status'] == 'disabled':
            self.state['pending_count'] = len(pending)
            self._save()
            return False
        if self.reader is None:
            self.state.update(online_status='reader_missing', pending_count=len(pending))
            self._save()
            return False
        error_kind = None
        try:
            remote = self._remote_ids()
            for event in pending:
                if event['id'] not in remote:
                    continue
                transport = self.state['transport'].get(event['id'])
                if transport is not None:
                    transport['state'] = 'confirmed'
                    transport['confirmed_wall_time'] = float(self.clock())
                self.state['acked'].append(event['id'])
                self._save()
            left = self.pending()
            in_flight = sum(event['id'] in self.state['transport'] for event in left)
            for event in left:
                if in_flight >= self.max_in_flight:
                    break
                if event['id'] in self.state['transport']:
                    continue
                step = self.state['next_transport_step']
                self.state['transport'][event['id']] = dict(
                    step=step, state='intent', created=float(self.clock()))
                self.state['next_transport_step'] = step + 1
                self._save()
                try:
                    self.backend.log(event['payload'] | {'event_id': event['id'],
                                                         'transport_step': step},
                                     step=event.get('step'))
                except Exception as error:  # noqa: BLE001
                    error_kind = type(error).__name__
                    self.state['last_error_type'] = error_kind
                    break
                self.state['transport'][event['id']]['state'] = 'enqueued'
                in_flight += 1
                self._save()
            if error_kind is None:
                left = self.pending()
        except RemoteReadError as error:
            error_kind = 'RemoteReadError'
            self.state['last_error_type'] = str(error)
            left = self.pending()
        if not left and error_kind is None:
            self.failed_since = None
            self.state.update(online_status='remote_verified', pending_count=0)
            self._save()
            return True
        if self.failed_since is None:
            self.failed_since = self.clock()
        oldest = max(0, self.clock() - min(event['created'] for event in left)) if left else 0
        self.state.update(online_status=error_kind or 'awaiting_remote_visibility',
                          pending_count=len(left), oldest_pending_seconds=oldest)
        if (self.clock() - self.failed_since >= self.delivery_timeout
                or oldest >= self.delivery_timeout):
            self.state['monitoring_blocked'] = True
        self._save()
        return False

    def drain(self, timeout=None, wait=time.sleep):
        deadline = self.clock() + (self.delivery_timeout if timeout is None else timeout)
        while True:
            if self.flush():
                return True
            if self.clock() >= deadline:
                return False
            wait(POLL_SECONDS)

    def heartbeat(self, metrics):
        if self.clock() - self.last_heartbeat >= HEARTBEAT_SECONDS:
            if not any(event['kind'] == HEARTBEAT_KIND for event in self.pending()):
                self.emit(HEARTBEAT_KIND, metrics)
            self.last_heartbeat = self.clock()
        return self.flush()

    def finish(self, result):
        summary = super().finish(result)
        self.emit('stop', {key: value for key, value in summary.items()
                           if isinstance(value, (str, int, float, type(None)))})
        verified = self.drain()
        receipt = dict(status='remote_verified' if verified else 'unconfirmed',
                       online_status=self.state['online_status'],
                       pending=self.state['pending_count'],
                       monitoring_blocked=self.state['monitoring_blocked'])
        self.state['delivery_finish_receipt'] = receipt
        self._save()
        if self.backend is not None:self.backend.finish(summary)
        return dict(summary, delivery=receipt)

    def close(self):
        if self.reader is not None:
            self.reader.close()
            self.reader = None


__all__ = ['BASE_URL', 'DELIVERY_TIMEOUT', 'GROUP', 'HEARTBEAT_KIND', 'HEARTBEAT_SECONDS',
           'MAX_IN_FLIGHT', 'MonitoringBlocked', 'POLL_SECONDS', 'ReliableConnection',
           'RemoteReadError', 'configure_network']
