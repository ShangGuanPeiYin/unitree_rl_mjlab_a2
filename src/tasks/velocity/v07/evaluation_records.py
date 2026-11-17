"""Q010: immutable current-result serialization and atomic evidence sidecars.

Uses the evaluation module's existing table/records/summary canonical hashes.
No policy execution, random-state changes, timestamps or history selection.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

from . import evaluation

SCHEMA = 'v07-first-episode-result-v1'
_KEYS = {'schema', 'identity', 'table', 'records', 'summary', 'isolation', 'evidence_sha256'}
_HASH_KEYS = {'case_table_sha256', 'records_sha256', 'summary_sha256'}


def _canonical(value):
    return evaluation._canonical(value)


def _plain(value):
    return json.loads(_canonical(value))


def _digest(value):
    return hashlib.sha256(_canonical(value).encode('utf-8')).hexdigest()


def restore_result(payload, expected_identity):
    """Validate all cached evidence and reconstruct exactly the same EvalResult."""
    try:
        if not isinstance(payload, dict) or set(payload) != _KEYS or payload['schema'] != SCHEMA:
            raise evaluation.EvaluationError('unknown or incomplete Q010 result envelope')
        if not isinstance(expected_identity, dict) or not expected_identity:
            raise evaluation.EvaluationError('Q010 expected identity must be explicit')
        if _canonical(payload['identity']) != _canonical(expected_identity):
            raise evaluation.EvaluationError('Q010 cached result identity mismatch')
        evidence = payload['evidence_sha256']
        if not isinstance(evidence, dict) or set(evidence) != _HASH_KEYS:
            raise evaluation.EvaluationError('Q010 result has incomplete evidence hashes')
        for value in evidence.values():
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise evaluation.EvaluationError('Q010 result hash is not SHA256')
        table = evaluation.table_from_json(payload['table'], expected_sha=evidence['case_table_sha256'])
        if _canonical(table.to_json()) != _canonical(payload['table']):
            raise evaluation.EvaluationError('Q010 table identity/metadata differs from current contract')
        records = _plain(payload['records'])
        summary = _plain(payload['summary'])
        isolation = _plain(payload['isolation'])
        if not isinstance(records, list) or not isinstance(summary, dict) or not isinstance(isolation, dict):
            raise evaluation.EvaluationError('Q010 result blocks have invalid types')
        if set(isolation) != {'statistics_unchanged','statistics_changed','snapshot_sha'}:
            raise evaluation.EvaluationError('Q010 isolation must contain exactly the three checked snapshot fields')
        snapshot_sha = isolation['snapshot_sha']
        if not isinstance(snapshot_sha,str) or len(snapshot_sha)!=64 or any(c not in '0123456789abcdef' for c in snapshot_sha):
            raise evaluation.EvaluationError('Q010 isolation snapshot SHA is not valid')
        if isolation['statistics_unchanged'] is not True or not isinstance(isolation['statistics_changed'],list) or isolation['statistics_changed'] != []:
            raise evaluation.EvaluationError('Q010 result did not preserve the checked training statistics')
        for key in ('phase','substage','kind'):
            if key not in expected_identity or expected_identity[key] != getattr(table,key):
                raise evaluation.EvaluationError('Q010 table context differs from its expected '+key)
        if expected_identity.get('case_table_sha256') != table.sha256:
            raise evaluation.EvaluationError('Q010 expected task does not bind this case table SHA')
        if _digest(records) != evidence['records_sha256'] or _digest(summary) != evidence['summary_sha256']:
            raise evaluation.EvaluationError('Q010 records or summary SHA mismatch')
        recomputed = evaluation.summarize(records, table)
        recomputed.update(isolation)
        if _canonical(recomputed) != _canonical(summary):
            raise evaluation.EvaluationError('Q010 cached summary disagrees with its actual case records')
        return evaluation.EvalResult(table=table, summary=summary, records=records,
                                     isolation=isolation, evidence_sha256=deepcopy(evidence))
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise evaluation.EvaluationError('invalid Q010 result data: '+str(error)) from error


def pack_result(result, identity):
    """Deep-freeze the actual return value; never regenerate or evaluate cases."""
    payload = _plain(dict(schema=SCHEMA, identity=identity, table=result.table.to_json(),
                          records=result.records, summary=result.summary,
                          isolation=result.isolation, evidence_sha256=result.evidence_sha256))
    restore_result(payload, identity)
    return payload


def _existing_matches(path, payload):
    try:
        existing = json.loads(path.read_text(encoding='utf-8'))
        if _canonical(existing) != _canonical(payload):
            raise evaluation.EvaluationError('conflicting Q010 evidence already exists: '+str(path))
        restore_result(existing, payload['identity'])
    except (ValueError, UnicodeError) as error:
        raise evaluation.EvaluationError('existing Q010 evidence is invalid: '+str(path)) from error


def _binding(path, payload):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                **deepcopy(payload['evidence_sha256']))


def verify_sidecar(binding, payload):
    """Read-only check of the actual file referenced by an applied result."""
    if not isinstance(binding,dict) or set(binding)!={'path','sha256',*_HASH_KEYS}:
        raise evaluation.EvaluationError('Q010 sidecar binding is incomplete')
    if not isinstance(binding['path'],str) or not binding['path']:
        raise evaluation.EvaluationError('Q010 sidecar path is missing')
    path=Path(binding['path'])
    _existing_matches(path,payload)
    if _canonical(_binding(path,payload))!=_canonical(binding):
        raise evaluation.EvaluationError('Q010 bound sidecar file SHA or evidence differs')
    return True


def _preserve_incomplete(path):
    """Retain a failed, non-JSON partial before retrying the same cached result."""
    for index in range(10000):
        archive = path.with_name(path.name+f'.failed.{index:04d}')
        try:
            os.link(path, archive)
        except FileExistsError:
            continue
        path.unlink()
        return
    raise evaluation.EvaluationError('too many retained Q010 partial attempts')


def _sync_file_and_directory(path):
    # Retry durability even when a previous attempt left complete JSON/final bytes.
    with Path(path).open('rb') as stream:
        os.fsync(stream.fileno())
    descriptor = os.open(Path(path).parent,os.O_RDONLY | getattr(os,'O_DIRECTORY',0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_sidecar(path, payload):
    """Atomically install a validated result, refuse conflicts, retain failures.

    A hard link installs the fsynced temporary without replacing an existing
    final path. A complete identical temporary is reused after installation
    failure; incomplete bytes are retained under a numbered failure path.
    """
    path = Path(path)
    restore_result(payload, payload.get('identity') if isinstance(payload, dict) else None)
    if path.exists():
        _existing_matches(path, payload)
        _sync_file_and_directory(path)
        _existing_matches(path, payload)
        return _binding(path, payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.partial')
    encoded = (_canonical(payload)+'\n').encode('utf-8')
    if temporary.exists():
        try:
            json.loads(temporary.read_text(encoding='utf-8'))
        except (ValueError, UnicodeError):
            _preserve_incomplete(temporary)
        else:
            _existing_matches(temporary, payload)
    if not temporary.exists():
        # No finally cleanup: any write/fsync/install failure retains evidence.
        with temporary.open('xb') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    _sync_file_and_directory(temporary)
    try:
        os.link(temporary, path)
    except FileExistsError:
        _existing_matches(path, payload)
    # Persist the final directory entry before treating the sidecar as committed.
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _existing_matches(path, payload)
    temporary.unlink()
    return _binding(path, payload)
