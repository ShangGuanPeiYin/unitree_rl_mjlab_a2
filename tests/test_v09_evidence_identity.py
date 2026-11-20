"""Direct-reuse boundaries: real files/ledger; no simulation or CUDA."""
from copy import deepcopy
from dataclasses import asdict
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v07_bootstrap

a2_v07_bootstrap.activate(ROOT)
from src.tasks.velocity.v09 import cases
from v09_identity_fixtures import OLD_CODE, file_sha, make_identity_fixture


def context_type():
    name = 'src.tasks.velocity.v09.evidence_identity'
    assert importlib.util.find_spec(name) is not None, 'approved reuse needs an actual identity validator'
    return importlib.import_module(name).EvidenceIdentityContext


def context(fixture):
    return context_type().from_approval(fixture.approval_path, fixture.approval_sha256,
                                       **fixture.context_kwargs())


def test_verified_old_calibration_keeps_producer_and_records_current_consumer(tmp_path):
    f = make_identity_fixture(tmp_path)
    before = f.ledger_path.read_bytes()
    ctx = context(f)
    data, ref = ctx.read_legacy_json(f.calibration_path, kind='calibration',
        suite='train_calibration', manifest_sha256='a' * 64)
    assert data['code_sha256'] == OLD_CODE
    assert data['value_scale'] == 2.
    assert ref.producer_code_sha256 == OLD_CODE
    assert ref.consumer_code_sha256 == 'c' * 64
    assert ref.file_sha256 == file_sha(f.calibration_path)
    assert ref.approval_sha256 == f.approval_sha256
    assert json.loads(json.dumps(asdict(ref)))['path'] == str(f.calibration_path.resolve())
    assert ctx.approval_path == f.approval_path.resolve()
    assert ctx.run_dir == f.run_dir.resolve()
    assert f.ledger_path.read_bytes() == before


@pytest.mark.parametrize('field,value', [
    ('approved', False), ('scope', ['S1.1_learning']),
    ('producer_code_sha256', 'd' * 64), ('consumer_code_sha256', 'e' * 64),
    ('run_id', 'a-different-run'), ('run_dir', '/tmp/not-this-run'),
    ('sds_sha256', '0' * 64), ('source_sha256', '0' * 64),
])
def test_approval_cannot_widen_or_change_its_bound_identity(tmp_path, field, value):
    f = make_identity_fixture(tmp_path)
    f.approval[field] = value
    f.seal()
    with pytest.raises(ValueError):
        context(f)


@pytest.mark.parametrize('field', ['approved', 'old_producer_compatible'])
def test_unsigned_or_unapproved_review_cannot_authorize_reuse(tmp_path, field):
    f = make_identity_fixture(tmp_path)
    f.approval['review'][field] = False
    f.seal()
    with pytest.raises(ValueError):
        context(f)


def test_approval_file_hash_is_an_external_trust_anchor(tmp_path):
    f = make_identity_fixture(tmp_path)
    with pytest.raises(ValueError):
        context_type().from_approval(f.approval_path, '0' * 64, **f.context_kwargs())


def test_unlisted_path_or_mismatched_suite_and_manifest_never_get_raw_bytes(tmp_path):
    f = make_identity_fixture(tmp_path)
    ctx = context(f)
    for kwargs in (dict(suite='gate_A', manifest_sha256='a' * 64),
                   dict(suite='train_calibration', manifest_sha256='b' * 64)):
        with pytest.raises(ValueError):
            ctx.read_legacy_bytes(f.calibration_path, kind='calibration', **kwargs)
    with pytest.raises(ValueError):
        ctx.read_legacy_json(f.run_dir / 'not-approved.json', kind='calibration',
                             suite='train_calibration')


def test_expected_ref_does_not_bootstrap_trust_from_modified_artifact(tmp_path):
    f = make_identity_fixture(tmp_path)
    ctx = context(f)
    original = file_sha(f.calibration_path)
    f.calibration_path.write_text('not JSON; must fail hash before parsing')
    ref = ctx.expected_ref(f.calibration_path, kind='calibration', suite='train_calibration')
    assert ref.file_sha256 == original
    with pytest.raises(ValueError):
        ctx.read_legacy_json(f.calibration_path, kind='calibration', suite='train_calibration')


def test_approved_json_cannot_claim_another_original_producer(tmp_path):
    f = make_identity_fixture(tmp_path)
    changed = dict(f.calibration, code_sha256='d' * 64)
    f.put_json(f.calibration_path, changed)
    f.bind_file(f.calibration_path, kind='calibration', suite='train_calibration',
                manifest_sha256='a' * 64)
    f.seal()
    with pytest.raises(ValueError):
        context(f).read_legacy_json(f.calibration_path, kind='calibration', suite='train_calibration')


@pytest.mark.parametrize('kind', ['checkpoint', 'resource_report', 'learning_approval', 'diagnostic_result'])
def test_catalog_never_grants_other_old_output_types(tmp_path, kind):
    f = make_identity_fixture(tmp_path)
    path = f.put_json(f.run_dir / (kind + '.json'), {'code_sha256': OLD_CODE})
    f.bind_file(path, kind=kind, suite=None)
    f.seal()
    with pytest.raises(ValueError):
        context(f)


def test_duplicate_or_outside_run_catalog_entries_are_rejected(tmp_path):
    f = make_identity_fixture(tmp_path)
    f.approval['runtime_evidence'].append(deepcopy(f.approval['runtime_evidence'][0]))
    f.seal()
    with pytest.raises(ValueError):
        context(f)
    f.approval['runtime_evidence'].pop()
    outside = f.put_json(tmp_path / 'outside.json', f.calibration)
    f.bind_file(outside, kind='calibration', suite='train_calibration', manifest_sha256='a' * 64)
    f.seal()
    with pytest.raises(ValueError):
        context(f)


def test_live_ledger_may_append_without_changing_frozen_identity(tmp_path):
    f = make_identity_fixture(tmp_path)
    f.ledger.advance_state('S1', episode_id_highwater=1)
    data, _ = context(f).read_legacy_json(f.calibration_path, kind='calibration', suite='train_calibration')
    assert data['value_scale'] == 2.
    assert f.ledger.snapshot()['stages']['S1']['progress']['episode_id_highwater'] == 1


@pytest.mark.parametrize('mutation', ['truncate', 'rewrite'])
def test_ledger_prefix_cannot_be_rolled_back_or_changed(tmp_path, mutation):
    f = make_identity_fixture(tmp_path)
    raw = f.ledger_path.read_bytes()
    f.ledger_path.write_bytes(raw[:-1] if mutation == 'truncate' else raw.replace(b'S1', b'SX', 1))
    with pytest.raises(ValueError):
        context(f)


def test_audit_material_is_hash_only_even_with_candidate_token(tmp_path):
    f = make_identity_fixture(tmp_path)
    path = f.run_dir / 'old-private-audit.pt'
    path.write_bytes(b'do-not-return-this-old-audit-payload')
    f.bind_file(path, kind='audit_material', suite=None, category='audit_material')
    f.seal()
    ctx = context(f)
    ref = ctx.verify_ref(path, kind='audit_material', suite=None)
    assert ref.file_sha256 == file_sha(path)
    assert 'do-not-return' not in json.dumps(asdict(ref))
    with pytest.raises(PermissionError):
        ctx.read_legacy_bytes(path, kind='audit_material', suite=None, candidate_sha256='b' * 64)


def test_final_hash_check_never_implies_formal_unseal(tmp_path):
    f = make_identity_fixture(tmp_path)
    path = f.run_dir / 'evidence/teacher_final_1/summary.json'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'invalid JSON, sealed until real same-SHA A/B qualification')
    f.bind_file(path, kind='teacher_summary', suite='final_1', manifest_sha256='a' * 64)
    f.seal()
    ctx = context(f)
    assert ctx.verify_ref(path, kind='teacher_summary', suite='final_1').file_sha256 == file_sha(path)
    for candidate in (None, 'b' * 64):
        with pytest.raises(PermissionError):
            ctx.read_legacy_json(path, kind='teacher_summary', suite='final_1', candidate_sha256=candidate)


def test_frozen_manifest_uses_verified_input_without_inventing_producer_field(tmp_path):
    f = make_identity_fixture(tmp_path)
    manifest = cases.build_s1_manifest('train_calibration', calibration_cases_per_group=8)
    path = f.put_json(f.run_dir / 'cases/train_calibration.json', manifest)
    f.bind_file(path, kind='frozen_manifest', suite='train_calibration',
                manifest_sha256=manifest['manifest_sha256'], category='frozen_inputs')
    f.seal()
    result, ref = context(f).read_manifest(path, purpose='training', suite='train_calibration')
    assert len(result['cases']) == 600
    assert 'code_sha256' not in result
    assert ref.manifest_sha256 == manifest['manifest_sha256']
    with pytest.raises(PermissionError):
        context(f).read_manifest(path, purpose='diagnostic', suite='train_calibration')


def test_jsonl_old_producer_is_proved_by_bound_original_summary(tmp_path):
    from src.tasks.velocity.v09 import spec
    f = make_identity_fixture(tmp_path)
    folder = f.run_dir / 'evidence/teacher_gate_A'
    folder.mkdir(parents=True)
    episode = folder / 'episodes.jsonl'
    episode.write_bytes(b'{"case_id":0}\n')
    summary = f.put_json(folder / 'summary.json', dict(code_sha256=OLD_CODE,
        checkpoint_sha256=spec.SOURCE_SHA256, manifest_sha256='a' * 64))
    f.bind_file(episode, kind='teacher_episodes', suite='gate_A', manifest_sha256='a' * 64)
    f.bind_file(summary, kind='teacher_summary', suite='gate_A', manifest_sha256='a' * 64)
    f.seal()
    raw, ref = context(f).read_legacy_bytes(episode, kind='teacher_episodes', suite='gate_A')
    assert raw == b'{"case_id":0}\n'
    assert ref.producer_code_sha256 == OLD_CODE
    f.put_json(summary, dict(code_sha256='d' * 64,
        checkpoint_sha256=spec.SOURCE_SHA256, manifest_sha256='a' * 64))
    f.bind_file(summary, kind='teacher_summary', suite='gate_A', manifest_sha256='a' * 64)
    f.seal()
    with pytest.raises(ValueError):
        context(f).read_legacy_bytes(episode, kind='teacher_episodes', suite='gate_A')


def test_long_lived_context_rechecks_original_ledger_prefix_before_read(tmp_path):
    f = make_identity_fixture(tmp_path)
    ctx = context(f)
    raw = f.ledger_path.read_bytes()
    f.ledger_path.write_bytes(raw.replace(b'S1', b'SX', 1))
    with pytest.raises(ValueError):
        ctx.read_legacy_json(f.calibration_path, kind='calibration', suite='train_calibration')


def test_cached_context_rejects_tampered_new_tail_even_when_original_prefix_is_intact(tmp_path):
    f = make_identity_fixture(tmp_path)
    ctx = context(f)
    original = f.ledger_path.read_bytes()
    f.ledger.advance_state('S1', episode_id_highwater=1)
    lines = f.ledger_path.read_bytes().splitlines(keepends=True)
    tail = json.loads(lines[-1])
    tail['event_sha256'] = '0' * 64
    lines[-1] = (json.dumps(tail, sort_keys=True) + '\n').encode()
    f.ledger_path.write_bytes(b''.join(lines))
    assert f.ledger_path.read_bytes().startswith(original)
    with pytest.raises(ValueError):
        ctx.read_legacy_json(f.calibration_path, kind='calibration', suite='train_calibration')


def test_approval_schema_identifier_requires_json_integer_not_float(tmp_path):
    f = make_identity_fixture(tmp_path)
    f.approval['schema'] = 1.0
    f.seal()
    with pytest.raises(ValueError):
        context(f)


def test_prefix_and_live_state_are_never_verified_from_different_file_versions(tmp_path, monkeypatch):
    import hashlib
    from src.tasks.velocity.v09.budget import AppendOnlyLedger
    f = make_identity_fixture(tmp_path)
    f.ledger.advance_state('S1', episode_id_highwater=1)
    original = f.ledger_path.read_bytes()
    state = f.ledger.snapshot()
    f.approval['ledger_prefix'].update(byte_length=len(original),
        sha256=hashlib.sha256(original).hexdigest(), sequence=state['sequence'],
        event_sha256=state['ledger_sha256'])
    f.seal()
    ctx = context(f)
    replacement = AppendOnlyLedger(tmp_path / 'replacement.jsonl', run_id=f.run_id)
    replacement.advance_state('S1', episode_id_highwater=2)
    other_image = replacement.path.read_bytes()
    assert len(original) == len(other_image) and original != other_image
    original_snapshot = AppendOnlyLedger.snapshot

    def swap_before_second_file_snapshot(ledger):
        if ledger.path.resolve() == f.ledger_path.resolve():
            f.ledger_path.write_bytes(other_image)
        return original_snapshot(ledger)

    monkeypatch.setattr(AppendOnlyLedger, 'snapshot', swap_before_second_file_snapshot)
    try:
        data, _ = ctx.read_legacy_json(f.calibration_path, kind='calibration', suite='train_calibration')
    except ValueError:
        return
    assert data['value_scale'] == 2.
    assert f.ledger_path.read_bytes()[:len(original)] == original, (
        'returned data after accepting the prefix of one ledger and the state of a different ledger')
