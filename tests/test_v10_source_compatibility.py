"""CPU-only explicit inheritance; no old receipt is relabelled or executed."""
import copy
import json
from pathlib import Path
import pytest
from src.tasks.velocity.v10 import source_compatibility as c


@pytest.fixture
def bridge(tmp_path,monkeypatch):
    old=tmp_path/'old';new=tmp_path/'new'
    for root in (old,new):
        (root/'src/tasks/velocity/v10').mkdir(parents=True)
        (root/'src/tasks/velocity/v10/runtime.py').write_text('old')
        (root/'src/tasks/velocity/v10/rewards.py').write_text('unchanged reward')
    (new/'src/tasks/velocity/v10/runtime.py').write_text('handoff only')
    (old/'manifest.json').write_text(json.dumps(c._files(old)))
    from src.tasks.velocity.v10.identity import code_fingerprint
    monkeypatch.setattr(c.spec,'ROOT',new)
    monkeypatch.setattr(c,'code_fingerprint',lambda root=None:code_fingerprint(root or new))
    monkeypatch.setattr(c,'ORIGIN_SOURCE',code_fingerprint(old))
    approval=tmp_path/'ruling.md';approval.write_text('explicit handoff-only ruling')
    review=tmp_path/'review.md';review.write_text('independent source scope review')
    artifact=tmp_path/'selection.json'
    artifact.write_text(json.dumps(dict(code_sha256=code_fingerprint(old),sds_sha256=c.spec.SDS_SHA256,rulings_sha256=c._sha(c.RULINGS),physical_sha256='a'*64)))
    path=tmp_path/'compat.json'
    row=c.build_compatibility_template(path,old,[dict(path=str(artifact),sha256=c._sha(artifact),purpose='resource_selection',operation_ids=['prepare_resource'])],approval,'a'*64)
    row.update(approved=True,independent_review=dict(passed=True,report_path=str(review),report_sha256=c._sha(review)))
    path.write_text(json.dumps(row))
    return path,row,artifact,old,new


def test_compatible_artifact_exact_scope_original_bytes_retained(bridge):
    path,row,artifact,old,new=bridge;before=artifact.read_bytes()
    checked=c.validate_compatibility(path)
    result=c.validate_compatible_artifact(artifact,checked,'resource_selection','prepare_resource')
    assert result['code_sha256']==row['from_code_sha256']!=row['to_code_sha256']
    assert artifact.read_bytes()==before
    with pytest.raises(ValueError):c.validate_compatible_artifact(artifact,checked,'resource_selection','another_operation')
    with pytest.raises(ValueError):c.validate_compatible_artifact(artifact,checked,'diagnostic','prepare_resource')


@pytest.mark.parametrize('damage',['source','baseline','artifact','review','scope','approval'])
def test_compatibility_tamper_or_unapproved_source_rejected(bridge,damage):
    path,row,artifact,old,new=bridge
    if damage=='source':(new/'src/tasks/velocity/v10/rewards.py').write_text('changed recipe')
    elif damage=='baseline':(old/'src/tasks/velocity/v10/runtime.py').write_text('tampered')
    elif damage=='artifact':artifact.write_text('{}')
    elif damage=='review':Path(row['independent_review']['report_path']).write_text('tampered')
    elif damage=='scope':row['source_changes'][0]['purpose']='physical_change';path.write_text(json.dumps(row))
    else:row['approved']=False;path.write_text(json.dumps(row))
    with pytest.raises((ValueError,KeyError)):c.validate_compatibility(path)


def test_resource_measurement_functions_ast_unchanged_from_frozen_baseline():
    import ast
    from src.tasks.velocity.v10 import spec
    baseline=spec.ROOT/'logs/a2_v10/formal_handoff_baseline/src/tasks/velocity/v10/resource_a1.py'
    if not baseline.exists():pytest.skip('local immutable baseline evidence unavailable')
    before=ast.parse(baseline.read_text());after=ast.parse((spec.ROOT/'src/tasks/velocity/v10/resource_a1.py').read_text())
    for name in ('run_a1_probes','a1_probe_stack','run_a1_v2_probes','a1_v2_probe_stack'):
        a=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(a)==ast.dump(b),name
