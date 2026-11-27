"""Fixed A1 diagnostics are bounded subsets, never substitute formal admission."""
import importlib
import math


def test_fixed_diagnostic_has_112_original_cases_and_exact_bound():
    diag=importlib.import_module('src.tasks.velocity.v10.diagnostic_a1')
    manifest=diag.diagnostic_manifest()
    assert len(manifest)==112
    assert len({c['case_sha256'] for c in manifest})==112
    assert sum(c['max_steps'] for c in manifest)==251288
    grouped={}
    for row in manifest:grouped.setdefault(row['group_id'],[]).append(row)
    assert len(grouped)==28
    assert all(len(rows)==4 and [c['index'] for c in rows]==[0,1,2,3] for rows in grouped.values())
    assert sum(rows[0]['task'].startswith('F_') for rows in grouped.values())==10
    for width in (.8,1.2):
        rows=[rows for rows in grouped.values() if rows[0]['task'].startswith('P_') and rows[0]['width']==width]
        assert len(rows)==9
        assert all(r['height']==2 and r['encoding']=='G' for rs in rows for r in rs)


def test_diagnostic_preserves_original_case_identity_and_timeout():
    diag=importlib.import_module('src.tasks.velocity.v10.diagnostic_a1')
    from src.tasks.velocity.v10 import cases
    originals={r['case_sha256']:r for course in ('D1','D2') for r in cases.suite(course,'A',4)}
    for row in diag.diagnostic_manifest():
        assert row==originals[row['case_sha256']]
        assert row['max_steps']==cases.max_steps(row)
    receipt=diag.diagnostic_plan()
    assert receipt['diagnostic_only'] is True
    assert receipt['max_samples']==251288
    assert receipt['group_threshold']=='4/4'
    assert receipt['formal_group_threshold']=='18/20'


def test_capacity_preserves_historical_charges_and_releases_only_capacity(tmp_path):
    from src.tasks.velocity.v10.checkpoint import BudgetLedger
    cap=importlib.import_module('src.tasks.velocity.v10.evaluation_capacity')
    ledger=tmp_path/'eval.jsonl'
    costs=BudgetLedger(ledger)
    costs.charge('ALL',30,limit=100,kind='evaluation')
    with cap.evaluation_capacity(ledger,'diag',70,100):
        costs.reserve('one','ALL',4,limit=100,kind='evaluation');costs.mark_executed('one')
    assert costs.totals()['evaluation:ALL']==34
    with cap.evaluation_capacity(ledger,'next',66,100):pass
    assert costs.totals()['evaluation:ALL']==34


def test_capacity_refuses_shortfall_and_competing_stage(tmp_path):
    import pytest
    cap=importlib.import_module('src.tasks.velocity.v10.evaluation_capacity')
    ledger=tmp_path/'eval.jsonl'
    with cap.evaluation_capacity(ledger,'a',100,100):
        with pytest.raises(RuntimeError,match='capacity.*busy'):
            with cap.evaluation_capacity(ledger,'b',1,100):pass
    with pytest.raises(RuntimeError,match='shortfall=1'):
        with cap.evaluation_capacity(ledger,'c',101,100):pass


def test_diagnostic_claim_is_persistent_across_output_paths(tmp_path):
    import pytest
    diag=importlib.import_module('src.tasks.velocity.v10.diagnostic_a1')
    ledger=tmp_path/'eval.jsonl'
    diag.claim_diagnostic_once(ledger,tmp_path/'first',diag.diagnostic_plan())
    with pytest.raises(RuntimeError,match='already claimed'):
        diag.claim_diagnostic_once(ledger,tmp_path/'different',diag.diagnostic_plan())


def test_diagnostic_rejects_unlinked_resource_receipt(tmp_path):
    import pytest
    diag=importlib.import_module('src.tasks.velocity.v10.diagnostic_a1')
    receipt=tmp_path/'selection.json';receipt.write_text('{}')
    with pytest.raises(ValueError,match='resource'):
        diag.validate_diagnostic_resource(receipt)
