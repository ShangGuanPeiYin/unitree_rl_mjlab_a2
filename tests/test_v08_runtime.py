from test_v08_evaluation import failure_raw
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import a2_v08_bootstrap
a2_v08_bootstrap.activate(ROOT)

def test_boundary_plan_probe_is_separate_and_formal_unchanged():
    from src.tasks.velocity.v08.runner import boundary_plan
    assert boundary_plan('formal',6450,50)==('diagnostic',)
    assert boundary_plan('formal',6500,100)==('diagnostic','gate','save')
    assert boundary_plan('formal',6600,200)==('diagnostic','gate','h00','save')
    assert boundary_plan('probe',6450,50)==('diagnostic',)
    assert boundary_plan('probe',6455,55)==('save_reload',)
    assert boundary_plan('probe',6456,56)==()

def test_ledger_preserves_source_and_counts_only_new_samples():
    from src.tasks.velocity.v08.ledger import CostLedger
    ledger=CostLedger(mode='smoke',num_envs=8)
    ledger.commit(6401)
    assert ledger.state['engineering_updates']==1 and ledger.state['formal_updates']==0
    assert ledger.state['control_samples']==384
    assert ledger.learning_samples==157286400+384
    with pytest.raises(ValueError):ledger.commit(6401)
    assert CostLedger(mode='formal',num_envs=512).state['formal_updates']==0

def test_signed_cli_and_modes_are_hard_contracts(tmp_path):
    from src.tasks.velocity.v08.runtime import build_parser,resolve_request
    from src.tasks.velocity.v08.continuation_v08 import SOURCE_PATH,SOURCE_SHA256
    base=['--source-checkpoint',SOURCE_PATH,'--source-sha256',SOURCE_SHA256,
          '--output-dir',str(tmp_path/'smoke'),'--device','cpu','--seed','42',
          '--num-envs','8','--updates','2','--mode','smoke']
    contract=resolve_request(build_parser().parse_args(base))
    assert contract['num_envs']==8
    with pytest.raises(ValueError):resolve_request(build_parser().parse_args(base+['--request-speed','nan']))
    with pytest.raises(ValueError):resolve_request(build_parser().parse_args(base+['--request-speed','0.61']))
    assert resolve_request(build_parser().parse_args(base+['--request-speed','-0.6']))['request_speed']==-.6

def test_baseline_identity_cannot_claim_another_source(tmp_path):
    import json
    from src.tasks.velocity.v08 import recovery,orchestration
    from src.tasks.velocity.v08.evaluation import make_cases,evaluate_cases
    from src.tasks.velocity.v08.evaluation_records import pack_result,write_sidecar
    from src.tasks.velocity.v08.continuation_v08 import SOURCE_SHA256
    table=make_cases('A',kind='baseline')
    def physical_failure(c):return dict(failure_raw(c),reason='physical_failure')
    result=evaluate_cases(None,table,physical_failure)
    frozen=recovery.build_freeze();cases=recovery.freeze_case_tables()
    identity=dict(update=6400,phase='A',kind='baseline',case_table_sha256=table.sha256,
        source_sha256='1'*64,recipe_sha256=recovery.digest(recovery.jsonable(__import__('src.tasks.velocity.v08.spec',fromlist=['contract']).contract())),
        freeze_sha256=frozen['sha256'])
    payload=pack_result(result,identity);binding=write_sidecar(tmp_path/'rows.json',payload)
    report=dict(mode='baseline',complete=True,source_checkpoint_sha256=SOURCE_SHA256,freeze_sha256=frozen['sha256'],
        case_tables_sha256=recovery.digest(cases),acceptance_sha256='2'*64,evaluation=dict(payload=payload,binding=binding))
    path=tmp_path/'baseline.json';path.write_text(json.dumps(report))
    with pytest.raises(ValueError):orchestration.verify_baseline(path,frozen,cases,{'sha256':'2'*64})

def test_existing_output_requires_latest_native_receipt(tmp_path):
    import json
    from src.tasks.velocity.v08.runtime import admit_output
    from src.tasks.velocity.v08 import recovery
    output=tmp_path/'run';output.mkdir();checkpoints=output/'checkpoints';checkpoints.mkdir()
    old=checkpoints/'ready_006402_0001.pt';latest=checkpoints/'exit_006402_0002.pt'
    oldsha=recovery.atomic_save(old,{'marker':1});newsha=recovery.atomic_save(latest,{'marker':2})
    for path,sha,update in ((old,oldsha,6402),(latest,newsha,6402)):
        recovery.write_json(path.with_suffix('.json'),dict(path=str(path),sha256=sha,update=update,mode='smoke'))
    with pytest.raises(ValueError):admit_output(dict(output_dir=str(output),resume_checkpoint=None,resume_sha256=None,mode='smoke'))
    with pytest.raises(ValueError):admit_output(dict(output_dir=str(output),resume_checkpoint=str(old),resume_sha256=oldsha,mode='smoke'))
    admit_output(dict(output_dir=str(output),resume_checkpoint=str(latest),resume_sha256=newsha,mode='smoke'))

def test_learning_journal_refuses_replaying_spent_or_partial_updates(tmp_path):
    from src.tasks.velocity.v08 import recovery
    from src.tasks.velocity.v08.runtime import verify_learning_journal
    directory=tmp_path/'transactions';directory.mkdir()
    recovery.write_json(directory/'update_006401.intent.json',dict(update=6401))
    with pytest.raises(ValueError):verify_learning_journal(tmp_path,6400)
    recovery.write_json(directory/'update_006401.commit.json',dict(update=6401))
    with pytest.raises(ValueError):verify_learning_journal(tmp_path,6400)
    verify_learning_journal(tmp_path,6401)
