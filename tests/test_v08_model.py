from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import a2_v08_bootstrap
a2_v08_bootstrap.activate(ROOT)

def test_native_model_and_ppo_exist():
    assert (ROOT/'src/tasks/velocity/v08/model.py').exists()
    assert (ROOT/'src/tasks/velocity/v08/ppo.py').exists()

def test_source_output_and_optimizer_exact_equivalence():
    from src.tasks.velocity.v08 import model,ppo,continuation_v08 as migration
    from src.tasks.velocity.v07.model import V07ActorCritic
    source=migration.load_designated_source()
    old=V07ActorCritic();old.load_state_dict(source['model'])
    new=model.V08ActorCritic();new.load_state_dict(source['model'])
    obs={key:torch.zeros((2,*shape),dtype=torch.float32) for key,shape in model.OBSERVATION_SHAPES.items()}
    assert torch.equal(old(obs),new(obs)) and torch.equal(old.evaluate(obs),new.evaluate(obs))
    opt=ppo.build_optimizer(new);opt.load_state_dict(source['optimizer'])
    assert ppo.optimizer_membership(opt,new)['exactly_once']
    assert len(opt.state)==45
    assert migration.tree_digest(opt.state_dict())==migration.tree_digest(source['optimizer'])
    assert migration.tree_digest(new.actor.normalizer.state_dict())==migration.tree_digest(old.actor.normalizer.state_dict())

def test_native_v08_cannot_initialize_a_new_eth_run():
    import pytest
    from src.tasks.velocity.v08.model import V08ActorCritic,EthMigrationError
    with pytest.raises(EthMigrationError,match='6400'):
        V08ActorCritic.from_eth('/tmp/unapproved_eth.pt',expected_sha='0'*64,seed=42,device='cpu')
