import unittest
import torch
from src.tasks.pointcloud.model import PointCloudActorCritic
from src.tasks.pointcloud.algorithm import JointPPO, compute_gae, masked_mse

class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model=PointCloudActorCritic(10)
        self.inputs=dict(obs_history=torch.randn(2,5,45),points=torch.randn(2,5,70,3),mask=torch.zeros(2,5,70,dtype=torch.bool),ages=torch.arange(5).float().expand(2,-1)*.02)
    def test_initial_identity_and_empty(self):
        for scale in [0,1,1e6]:
            self.inputs['obs_history']*=scale
            actual=self.model.actor(**self.inputs)
            expected=self.model.actor.eth(self.inputs['obs_history'][:,-1])
            self.assertTrue(torch.equal(actual,expected))
    def test_masked_nan_labels(self):
        value=masked_mse(torch.ones(2,4),torch.full((2,4),float('nan')),torch.zeros(2,4,dtype=torch.bool))
        self.assertEqual(float(value),0)
    def test_timeout_bootstrap(self):
        f=torch.tensor([[False,True,False]])
        s=torch.tensor([[False,False,True]])
        t=torch.tensor([[True,False,False]])
        a,_=compute_gae(torch.ones(1,3),torch.zeros(1,3),torch.ones(1,3),f,s,t)
        torch.testing.assert_close(a,torch.tensor([[1.99,1,1]]))
    def test_synthetic_single_update(self):
        self.model.freeze_eth(True)
        ppo=JointPPO(self.model)
        with torch.no_grad():
            d=self.model.distribution(self.inputs); actions=d.sample(); log=d.log_prob(actions).sum(-1)
        batch=dict(actor=self.inputs,actions=actions,old_log_prob=log,advantages=torch.ones(2),privileged=torch.randn(2,10),old_value=torch.zeros(2),returns=torch.ones(2),velocity=torch.zeros(2,3),clearance=torch.zeros(2,4),clearance_mask=torch.ones(2,4,dtype=torch.bool))
        result=ppo.update_batch(batch)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in result.values()))
        self.assertTrue(all(p.grad is None for p in self.model.actor.eth.mlp.parameters()))


class IntegrationTests(unittest.TestCase):
    def test_real_eth_and_serialization(self):
        import tempfile
        from pathlib import Path
        from src.tasks.velocity.rl.eth_actor import EthActor
        from src.tasks.pointcloud.checkpoint import save_checkpoint, restore_checkpoint
        weight=Path('/home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt')
        if not weight.exists(): self.skipTest('External ETH checkpoint absent')
        torch.set_num_threads(1)
        eth=EthActor.load(weight)
        model=PointCloudActorCritic(10,eth)
        inputs=dict(obs_history=torch.randn(2,5,45),points=torch.full((2,5,70,3),float('nan')),mask=torch.zeros(2,5,70,dtype=torch.bool),ages=torch.zeros(2,5))
        before=model.actor(**inputs).detach()
        torch.testing.assert_close(before,eth(inputs['obs_history'][:,-1]),rtol=0,atol=0)
        ppo=JointPPO(model)
        # Synthetic optimizer state only: no environment sampling or model artifact.
        model.actor(**inputs).square().mean().backward()
        ppo.optimizer.step(); ppo.optimizer.zero_grad(set_to_none=True)
        before=model.actor(**inputs).detach()
        expected_optimizer=__import__('copy').deepcopy(ppo.optimizer.state_dict())
        calls=[]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pt'
            save_checkpoint(path,model,ppo.optimizer,{'level':'P1'},0,{},update_complete=True)
            with torch.no_grad(): model.actor.delta.bias.add_(1)
            restored=restore_checkpoint(path,model,ppo.optimizer,lambda:calls.append('reset'))
            for key, value in expected_optimizer['state'].items():
                for name, tensor in value.items():
                    torch.testing.assert_close(tensor,ppo.optimizer.state_dict()['state'][key][name],rtol=0,atol=0)
            self.assertEqual(calls,['reset'])
            self.assertEqual(restored['event'],'resume_reset')
            torch.testing.assert_close(before,model.actor(**inputs),rtol=0,atol=0)
    def test_onnx(self):
        import tempfile
        from pathlib import Path
        try: import onnxruntime
        except ImportError: self.skipTest('onnxruntime unavailable')
        from src.tasks.pointcloud.export import export_actor
        torch.set_num_threads(1)
        model=PointCloudActorCritic(10)
        inputs=dict(obs_history=torch.randn(2,5,45),points=torch.randn(2,5,70,3),mask=torch.rand(2,5,70)>.5,ages=torch.zeros(2,5))
        with tempfile.TemporaryDirectory() as directory:
            self.assertLessEqual(export_actor(model.actor,Path(directory)/'actor.onnx',inputs),1e-6)
            with torch.no_grad(): model.actor.delta.weight.normal_(std=.01)
            self.assertLessEqual(export_actor(model.actor,Path(directory)/'nonzero.onnx',inputs),1e-6)
            blind=PointCloudActorCritic(10,use_points=False)
            self.assertLessEqual(export_actor(blind.actor,Path(directory)/'blind.onnx',inputs),1e-6)

class ContractTests(unittest.TestCase):
    def test_blind_has_only_eth(self):
        model=PointCloudActorCritic(10,use_points=False)
        self.assertEqual(list(dict(model.actor.named_children())),['eth'])
    def test_proprio_time_independent_of_delay(self):
        torch.set_num_threads(1)
        model=PointCloudActorCritic(10)
        inputs=dict(obs_history=torch.randn(2,5,45),points=torch.zeros(2,5,70,3),mask=torch.zeros(2,5,70,dtype=torch.bool),ages=torch.zeros(2,5))
        before=model.actor.features(**inputs)
        inputs['ages']+=.06
        torch.testing.assert_close(before,model.actor.features(**inputs),rtol=0,atol=0)
    def test_unfrozen_eth_and_encoder_gradient(self):
        torch.set_num_threads(1)
        model=PointCloudActorCritic(10)
        model.freeze_eth(False)
        with torch.no_grad(): model.actor.delta.weight.fill_(.01)
        inputs=dict(obs_history=torch.randn(2,5,45),points=torch.randn(2,5,70,3),mask=torch.ones(2,5,70,dtype=torch.bool),ages=torch.zeros(2,5))
        model.actor(**inputs).square().mean().backward()
        self.assertGreater(float(model.actor.eth.mlp[0].weight.grad.abs().sum()),0)
        self.assertGreater(float(model.actor.embed.weight.grad.abs().sum()),0)

if __name__=='__main__': unittest.main()
