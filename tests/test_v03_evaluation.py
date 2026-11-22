"""CPU-only retained raw evidence; synthetic trajectories are not robot trials."""
import copy
import importlib
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.tasks.velocity.v03.spec import (TASK, PROTOCOL_ID, CODE_VERSION, SCHEMA_VERSION,
    INIT_SHA256, STAGE_BY_ID, behavior_hashes, sha256)
from src.tasks.velocity.v03.speed import SegmentSpeedState

try:
    evaluation = importlib.import_module('src.tasks.velocity.v03.evaluation')
except ModuleNotFoundError:
    evaluation = None


def terminal_source(case, identity, *, step, climb_step=None):
    """Mirror actual V2 terminal field types, including its checkpoint dictionary."""
    stage = STAGE_BY_ID[identity['stage']]
    return dict(stage=stage.id, N=stage.N, h=float(np.float32(stage.h)),
        vx=float(np.float32(case['vx'])), seed=identity['eval_seed'],
        checkpoint=dict(path='/synthetic/cpu_fixture.pt', sha256=identity['checkpoint_sha256']),
        protocol_id=PROTOCOL_ID, code_version=CODE_VERSION,
        speed_source='evaluation_online_preview', terminal_step=step, climb_step=climb_step,
        parking_diagnostic=identity['parking_diagnostic'])


def make_gate_fixture(directory, *, stage='S1', kind='validation', iteration=200,
                      stage_iteration=200, checkpoint_sha='1' * 64, num_envs=258, passed=True):
    """Write genuine collector/NumPy-verifiable synthetic raw files, never mock a gate."""
    from src.tasks.velocity.v03.evaluation import (make_cases, RawEvidenceCollector)
    table = make_cases(stage, kind, 100 if kind == 'test' else 20)
    identity = dict(task=TASK, protocol_id=PROTOCOL_ID, code_version=CODE_VERSION,
        evidence_mode='cpu_fixture', evidence_schema='v03-cpu-fixture-evidence-1',
        checkpoint_schema=SCHEMA_VERSION, source_c_sha256=INIT_SHA256,
        source_sha256=behavior_hashes(), seed=42, eval_seed=table['eval_seed'],
        stage=stage, kind=kind, iteration=iteration, global_iteration=iteration,
        stage_iteration=stage_iteration, checkpoint_sha256=checkpoint_sha,
        training_num_envs=num_envs, actual_num_envs=8, statistics_unchanged=True,
        parking_diagnostic=False)
    collector = RawEvidenceCollector(directory, identity, table)
    for offset in range(0, len(table['cases']), 8):
        batch = table['cases'][offset:offset + 8]
        collector.begin_batch(batch)
        state = SegmentSpeedState(len(batch))
        n = STAGE_BY_ID[stage].N
        for j in range(27):
            # One approach, 25 inclusive body samples, then a genuine top sample.
            x = -.01 if j == 0 else ((j - 1) * n * .2 / 24 if j <= 25 else n * .2 + .01)
            samples = dict(step=torch.full((len(batch),), j + 1, dtype=torch.int64),
                local_x=torch.full((len(batch),), x, dtype=torch.float32),
                body_vx=torch.tensor([c['vx'] for c in batch], dtype=torch.float32),
                executed_c=torch.tensor([c['vx'] for c in batch], dtype=torch.float32),
                num_steps=torch.full((len(batch),), n, dtype=torch.int64))
            samples.update(state.update(**samples))
            collector.append_samples(samples, speed_snapshot=state.snapshot())
        for env_id, case in enumerate(batch):
            fail = not passed and case['vx'] == .2 and case['episode_index'] < (6 if kind == 'test' else 3)
            terminal = dict(state.episode_metrics(env_id, climb_success=not fail, failure=fail),
                env_id=env_id, initial_pose=np.asarray(case['initial_pose'], dtype=np.float32).tolist(),
                reason='body_contact' if fail else 'success', duration=27 * .02,
                **terminal_source(case, identity, step=27, climb_step=None if fail else 27))
            collector.finish_episode(env_id, terminal)
        collector.commit_batch()
    summary = collector.finalize()
    return summary, summary['case_table_sha256']


class EvaluationTests(unittest.TestCase):
    def require(self):
        self.assertIsNotNone(evaluation, 'V4 evaluation evidence API absent')
        return evaluation

    def test_public_api_exists(self):
        api = self.require()
        for name in ('make_cases', 'RawEvidenceCollector', 'verify_summary_evidence'):
            self.assertTrue(callable(getattr(api, name)))

    def test_cases_exact_seed_order_and_rng_unchanged(self):
        api = self.require()
        before = copy.deepcopy(np.random.get_state())
        table = api.make_cases('S1', 'perturbed')
        rng = np.random.default_rng(300342)
        for c in table['cases']:
            self.assertEqual(c['initial_pose'], [-float(rng.uniform(.9, 1.1)),
                float(rng.uniform(-.05, .05)), float(rng.uniform(-math.pi / 36, math.pi / 36))])
        after = np.random.get_state()
        self.assertEqual(before[0], after[0]); np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertEqual([c['vx'] for c in table['cases']], [.2] * 20 + [.4] * 20 + [.6] * 20)
        for stage in ('S1', 'S2', 'S3', 'S4'):
            self.assertTrue(all(c['initial_pose'] == [-1., 0., 0.] for c in api.make_cases(stage, 'validation')['cases']))
        for args in [('flat', 'validation', 20), ('R0', 'validation', 20), ('S2', 'perturbed', 20),
                     ('S1', 'test', 100), ('S5', 'test', 20), ('S1', 'validation', 19)]:
            with self.assertRaises(ValueError): api.make_cases(*args)
        self.assertEqual(len(api.make_cases('S5', 'test', 100)['cases']), 300)
        self.assertEqual(api.make_cases('S1', 'baseline')['eval_seed'], 200342)

    def test_full_raw_roundtrip_and_trusted_counts_ignore_summary_passed(self):
        api = self.require()
        with tempfile.TemporaryDirectory() as directory:
            summary, case_sha = make_gate_fixture(directory, passed=False)
            summary['passed'] = True
            summary['speeds'][0]['joint_successes'] = 20
            trusted = api.verify_summary_evidence(summary, expected_checkpoint_sha='1' * 64,
                                                  expected_case_sha256=case_sha)
            self.assertEqual(trusted['speeds'][0]['joint_successes'], 17)
            self.assertEqual(trusted['speeds'][0]['speed_pass_count'], 20)
            self.assertEqual(trusted['speeds'][0]['failures'], 3)
            self.assertFalse(trusted['passed'])
            self.assertTrue(trusted['statistics_unchanged'])
            self.assertEqual(trusted['checkpoint_sha256'], '1' * 64)
            index = json.loads(Path(summary['raw_index_path']).read_text())
            self.assertEqual(len(index['records']), 60)
            for batch in index['batches']:
                with np.load(batch['path'], allow_pickle=False) as raw:
                    self.assertEqual(raw['body_vx'].dtype, np.float32)
                    self.assertEqual(raw['executed_c'].dtype, np.float32)
                    self.assertEqual(raw['local_x'].dtype, np.float32)
                    self.assertEqual(raw['step'].dtype, np.int64)
                    self.assertEqual(raw['offsets'].dtype, np.int64)
            self.assertGreater(summary['raw_write_bytes'], 0)

    def test_final_fixture_has_100_per_speed_and_94_failure_is_reported(self):
        api = self.require()
        with tempfile.TemporaryDirectory() as directory:
            summary, case_sha = make_gate_fixture(directory, stage='S5', kind='test',
                iteration=2000, stage_iteration=400, passed=False)
            trusted = api.verify_summary_evidence(summary, expected_checkpoint_sha='1'*64,
                expected_case_sha256=case_sha)
            self.assertEqual([s['episodes'] for s in trusted['speeds']], [100,100,100])
            self.assertEqual(trusted['speeds'][0]['joint_successes'],94)
            self.assertEqual(trusted['speeds'][0]['speed_pass_count'],100)
            self.assertFalse(trusted['passed'])
            self.assertEqual(len(trusted['speeds'][0]['wilson95']['joint_successes']),2)

    def test_missing_corrupt_identity_and_missing_case_reject(self):
        api = self.require()
        with tempfile.TemporaryDirectory() as directory:
            summary, case_sha = make_gate_fixture(directory)
            for key, bad in [('checkpoint_sha256', '2' * 64), ('case_table_sha256', '2' * 64),
                             ('protocol_id', 'old'), ('statistics_unchanged', False), ('stage', 'S2')]:
                altered = copy.deepcopy(summary); altered[key] = bad
                with self.assertRaises((ValueError, OSError)):
                    api.verify_summary_evidence(altered, expected_checkpoint_sha='1' * 64, expected_case_sha256=case_sha)
            index_path = Path(summary['raw_index_path']); index = json.loads(index_path.read_text())
            raw_path = Path(index['batches'][0]['path']); original = raw_path.read_bytes()
            raw_path.unlink()
            with self.assertRaises((ValueError, OSError)):
                api.verify_summary_evidence(summary, expected_checkpoint_sha='1' * 64, expected_case_sha256=case_sha)
            raw_path.write_bytes(original[:-30])
            with self.assertRaises((ValueError, OSError)):
                api.verify_summary_evidence(summary, expected_checkpoint_sha='1' * 64, expected_case_sha256=case_sha)
            raw_path.write_bytes(original)
            index['records'].pop(); api.atomic_json(index_path, index)
            summary['raw_index_sha256'] = sha256(index_path)
            with self.assertRaises(ValueError):
                api.verify_summary_evidence(summary, expected_checkpoint_sha='1' * 64, expected_case_sha256=case_sha)

    def test_raw_crop_and_missing_frame_even_rehashed_reject(self):
        api = self.require()
        for mutation in ('crop', 'gap', 'wrong_command'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                summary, case_sha = make_gate_fixture(directory)
                path = Path(summary['raw_index_path']); index = json.loads(path.read_text())
                batch = index['batches'][0]; raw_path = Path(batch['path'])
                with np.load(raw_path, allow_pickle=False) as file: raw = {k: file[k] for k in file.files}
                if mutation == 'crop':
                    for key in ('body_vx', 'executed_c', 'local_x', 'step'): raw[key] = raw[key][1:]
                    raw['offsets'][1:] -= 1
                    index['records'][0]['raw_length'] -= 1
                elif mutation == 'gap': raw['step'][10] += 2
                else: raw['executed_c'][:25] = np.float32(.6)
                with raw_path.open('wb') as file: np.savez(file, **raw)
                batch['sha256'] = sha256(raw_path); api.atomic_json(path, index)
                summary['raw_index_sha256'] = sha256(path)
                with self.assertRaises(ValueError):
                    api.verify_summary_evidence(summary, expected_checkpoint_sha='1' * 64, expected_case_sha256=case_sha)

    def test_collector_rejects_wrong_mask_snapshot_and_extreme_preview(self):
        api = self.require()
        for mode in ('mask', 'snapshot', 'extreme'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                table = api.make_cases('S1', 'validation')
                identity = dict(task=TASK, protocol_id=PROTOCOL_ID, code_version=CODE_VERSION,
                    evidence_mode='cpu_fixture', evidence_schema='v03-cpu-fixture-evidence-1',
                    checkpoint_schema=SCHEMA_VERSION, source_c_sha256=INIT_SHA256,
                    source_sha256=behavior_hashes(), seed=42, eval_seed=200342, stage='S1', kind='validation',
                    iteration=200, global_iteration=200, stage_iteration=200, checkpoint_sha256='1' * 64,
                    training_num_envs=258, actual_num_envs=1, statistics_unchanged=True, parking_diagnostic=False)
                collector = api.RawEvidenceCollector(directory, identity, table)
                selected_case = table['cases'][20 if mode == 'extreme' else 0]
                collector.begin_batch([selected_case]); command = selected_case['vx']
                state = SegmentSpeedState(1); rejected = False
                try:
                    for j in range(101):
                        x = -.01 if j == 0 else (j - 1) * .8 / 99
                        samples = dict(step=torch.tensor([j + 1], dtype=torch.int64),
                            local_x=torch.tensor([x], dtype=torch.float32),
                            body_vx=torch.tensor([1e9 if mode == 'extreme' and j == 1 else command], dtype=torch.float32),
                            executed_c=torch.tensor([command], dtype=torch.float32), num_steps=torch.tensor([4], dtype=torch.int64))
                        samples.update(state.update(**samples)); snapshot = state.snapshot()
                        if mode == 'mask' and j == 1: samples['body_mask'][0] = False
                        if mode == 'snapshot' and j == 2: snapshot['sample_count'][0] -= 1
                        collector.append_samples(samples, speed_snapshot=snapshot)
                    terminal = dict(state.episode_metrics(0, climb_success=True), env_id=0,
                        initial_pose=[-1., 0., 0.], reason='success', duration=101 * .02,
                        **terminal_source(selected_case, identity, step=101, climb_step=101))
                    collector.finish_episode(0, terminal)
                except ValueError: rejected = True
                self.assertTrue(rejected, 'engineering faults must reject, never sign ordinary speed failure')

    def test_streaming_incomplete_batches_seen_mask_and_invalid_episode_denominators(self):
        api = self.require()
        table = api.make_cases('S1', 'validation')
        identity = dict(task=TASK, protocol_id=PROTOCOL_ID, code_version=CODE_VERSION,
            evidence_mode='cpu_fixture', evidence_schema='v03-cpu-fixture-evidence-1',
            checkpoint_schema=SCHEMA_VERSION, source_c_sha256=INIT_SHA256,
            source_sha256=behavior_hashes(), seed=42, eval_seed=200342, stage='S1', kind='validation',
            iteration=200, global_iteration=200, stage_iteration=200, checkpoint_sha256='1' * 64,
            training_num_envs=258, actual_num_envs=1, statistics_unchanged=True, parking_diagnostic=False)
        with tempfile.TemporaryDirectory() as directory:
            collector = api.RawEvidenceCollector(directory, identity, table)
            with self.assertRaises(ValueError): collector.finalize()
            collector.begin_batch(table['cases'][:1]); state = SegmentSpeedState(1)
            with self.assertRaises(ValueError): collector.commit_batch()
            sample = dict(step=torch.tensor([1], dtype=torch.int64), local_x=torch.tensor([-.5]),
                body_vx=torch.tensor([.2]), executed_c=torch.tensor([.2]), num_steps=torch.tensor([4], dtype=torch.int64))
            sample.update(state.update(**sample)); collector.append_samples(sample, speed_snapshot=state.snapshot())
            terminal = dict(state.episode_metrics(0, failure=True), env_id=0,
                initial_pose=[-1.,0.,0.], reason='body_contact', duration=.02,
                **terminal_source(table['cases'][0], identity, step=1))
            collector.finish_episode(0, terminal)
            # A reset episode in the same physical env cannot contaminate case1.
            collector.append_samples(sample, speed_snapshot=state.snapshot())
            collector.commit_batch()
            index = json.loads((Path(directory) / 'raw_index.json').read_text())
            self.assertFalse(index['complete']); self.assertEqual(index['records'][0]['raw_length'], 0)
            expected_bytes = sum(path.stat().st_size for path in Path(directory).iterdir() if path.is_file())
            self.assertEqual(collector.write_bytes, expected_bytes)
            self.assertTrue(Path(index['batches'][0]['path']).is_file())
            self.assertEqual(index['records'][0]['observed']['terminal_sample']['step'], 1)
            with self.assertRaises(ValueError): collector.begin_batch(table['cases'][:1])
            with self.assertRaises(ValueError): collector.finalize()
        rows = []
        for case in table['cases']:
            rows.append(dict(case_id=case['case_id'], vx=case['vx'], episode_index=case['episode_index'],
                climb_success=False, speed_valid=False, speed_pass=False, joint_success=False,
                failure=True, reason='body_contact', sample_count=0, mean_vx=None,
                mean_relative_error=None, smooth_mae_relative=None, longest_deviation_seconds=None,
                longest_deviation_samples=None))
        summary = api.summarize_records(rows, table)
        self.assertTrue(all(row['episodes']==20 and row['failures']==20 and row['joint_successes']==0 for row in summary))
        self.assertTrue(all(value is None for row in summary for value in row['metrics']['mean_vx']))
        self.assertIsNone(api.wilson(0,0))

    def test_terminal_source_contradictions_and_missing_fields_reject_at_collector_and_disk(self):
        api = self.require()
        mutations = dict(
            training_source={'speed_source':'training_preview'},
            foreign_checkpoint={'checkpoint':{'path':'/foreign.pt','sha256':'2'*64}},
            foreign_stage_case={'stage':'S5','N':40,'h':float(np.float32(.2)),
                                'vx':float(np.float32(.6)),'seed':777},
            foreign_protocol={'protocol_id':'v02-formal','code_version':'0.2.0'},
            wrong_terminal_step={'terminal_step':1},
            training_seed={'seed':42}, wrong_geometry={'h':.08},
            wrong_parking={'parking_diagnostic':True}, malformed_checkpoint={'checkpoint':'1'*64})
        required = ('stage','N','h','vx','seed','checkpoint','protocol_id','code_version',
                    'speed_source','terminal_step','climb_step','parking_diagnostic')
        for key in required: mutations['missing_' + key] = {key:None}
        mutations['missing_checkpoint_sha'] = {'checkpoint':{'path':'/synthetic/cpu_fixture.pt'}}
        with tempfile.TemporaryDirectory() as directory:
            summary, case_sha = make_gate_fixture(Path(directory)/'complete')
            path = Path(summary['raw_index_path']); original = json.loads(path.read_text())
            table = json.loads(Path(original['cases_path']).read_text())
            row = original['records'][0]
            identity = original['identity']
            for name, changed in mutations.items():
                with self.subTest(location='disk',mutation=name):
                    index = copy.deepcopy(original)
                    terminal = index['records'][0]['terminal']
                    for key, value in changed.items():
                        if name.startswith('missing_') and key != 'checkpoint': terminal.pop(key)
                        else: terminal[key] = value
                    if name == 'missing_checkpoint': terminal.pop('checkpoint')
                    api.atomic_json(path,index); altered = copy.deepcopy(summary)
                    altered['raw_index_sha256'] = sha256(path)
                    with self.assertRaises(ValueError):
                        api.verify_summary_evidence(altered,expected_checkpoint_sha='1'*64,expected_case_sha256=case_sha)
                with self.subTest(location='collector',mutation=name):
                    run_identity = copy.deepcopy(identity); run_identity['actual_num_envs']=1
                    collector = api.RawEvidenceCollector(Path(directory)/('collector_'+name),run_identity,table)
                    collector.begin_batch([row['case']]); state=SegmentSpeedState(1)
                    for j in range(27):
                        x = -.01 if j==0 else ((j-1)*.8/24 if j<=25 else .81)
                        sample = dict(step=torch.tensor([j+1],dtype=torch.int64),
                            local_x=torch.tensor([x],dtype=torch.float32),body_vx=torch.tensor([.2]),
                            executed_c=torch.tensor([.2]),num_steps=torch.tensor([4],dtype=torch.int64))
                        sample.update(state.update(**sample))
                        collector.append_samples(sample,speed_snapshot=state.snapshot())
                    terminal=copy.deepcopy(row['terminal'])
                    for key,value in changed.items():
                        if name.startswith('missing_') and key != 'checkpoint': terminal.pop(key)
                        else: terminal[key]=value
                    if name == 'missing_checkpoint': terminal.pop('checkpoint')
                    with self.assertRaises(ValueError): collector.finish_episode(0,terminal)
            # Path names may differ; the exact loaded model bytes SHA is binding.
            index=copy.deepcopy(original);index['records'][0]['terminal']['checkpoint']={'sha256':'1'*64}
            api.atomic_json(path,index);summary['raw_index_sha256']=sha256(path)
            self.assertTrue(api.verify_summary_evidence(summary,expected_checkpoint_sha='1'*64,
                expected_case_sha256=case_sha)['passed'])

    def test_parking_post_arrival_failure_requires_strict_context_and_never_joint(self):
        api = self.require()
        with tempfile.TemporaryDirectory() as directory:
            summary, _ = make_gate_fixture(directory)
            index = json.loads(Path(summary['raw_index_path']).read_text()); original = index['records'][0]
            with np.load(index['batches'][0]['path'], allow_pickle=False) as storage:
                raw = {k: storage[k][:25] for k in ('body_vx','executed_c','local_x','step')}
            row = copy.deepcopy(original)
            row['terminal'].update(failure=True, joint_success=False, reason='body_contact',
                post_climb_failure=True, parking_entered=True, climb_time=.52,
                climb_step=26, terminal_step=27, parking_diagnostic=True, parking_success=False, parking_time=.02)
            row['preview'].update(failure=True, joint_success=False)
            parking_identity = dict(index['identity'], parking_diagnostic=True)
            result = api._oracle_record(row, raw, run_identity=parking_identity)
            self.assertTrue(result['climb_success']); self.assertTrue(result['failure'])
            self.assertFalse(result['joint_success'])
            with self.assertRaises(ValueError): api._oracle_record(row, raw, run_identity=index['identity'])
            for key, value in [('parking_entered',False),('post_climb_failure',False),
                               ('climb_step',27),('climb_step',26.),('climb_step',None),('parking_diagnostic',False),
                               ('terminal_step',28),('climb_time',.521),('climb_event',True)]:
                altered = copy.deepcopy(row); altered['terminal'][key] = value
                with self.assertRaises(ValueError): api._oracle_record(altered, raw, run_identity=parking_identity)

    def test_disk_pause_and_no_global_rng_consumption(self):
        api = self.require()
        before = (random.getstate(), copy.deepcopy(np.random.get_state()), torch.get_rng_state().clone())
        with tempfile.TemporaryDirectory() as directory:
            summary, case_sha = make_gate_fixture(directory)
            api.verify_summary_evidence(summary, expected_checkpoint_sha='1' * 64, expected_case_sha256=case_sha)
        self.assertEqual(random.getstate(), before[0])
        np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
        self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(api.shutil, 'disk_usage', return_value=(20, 15, 5)):
                with self.assertRaises(api.DiskSpacePause): api.ensure_disk_space(directory)



# Runtime CPU fixtures replace only physical/model backends, never the raw
# collector, filesystem, speed state, oracle, case generator or verifier.
class RuntimeFakeModel(torch.nn.Module):
    mutate = False
    calls = 0
    def __init__(self):
        super().__init__(); self.weight = torch.nn.Parameter(torch.rand(()))
        self.register_buffer('statistics', torch.zeros(()))
    def forward(self, obs):
        type(self).calls += 1
        if self.training: raise AssertionError('evaluation must call model.eval()')
        if type(self).mutate: self.statistics.add_(1)
        return self.weight.expand(obs['batch'], 1)
    def sample(self, *_): raise AssertionError('std sampling forbidden')


class RuntimeFakeEnv:
    instances = []
    fail_step = None
    missing_checkpoint = False
    def __init__(self, **kwargs):
        from types import SimpleNamespace
        self.kwargs = kwargs; self.num_envs = kwargs['num_envs']; self.closed = False
        self.run_seed = kwargs['seed']; self.stage = kwargs['stage']; self.fixed_vx = kwargs['fixed_vx']
        self.initial_cases = kwargs['initial_cases']; self.checkpoint_identity = None
        self.steps = torch.zeros(self.num_envs, dtype=torch.int64); self.calls = 0
        self.state = SegmentSpeedState(self.num_envs); self._segment_samples = None
        self._speed_snapshot = None; self._done = torch.zeros(self.num_envs, dtype=torch.bool)
        self.sim = SimpleNamespace(data=SimpleNamespace(qpos=torch.zeros(self.num_envs, 3), qvel=torch.zeros(self.num_envs, 3)))
        self.rng = np.random.default_rng(self.run_seed)
        # Deliberately spend all global CPU RNGs: evaluate must restore them.
        random.random(); np.random.random(); torch.rand(1)
        self.original_hook = self.diagnostics_snapshot; type(self).instances.append(self)
    def reset(self): return {'batch':self.num_envs}, {}
    def diagnostics_snapshot(self, env_id=0):
        metrics = self.state.episode_metrics(env_id, climb_success=bool(self._done[env_id]))
        return dict(env_id=env_id, step=self.steps[env_id].clone(),
            position=torch.tensor([float(self._segment_samples['local_x'][env_id]),0.,0.]),
            command_executed=torch.tensor([self.fixed_vx,0.,0.]),
            linear_velocity_body=torch.tensor([self.fixed_vx,0.,0.]),
            speed_metrics=metrics,sample_phase=self._segment_samples['sample_phase'][env_id].clone(),
            joint_status=('passed' if metrics['joint_success'] else 'failed') if self._done[env_id] else 'pending',
            success=self._done[env_id].clone(), failure=torch.tensor(False), truncated=torch.tensor(False),
            foot_force=torch.zeros(4,3),nonfoot_force=torch.zeros(2,3))
    def render(self):
        # Returning after reset would produce pixel0=0; true terminal frame is27.
        return np.full((90, 240, 3), int(self.steps[0]), dtype=np.uint8)
    def step(self, action):
        if not torch.equal(action,RuntimeFakeModel.expected_weight.expand(self.num_envs,1)):
            raise AssertionError('environment must receive loaded Actor deterministic mean')
        self.calls += 1; self.steps += 1
        n = STAGE_BY_ID[self.stage].N
        x = torch.where(self.steps==1,-.01,(self.steps.to(torch.float32)-2)*n*.2/24)
        self._segment_samples = dict(step=self.steps.clone(),local_x=x.to(torch.float32),
            body_vx=torch.full((self.num_envs,),self.fixed_vx),
            executed_c=torch.full((self.num_envs,),self.fixed_vx),
            num_steps=torch.full((self.num_envs,),n,dtype=torch.int64))
        self._segment_samples.update(self.state.update(**self._segment_samples))
        self._speed_snapshot = self.state.snapshot()
        self._done = self.steps >= (27 + torch.arange(self.num_envs)%2)
        self.sim.data.qpos[:,0]=self.steps; self.sim.data.qvel[:,0]=self.fixed_vx
        diagnostics = self.diagnostics_snapshot(0)  # Mandatory pre-reset hook.
        if type(self).fail_step == self.calls: raise RuntimeError('physics advanced before failed return')
        episodes = []
        for i in self._done.nonzero().flatten().tolist():
            metrics = self.state.episode_metrics(i,climb_success=True)
            case = dict(vx=self.fixed_vx)
            identity = dict(stage=self.stage,eval_seed=self.run_seed,
                checkpoint_sha256=(self.checkpoint_identity or {}).get('sha256',''),
                parking_diagnostic=self.kwargs['parking_diagnostic'])
            terminal = dict(metrics,env_id=i,initial_pose=self.initial_cases[i],reason='success',
                duration=int(self.steps[i])*.02,
                **terminal_source(case,identity,step=int(self.steps[i]),climb_step=int(self.steps[i])))
            terminal['checkpoint'] = copy.deepcopy(self.checkpoint_identity)
            if type(self).missing_checkpoint: terminal.pop('checkpoint')
            episodes.append(terminal)
        self.state.reset(self._done); self.steps[self._done]=0
        return {'batch':self.num_envs},torch.zeros(self.num_envs),self._done.clone(),torch.zeros(self.num_envs,dtype=torch.bool),dict(
            episodes=episodes,segment_samples=self._segment_samples,speed_snapshot=self._speed_snapshot,diagnostics=diagnostics)
    def close(self):
        self.closed = True
        if self.diagnostics_snapshot != self.original_hook: raise AssertionError('instance pre-reset hook not restored')


class RuntimeFakeMonitor:
    def __init__(self,close_error=False): self.closed=False; self.close_error=close_error; self.polls=0
    def start(self): return self
    def poll(self): self.polls+=1
    def close(self):
        self.closed=True
        if self.close_error: raise RuntimeError('resource monitor close failed')


class RuntimeFakeWriter:
    instances=[]
    def __init__(self,path): self.path=path;self.frames=[];self.closed=False;type(self).instances.append(self)
    def append_data(self,frame): self.frames.append(np.asarray(frame).copy())
    def close(self):
        self.closed=True;Path(self.path).write_bytes(b'controlled CPU fake video; not a decoded movie')


class RuntimeEvaluationTests(unittest.TestCase):
    def require_runtime(self):
        self.assertTrue(callable(getattr(evaluation,'evaluate',None)),'runtime evaluate absent')
        self.assertTrue(callable(getattr(evaluation,'parser',None)),'runtime parser absent')
        self.assertTrue(callable(getattr(evaluation,'main',None)),'runtime main absent')
        return evaluation
    def setUp(self):
        RuntimeFakeEnv.instances=[];RuntimeFakeEnv.fail_step=None;RuntimeFakeEnv.missing_checkpoint=False
        RuntimeFakeModel.calls=0;RuntimeFakeModel.mutate=False;RuntimeFakeWriter.instances=[]
    def runtime(self,root,*,kind='validation',stage='S1',video=False):
        from types import SimpleNamespace
        from src.tasks.velocity.v03.spec import contract
        api=self.require_runtime(); root=Path(root)
        checkpoint=root/'checkpoint.pt';checkpoint.write_bytes(b'controlled CPU fake checkpoint')
        checkpoint_sha=sha256(checkpoint)
        table=api.make_cases(stage,kind,20)
        cases=root/'input_cases.json'
        # Preserve exact external bytes/SHA, even when not canonical JSON.
        cases.write_text(json.dumps(table,separators=(',',':')))
        args=SimpleNamespace(task=TASK,checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,
            stage=stage,kind=kind,seed=42,num_envs=8,episodes=20,cases=str(cases),device='cpu',
            output_dir=str(root/'evaluation'),parking=False,video=video,vy=0.,wz=0.,vx=None,execution=None)
        model=RuntimeFakeModel()
        RuntimeFakeModel.expected_weight=model.weight.detach().clone()
        state=dict(contract=contract(),execution='probe' if kind=='probe' else 'formal',
            model=copy.deepcopy(model.state_dict()),metadata={'seed':42},num_envs=258,
            curriculum=dict(stage=stage,iteration=0,stage_iteration=0,environment_stage_iteration=400 if kind=='probe' else 0))
        preload_calls=[]
        def preload(path,expected_sha,*,execution='formal'):
            preload_calls.append((path,expected_sha,execution));return copy.deepcopy(state)
        components=dict(env_class=RuntimeFakeEnv,model_class=RuntimeFakeModel,preload_checkpoint=preload)
        return api,args,state,components,preload_calls
    def test_runtime_frozen_actor_partial_batches_raw_rng_and_pre_reset_video(self):
        with tempfile.TemporaryDirectory() as root:
            api,args,_,components,calls=self.runtime(root,video=True)
            before=(random.getstate(),copy.deepcopy(np.random.get_state()),torch.get_rng_state().clone())
            monitor=RuntimeFakeMonitor()
            with patch.object(api,'_runtime_components',return_value=components),patch.object(api,'_video_writer',side_effect=RuntimeFakeWriter):
                summary=api.evaluate(args,monitor)
            self.assertEqual([e.num_envs for e in RuntimeFakeEnv.instances],[8,8,4]*3)
            self.assertTrue(all(e.closed for e in RuntimeFakeEnv.instances))
            self.assertTrue(all(e.kwargs['evaluation'] and e.run_seed==200342 for e in RuntimeFakeEnv.instances))
            self.assertEqual(calls[0][2],'formal')
            self.assertEqual(summary['case_table_sha256'],sha256(args.cases))
            self.assertEqual(summary['actual_num_envs'],8)
            self.assertEqual(summary['physical_control_samples'],sum(e.calls*e.num_envs for e in RuntimeFakeEnv.instances))
            self.assertGreater(summary['unused_autoreset_samples'],0)
            self.assertEqual(summary['physical_control_samples'],summary['scored_episode_samples']+summary['unused_autoreset_samples'])
            self.assertTrue(summary['statistics_unchanged']);self.assertTrue(summary['evidence_verified'])
            self.assertGreater(RuntimeFakeModel.calls,0)
            self.assertEqual(random.getstate(),before[0]);np.testing.assert_array_equal(np.random.get_state()[1],before[1][1])
            self.assertTrue(torch.equal(torch.get_rng_state(),before[2]))
            self.assertEqual(len(RuntimeFakeWriter.instances),3)
            self.assertTrue(all(int(frame[8,4,0])==0 for writer in RuntimeFakeWriter.instances for frame in writer.frames))
            self.assertTrue(all(w.closed and any(int(f[0,0,0])==27 for f in w.frames) for w in RuntimeFakeWriter.instances))
            trace=[json.loads(line) for line in (Path(args.output_dir)/'steps_env0.jsonl').read_text().splitlines()]
            self.assertTrue(all('qpos' in row and 'qvel' in row and 'diagnostics' in row for row in trace))
            self.assertTrue(any(row['diagnostics']['joint_status']=='pending' for row in trace))
            self.assertTrue(any(row['diagnostics']['joint_status']=='passed' for row in trace))
    def test_runtime_rejects_mutation_missing_terminal_identity_and_preserves_unknown_tail(self):
        for mode in ('statistics','terminal','physics'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as root:
                api,args,_,components,_=self.runtime(root)
                RuntimeFakeModel.mutate=mode=='statistics';RuntimeFakeEnv.missing_checkpoint=mode=='terminal'
                RuntimeFakeEnv.fail_step=1 if mode=='physics' else None
                with patch.object(api,'_runtime_components',return_value=components):
                    with self.assertRaises((ValueError,RuntimeError)):api.evaluate(args,RuntimeFakeMonitor())
                self.assertTrue(all(e.closed for e in RuntimeFakeEnv.instances))
                self.assertFalse((Path(args.output_dir)/'summary.json').exists())
                evidence=json.loads((Path(args.output_dir)/'physical_samples.json').read_text())
                if mode=='physics':
                    self.assertTrue(evidence['unknown_tail']);self.assertIsNone(evidence['physical_control_samples'])
                    self.assertEqual(evidence['physical_control_samples_lower_bound'],0)
                self.setUp()
    def test_probe_execution_and_replay_are_separate_non_gate_cases(self):
        with tempfile.TemporaryDirectory() as root:
            api,args,state,components,calls=self.runtime(root,kind='probe',stage='S5')
            with patch.object(api,'_runtime_components',return_value=components):summary=api.evaluate(args,RuntimeFakeMonitor())
            self.assertEqual(calls[0][2],'probe');self.assertEqual(summary['stage_iteration'],0)
            self.assertFalse(summary['gate_eligible'])
        with tempfile.TemporaryDirectory() as root:
            api,args,state,components,calls=self.runtime(root)
            state['execution']='probe'
            with patch.object(api,'_runtime_components',return_value=components):
                with self.assertRaises(ValueError):api.evaluate(args,RuntimeFakeMonitor())
            self.assertFalse(RuntimeFakeEnv.instances[-1].kwargs['stage']=='S1')
        with tempfile.TemporaryDirectory() as root:
            api,args,_,components,_=self.runtime(root)
            table=api.make_cases('S1','replay',1,replay_vx=.4)
            Path(args.cases).write_text(json.dumps(table));args.kind='replay';args.episodes=1;args.vx=.4;args.num_envs=1
            with patch.object(api,'_runtime_components',return_value=components):summary=api.evaluate(args,RuntimeFakeMonitor())
            self.assertEqual(len(table['cases']),1);self.assertEqual(summary['speeds'][0]['episodes'],1)
            self.assertFalse(summary['gate_eligible']);self.assertIsNone(summary['passed'])
    def test_runtime_failure_retains_current_raw_batch_and_bad_terminal(self):
        with tempfile.TemporaryDirectory() as root:
            api,args,_,components,_=self.runtime(root)
            RuntimeFakeEnv.missing_checkpoint=True
            with patch.object(api,'_runtime_components',return_value=components):
                with self.assertRaises(ValueError):api.evaluate(args,RuntimeFakeMonitor())
            path=Path(args.output_dir)/'fault_diagnostics.pt'
            self.assertTrue(path.is_file(),'retain invalid terminal and current raw batch for failure audit')
            archive=torch.load(path,map_location='cpu',weights_only=False)['evaluation_runtime']
            self.assertEqual(len(archive['active_batch'][0]['body_vx']),25)
            self.assertNotIn('checkpoint',archive['last_info']['episodes'][0])
            self.assertEqual(archive['physics']['physical_control_samples_lower_bound'],8*27)

    def test_runtime_preflight_rejects_wrong_checkpoint_sha_and_replay_cropping(self):
        for mode in ('sha','selected_gate_speed','cropped_replay'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as root:
                api,args,_,components,calls=self.runtime(root)
                before=len(RuntimeFakeEnv.instances)
                if mode=='sha':args.checkpoint_sha256='2'*64
                elif mode=='selected_gate_speed':args.vx=.2
                else:args.kind='replay';args.episodes=1;args.vx=.4
                with patch.object(api,'_runtime_components',return_value=components):
                    with self.assertRaises(ValueError):api.evaluate(args,RuntimeFakeMonitor())
                self.assertEqual(len(RuntimeFakeEnv.instances),before);self.assertEqual(calls,[])

    def test_main_always_closes_resource_monitor_and_close_failure_is_not_success(self):
        with tempfile.TemporaryDirectory() as root:
            api,args,_,components,_=self.runtime(root)
            monitor=RuntimeFakeMonitor(close_error=True)
            argv=['--task',TASK,'--checkpoint',args.checkpoint,'--checkpoint-sha256',args.checkpoint_sha256,
                '--stage','S1','--kind','validation','--seed','42','--num-envs','8','--episodes','20',
                '--cases',args.cases,'--device','cpu','--output-dir',args.output_dir,'--vy','0','--wz','0']
            with patch.object(api,'_runtime_components',return_value=components),patch.object(api,'_resource_monitor',return_value=monitor):
                with self.assertRaisesRegex(RuntimeError,'monitor close'):api.main(argv)
            self.assertTrue(monitor.closed)
            self.assertNotEqual(json.loads((Path(args.output_dir)/'result.json').read_text())['status'],'evaluation_complete')

    def test_main_success_closes_monitor_before_publishing_replay_result(self):
        with tempfile.TemporaryDirectory() as root:
            api,args,_,components,_=self.runtime(root)
            table=api.make_cases('S1','replay',1,replay_vx=.4);Path(args.cases).write_text(json.dumps(table))
            monitor=RuntimeFakeMonitor()
            argv=['--task',TASK,'--checkpoint',args.checkpoint,'--checkpoint-sha256',args.checkpoint_sha256,
                '--stage','S1','--kind','replay','--seed','42','--num-envs','1','--episodes','1',
                '--cases',args.cases,'--device','cpu','--output-dir',args.output_dir,'--vy','0','--wz','0','--vx','.4','--execution','formal']
            with patch.object(api,'_runtime_components',return_value=components),patch.object(api,'_resource_monitor',return_value=monitor):
                summary=api.main(argv)
            self.assertTrue(monitor.closed);self.assertIsNone(summary['passed']);self.assertFalse(summary['gate_eligible'])
            self.assertEqual(json.loads((Path(args.output_dir)/'result.json').read_text())['status'],'evaluation_complete')

    def test_runtime_max_steps_never_truncates_a_protocol_case(self):
        with tempfile.TemporaryDirectory() as root:
            api,args,_,components,calls=self.runtime(root)
            argv=['--checkpoint',args.checkpoint,'--checkpoint-sha256',args.checkpoint_sha256,
                '--stage','S1','--kind','validation','--cases',args.cases,'--output-dir',args.output_dir,'--max-steps','10000']
            self.assertEqual(api.parser().parse_args(argv).max_steps,10000)
            args.max_steps=1
            with patch.object(api,'_runtime_components',return_value=components):
                with self.assertRaises(ValueError):api.evaluate(args,RuntimeFakeMonitor())
            self.assertEqual(calls,[])

    def test_parking_records_and_conditional_summary_are_separate_from_joint(self):
        api=self.require_runtime()
        with tempfile.TemporaryDirectory() as directory:
            summary,_=make_gate_fixture(directory)
            index=json.loads(Path(summary['raw_index_path']).read_text());row=copy.deepcopy(index['records'][0])
            with np.load(index['batches'][0]['path'],allow_pickle=False) as storage:
                raw={k:storage[k][:25] for k in ('body_vx','executed_c','local_x','step')}
            row['terminal'].update(parking_diagnostic=True,parking_entered=True,parking_success=False,
                post_climb_failure=False,climb_step=26,climb_time=.52,parking_time=.02,reason='timeout')
            parking_identity=dict(index['identity'],parking_diagnostic=True)
            record=api._oracle_record(row,raw,run_identity=parking_identity)
            self.assertFalse(record['parking_success']);self.assertTrue(record['joint_success'])
            rows=[copy.deepcopy(record) for speed in summary['speeds'] for record in speed['records']]
            for record in rows:
                record.update(parking_diagnostic=True,parking_entered=record['vx']!=.6,
                    parking_success=True if record['vx']==.2 else False if record['vx']==.4 else None,
                    post_climb_failure=False,parking_time=3. if record['vx']==.2 else None)
            table=json.loads(Path(index['cases_path']).read_text());speeds=api.summarize_records(rows,table)
            self.assertEqual(speeds[0]['parking_successes'],20);self.assertEqual(speeds[1]['parking_successes'],0)
            self.assertEqual(speeds[0]['parking_conditional_success_rate'],1.)
            self.assertIsNone(speeds[2]['parking_conditional_success_rate'])
            self.assertEqual(speeds[2]['parking_not_entered'],20)

    def test_actual_environment_count_must_match_before_any_control(self):
        class WrongCountEnv(RuntimeFakeEnv):
            def __init__(self,**kwargs):
                super().__init__(**kwargs);self.num_envs+=1
        with tempfile.TemporaryDirectory() as root:
            api,args,_,components,_=self.runtime(root);components['env_class']=WrongCountEnv
            with patch.object(api,'_runtime_components',return_value=components):
                with self.assertRaises(ValueError):api.evaluate(args,RuntimeFakeMonitor())
            self.assertEqual(WrongCountEnv.instances[-1].calls,0,'mismatched batch must fail before physical control')


class RuntimeEvidenceSemanticTests(unittest.TestCase):
    """Rehash bytes after contradicting retained raw evidence; never mock verifier."""
    def setUp(self):
        self.harness=RuntimeEvaluationTests();self.harness.setUp()
    def actual(self,directory,*,replay=True,video=False):
        api,args,_,components,_=self.harness.runtime(directory,video=video)
        if replay:
            table=api.make_cases('S1','replay',1,replay_vx=.4)
            Path(args.cases).write_text(json.dumps(table));args.kind='replay';args.episodes=1;args.vx=.4;args.num_envs=1
        with patch.object(api,'_runtime_components',return_value=components),patch.object(api,'_video_writer',side_effect=RuntimeFakeWriter):
            summary=api.evaluate(args,RuntimeFakeMonitor())
        index_path=Path(summary['raw_index_path']);index=json.loads(index_path.read_text())
        return api,summary,index_path,index
    def verify(self,api,summary):
        return api.verify_summary_evidence(summary,expected_checkpoint_sha=summary['checkpoint_sha256'],
            expected_case_sha256=summary['case_table_sha256'])
    def rehash(self,api,summary,path,index):
        if 'runtime_evidence' in index:
            summary['runtime_evidence']=copy.deepcopy(index['runtime_evidence'])
        api.atomic_json(path,index);summary['raw_index_sha256']=sha256(path)

    def test_runtime_trace_modes_identity_completeness_and_video_links_reject(self):
        modes=('empty','foreign','missing_middle','missing_terminal','wrong_step','foreign_checkpoint',
            'foreign_case','foreign_source','wrong_batch','wrong_n','wrong_env','missing_qpos','empty_qvel',
            'missing_contact','nonfinite_contact','premature_joint','pending_terminal','wrong_terminal_metrics',
            'wrong_video_frame','wrong_video_case','wrong_video_vx','drop_video','drop_runtime',
            'drop_mode','drop_schema','drop_runtime_schema','downgrade_to_fixture')
        with tempfile.TemporaryDirectory() as root:
            api,summary,path,index=self.actual(root,video=True)
            self.assertTrue(self.verify(api,summary)['evidence_verified'])
            trace_path=Path(index['runtime_evidence']['trace']['path'])
            original=[json.loads(line) for line in trace_path.read_text().splitlines()]
            for mode in modes:
                with self.subTest(mode=mode):
                    current=copy.deepcopy(index);altered=copy.deepcopy(summary);lines=copy.deepcopy(original)
                    if mode=='empty':lines=[]
                    elif mode=='foreign':lines=[{'checkpoint_sha256':'f'*64,'case_id':'S5:test:0.2:099'}]
                    elif mode=='missing_middle':lines.pop(10)
                    elif mode=='missing_terminal':lines.pop()
                    elif mode=='wrong_step':lines[1]['diagnostics']['step']=3
                    elif mode=='foreign_checkpoint':lines[1]['checkpoint_sha256']='f'*64
                    elif mode=='foreign_case':lines[1]['case_id']='S5:test:0.2:099'
                    elif mode=='foreign_source':lines[1]['source_sha256']={'foreign.py':'f'*64}
                    elif mode=='wrong_batch':lines[1]['batch_index']=1
                    elif mode=='wrong_n':lines[1]['actual_num_envs']=8
                    elif mode=='wrong_env':lines[1]['diagnostics']['env_id']=1
                    elif mode=='missing_qpos':lines[1].pop('qpos')
                    elif mode=='empty_qvel':lines[1]['qvel']=[]
                    elif mode=='missing_contact':lines[1]['diagnostics'].pop('foot_force')
                    elif mode=='nonfinite_contact':lines[1]['diagnostics']['nonfoot_force'][0][0]=float('nan')
                    elif mode=='premature_joint':lines[1]['diagnostics']['joint_status']='passed'
                    elif mode=='pending_terminal':lines[-1]['diagnostics']['joint_status']='pending'
                    elif mode=='wrong_terminal_metrics':lines[-1]['diagnostics']['speed_metrics']['sample_count']-=1
                    elif mode=='wrong_video_frame':lines[4]['video_frame_index']=99
                    elif mode=='wrong_video_case':current['runtime_evidence']['videos'][0]['case_id']='S5:test:0.2:099'
                    elif mode=='wrong_video_vx':current['runtime_evidence']['videos'][0]['vx']=.2
                    elif mode=='drop_video':current['runtime_evidence']['videos']=[]
                    elif mode=='drop_runtime':current.pop('runtime_evidence');altered.pop('runtime_evidence')
                    elif mode in ('drop_mode','drop_schema'):
                        key='evidence_mode' if mode=='drop_mode' else 'evidence_schema'
                        current['identity'].pop(key,None);altered.pop(key,None)
                    elif mode=='drop_runtime_schema':current['runtime_evidence'].pop('schema',None)
                    elif mode=='downgrade_to_fixture':
                        current['identity'].update(evidence_mode='cpu_fixture',evidence_schema='v03-cpu-fixture-evidence-1')
                        altered.update(evidence_mode='cpu_fixture',evidence_schema='v03-cpu-fixture-evidence-1')
                        current.pop('runtime_evidence');altered.pop('runtime_evidence')
                    trace_path.write_text(''.join(json.dumps(line,allow_nan=True)+'\n' for line in lines))
                    if 'runtime_evidence' in current:current['runtime_evidence']['trace']['sha256']=sha256(trace_path)
                    self.rehash(api,altered,path,current)
                    with self.assertRaises(ValueError):self.verify(api,altered)

    def test_batch_physics_must_match_raw_maximum_not_just_global_totals(self):
        with tempfile.TemporaryDirectory() as root:
            api,summary,path,index=self.actual(root)
            physics_path=Path(index['runtime_evidence']['physics']['path']);physics=json.loads(physics_path.read_text())
            self.assertEqual(physics['batches'][0]['returned_controls'],27)
            physics['batches'][0]['returned_controls']+=1
            for key in ('physical_control_samples','physical_control_samples_lower_bound','unused_autoreset_samples'):
                physics[key]+=1;summary[key]+=1
            api.atomic_json(physics_path,physics);index['runtime_evidence']['physics']['sha256']=sha256(physics_path)
            self.rehash(api,summary,path,index)
            with self.assertRaises(ValueError):self.verify(api,summary)

    def test_equal_n_batches_cannot_exchange_controls_preserving_global_totals(self):
        with tempfile.TemporaryDirectory() as root:
            api,summary,path,index=self.actual(root,replay=False)
            physics_path=Path(index['runtime_evidence']['physics']['path']);physics=json.loads(physics_path.read_text())
            self.assertEqual(physics['batches'][0]['num_envs'],physics['batches'][1]['num_envs'])
            physics['batches'][0]['returned_controls']+=1;physics['batches'][1]['returned_controls']-=1
            api.atomic_json(physics_path,physics);index['runtime_evidence']['physics']['sha256']=sha256(physics_path)
            self.rehash(api,summary,path,index)
            with self.assertRaises(ValueError):self.verify(api,summary)

    def test_cpu_fixture_mode_is_explicit_and_cannot_omit_provenance(self):
        with tempfile.TemporaryDirectory() as root:
            summary,case_sha=make_gate_fixture(root)
            self.assertEqual(summary.get('evidence_mode'),'cpu_fixture')
            self.assertEqual(summary.get('evidence_schema'),'v03-cpu-fixture-evidence-1')
            self.assertTrue(self.verify(evaluation,summary)['evidence_verified'])
            path=Path(summary['raw_index_path']);index=json.loads(path.read_text())
            index['identity'].pop('evidence_mode');summary.pop('evidence_mode')
            self.rehash(evaluation,summary,path,index)
            with self.assertRaises(ValueError):self.verify(evaluation,summary)


class EvaluationCliQueueTests(unittest.TestCase):
    """Real core CLI/guard with a CPU stop before any NVML/model/environment."""
    class StopBeforeGpu(RuntimeError): pass
    def args(self, root, *, device):
        return ['--task',TASK,'--checkpoint',str(root/'not_loaded.pt'),'--checkpoint-sha256','a'*64,
            '--stage','S1','--kind','baseline','--seed','42','--num-envs','8','--episodes','20',
            '--cases',str(root/'not_loaded_cases.json'),'--device',device,
            '--output-dir',str(root/'evaluation'),'--vy','0','--wz','0']
    def test_held_queue_rejects_real_module_main_before_resource_monitor(self):
        import os
        from src.tasks.velocity.v03 import orchestration as orch
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{},clear=False):
            root=Path(directory);entered=[];os.environ.pop('V03_GPU_QUEUE_LOCK_FD',None)
            def stop(*args):
                entered.append(args);raise self.StopBeforeGpu('CPU_SENTINEL_STOP_BEFORE_NVML')
            with patch.object(orch,'ROOT',root),orch._new_gpu_queue_lease():
                rejected=False
                with patch.object(evaluation,'_resource_monitor',side_effect=stop):
                    try:evaluation.main(self.args(root,device='cuda:0'))
                    except BlockingIOError:rejected=True
                    except self.StopBeforeGpu:pass
                self.assertEqual(entered,[],'actual module main bypassed the held shared queue')
                self.assertTrue(rejected)
            self.assertFalse(torch.cuda.is_initialized())
    def test_inherited_real_description_reaches_cpu_stop_and_is_cloexec_consumed_once(self):
        import os
        from src.tasks.velocity.v03 import orchestration as orch
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);entered=[]
            with patch.object(orch,'ROOT',root),orch._new_gpu_queue_lease() as owner:
                fd=owner.fileno();os.set_inheritable(fd,True)
                with patch.dict(os.environ,{'V03_GPU_QUEUE_LOCK_FD':str(fd)},clear=False):
                    def stop(*args):
                        entered.append(dict(inheritable=os.get_inheritable(fd),environment_present='V03_GPU_QUEUE_LOCK_FD' in os.environ))
                        raise self.StopBeforeGpu('CPU_SENTINEL_STOP_BEFORE_NVML')
                    with patch.object(evaluation,'_resource_monitor',side_effect=stop):
                        with self.assertRaises(self.StopBeforeGpu):evaluation.main(self.args(root,device='cuda:0'))
                    self.assertEqual(len(entered),1)
                    self.assertFalse(entered[0]['inheritable'],'real inherited queue FD must be close-on-exec')
                    self.assertFalse(entered[0]['environment_present'],'core CLI must consume inherited FD exactly once')
                with self.assertRaises(BlockingIOError):orch._new_gpu_queue_lease()
            self.assertFalse(torch.cuda.is_initialized())
    def test_cpu_main_bypasses_gpu_queue_even_when_held_and_fd_environment_is_irrelevant(self):
        import os
        from src.tasks.velocity.v03 import orchestration as orch
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);entered=[]
            def stop(*args):
                entered.append(args);raise self.StopBeforeGpu('CPU_SENTINEL_STOP_BEFORE_NVML')
            with patch.object(orch,'ROOT',root),orch._new_gpu_queue_lease(),                    patch.dict(os.environ,{'V03_GPU_QUEUE_LOCK_FD':'not-a-gpu-description'},clear=False),                    patch.object(evaluation,'_resource_monitor',side_effect=stop):
                with self.assertRaises(self.StopBeforeGpu):evaluation.main(self.args(root,device='cpu'))
            self.assertEqual(len(entered),1);self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__": unittest.main()
