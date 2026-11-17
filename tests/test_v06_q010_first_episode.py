"""Q010: exactly one immutable first episode/window per explicit case slot."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import a2_v06_bootstrap
a2_v06_bootstrap.activate(ROOT)
from src.tasks.velocity.v06 import runner as target,evaluation

class Policy:
    def act(self,obs,deterministic=False):
        assert deterministic
        return torch.zeros(int(obs['n']),12)


def cases(count=8,kind='validation',flat=False):
    return [dict(case_id=f'{kind}:{i}',vx=.3,kind=kind,phase='F0' if flat else 'D',
                 measurement_stage='F0' if flat else 'D',episode_index=i,
                 initial_pose=[-1.,0.,0.],tile_id=0 if flat else 17) for i in range(count)]


def episode(index,step,success=True,**overrides):
    value=dict(env_id=index,terminal_step=step,control_samples=step,
        success=success,failure=not success,timeout=False,reason='success' if success else 'nonfoot',
        active_samples=step,stall_samples=0,first_step_valid=True,
        max_abs_y=.01,max_abs_yaw_deg=1.,mean_vx=.35,mean_speed=.36,
        forward_distance_start=-1.,forward_distance_end=.5,forward_distance=1.5,
        reward_sums={'tracking_lin_vel':.1})
    value.update(overrides)
    return value


class AsyncEnv:
    def __init__(self,first_success=False,malformed=None):
        self.num_envs=8;self.u=0;self.first_success=first_success;self.malformed=malformed
        self.reused=episode(0,1,first_success,stall_samples=1,first_step_valid=False)
        self.actions=[]
    def reset(self):return {'n':8},{}
    def step(self,action):
        self.u+=1;self.actions.append(action.clone())
        if self.u==1:
            rows=[self.reused]
            if self.malformed=='missing_id':rows[0].pop('env_id')
            if self.malformed=='out_of_bounds':rows[0]['env_id']=8
            if self.malformed=='negative_id':rows[0]['env_id']=-1
            if self.malformed=='bool_id':rows[0]['env_id']=True
            if self.malformed=='string_id':rows[0]['env_id']='0'
            if self.malformed=='step_mismatch':rows[0]['terminal_step']=2
            if self.malformed=='nan_speed':rows[0]['mean_speed']=float('nan')
            if self.malformed=='nan_yaw':rows[0]['max_abs_yaw_deg']=float('nan')
            if self.malformed=='duplicate':rows.append(deepcopy(rows[0]))
            if self.malformed=='missing_first_counter':rows[0].pop('active_samples')
            if self.malformed=='nan_mean':rows[0]['mean_vx']=float('nan')
            if self.malformed=='missing_done':rows=[]
        else:
            self.reused.update(success=not self.first_success,failure=self.first_success,
                mean_vx=.9,active_samples=1,stall_samples=0,first_step_valid=True)
            self.reused['reward_sums']['tracking_lin_vel']=99.
            rows=[self.reused,episode(self.u-1,self.u,True)]
        terminated=torch.zeros(8,dtype=torch.bool)
        if self.u==1 and self.malformed=='missing_done':terminated[0]=True
        for row in rows:
            i=row.get('env_id',0)
            if isinstance(i,int) and 0<=i<8:terminated[i]=True
        return {'n':8},torch.zeros(8),terminated,torch.zeros(8,dtype=torch.bool),{'episodes':rows}
    def diagnostics_cases(self):
        raise AssertionError('finished stairs must never consult live diagnostics')


@pytest.mark.parametrize('first_success',[False,True])
@pytest.mark.parametrize('kind',['validation','diagnostic','stair_retention','test'])
def test_first_asynchronous_episode_is_immutable_and_reports_actual_steps(first_success,kind):
    env=AsyncEnv(first_success)
    rows=target.environment_case_runner(Policy(),lambda _:env,step_limit=8)(cases(kind=kind))
    assert len(rows)==8 and rows[0]['climb_success'] is first_success
    assert rows[0]['failure'] is (not first_success)
    assert rows[0]['steps']==1 and rows[-1]['steps']==8
    assert rows[0]['mean_vx']==.35
    assert rows[0]['active_samples']==1 and rows[0]['stall_samples']==1
    assert rows[0]['first_step_valid'] is False
    assert rows[0]['first_episode']['reward_sums']['tracking_lin_vel']==.1
    assert all(row['evidence_kind']=='terminal' for row in rows)
    assert sum(row['physical_control_samples'] for row in rows)==64
    assert sum(row['extra_control_samples_after_case'] for row in rows)==28
    assert all(torch.equal(action,torch.zeros(8,12)) for action in env.actions)


@pytest.mark.parametrize('malformed',['missing_id','out_of_bounds','negative_id','duplicate',
    'missing_first_counter','nan_mean','missing_done','bool_id','string_id',
    'step_mismatch','nan_speed','nan_yaw'])
def test_bad_first_episode_is_invalid_and_cannot_be_filled_from_later_episode(malformed):
    env=AsyncEnv(malformed=malformed)
    with pytest.raises(evaluation.EvaluationError):
        target.environment_case_runner(Policy(),lambda _:env,step_limit=8)(cases())


class FlatEnv:
    def __init__(self,failure_step=None,missing_window=False):
        self.num_envs=2;self.u=0;self.failure_step=failure_step;self.missing_window=missing_window
    def reset(self):return {'n':2},{}
    def step(self,action):
        self.u+=1;rows=[];done=torch.zeros(2,dtype=torch.bool)
        if self.u==self.failure_step:
            rows=[episode(0,self.u,False,active_samples=self.u,forward_distance=.7,
                          forward_distance_end=-.3,mean_vx=.2)];done[0]=True
        info=dict(episodes=rows,success=torch.zeros(2,dtype=torch.bool),failure=done,
                  timeout=torch.zeros(2,dtype=torch.bool),reason=torch.zeros(2,dtype=torch.long))
        return {'n':2},torch.zeros(2),done,torch.zeros(2,dtype=torch.bool),info
    def diagnostics_cases(self):
        rows=[]
        for i in range(2):
            # Slot0 may already be in a later episode. Its live endpoint must
            # never replace the first failure snapshot.
            step=1 if i==0 and self.failure_step and self.u>self.failure_step else self.u
            row=dict(env_id=i,step=step,active_samples=step,stall_samples=0,first_step_valid=False,
                max_abs_y=.01,max_abs_yaw_deg=1.,forward_distance_start=-1.,
                forward_distance_end=.5,forward_distance=1.5,mean_vx=.4,mean_speed=.41)
            if self.missing_window:row.pop('mean_vx')
            rows.append(row)
        return rows


@pytest.mark.parametrize('kind',['validation','flat_retention','diagnostic'])
@pytest.mark.parametrize('failure_step',[None,499,500])
def test_flat_first_terminal_or_exact_500_window_uses_one_episode(kind,failure_step):
    env=FlatEnv(failure_step)
    rows=target.environment_case_runner(Policy(),lambda _:env,step_limit=500)(cases(2,kind,True))
    assert rows[1]['steps']==500 and rows[1]['evidence_kind']=='window_end'
    assert rows[1]['flat_pass'] is True and rows[1]['mean_vx']==.4
    if failure_step:
        assert rows[0]['steps']==failure_step and rows[0]['evidence_kind']=='terminal'
        assert rows[0]['flat_pass'] is False and rows[0]['forward_distance']==pytest.approx(.7)
        assert rows[0]['mean_vx']==.2
    else:assert rows[0]['evidence_kind']=='window_end' and rows[0]['steps']==500


def test_flat_499_alive_and_incomplete_500_window_are_invalid():
    with pytest.raises(evaluation.EvaluationError):
        target.environment_case_runner(Policy(),lambda _:FlatEnv(),step_limit=499)(cases(2,flat=True))
    with pytest.raises(evaluation.EvaluationError):
        target.environment_case_runner(Policy(),lambda _:FlatEnv(missing_window=True),step_limit=500)(cases(2,flat=True))


def test_real_env_readonly_mean_exports_match_existing_episode_accumulators():
    from src.tasks.velocity.v06.env import V06Env
    env=V06Env(num_envs=2,device='cpu',seed=42,phase='F0',evaluation=True,
               fixed_vx=.3,initial_cases=cases(2,flat=True),frontier_tile_id=0)
    env.reset()
    for _ in range(2):env.step(torch.zeros(2,12))
    rows=env.diagnostics_cases()
    for i,row in enumerate(rows):
        count=env.episode_length_buf[i].clamp_min(1).to(torch.float32)
        assert row['mean_vx']==float(env.sum_vx[i]/count)
        assert row['mean_speed']==float(env.sum_speed[i]/count)
    env.reset()
    assert all(row['mean_vx']==0. and row['mean_speed']==0. for row in env.diagnostics_cases())
    env.close()


def test_scalar_batch_masks_and_window_flags_or_missing_reason_are_invalid():
    class ScalarMask(AsyncEnv):
        def step(self,action):
            values=list(super().step(action));values[2]=True;return tuple(values)
    with pytest.raises(evaluation.EvaluationError):
        target.environment_case_runner(Policy(),lambda _:ScalarMask(),step_limit=8)(cases())
    for broken in ['failure','reason']:
        class BadWindow(FlatEnv):
            def step(self,action):
                values=super().step(action)
                if broken=='reason':values[4].pop('reason')
                else:values[4]['failure']=False
                return values
        with pytest.raises(evaluation.EvaluationError):
            target.environment_case_runner(Policy(),lambda _:BadWindow(),step_limit=500)(cases(2,flat=True))
    rows=target.environment_case_runner(Policy(),lambda _:FlatEnv(),step_limit=500)(cases(2,flat=True))
    assert all(row['reason']==0 and row['first_episode']['reason']==0
               and row['first_episode']['pre_reset_reason']==0 for row in rows)
