"""Execute the approved W7 CLI sequence only after the unique formal terminal.

Operational companion, outside the frozen training closure. No model, recipe,
formal ledger, control file, checkpoint or historical evidence is modified.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import datetime
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/home/xxs/dog/unitree_rl_mjlab_a2_v08')
CANONICAL = Path('/home/xxs/dog/unitree_rl_mjlab_a2')
BASE = CANONICAL/'logs/a2_ideal_stairs_v08'
IMPL = BASE/'implementation'
FORMAL = BASE/'continue_seed42_from006400_trial01'
PYTHON = '/home/xxs/anaconda3/envs/a2/bin/python'
SOURCE = CANONICAL/'logs/a2_ideal_stairs_v07/continue_seed42_from004817_trial01/checkpoints/stop_006400.pt'
SOURCE_SHA = '30ed23ba16f7e490afca001fbbecadb4a11ff4b11733a5ae7ac7d0c841f9a8b2'
FREEZE = IMPL/'freeze/v2/FROZEN_INPUTS.json'
FREEZE_SHA = 'ba85e965723050fd335c492a15385f79a41f6d42d439d9d16a1cc6d3617cc783'
ACCEPTANCE = IMPL/'coordination/reports/ENGINEERING_ACCEPTANCE.json'
BASELINE = BASE/'baseline_seed42_trial01/baseline_result.json'
CONTROLLER = IMPL/'coordination/job_controller_20261118.py'
FORMAL_RECEIPT = IMPL/'gpu/FORMAL_01_JOB/PROCESS_RECEIPT.json'
PIPELINE = IMPL/'postprocess_v1'
CONTROL = IMPL/'POSTPROCESS_CONTROL.json'
CAPS = dict(A=4000, B=2000, C=2000, D=4000, E=6000)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def cancelled():
    if not CONTROL.exists():
        return False
    value = read(CONTROL)
    if set(value)-{'cancel', 'reason'} or type(value.get('cancel')) is not bool:
        raise ValueError('invalid explicit postprocess control')
    return value['cancel']


def event(kind, **data):
    value = dict(time=datetime.datetime.now().astimezone().isoformat(), kind=kind, **data)
    with (PIPELINE/'events.jsonl').open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())
    write(PIPELINE/'status.json', value)


def journal(message):
    date = datetime.datetime.now().strftime('%Y-%m-%d')
    entry = '\n### '+date+'｜v08授权终态后处理\n\n'+message+'\n'
    for root in (ROOT, CANONICAL):
        with (root/'logs/codex_changes.md').open('a') as stream:
            stream.write(entry)


def terminal_plan(process, report, state):
    """Pure allow-list; operator/error/running/foreign outcomes never launch W7."""
    if process.get('status') != 'finished' or process.get('exit_code') != 0:
        raise ValueError('formal process is not a successful finished process')
    if report.get('mode') != 'formal' or state.get('mode') != 'formal':
        raise ValueError('foreign container')
    learning, cost, cur = state['learning'], state['ledger'], state['curriculum']['state']
    if report['learning'] != learning or report['cost'] != cost or report['phase'] != cur['phase']:
        raise ValueError('formal report differs from actual final checkpoint')
    if cost.get('pending_update') is not None or cost.get('pending_boundary') is not None:
        raise ValueError('formal boundary incomplete')
    n = cost['formal_updates']
    if type(n) is not int or not 1 <= n <= 18000 or cost['engineering_updates'] != 0:
        raise ValueError('mixed or invalid learning budget')
    if learning['update'] != 6400+n or learning['adam_steps'] != 128000+20*n:
        raise ValueError('learning counters differ')
    if cost['control_samples'] != n*512*48 or cost['physics_substeps'] != n*512*48*4:
        raise ValueError('formal sample counters differ')
    spent = cur['spent_by_phase']
    if set(spent) != set(CAPS) or any(type(v) is not int or not 0 <= v <= CAPS[k] for k, v in spent.items()):
        raise ValueError('invalid phase budget')
    if sum(spent.values()) != n or not cur['finished'] or cur['last_update'] != learning['update']:
        raise ValueError('curriculum is not the actual completed budget ledger')
    if report['stop_status'] != cost['stop_state']:
        raise ValueError('terminal reason differs')
    if report['stop_status'] == 'curriculum_gates_passed':
        history = cur['gate_history']
        if cur['phase'] != 'E' or cur['stop_status'] != 'complete' or cur['gate_streak'] < 2:
            raise ValueError('E completion not established')
        if len(history) < 2 or any(v['phase'] != 'E' or v['passed'] is not True for v in history[-2:]):
            raise ValueError('E consecutive complete gates missing')
        if history[-1]['update'] != learning['update'] or history[-1]['update']-history[-2]['update'] != 100:
            raise ValueError('E gates are not consecutive current boundaries')
        # native_inputs has already recomputed every referenced raw sidecar.
        # Bind the curriculum flags to those verified complete E gate results.
        case_sha = state['case_tables']['gate:E']['sha256']
        for gate in history[-2:]:
            reference = state.get('evaluation', {}).get('gate:'+str(gate['update']), {})
            identity, summary = reference.get('identity', {}), reference.get('summary', {})
            expected = dict(kind='gate', phase='E', update=gate['update'],
                case_table_sha256=case_sha, source_sha256=SOURCE_SHA, freeze_sha256=FREEZE_SHA)
            if any(identity.get(k) != v for k, v in expected.items()):
                raise ValueError('actual consecutive E gate reference missing or foreign')
            expected_summary = dict(kind='gate', phase='E', case_table_sha256=case_sha,
                expected=420, valid=420, invalid=0, error=0, engineering_error=False, gate_pass=True)
            if any(summary.get(k) != v for k, v in expected_summary.items()):
                raise ValueError('actual E gate was not a complete valid passing denominator')
        return dict(kind='curriculum_gates_passed', run_final2100=True, phase='E')
    if report['stop_status'] == 'budget_exhausted':
        phase = cur['phase']
        if cur['stop_status'] != 'budget_exhausted' or spent[phase] != CAPS[phase] or cur['generation_updates'] != CAPS[phase]:
            raise ValueError('phase cap not established')
        if cur['gate_streak'] >= 2 or cur['last_gate_update'] != learning['update']:
            raise ValueError('phase-cap gate was not settled')
        return dict(kind='budget_exhausted', run_final2100=False, phase=phase)
    raise ValueError('operator, engineering, running or unknown stop is not a contract terminal')


def native_inputs(report):
    """Use the existing complete validators; never rewrite a frozen lineage."""
    sys.path.insert(0, str(ROOT/'scripts'))
    import a2_v08_bootstrap
    a2_v08_bootstrap.activate(ROOT)
    from src.tasks.velocity.v08 import recovery, orchestration
    import torch
    assert not torch.cuda.is_initialized()
    checkpoint = Path(report['checkpoint']).resolve()
    if checkpoint.parent != (FORMAL/'checkpoints').resolve():
        raise ValueError('final checkpoint is outside the unique formal directory')
    metadata = read(checkpoint.with_suffix('.json'))
    if metadata['sha256'] != report['checkpoint_sha256']:
        raise ValueError('final checkpoint metadata SHA differs')
    receipts = [read(p) for p in (FORMAL/'checkpoints').glob('*.json')]
    latest = max(receipts, key=lambda v: (v['update'], v['sequence']))
    if Path(latest['path']).resolve() != checkpoint or latest['sha256'] != report['checkpoint_sha256']:
        raise ValueError('final policy is not latest durable checkpoint')
    frozen = read(FREEZE)
    if frozen['freeze']['sha256'] != FREEZE_SHA:
        raise ValueError('approved v2 freeze changed')
    recovery.verify_freeze(frozen['freeze']); recovery.verify_case_tables(frozen['case_tables'])
    orchestration.admit(acceptance_file=ACCEPTANCE, baseline_report=BASELINE,
                        frozen=frozen['freeze'], case_tables=frozen['case_tables'], formal=True)
    state = recovery.load_verified(checkpoint, report['checkpoint_sha256'])
    recovery.validate_checkpoint(state, mode='formal')
    recovery.verify_learning_journal(FORMAL, state['learning']['update'])
    transactions = recovery.verify_evaluation_journal(FORMAL, FREEZE_SHA)
    recovery.verify_evaluation_checkpoint_coverage(state, transactions)
    if state['source']['checkpoint_sha256'] != SOURCE_SHA:
        raise ValueError('source ancestry changed')
    assert not torch.cuda.is_initialized()
    return state, frozen


def stage(name, mode, output, policy, policy_sha, num_envs):
    if cancelled():
        raise InterruptedError('explicit postprocess cancellation')
    command = [PYTHON, '-B', str(ROOT/'scripts'/('play_a2_ideal_stairs_v08.py' if mode == 'replay' else 'evaluate_a2_ideal_stairs_v08.py')),
               '--source-checkpoint', str(SOURCE), '--source-sha256', SOURCE_SHA,
               '--policy-checkpoint', str(policy), '--policy-sha256', policy_sha,
               '--output-dir', str(output), '--device', 'cuda:0', '--seed', '42',
               '--num-envs', str(num_envs), '--updates', '0', '--mode', mode,
               '--phase', 'E', '--max-control-steps', '50000', '--freeze-file', str(FREEZE)]
    if mode == 'final':
        command += ['--acceptance-file', str(ACCEPTANCE), '--baseline-report', str(BASELINE)]
    else:
        command += ['--record-video']
    job = PIPELINE/(name+'_JOB')
    if output.exists() or job.exists():
        raise ValueError('existing postprocess output/attempt requires manual audit; no automatic retry')
    config = dict(command=command, cwd=str(ROOT), job_directory=str(job),
        single_worker_lock=str(IMPL/'GPU_SINGLE_WORKER.lock'),
        environment_overrides=dict(CUDA_VISIBLE_DEVICES='0', A2_V08_WANDB='0', MUJOCO_GL='egl',
             OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1'))
    config_path = PIPELINE/(name+'_JOB_CONFIG.json'); write(config_path, config)
    with (PIPELINE/(name+'_CONTROLLER.log')).open('xb') as stream:
        child = subprocess.Popen([PYTHON, str(CONTROLLER), str(config_path)], cwd=ROOT,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    event('stage_running', stage=name, controller_pid=child.pid, command=command)
    journal('按SDS合同终态启动'+name+'，原生'+mode+'、seed42、cuda:0、环境'+str(num_envs)+'、updates0；policy SHA '+policy_sha+'。实际argv/回执在'+str(job)+'，未新增正式学习预算。')
    while child.poll() is None:
        if cancelled():
            os.killpg(child.pid, signal.SIGTERM)
            child.wait()
            raise InterruptedError('explicit cancellation terminated only the owned postprocess process group; partial evidence retained')
        time.sleep(2)
    receipt = read(job/'PROCESS_RECEIPT.json')
    if child.returncode != 0 or receipt.get('status') != 'finished' or receipt.get('exit_code') != 0:
        raise RuntimeError('postprocess failed; partial files/cost intents retained; no automatic rerun: '+name)
    event('stage_finished', stage=name, receipt=receipt)
    return receipt


def expected_video_cases(frozen):
    cases = frozen['case_tables']['final:E']['table']['cases']
    wanted = [('flat_stop', .5, None), ('up_stop', .5, None), ('E1', .5, None), ('E2', .5, None),
              ('E3', .5, 'up_reverse'), ('E4', .5, None), ('E4', .3, None), ('E4', .6, None)]
    return [next(c['case_id'] for c in cases if c['task'] == task and c['speed'] == speed
                 and (script is None or c['script'] == script)) for task, speed, script in wanted]


def decode_videos(report, output, expected_ids, policy_sha):
    import imageio_ffmpeg
    if report.get('learning_updates') != 0 or report['policy_checkpoint_sha256'] != policy_sha:
        raise ValueError('video policy or zero-learning identity differs')
    if report['prescribed_case_ids'] != expected_ids or [r['case_id'] for r in report['records']] != expected_ids:
        raise ValueError('the eight prescribed cases were changed')
    if len(expected_ids) != 8 or any(r['status'] not in ('success', 'failure') for r in report['records']):
        raise ValueError('video engineering error or incomplete protocol')
    if report['control_samples'] != sum(r['control_steps'] for r in report['records']) or report['physics_substeps'] != report['control_samples']*4:
        raise ValueError('video physical costs differ')
    settings = dict(width=1280, height=720, fps=25, control_dt=.02, frame_stride=2,
        azimuth='90+body_yaw_degrees', elevation=-28., distance=3.8, speed=1.)
    if report.get('video_settings') != settings or report.get('failures_retained') is not True or report.get('protocol_override') is not None:
        raise ValueError('prescribed recording protocol differs')
    media = []
    for row in report['records']:
        if cancelled(): raise InterruptedError('explicit cancellation before decode')
        path = Path(row['video_path']).resolve()
        if path.parent != output.resolve() or path.name != row['case_id']+'.mp4' or not path.is_file() or not path.stat().st_size:
            raise ValueError('missing or foreign video path')
        if row.get('video_settings') != settings or type(row['control_steps']) is not int or row['control_steps'] < 1:
            raise ValueError('case recording protocol differs')
        frames = imageio_ffmpeg.read_frames(str(path), input_params=['-threads','2'])
        try:
            metadata = next(frames)
        finally:
            frames.close()
        expected_frames = (row['control_steps']+1)//2+1
        if metadata['source_size'] != (1280,720) or not math.isclose(metadata['fps'],25.,abs_tol=1e-6):
            raise ValueError('video encoded size or fps differs: '+path.name)
        # MP4 duration metadata is printed at 0.01s resolution by this reader.
        if not math.isclose(metadata['duration'],expected_frames/25.,rel_tol=0.,abs_tol=.011):
            raise ValueError('video duration differs from recorded physical steps: '+path.name)
        log = PIPELINE/(path.stem+'_decode.log')
        progress = PIPELINE/(path.stem+'_decode_progress.txt')
        with log.open('xb') as errors, progress.open('xb') as stream:
            decoded = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-xerror', '-threads', '2',
                '-i', str(path), '-map', '0:v:0', '-an', '-fps_mode', 'passthrough',
                '-progress', 'pipe:1', '-nostats', '-f', 'null', '-'], stdout=stream, stderr=errors)
        if decoded.returncode != 0 or log.stat().st_size:
            raise ValueError('video full decode failed: '+path.name)
        decoded_progress = dict(line.split('=',1) for line in progress.read_text().splitlines() if '=' in line)
        if decoded_progress.get('progress') != 'end' or int(decoded_progress.get('frame',-1)) != expected_frames:
            raise ValueError('video decoded frame count differs from complete physical recording: '+path.name)
        decoded_seconds = int(decoded_progress.get('out_time_us',-1))/1e6
        if not math.isclose(decoded_seconds,expected_frames/25.,rel_tol=0.,abs_tol=.011):
            raise ValueError('decoded video timeline differs: '+path.name)
        media.append(dict(case_id=row['case_id'],path=str(path),sha256=sha(path),bytes=path.stat().st_size,
                          outcome=row['status'],reason=row['reason'],full_decode_exit_code=0,
                          decoded_frames=expected_frames,expected_frames=expected_frames,
                          fps=metadata['fps'],size=metadata['source_size'],duration_seconds=metadata['duration'],
                          decoded_seconds=decoded_seconds,decode_error_log=str(log),decode_progress=str(progress)))
    return media


def execute():
    # Failed lock acquisition or duplicate directory must never write an event
    # into another instance's running/completed evidence.
    with (IMPL/'POSTPROCESS_DRIVER.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        PIPELINE.mkdir(parents=True, exist_ok=False)
        try:
            execute_owned()
        except BaseException as error:
            event('postprocess_cancelled' if isinstance(error,InterruptedError) else 'postprocess_blocked',
                  error_type=type(error).__name__,error=str(error),no_automatic_retry=True)
            raise


def execute_owned():
    event('waiting_formal', formal_receipt=str(FORMAL_RECEIPT), driver_pid=os.getpid())
    while True:
        if cancelled(): raise InterruptedError('explicit cancellation before contract terminal')
        process = read(FORMAL_RECEIPT)
        if process['status'] == 'finished': break
        time.sleep(10)
    if process.get('exit_code') != 0:
        raise ValueError('formal process error; no automatic postprocessing')
    report = read(FORMAL/'run_result.json')
    # The original controller may still hold the GPU lock for a few milliseconds.
    time.sleep(2)
    state, frozen = native_inputs(report)
    plan = terminal_plan(process, report, state)
    expected = expected_video_cases(frozen)
    policy, policy_sha = Path(report['checkpoint']), report['checkpoint_sha256']
    summary = dict(plan=plan, formal_result=report, curriculum=state['curriculum']['state'],
                   policy_checkpoint=str(policy), policy_sha256=policy_sha, final2100=None)
    write(PIPELINE/'CONTRACT_TERMINAL.json', summary)
    event('contract_terminal_verified', plan=plan, policy_checkpoint=str(policy), policy_sha256=policy_sha)
    journal('已实际核验唯一正式run合同终态'+plan['kind']+'，阶段'+plan['phase']+'，update'+str(report['learning']['update'])+'，policy SHA '+policy_sha+'。按SDS执行后处理；普通预算耗尽不运行伪造final2100，失败录像仍保留。')
    del state; gc.collect()
    if plan['run_final2100']:
        output = BASE/'final_seed42_trial01'
        stage('FINAL_01', 'final', output, policy, policy_sha, 8)
        result = read(output/'final_result.json'); data = result['evaluation']['payload']['summary']
        if not result['complete'] or result['learning_updates'] != 0 or result['policy_checkpoint_sha256'] != policy_sha:
            raise ValueError('final2100 incomplete or wrong policy')
        if data['expected'] != 2100 or data['valid'] != 2100 or data['invalid'] or data['error'] or data['engineering_error']:
            raise ValueError('final2100 full valid denominator missing')
        summary['final2100'] = dict(path=str(output/'final_result.json'),sha256=sha(output/'final_result.json'),
            summary=deepcopy(data),policy_passed=bool(result['policy_capability_established']))
        write(PIPELINE/'FINAL_2100_SUMMARY.json', summary['final2100'])
        del result; gc.collect()
    else:
        write(PIPELINE/'FINAL_2100_NOT_RUN.json', dict(reason='phase_budget_exhausted',phase=plan['phase'],
              policy_checkpoint=str(policy),policy_sha256=policy_sha,final2100_evaluated=False))
    output = BASE/'videos_final_seed42_trial01'
    stage('REPLAY_01', 'replay', output, policy, policy_sha, 1)
    replay = read(output/'replay_result.json')
    summary['videos'] = decode_videos(replay, output, expected, policy_sha)
    summary['replay_control_samples'] = replay['control_samples']
    summary['complete'] = True
    summary['qualification'] = 'Authorized terminal CLI sequence and media integrity complete; independent final results review and root substantive report still required. No success was substituted for failed cases.'
    write(PIPELINE/'POSTPROCESS_RESULT.json', summary)
    event('postprocess_complete', result=str(PIPELINE/'POSTPROCESS_RESULT.json'), videos=len(summary['videos']))
    journal('合同终态后原生零学习后处理已实际完成，8个预指定视频保留全部成功/失败并逐条完整解码exit0；最终2100状态及完整分母、policy/video SHA见'+str(PIPELINE/'POSTPROCESS_RESULT.json')+'。独立最终结果复核与root完整工作汇报仍待完成，不能以此自动宣称策略目标通过。')


def self_test():
    import unittest
    class ContractTerminalTests(unittest.TestCase):
        def fixture(self, complete=False):
            spent = deepcopy(CAPS) if complete else dict(A=4000,B=0,C=0,D=0,E=0)
            n = sum(spent.values()); update = 6400+n
            cur = dict(phase='E' if complete else 'A',finished=True,last_update=update,
                spent_by_phase=spent,generation_updates=6000 if complete else 4000,
                stop_status='complete' if complete else 'budget_exhausted',gate_streak=2 if complete else 1,
                last_gate_update=update,gate_history=[dict(phase='E',passed=True,update=update-100),dict(phase='E',passed=True,update=update)])
            learning = dict(update=update,adam_steps=128000+20*n)
            cost = dict(formal_updates=n,engineering_updates=0,control_samples=n*512*48,
                physics_substeps=n*512*48*4,pending_update=None,pending_boundary=None,
                stop_state='curriculum_gates_passed' if complete else 'budget_exhausted')
            state = dict(mode='formal',learning=learning,ledger=cost,curriculum=dict(state=cur))
            if complete:
                state['case_tables']={'gate:E':dict(sha256='fixture-E')}
                state['evaluation']={}
                for gate in cur['gate_history']:
                    state['evaluation']['gate:'+str(gate['update'])]=dict(
                        identity=dict(kind='gate',phase='E',update=gate['update'],case_table_sha256='fixture-E',source_sha256=SOURCE_SHA,freeze_sha256=FREEZE_SHA),
                        summary=dict(kind='gate',phase='E',case_table_sha256='fixture-E',expected=420,valid=420,invalid=0,error=0,engineering_error=False,gate_pass=True))
            report = dict(mode='formal',learning=deepcopy(learning),cost=deepcopy(cost),phase=cur['phase'],stop_status=cost['stop_state'])
            return dict(status='finished',exit_code=0),report,state
        def test_cap_with_only_one_passing_gate_skips_final_but_keeps_videos(self):
            p,r,s=self.fixture();self.assertFalse(terminal_plan(p,r,s)['run_final2100'])
        def test_only_complete_E_two_consecutive_gates_runs_final(self):
            p,r,s=self.fixture(True);self.assertTrue(terminal_plan(p,r,s)['run_final2100'])
            s['curriculum']['state']['gate_history'][-2]['update']-=100
            with self.assertRaises(ValueError):terminal_plan(p,r,s)
        def test_operator_error_running_and_engineering_containers_never_trigger(self):
            for reason in ('operator_stop','operator_pause','engineering_error','running','monitoring_blocked'):
                p,r,s=self.fixture();r['stop_status']=s['ledger']['stop_state']=r['cost']['stop_state']=reason
                with self.assertRaises(ValueError):terminal_plan(p,r,s)
            p,r,s=self.fixture();p['exit_code']=1
            with self.assertRaises(ValueError):terminal_plan(p,r,s)
            p,r,s=self.fixture();p['status']='running'
            with self.assertRaises(ValueError):terminal_plan(p,r,s)
            p,r,s=self.fixture();r['mode']=s['mode']='probe'
            with self.assertRaises(ValueError):terminal_plan(p,r,s)
        def test_partial_and_foreign_budget_never_trigger(self):
            p,r,s=self.fixture();r['cost']['pending_update']=s['ledger']['pending_update']={'update':10401}
            with self.assertRaises(ValueError):terminal_plan(p,r,s)
            p,r,s=self.fixture();r['cost']['engineering_updates']=s['ledger']['engineering_updates']=1
            with self.assertRaises(ValueError):terminal_plan(p,r,s)
            p,r,s=self.fixture();s['curriculum']['state']['spent_by_phase']['A']-=1
            with self.assertRaises(ValueError):terminal_plan(p,r,s)
        def test_unsettled_cap_and_nonE_success_never_trigger(self):
            p,r,s=self.fixture();s['curriculum']['state']['last_gate_update']-=100
            with self.assertRaises(ValueError):terminal_plan(p,r,s)
            p,r,s=self.fixture(True);r['phase']=s['curriculum']['state']['phase']='D'
            with self.assertRaises(ValueError):terminal_plan(p,r,s)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ContractTerminalTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');parser.add_argument('--inspect-only',action='store_true')
    args=parser.parse_args()
    if args.self_test:return self_test()
    if args.inspect_only:
        print(json.dumps(dict(formal_process=read(FORMAL_RECEIPT),pipeline_exists=PIPELINE.exists(),cancelled=cancelled()),indent=2));return 0
    execute()
    return 0

if __name__=='__main__':raise SystemExit(main())
