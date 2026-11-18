from pathlib import Path
import subprocess
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
ENTRIES=('run','train','evaluate','play','test','validate')
@pytest.mark.parametrize('entry',ENTRIES)
def test_native_entry_help_without_task_registry_scan(entry):
    path=ROOT/'scripts'/f'{entry}_a2_ideal_stairs_v08.py'
    assert path.is_file()
    result=subprocess.run([sys.executable,str(path),'--help'],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert '--source-checkpoint' in result.stdout and '--source-sha256' in result.stdout
    assert '--num-envs' in result.stdout and '--updates' in result.stdout

def test_final_requires_native_sha_bound_policy(tmp_path):
    sys.path.insert(0,str(ROOT/'scripts'))
    import a2_v08_bootstrap;a2_v08_bootstrap.activate(ROOT)
    from src.tasks.velocity.v08.runtime import build_parser,resolve_request
    from src.tasks.velocity.v08.continuation_v08 import SOURCE_PATH,SOURCE_SHA256
    args=['--source-checkpoint',SOURCE_PATH,'--source-sha256',SOURCE_SHA256,'--output-dir',str(tmp_path/'final'),
        '--device','cpu','--seed','42','--num-envs','8','--updates','0','--mode','final']
    with pytest.raises(ValueError):resolve_request(build_parser().parse_args(args))

def test_replay_cases_are_prescribed_and_cover_required_videos():
    from src.tasks.velocity.v08.evaluation import make_cases
    from src.tasks.velocity.v08.runtime import prescribed_replay_cases
    table=make_cases('E',kind='final')
    cases=prescribed_replay_cases(table)
    assert len(cases)==8
    assert [(c['task'],c['speed']) for c in cases]==[('flat_stop',.5),('up_stop',.5),('E1',.5),('E2',.5),('E3',.5),('E4',.5),('E4',.3),('E4',.6)]
    assert cases[4]['script']=='up_reverse'
