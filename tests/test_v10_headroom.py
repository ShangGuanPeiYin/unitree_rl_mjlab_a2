"""Pure-stdlib headroom admission; no real sleeps or physics."""
import importlib.util
from pathlib import Path
import pytest


def load_helper():
    path=Path(__file__).resolve().parents[1]/'src/tasks/velocity/v10/headroom.py'
    spec=importlib.util.spec_from_file_location('v10_headroom_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_continuous_sixty_seconds_and_dip_refuses():
    h=load_helper();clock=[0.]
    def sleep(t):clock[0]+=t
    report=h.wait_for_headroom(read_available=lambda:12*1024**3,monotonic=lambda:clock[0],sleep=sleep)
    assert report['passed'] and report['observed_seconds']>=60 and clock[0]==60
    clock[0]=0
    report=h.wait_for_headroom(read_available=lambda:11*1024**3 if clock[0]==17 else 12*1024**3,monotonic=lambda:clock[0],sleep=sleep)
    assert not report['passed'] and clock[0]==17


def test_prefill_guard_threshold_no_delay():
    h=load_helper()
    assert h.prefill_headroom(lambda:int(9.5*1024**3))['passed']
    assert not h.prefill_headroom(lambda:int(9.5*1024**3)-1)['passed']


def test_real_cli_failed_startup_imports_no_src_or_torch_and_claims_nothing(tmp_path):
    import subprocess,sys,textwrap
    root=Path(__file__).resolve().parents[1]
    code=textwrap.dedent('''
        import importlib.util,runpy,sys,json
        from pathlib import Path
        root=Path(sys.argv[1]);out=Path(sys.argv[2]);original=importlib.util.spec_from_file_location
        def patched(name,path,*args,**kwargs):
            spec=original(name,path,*args,**kwargs)
            if name=='v10_headroom_preimport':
                execute=spec.loader.exec_module
                def load(module):
                    execute(module);actual=module.wait_for_headroom
                    module.wait_for_headroom=lambda:actual(read_available=lambda:0)
                spec.loader.exec_module=load
            return spec
        importlib.util.spec_from_file_location=patched
        module=runpy.run_path(str(root/'scripts/run_a2_v10.py'),run_name='cli_cpu_test')
        sys.argv=['run_a2_v10.py','probe-a1-v2','--output-dir',str(out),'--linkage-contract','unused',
            '--global-probe-ledger',str(out)+'.ledger','--probe-budget-authorization','unused','--probe-sequence-id','unused']
        assert module['main']()==1
        assert 'torch' not in sys.modules and 'src.tasks' not in sys.modules
        assert not Path(str(out)+'.ledger').exists()
        print('PURE_STDLIB_NO_CLAIM')
    ''')
    result=subprocess.run([sys.executable,'-c',code,str(root),str(tmp_path/'attempt')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'PURE_STDLIB_NO_CLAIM' in result.stdout
