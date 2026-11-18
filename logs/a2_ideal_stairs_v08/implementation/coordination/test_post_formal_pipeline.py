"""CPU-only regression for terminal queue decisions; no physics or formal writes."""
from copy import deepcopy
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('post_pipeline', Path(__file__).with_name('post_formal_pipeline_20261118.py'))
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def terminal_fixture():
    # Logical full-budget E fixture only; never a physical success claim.
    n, update = 18000, 24400
    cur = dict(phase='E', finished=True, last_update=update, spent_by_phase=deepcopy(m.CAPS),
        generation_updates=6000, stop_status='complete', gate_streak=2, last_gate_update=update,
        gate_history=[dict(phase='E', passed=True, update=24300), dict(phase='E', passed=True, update=24400)])
    learning = dict(update=update, adam_steps=488000)
    cost = dict(formal_updates=n, engineering_updates=0, control_samples=442368000,
        physics_substeps=1769472000, pending_update=None, pending_boundary=None,
        stop_state='curriculum_gates_passed')
    refs = {}
    for u in (24300,24400):
        refs['gate:'+str(u)] = dict(identity=dict(update=u,kind='gate',phase='E',
            case_table_sha256='frozen-E',source_sha256=m.SOURCE_SHA,freeze_sha256=m.FREEZE_SHA),
            summary=dict(kind='gate',phase='E',case_table_sha256='frozen-E',expected=420,valid=420,
                         invalid=0,error=0,engineering_error=False,gate_pass=True))
    state = dict(mode='formal',learning=learning,ledger=cost,curriculum=dict(state=cur),
                 evaluation=refs,case_tables={'gate:E':dict(sha256='frozen-E')})
    report = dict(mode='formal',learning=deepcopy(learning),cost=deepcopy(cost),phase='E',stop_status='curriculum_gates_passed')
    return dict(status='finished',exit_code=0), report, state


class OwnershipTests(unittest.TestCase):
    def test_duplicate_running_or_complete_state_is_unchanged(self):
        for kind in ('waiting_formal','postprocess_complete'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                impl = Path(tmp); pipeline = impl/'post'; pipeline.mkdir()
                (pipeline/'status.json').write_text(json.dumps(dict(kind=kind,original=True)))
                (pipeline/'events.jsonl').write_text('original\n')
                before = {p.name:p.read_bytes() for p in pipeline.iterdir()}
                with patch.object(m,'IMPL',impl), patch.object(m,'PIPELINE',pipeline), patch.object(sys,'argv',['post']):
                    with self.assertRaises(FileExistsError): m.main()
                self.assertEqual(before,{p.name:p.read_bytes() for p in pipeline.iterdir()})

    def test_lock_contention_does_not_create_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            impl=Path(tmp); pipeline=impl/'post'
            with (impl/'POSTPROCESS_DRIVER.lock').open('a+') as held:
                fcntl.flock(held.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                with patch.object(m,'IMPL',impl),patch.object(m,'PIPELINE',pipeline),patch.object(sys,'argv',['post']):
                    with self.assertRaises(BlockingIOError):m.main()
            self.assertFalse(pipeline.exists())


class GateProofTests(unittest.TestCase):
    def test_complete_actual_E_summaries_allow_final(self):
        self.assertTrue(m.terminal_plan(*terminal_fixture())['run_final2100'])

    def test_missing_false_foreign_or_partial_actual_E_gate_refuses_final(self):
        for change in ('missing','false','kind','phase','update','case_sha','source','freeze','partial','error','engineering'):
            with self.subTest(change=change):
                p,r,s=terminal_fixture();ref=s['evaluation']['gate:24300']
                if change=='missing':del s['evaluation']['gate:24300']
                elif change=='false':ref['summary']['gate_pass']=False
                elif change in ('kind','phase','update'):ref['identity'][change]='foreign'
                elif change=='case_sha':ref['identity']['case_table_sha256']='foreign'
                elif change=='source':ref['identity']['source_sha256']='foreign'
                elif change=='freeze':ref['identity']['freeze_sha256']='foreign'
                elif change=='partial':ref['summary']['valid']=419
                elif change=='error':ref['summary']['error']=1
                elif change=='engineering':ref['summary']['engineering_error']=True
                with self.assertRaises(ValueError):m.terminal_plan(p,r,s)


class MediaProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import imageio_ffmpeg
        cls.ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        # Synthetic black clips test media integrity only; they are not robot evidence.
        cls.clips={}
        for name,frames,fps,size in [('full',4,25,'1280x720'),('short',2,25,'1280x720'),
                                    ('fps',4,30,'1280x720'),('size',4,25,'640x360')]:
            path=cls.root/(name+'.mp4')
            subprocess.run([cls.ffmpeg,'-v','error','-f','lavfi','-i','color=c=black:s='+size+':r='+str(fps),
                '-frames:v',str(frames),'-c:v','libx264','-threads','1','-pix_fmt','yuv420p',str(path)],check=True)
            cls.clips[name]=path.read_bytes()

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def fixture(self,root,clip='full',steps=6):
        output=root/'videos';output.mkdir();pipeline=root/'post';pipeline.mkdir()
        ids=['case-'+str(i) for i in range(8)]
        settings=dict(width=1280,height=720,fps=25,control_dt=.02,frame_stride=2,
            azimuth='90+body_yaw_degrees',elevation=-28.,distance=3.8,speed=1.)
        rows=[]
        for case in ids:
            path=output/(case+'.mp4');path.write_bytes(self.clips[clip])
            rows.append(dict(case_id=case,status='failure',reason='fixture',control_steps=steps,
                video_path=str(path),video_settings=deepcopy(settings)))
        report=dict(learning_updates=0,policy_checkpoint_sha256='policy',prescribed_case_ids=ids,
            records=rows,control_samples=steps*8,physics_substeps=steps*32,video_settings=settings,
            failures_retained=True,protocol_override=None)
        return pipeline,output,report,ids

    def test_real_complete_even_and_odd_terminal_clips_decode(self):
        for steps in (5,6):
            with self.subTest(steps=steps),tempfile.TemporaryDirectory() as tmp:
                pipeline,out,report,ids=self.fixture(Path(tmp),steps=steps)
                with patch.object(m,'PIPELINE',pipeline),patch.object(m,'cancelled',return_value=False):
                    media=m.decode_videos(report,out,ids,'policy')
                self.assertEqual(len(media),8)
                self.assertTrue(all(row.get('decoded_frames')==4 for row in media))

    def test_clean_short_wrong_fps_or_size_clips_are_rejected(self):
        for clip in ('short','fps','size'):
            with self.subTest(clip=clip),tempfile.TemporaryDirectory() as tmp:
                pipeline,out,report,ids=self.fixture(Path(tmp),clip=clip)
                with patch.object(m,'PIPELINE',pipeline),patch.object(m,'cancelled',return_value=False):
                    with self.assertRaises(ValueError):m.decode_videos(report,out,ids,'policy')

    def test_zero_exit_with_error_log_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline,out,report,ids=self.fixture(Path(tmp))
            def recoverable_error(*args,**kwargs):
                stream=kwargs['stdout'] if kwargs['stderr']==subprocess.STDOUT else kwargs['stderr']
                stream.write(b'mocked recoverable decode error\n')
                return subprocess.CompletedProcess(args[0],0)
            with patch.object(m,'PIPELINE',pipeline),patch.object(m,'cancelled',return_value=False),patch.object(m.subprocess,'run',side_effect=recoverable_error):
                with self.assertRaises(ValueError):m.decode_videos(report,out,ids,'policy')

    def test_foreign_case_video_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline,out,report,ids=self.fixture(Path(tmp))
            foreign=out/'another-case.mp4';foreign.write_bytes(self.clips['full'])
            report['records'][0]['video_path']=str(foreign)
            with patch.object(m,'PIPELINE',pipeline),patch.object(m,'cancelled',return_value=False):
                with self.assertRaises(ValueError):m.decode_videos(report,out,ids,'policy')

    def test_nonzero_decode_exit_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline,out,report,ids=self.fixture(Path(tmp))
            with patch.object(m,'PIPELINE',pipeline),patch.object(m,'cancelled',return_value=False),patch.object(m.subprocess,'run',return_value=subprocess.CompletedProcess([],1)):
                with self.assertRaises(ValueError):m.decode_videos(report,out,ids,'policy')


if __name__=='__main__':unittest.main(verbosity=2)
