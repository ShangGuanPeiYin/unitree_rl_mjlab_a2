"""Delivery regressions: asynchronous visibility, outages, and crash-safe intents."""
import tempfile
import unittest
from unittest.mock import patch
from test_v03_telemetry import Backend, Clock
from scripts.tools.wandb_connection import ReliableTelemetry, DeliveryPump, configure_network, ReliableWandbBackend

class Delayed(Backend):
    def __init__(self, clock, delay=3):
        super().__init__();self.clock=clock;self.delay=delay;self.pending=[]
    def send(self,event):
        self.sent.append(event);self.pending.append((self.clock()+self.delay,event['id']))
    def remote_ids(self):
        if self.down:raise ConnectionError('offline')
        return self.remote|{eid for t,eid in self.pending if t<=self.clock()}

class DeliveryTests(unittest.TestCase):
    def monitor(self,path,delay=3):
        c=Clock();b=Delayed(c,delay);pauses=[]
        m=ReliableTelemetry(path,b,{},clock=c,wall_clock=c,pause=pauses.append)
        return m,b,c,pauses
    def test_progress_without_empty_queue_does_not_timeout(self):
        with tempfile.TemporaryDirectory() as p:
            m,b,c,pauses=self.monitor(p)
            for i in range(50):
                m.emit('train',{'i':i},flush=False);m.flush();c.advance(4)
            self.assertFalse(pauses)
            self.assertEqual(len(m.state['acked']),49)
            c.advance(4);self.assertTrue(m.flush());self.assertEqual(len(b.sent),50)
    def test_real_outage_pauses_at_120_and_recovers_without_resend(self):
        with tempfile.TemporaryDirectory() as p:
            m,b,c,pauses=self.monitor(p);m.emit('train',{})
            b.down=True;c.advance(119);m.flush();self.assertFalse(pauses)
            c.advance(1);m.flush();self.assertEqual(pauses,['monitoring_blocked'])
            b.down=False;self.assertTrue(m.flush());self.assertEqual(len(b.sent),1)
            self.assertNotIn('monitoring_blocked',m.state)
    def test_old_session_intent_is_never_relogged(self):
        with tempfile.TemporaryDirectory() as p:
            m,b,c,_=self.monitor(p);m.emit('train',{})
            new=ReliableTelemetry(p,b,{},clock=c,wall_clock=c)
            self.assertFalse(new.flush());self.assertTrue(new.state['queue_recovery_required'])
            self.assertEqual(len(b.sent),1)
            c.advance(4);self.assertTrue(new.flush());self.assertEqual(len(b.sent),1)
    def test_pump_retries_without_creating_heartbeat_every_poll(self):
        with tempfile.TemporaryDirectory() as p:
            m,b,c,_=self.monitor(p);m.emit('train',{},flush=False)
            pump=DeliveryPump(m,lambda:{},clock=c)
            for i in range(60):pump.poll();c.advance(2)
            self.assertEqual(len([e for e in m.events if e['kind']=='authorized_fork_heartbeat']),3)
            self.assertLessEqual(len(m.events)-len(m.state['acked']),1)
            self.assertFalse(m.state.get('monitoring_blocked'))
    def test_age_limit_stops_unbounded_backlog_despite_some_acks(self):
        with tempfile.TemporaryDirectory() as p:
            m,b,c,pauses=self.monitor(p,delay=90)
            for i in range(64):m.emit('train',{'i':i},flush=False)
            for i in range(14):m.flush();c.advance(10)
            self.assertTrue(pauses)
    def test_thirty_second_visibility_lag_with_live_producer_does_not_build_backlog(self):
        with tempfile.TemporaryDirectory() as p:
            m,b,c,pauses=self.monitor(p,delay=30)
            for i in range(36):
                m.emit('train',{'i':i},flush=False);m.flush();c.advance(5)
            self.assertFalse(pauses)
            self.assertGreaterEqual(len(m.state['acked']),30)
            self.assertEqual(len(b.sent),36)
    def test_uncertain_send_exception_still_blocks_later_events(self):
        with tempfile.TemporaryDirectory() as p:
            m,b,c,_=self.monitor(p)
            def fail(event):raise ConnectionError('send interrupted before proof')
            b.send=fail
            m.emit('train',{});m.emit('train',{})
            self.assertEqual(len(m.state['transport']),1)
    def test_finish_waits_for_stop_visibility(self):
        with tempfile.TemporaryDirectory() as p:
            m,b,c,_=self.monitor(p)
            result=m.finish({'result/status':'test'},wait=c.advance)
            self.assertEqual(result['status'],'remote_verified')
            self.assertEqual(result['sdk_finish'],'completed')
    def test_ack_reader_never_queries_sdk_step_rpc(self):
        class ClosedRun:
            @property
            def step(self):raise AssertionError('SDK step RPC is forbidden in readback')
        class Reader:
            def rows(self,lo,hi):return [{'event_id':'one','_step':0}]
        backend=ReliableWandbBackend.__new__(ReliableWandbBackend)
        backend.run=ClosedRun();backend.reader=Reader();backend.seen=set()
        backend.remote_floor=0;backend.transport_next=1
        self.assertEqual(backend.remote_ids(),{'one'})
        self.assertEqual(backend.next_transport_step(),1)
    def test_explicit_proxy_is_process_scoped_and_target_cannot_drift(self):
        with patch.dict('os.environ',{},clear=True):
            import os
            configure_network()
            self.assertEqual(os.environ['HTTPS_PROXY'],'http://127.0.0.1:7897')
            self.assertEqual(os.environ['WANDB_BASE_URL'],'https://api.forge.coreweave.com')
            os.environ['WANDB_BASE_URL']='https://example.invalid'
            with self.assertRaises(ValueError):configure_network()

if __name__=='__main__':unittest.main()
