"""Exercise only the actual observer snapshot expression and restricted argv helper."""
import ast
from copy import deepcopy
from pathlib import Path
import unittest

PATH=Path(__file__).resolve().parents[1]/'run_engineering.py'
TREE=ast.parse(PATH.read_text())

def function(name):
    return next(node for node in TREE.body if isinstance(node,ast.FunctionDef) and node.name==name)

class ObserverSnapshotTests(unittest.TestCase):
    def test_actual_return_expression_freezes_nested_reset_events(self):
        returned=next(node.value for node in ast.walk(function('environment_evidence'))
                      if isinstance(node,ast.Return))
        expression=next(k.value for k in returned.keywords if k.arg=='reset_events')
        events=[dict(ids=[1,2],count=2,context=dict(substage='D25',generation=68)),
                dict(ids=[0,1,2],count=3,context=dict(substage='D25',generation=68))]
        snapshot=eval(compile(ast.Expression(expression),str(PATH),'eval'),dict(events=events,deepcopy=deepcopy))
        events[0]['ids'].append(99)
        events[0]['context']['substage']='mutated'
        events.append(dict(ids=[0,1,2],count=3,context=dict(substage='D20',generation=69)))
        self.assertEqual(len(snapshot),2)
        self.assertEqual(snapshot[0]['ids'],[1,2])
        self.assertEqual(snapshot[0]['context']['substage'],'D25')

    def test_recheck_suffix_is_limited_to_one_explicit_source_reload_stage(self):
        ns={}
        exec(compile(ast.Module(body=[function('observer_arguments')],type_ignores=[]),str(PATH),'exec'),ns)
        parse=ns['observer_arguments'];base=['--stage','source-reload','--device','cuda:0','--seed','42','--execute']
        self.assertEqual(parse(base),(base,None))
        self.assertEqual(parse(base+['--observer-snapshot-recheck']),(base,'observer_snapshot_v1'))
        for bad in [
            ['--stage','baseline','--observer-snapshot-recheck'],
            ['--stage','probe','--observer-snapshot-recheck'],
            base+['--stage','baseline','--observer-snapshot-recheck'],
            base+['--observer-snapshot-recheck','--observer-snapshot-recheck']]:
            with self.assertRaises(ValueError):parse(bad)

if __name__=='__main__':unittest.main(verbosity=2)
