"""Global nonrefundable probe cost journal, never an implicit budget approval.

Historical migration records facts without permitting a future physical step.
Future steps require an explicit user-approved, cumulative probe-only cap.
The journal is shared across controller run IDs and all child probe stacks.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
from . import spec

RULINGS_PATH=spec.ROOT/'doc/sds/reviews/v10_execution_rulings.md'
DEFAULT_GLOBAL_LEDGER=spec.ROOT/'logs/a2_v10/global_probe_costs.jsonl'
AUTH_SCHEMA='a2-v10-global-probe-budget-authorization-v1'
# This is workload demand for one complete 1024/512, DAgger/PPO sequence,
# NOT an authorized cap and NOT a claim that the original SDS froze a cap.
FULL_SEQUENCE_DEMAND=26*48*(2*1024+2*512)

class ProbeBudgetError(RuntimeError):pass


class ProbeBudgetInsufficient(ProbeBudgetError):
    def __init__(self,required,remaining):
        self.required_samples=required;self.remaining_samples=remaining
        self.shortfall_samples=max(required-remaining,0)
        super().__init__(f'global probe remaining cap insufficient: required={required}, remaining={remaining}, shortfall={self.shortfall_samples}')


def streaming_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _positive(value):return isinstance(value,int) and not isinstance(value,bool) and value>0


def _empty_cost():return dict(charged={},executed={},uncertain={},updated={})


def _add(target,values):
    for key,value in values.items():target[key]=target.get(key,0)+value


def _cost_sum(cost):return {field:sum(cost[field].values()) for field in ('charged','executed','uncertain','updated')}


def read_legacy_ledger(path):
    """Validate every original reservation/finalization/update before migration."""
    path=Path(path).resolve();before=streaming_sha256(path)
    cost=_empty_cost();entries={};seq=0;raw_reservations=[]
    with path.open() as f:
        for line in f:
            try:row=json.loads(line)
            except ValueError as error:raise ProbeBudgetError('invalid legacy journal JSON') from error
            seq+=1
            if row.get('seq')!=seq:raise ProbeBudgetError('legacy ledger sequence damaged')
            uid=row.get('id');key=row.get('key','');n=row.get('samples');op=row.get('op')
            if not key.startswith('probe:') or not _positive(n):raise ProbeBudgetError('legacy ledger contains non-probe or invalid cost')
            if op in ('reserve','charge'):
                if uid in entries:raise ProbeBudgetError('legacy duplicate physical debit')
                entries[uid]=dict(key=key,samples=n,status='pending' if op=='reserve' else 'executed');raw_reservations.append(uid)
                _add(cost['charged'],{key:n})
                if op=='charge':_add(cost['executed'],{key:n})
            elif op in ('executed','uncertain'):
                e=entries.get(uid)
                if not e or e['status']!='pending' or e['key']!=key or e['samples']!=n:raise ProbeBudgetError('legacy invalid physical finalization')
                e['status']=op
                if op=='executed':_add(cost['executed'],{key:n})
            elif op=='update':
                if uid in entries or cost['updated'].get(key,0)+n>cost['executed'].get(key,0):raise ProbeBudgetError('legacy invalid update evidence')
                entries[uid]=dict(key=key,samples=n,status='updated');_add(cost['updated'],{key:n})
            else:raise ProbeBudgetError('unknown legacy journal operation')
    if streaming_sha256(path)!=before:raise ProbeBudgetError('legacy source changed while reading')
    cost['uncertain']={k:n-cost['executed'].get(k,0) for k,n in cost['charged'].items()}
    return dict(source_path=str(path),source_sha256=before,source_rows=seq,cost=cost,
                reservation_ids=raw_reservations,pending_conservatively_uncertain=sum(e['status']=='pending' for e in entries.values()))


class GlobalProbeBudget:
    def __init__(self,path=DEFAULT_GLOBAL_LEDGER,authorization_path=None,scope_id=None):
        self.path=Path(path).resolve();self.path.parent.mkdir(parents=True,exist_ok=True)
        self.authorization_path=Path(authorization_path).resolve() if authorization_path else None
        self.scope_id=scope_id;self.sequence_id=None
        self._cache=None
        self._state=dict(cost=_empty_cost(),scopes={},entries={},pending=set(),sources={},source_paths={},
                         legacy_ids=set(),authorizations={},retired_authorizations=set(),sequences={},active_sequence=None,capacity_leases={},active_capacity=None,protocol_attempts={},last_sha='0'*64,seq=0)

    def _apply(self,row):
        s=self._state;p=row['payload'];op=row['op']
        if op=='migration':
            digest=p['source_sha256'];scope='legacy:'+digest
            if digest in s['sources']:raise ProbeBudgetError('duplicate migration event')
            if s['legacy_ids']&set(p['reservation_ids']):raise ProbeBudgetError('overlapping legacy physical IDs')
            s['sources'][digest]=p;s['source_paths'][p['source_path']]=digest;s['legacy_ids'].update(p['reservation_ids'])
            s['scopes'][scope]=copy.deepcopy(p['cost'])
            for field in s['cost']:_add(s['cost'][field],p['cost'][field])
        elif op=='source_alias':s['source_paths'][p['source_path']]=p['source_sha256']
        elif op=='authorization':
            s['authorizations'][p['sha256']]=p
            parent=p['document'].get('supersedes')
            if parent:s['retired_authorizations'].add(parent['sha256'])
        elif op=='sequence_begin':
            if s['active_sequence'] is not None:raise ProbeBudgetError('two active probe sequences in journal')
            s['sequences'][p['id']]=dict(p,status='active');s['active_sequence']=p['id']
        elif op=='sequence_end':
            if s['active_sequence']!=p['id']:raise ProbeBudgetError('invalid sequence close')
            s['sequences'][p['id']].update(p,status='finished');s['active_sequence']=None
        elif op=='protocol_attempt_claim':
            if p['key'] in s['protocol_attempts']:raise ProbeBudgetError('duplicate protocol attempt')
            s['protocol_attempts'][p['key']]=dict(p,status='active')
        elif op=='protocol_attempt_end':
            entry=s['protocol_attempts'].get(p['key'])
            if not entry or entry['status']!='active':raise ProbeBudgetError('invalid protocol attempt close')
            entry.update(p,status='finished')
        elif op=='stack_capacity_acquire':
            if s['active_capacity'] is not None:raise ProbeBudgetError('two active stack capacity leases')
            s['capacity_leases'][p['scope']]=dict(p,status='active');s['active_capacity']=p['scope']
        elif op=='stack_capacity_release':
            if s['active_capacity']!=p['scope']:raise ProbeBudgetError('invalid stack capacity release')
            s['capacity_leases'][p['scope']].update(p,status='released');s['active_capacity']=None
        elif op=='reserve':
            uid=p['id'];scope=p['scope'];key=p['key'];n=p['samples']
            if uid in s['entries'] or s['pending']:raise ProbeBudgetError('duplicate reserve or multiple in-flight steps')
            s['entries'][uid]=dict(p,status='pending');s['pending'].add(uid)
            local=s['scopes'].setdefault(scope,_empty_cost())
            for cost in (s['cost'],local):
                _add(cost['charged'],{key:n});_add(cost['uncertain'],{key:n})
        elif op in ('executed','uncertain'):
            e=s['entries'].get(p['id'])
            if not e or e['status']!='pending':raise ProbeBudgetError('invalid step finalization')
            e['status']=op;s['pending'].remove(p['id'])
            if op=='executed':
                for cost in (s['cost'],s['scopes'][e['scope']]):
                    _add(cost['executed'],{e['key']:e['samples']});_add(cost['uncertain'],{e['key']:-e['samples']})
        elif op=='update':
            uid=p['id'];scope=p['scope'];key=p['key'];n=p['samples']
            local=s['scopes'].setdefault(scope,_empty_cost())
            if uid in s['entries'] or local['updated'].get(key,0)+n>local['executed'].get(key,0):raise ProbeBudgetError('invalid updated cost')
            s['entries'][uid]=dict(p,status='updated')
            for cost in (s['cost'],local):_add(cost['updated'],{key:n})
        else:raise ProbeBudgetError('unknown global probe journal operation')
        s['seq']=row['seq'];s['last_sha']=row['sha256']

    def _scan(self,f):
        stat=os.fstat(f.fileno())
        offset=0
        if self._cache:
            inode,offset=self._cache
            if stat.st_ino!=inode or stat.st_size<offset:raise ProbeBudgetError('global probe ledger replaced or truncated')
        f.seek(offset)
        while True:
            line=f.readline()
            if not line:break
            row=json.loads(line);body={k:v for k,v in row.items() if k!='sha256'}
            if row.get('seq')!=self._state['seq']+1 or row.get('previous_sha256')!=self._state['last_sha'] or row.get('sha256')!=_digest(body):
                raise ProbeBudgetError('global probe journal identity damaged')
            self._apply(row)
        self._cache=(stat.st_ino,f.tell())

    def _locked(self,fn):
        import fcntl
        with self.path.open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX);self._scan(f)
            return fn(f,self._state)

    def _append(self,f,op,payload):
        s=self._state;row=dict(seq=s['seq']+1,previous_sha256=s['last_sha'],op=op,payload=payload);row['sha256']=_digest(row)
        f.seek(0,2);f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
        # Parse only after durable append, preserving exactly the same replay path.
        self._scan(f)

    def global_report(self):
        def report(f,s):
            result=copy.deepcopy(s['cost']);result['global_totals']=_cost_sum(s['cost']);result['pending']=sorted(s['pending'])
            result.update(retired_authorizations=sorted(s['retired_authorizations']),protocol_attempts=copy.deepcopy(s['protocol_attempts']),active_capacity=s['active_capacity'],capacity_leases=copy.deepcopy(s['capacity_leases']),authorizations=copy.deepcopy(s['authorizations']),sources=copy.deepcopy(s['sources']),source_paths=dict(s['source_paths']),sequences=copy.deepcopy(s['sequences']),active_sequence=s['active_sequence'],journal_sha256=s['last_sha'])
            return result
        return self._locked(report)

    def report(self):
        def report(f,s):
            cost=s['cost'] if self.scope_id is None else s['scopes'].get(self.scope_id,_empty_cost())
            result=copy.deepcopy(cost)
            result['pending']=[uid for uid in s['pending'] if self.scope_id is None or s['entries'][uid]['scope']==self.scope_id]
            result['global_totals']=_cost_sum(s['cost']);result['global_ledger']=str(self.path)
            return result
        return self._locked(report)
    def totals(self):return self.report()['charged']

    def migrate(self,path):
        source=read_legacy_ledger(path)
        def perform(f,s):
            old=s['source_paths'].get(source['source_path'])
            if old and old!=source['source_sha256']:raise ProbeBudgetError('previously migrated source changed')
            if source['source_sha256'] in s['sources']:
                if old is None:self._append(f,'source_alias',dict(source_path=source['source_path'],source_sha256=source['source_sha256']))
                return False
            if set(source['reservation_ids'])&s['legacy_ids']:raise ProbeBudgetError('legacy overlap conflict; cannot double count or silently merge')
            self._append(f,'migration',source);return True
        return self._locked(perform)

    def _authorization(self):
        if self.authorization_path is None:raise ProbeBudgetError('explicit probe budget authorization required')
        try:
            before=streaming_sha256(self.authorization_path);doc=json.loads(self.authorization_path.read_text())
            if streaming_sha256(self.authorization_path)!=before:raise ProbeBudgetError('authorization changed while reading')
        except (OSError,ValueError) as error:raise ProbeBudgetError('probe budget authorization unavailable') from error
        if (doc.get('schema')!=AUTH_SCHEMA or doc.get('approved') is not True or not doc.get('authorization_id')
            or doc.get('scope')!='probe_only' or doc.get('sds_sha256')!=spec.SDS_SHA256
            or doc.get('rulings_sha256')!=streaming_sha256(RULINGS_PATH)
            or doc.get('global_ledger_path')!=str(self.path) or not _positive(doc.get('max_probe_samples'))
            or not isinstance(doc.get('prior_charged_samples'),int) or isinstance(doc.get('prior_charged_samples'),bool) or doc['prior_charged_samples']<0
            or doc.get('approval_source',{}).get('kind')!='user' or not doc.get('approval_source',{}).get('reference')):
            raise ProbeBudgetError('explicit user-approved probe-only budget authorization invalid')
        parent=doc.get('supersedes')
        if parent is not None:
            try:
                parent_path=Path(parent['path']);parent_doc=json.loads(parent_path.read_text())
                valid=(parent_path.is_absolute() and streaming_sha256(parent_path)==parent['sha256']
                    and parent_doc.get('authorization_id')==parent['authorization_id']
                    and all(doc.get(k)==parent_doc.get(k) for k in ('max_probe_samples','global_ledger_path','sds_sha256','scope'))
                    and doc['authorization_id']!=parent['authorization_id'])
            except (OSError,ValueError,KeyError):valid=False
            if not valid:raise ProbeBudgetError('invalid supersedes parent SHA/id/cap/ledger/SDS identity')
        if self._locked(lambda f,s:before in s['retired_authorizations']):raise ProbeBudgetError('authorization retired by registered successor')
        return dict(document=doc,sha256=before,path=str(self.authorization_path))

    @staticmethod
    def _assert_current_authorization(s,auth):
        if auth['sha256'] in s['retired_authorizations']:
            raise ProbeBudgetError('authorization retired by registered successor')

    def _register_authorization(self,f,s,auth):
        self._assert_current_authorization(s,auth)
        doc=auth['document']
        existing=[r for r in s['authorizations'].values() if r['document']['authorization_id']==doc['authorization_id']]
        if existing and existing[0]['sha256']!=auth['sha256']:raise ProbeBudgetError('authorization ID reused with changed approval')
        if auth['sha256'] not in s['authorizations']:
            parent=doc.get('supersedes')
            if parent is not None:
                registered=s['authorizations'].get(parent['sha256'])
                if parent['sha256'] in s['retired_authorizations']:raise ProbeBudgetError('supersedes parent already retired')
                if not registered or registered['document']['authorization_id']!=parent['authorization_id']:
                    raise ProbeBudgetError('supersedes parent must already be registered in global history')
                if any(doc.get(k)!=registered['document'].get(k) for k in ('max_probe_samples','global_ledger_path','sds_sha256','scope')):
                    raise ProbeBudgetError('supersedes cannot change cap/ledger/SDS/scope')
                if s['active_sequence'] is not None:raise ProbeBudgetError('cannot supersede authorization during active sequence')
            if _cost_sum(s['cost'])['charged']!=doc['prior_charged_samples']:raise ProbeBudgetError('authorization prior charged baseline mismatch; migrate history first')
            self._append(f,'authorization',auth)

    def begin_sequence(self,sequence_id,required_samples=FULL_SEQUENCE_DEMAND):
        auth=self._authorization()
        if not sequence_id or not _positive(required_samples):raise ProbeBudgetError('explicit probe sequence identity/demand required')
        def perform(f,s):
            self._register_authorization(f,s,auth)
            old=s['sequences'].get(sequence_id)
            if old:
                if old['status']!='active':raise ProbeBudgetError('probe sequence already closed')
                if old['authorization_sha256']!=auth['sha256'] or old['required_samples']!=required_samples:raise ProbeBudgetError('probe sequence identity changed')
                return old
            if s['active_sequence'] is not None:raise ProbeBudgetError('another global probe sequence is active')
            if s['pending']:raise ProbeBudgetError('global in-flight step unresolved')
            charged=_cost_sum(s['cost'])['charged'];cap=auth['document']['max_probe_samples']
            if charged+required_samples>cap:raise ProbeBudgetInsufficient(required_samples,cap-charged)
            event=dict(id=sequence_id,authorization_sha256=auth['sha256'],required_samples=required_samples,charged_at_start=charged,cap=cap)
            self._append(f,'sequence_begin',event);return copy.deepcopy(s['sequences'][sequence_id])
        return self._locked(perform)

    def validate_authorization(self):
        auth=self._authorization()
        def perform(f,s):
            self._assert_current_authorization(s,auth)
            return auth
        return self._locked(perform)

    def assert_authorized(self,sequence_id,required_samples=0):
        if not isinstance(required_samples,int) or isinstance(required_samples,bool) or required_samples<0:raise ProbeBudgetError('nonnegative probe demand required')
        auth=self._authorization()
        def perform(f,s):
            self._assert_current_authorization(s,auth)
            seq=s['sequences'].get(sequence_id)
            if s['pending']:raise ProbeBudgetError('global in-flight physical step unresolved')
            if (not seq or seq['status']!='active' or s['active_sequence']!=sequence_id or seq['authorization_sha256']!=auth['sha256']):raise ProbeBudgetError('active authorized probe sequence required')
            if self.scope_id and not self.scope_id.startswith(sequence_id+':'):raise ProbeBudgetError('probe scope does not belong to sequence')
            remaining=auth['document']['max_probe_samples']-_cost_sum(s['cost'])['charged']
            lease=s['capacity_leases'].get(self.scope_id,{})
            demand=required_samples
            if lease.get('status')=='active' and required_samples==lease['required_samples']:
                used=_cost_sum(s['scopes'].get(self.scope_id,_empty_cost()))['charged']-lease['charged_at_start']
                demand=max(required_samples-used,0)
            if demand>remaining:raise ProbeBudgetInsufficient(demand,remaining)
            self.sequence_id=sequence_id;return auth
        return self._locked(perform)

    def acquire_stack_capacity(self,sequence_id,required_samples,scope_id=None):
        """Exclusive capacity, not a charge; parent and child share the same scope."""
        auth=self._authorization();scope=scope_id or self.scope_id
        if not scope or not scope.startswith(sequence_id+':') or not _positive(required_samples):raise ProbeBudgetError('explicit stack scope/capacity required')
        def perform(f,s):
            self._assert_current_authorization(s,auth)
            seq=s['sequences'].get(sequence_id)
            if not seq or seq['status']!='active' or seq['authorization_sha256']!=auth['sha256']:raise ProbeBudgetError('active authorized sequence required')
            old=s['capacity_leases'].get(scope)
            if old:
                if old['status']!='active' or old['required_samples']!=required_samples:raise ProbeBudgetError('stack capacity lease already released or identity changed')
                return copy.deepcopy(old)
            if s['active_capacity'] is not None:raise ProbeBudgetError('another stack scope holds exclusive capacity')
            if s['pending']:raise ProbeBudgetError('in-flight physical step unresolved')
            remaining=auth['document']['max_probe_samples']-_cost_sum(s['cost'])['charged']
            if required_samples>remaining:raise ProbeBudgetInsufficient(required_samples,remaining)
            start=_cost_sum(s['scopes'].get(scope,_empty_cost()))['charged']
            self._append(f,'stack_capacity_acquire',dict(scope=scope,sequence_id=sequence_id,required_samples=required_samples,charged_at_start=start))
            return copy.deepcopy(s['capacity_leases'][scope])
        return self._locked(perform)

    def release_stack_capacity(self,scope_id=None,reason=''):
        scope=scope_id or self.scope_id
        def perform(f,s):
            lease=s['capacity_leases'].get(scope)
            if not lease:raise ProbeBudgetError('unknown stack capacity lease')
            if lease['status']=='released':return False
            if s['pending']:raise ProbeBudgetError('cannot release capacity with unresolved in-flight step')
            used=_cost_sum(s['scopes'].get(scope,_empty_cost()))['charged']-lease['charged_at_start']
            # Executed AND uncertain are already charged, so neither is released.
            unused=lease['required_samples']-used
            if unused<0:raise ProbeBudgetError('stack capacity overconsumed')
            self._append(f,'stack_capacity_release',dict(scope=scope,released_unused_samples=unused,consumed_charged_samples=used,reason=str(reason)))
            return unused
        return self._locked(perform)

    def claim_protocol_attempt(self,protocol,contract_sha256,sequence_id,output_dir,admission_evidence=None,exclusive_protocol=False):
        if len(contract_sha256)!=64:raise ProbeBudgetError('protocol contract SHA required')
        auth=self._authorization();key=protocol+':'+contract_sha256
        def perform(f,s):
            self._assert_current_authorization(s,auth)
            seq=s['sequences'].get(sequence_id)
            if not seq or seq['status']!='active' or seq['authorization_sha256']!=auth['sha256']:raise ProbeBudgetError('authorized active protocol sequence required')
            previous=[entry for entry in s['protocol_attempts'].values() if entry['protocol']==protocol]
            if key in s['protocol_attempts'] or (previous and (exclusive_protocol or any(entry.get('exclusive_protocol',False) for entry in previous))):
                raise ProbeBudgetError('protocol validation attempt already claimed; no remeasure')
            self._append(f,'protocol_attempt_claim',dict(key=key,protocol=protocol,contract_sha256=contract_sha256,sequence_id=sequence_id,output_dir=str(Path(output_dir).resolve()),admission_evidence=admission_evidence,exclusive_protocol=bool(exclusive_protocol)))
            return copy.deepcopy(s['protocol_attempts'][key])
        return self._locked(perform)

    def require_protocol_attempt(self,protocol,contract_sha256,sequence_id,output_dir=None):
        key=protocol+':'+contract_sha256
        def perform(f,s):
            entry=s['protocol_attempts'].get(key)
            if not entry or entry['status']!='active' or entry['sequence_id']!=sequence_id:raise ProbeBudgetError('active original protocol attempt required')
            if output_dir is not None and str(Path(output_dir).resolve())!=entry['output_dir']:raise ProbeBudgetError('protocol attempt output changed')
            return copy.deepcopy(entry)
        return self._locked(perform)

    def finish_protocol_attempt(self,protocol,contract_sha256,passed=False):
        key=protocol+':'+contract_sha256
        def perform(f,s):
            entry=s['protocol_attempts'].get(key)
            if not entry:raise ProbeBudgetError('protocol attempt missing')
            if entry['status']=='finished':return False
            self._append(f,'protocol_attempt_end',dict(key=key,passed=bool(passed)));return True
        return self._locked(perform)

    def finish_sequence(self,sequence_id,passed=False,reason=''):
        def perform(f,s):
            seq=s['sequences'].get(sequence_id)
            if not seq:raise ProbeBudgetError('unknown probe sequence')
            if seq['status']=='finished':return False
            if s['pending']:raise ProbeBudgetError('cannot close sequence with in-flight physical step')
            if s['active_capacity'] is not None:raise ProbeBudgetError('cannot close sequence with active stack capacity lease')
            self._append(f,'sequence_end',dict(id=sequence_id,passed=bool(passed),reason=str(reason),totals=_cost_sum(s['cost'])))
            return True
        return self._locked(perform)

    def _uid(self,uid):
        if not self.scope_id:raise ProbeBudgetError('physical cost requires explicit probe scope')
        return self.scope_id+'|'+str(uid)

    def reserve(self,uid,course,samples,*,limit,kind,details=None):
        # Re-read the grant before every future physical step, never after it as
        # a precondition for recording an already-executed fact.
        auth=self._authorization();scoped=self._uid(uid)
        if kind!='probe' or not _positive(samples) or not _positive(limit):raise ProbeBudgetError('probe-only positive sample reservation required')
        key='probe:'+str(course)
        def perform(f,s):
            self._assert_current_authorization(s,auth)
            if scoped in s['entries']:
                e=s['entries'][scoped]
                if e['key']!=key or e['samples']!=samples:raise ProbeBudgetError('reservation identity mismatch')
                return False
            seq=s['sequences'].get(self.sequence_id)
            if not seq or seq['status']!='active' or s['active_sequence']!=self.sequence_id or seq['authorization_sha256']!=auth['sha256']:
                raise ProbeBudgetError('active authorization must be asserted before physical step')
            if s['pending']:raise ProbeBudgetError('one global step already in-flight')
            local=s['scopes'].get(self.scope_id,_empty_cost())
            lease=s['capacity_leases'].get(self.scope_id,{})
            if s['active_capacity']!=self.scope_id or lease.get('status')!='active':raise ProbeBudgetError('active stack capacity lease required')
            used=_cost_sum(local)['charged']-lease['charged_at_start']
            if used+samples>lease['required_samples']:raise ProbeBudgetError('stack capacity exhausted')
            if _cost_sum(local)['charged']+samples>limit:raise ProbeBudgetError('local probe stack scope cap exhausted')
            if _cost_sum(s['cost'])['charged']+samples>auth['document']['max_probe_samples']:raise ProbeBudgetError('global probe cap exhausted')
            self._append(f,'reserve',dict(id=scoped,scope=self.scope_id,key=key,samples=samples,details=details or {},authorization_sha256=auth['sha256']))
            return True
        return self._locked(perform)

    def mark_executed(self,uid):
        scoped=self._uid(uid)
        def perform(f,s):
            e=s['entries'].get(scoped)
            if not e:raise ProbeBudgetError('unknown physical reservation')
            if e['status']=='executed':return False
            if e['status']!='pending':raise ProbeBudgetError('uncertain reservation cannot be replayed')
            self._append(f,'executed',dict(id=scoped));return True
        return self._locked(perform)

    def reconcile_uncertain(self):
        def perform(f,s):
            ids=[uid for uid in s['pending'] if self.scope_id is None or s['entries'][uid]['scope']==self.scope_id]
            for uid in ids:self._append(f,'uncertain',dict(id=uid))
        return self._locked(perform)

    def record_update(self,uid,course,samples,*,kind='probe'):
        if kind!='probe' or not _positive(samples):raise ProbeBudgetError('probe-only completed update required')
        scoped=self._uid(uid);key='probe:'+str(course)
        def perform(f,s):
            if scoped in s['entries']:
                e=s['entries'][scoped]
                if e.get('status')!='updated' or e.get('samples')!=samples or e.get('key')!=key:raise ProbeBudgetError('update identity mismatch')
                return False
            if s['pending']:raise ProbeBudgetError('cannot update with in-flight step')
            local=s['scopes'].get(self.scope_id,_empty_cost())
            if local['updated'].get(key,0)+samples>local['executed'].get(key,0):raise ProbeBudgetError('completed update exceeds actually executed samples')
            self._append(f,'update',dict(id=scoped,scope=self.scope_id,key=key,samples=samples));return True
        return self._locked(perform)

    def assert_checkpoint_not_ahead(self,totals):
        current=self.totals()
        if any(current.get(k,0)<n for k,n in totals.items()):raise ProbeBudgetError('global ledger cannot refund checkpoint costs')


def migrate_legacy_ledgers(path,sources,expected_totals=None):
    ledger=GlobalProbeBudget(path)
    for source in sources:ledger.migrate(source)
    report=ledger.global_report()
    selected={streaming_sha256(source) for source in sources}
    migrated={field:sum(sum(report['sources'][digest]['cost'][field].values()) for digest in selected) for field in ('charged','executed','uncertain','updated')}
    report['migration_totals']=migrated
    if expected_totals is not None and migrated!=expected_totals:raise ProbeBudgetError('historical probe migration totals do not match independently audited facts')
    return report
