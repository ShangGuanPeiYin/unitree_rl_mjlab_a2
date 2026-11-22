"""S2 resource admission: 1024 first, documented fallback to 512 (§2.3)."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sys

from . import s2_spec

PROBE_EVAL_TIMEOUT_STEPS = 9472  # batch hard-timeout inclusive; 9471 is last index
REQUIREMENTS = dict(gpu_peak_mib=7168, host_free_mib=4096,
                    warmups=5, measurements=50, restore_checks=1,
                    rollout_steps=48, adam_updates_per_rollout=20,
                    independent_copy=True, allow_oom=False,
                    allow_nonfinite=False, allow_contact_overflow=False,
                    evaluation_envs=100,
                    evaluation_steps=PROBE_EVAL_TIMEOUT_STEPS,
                    video_envs=1, video_steps=200)


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha256_state(model):
    """Stable SHA of the entry actor state, independent of device placement."""
    import numpy as np
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(
            value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def _process_evidence():
    """Record the independent probe process and its launcher."""
    launcher = None
    try:
        raw = Path(f'/proc/{os.getppid()}/cmdline').read_bytes()
        launcher = [part.decode(errors='replace') for part in raw.split(b'\0') if part]
    except Exception:
        pass
    return dict(pid=os.getpid(), ppid=os.getppid(),
                argv=list(sys.argv), launcher_argv=launcher,
                python=sys.executable, cwd=str(Path.cwd()),
                independent_process=True)


@dataclass(frozen=True)
class ProbeResult:
    num_envs: int
    peak_mib: float
    host_free_mib: float
    oom: bool
    nonfinite: bool
    contact_overflow: bool
    warmups: int
    measurements: int
    restore_checks: int
    independent_copy: bool
    evaluation_envs: int
    evidence_sha256: str | None = None

    def admitted(self):
        return all((
            self.peak_mib <= REQUIREMENTS['gpu_peak_mib'],
            self.host_free_mib >= REQUIREMENTS['host_free_mib'],
            not self.oom, not self.nonfinite, not self.contact_overflow,
            self.warmups >= REQUIREMENTS['warmups'],
            self.measurements >= REQUIREMENTS['measurements'],
            self.restore_checks >= REQUIREMENTS['restore_checks'],
            self.independent_copy,
            self.evaluation_envs == REQUIREMENTS['evaluation_envs'],
        ))

    def failures(self):
        reasons = []
        if self.peak_mib > REQUIREMENTS['gpu_peak_mib']:
            reasons.append('gpu_peak_mib')
        if self.host_free_mib < REQUIREMENTS['host_free_mib']:
            reasons.append('host_free_mib')
        if self.oom:
            reasons.append('oom')
        if self.nonfinite:
            reasons.append('nonfinite')
        if self.contact_overflow:
            reasons.append('contact_overflow')
        if self.warmups < REQUIREMENTS['warmups']:
            reasons.append('warmups')
        if self.measurements < REQUIREMENTS['measurements']:
            reasons.append('measurements')
        if self.restore_checks < REQUIREMENTS['restore_checks']:
            reasons.append('restore')
        if not self.independent_copy:
            reasons.append('independent_copy')
        if self.evaluation_envs != REQUIREMENTS['evaluation_envs']:
            reasons.append('evaluation_envs')
        return reasons


def probe_report_failures(report, *, num_envs):
    """Re-derive every admission requirement from the probe report itself.

    This deliberately does not trust the report's ``admitted`` boolean: the
    independent copy and entry-checkpoint evidence are checked here too.
    """
    if not isinstance(report, dict):
        return ['report_schema']

    def as_int(key, default=-1):
        try:
            return int(report.get(key))
        except (TypeError, ValueError):
            return default

    def as_float(key, default=float('inf')):
        try:
            return float(report.get(key))
        except (TypeError, ValueError):
            return default

    failures = []
    if as_int('num_envs') != int(num_envs):
        failures.append('num_envs')
    if report.get('admitted') is not True:
        failures.append('admitted')
    if report.get('failure_reasons'):
        failures.append('failure_reasons')
    if as_float('peak_mib') > REQUIREMENTS['gpu_peak_mib']:
        failures.append('gpu_peak_mib')
    if as_float('host_free_mib', -1.0) < REQUIREMENTS['host_free_mib']:
        failures.append('host_free_mib')
    if report.get('oom') is not False:
        failures.append('oom')
    if report.get('nonfinite') is not False:
        failures.append('nonfinite')
    if report.get('contact_overflow') is not False:
        failures.append('contact_overflow')
    if as_int('warmups') < REQUIREMENTS['warmups']:
        failures.append('warmups')
    if as_int('measurements') < REQUIREMENTS['measurements']:
        failures.append('measurements')
    if len(report.get('timings_s') or []) < REQUIREMENTS['measurements']:
        failures.append('timings_s')
    if as_int('restore_checks') < REQUIREMENTS['restore_checks']:
        failures.append('restore_checks')
    if report.get('restore_ok') is not True:
        failures.append('restore_ok')
    if as_int('evaluation_envs') != REQUIREMENTS['evaluation_envs']:
        failures.append('evaluation_envs')
    if as_int('evaluation_steps') != REQUIREMENTS['evaluation_steps']:
        failures.append('evaluation_steps')
    if as_int('video_envs') != REQUIREMENTS['video_envs']:
        failures.append('video_envs')
    if as_int('video_steps') != REQUIREMENTS['video_steps']:
        failures.append('video_steps')
    if report.get('independent_copy') is not True:
        failures.append('independent_copy')
    # Material proof that the probe used a copy and did not mutate the entry.
    entry = report.get('entry_weight') or {}
    entry_state = report.get('entry_state') or {}
    source_before = report.get('entry_source_sha256_before',
                               entry.get('source_sha256_before'))
    source_after = report.get('entry_source_sha256_after',
                              entry.get('source_sha256_after'))
    copy_sha = report.get('entry_probe_copy_sha256',
                          entry.get('probe_copy_sha256'))
    state_before = report.get('entry_actor_state_sha256_before',
                              entry_state.get('sha256_before'))
    state_after = report.get('entry_actor_state_sha256_after_reload',
                             entry_state.get('sha256_after_reload'))
    unchanged = report.get('entry_source_unchanged', entry.get('source_unchanged'))
    if unchanged is not True:
        failures.append('entry_source_unchanged')
    if not source_before or source_before != source_after:
        failures.append('entry_source_sha256')
    if not copy_sha or copy_sha != source_before:
        failures.append('entry_probe_copy_sha256')
    if not state_before or state_before != state_after:
        failures.append('entry_actor_state_sha256')
    process = report.get('process') or {}
    if not process.get('pid') or process.get('independent_process') is not True:
        failures.append('process')
    return failures


def probe_plan(num_envs, *, evaluation_timeout_steps=PROBE_EVAL_TIMEOUT_STEPS):
    if num_envs not in s2_spec.S2_ENV_CANDIDATES:
        raise ValueError('S2 probe is 1024 then 512 only')
    samples = num_envs * REQUIREMENTS['warmups'] * REQUIREMENTS['rollout_steps']
    samples += num_envs * REQUIREMENTS['measurements'] * REQUIREMENTS['rollout_steps']
    samples += num_envs * REQUIREMENTS['rollout_steps']  # restore rollout
    samples += REQUIREMENTS['evaluation_envs'] * int(evaluation_timeout_steps)
    samples += 1 * int(evaluation_timeout_steps)  # one-env video process
    return dict(num_envs=num_envs, requirements=dict(REQUIREMENTS),
                evaluation_timeout_steps=int(evaluation_timeout_steps),
                evaluation_samples=REQUIREMENTS['evaluation_envs'] * int(evaluation_timeout_steps),
                video_samples=int(evaluation_timeout_steps),
                control_samples=samples, physics_substeps=4 * samples,
                warmups=REQUIREMENTS['warmups'],
                measurements=REQUIREMENTS['measurements'],
                restore_checks=REQUIREMENTS['restore_checks'])


def choose_resource(probes, *, require_1024_failure_evidence=True):
    """Return 1024, or 512 only after an explicit 1024 failure record."""
    probes = list(probes)
    if require_1024_failure_evidence:
        if not probes or probes[0].num_envs != 1024:
            raise ValueError('512 fallback requires a 1024 probe first')
        if probes[0].admitted():
            raise ValueError('1024 already admitted; 512 fallback is not allowed')
    evidence = []
    for result in probes:
        evidence.append(dict(num_envs=result.num_envs, admitted=result.admitted(),
                             failures=result.failures(),
                             evidence_sha256=result.evidence_sha256))
        if result.admitted():
            return dict(selected=result.num_envs, fallback_evidence=evidence)
    return dict(selected=None, fallback_evidence=evidence,
                reason='both 1024 and 512 failed; stop and report')


def run_probe(*, num_envs, device='cuda:0', output_dir=None,
              independent_copy=True):
    """Measured §2.3 probe against the real S2TrainEnv (GPU call only).

    Five warmups, fifty measured full 48-step/20-Adam rollouts and one
    save→load restore.  The caller should run this only in an approved GPU
    window; no ledger write happens here.
    """
    import json
    import shutil
    import tempfile
    import time
    from pathlib import Path
    import torch
    try:
        import psutil
        host_free = psutil.virtual_memory().available / 2 ** 20
    except Exception:
        host_free = None
    from . import s2_physics, s2_train, s2_train_real
    entry_path = Path(s2_spec.S1_ENTRY_WEIGHT)
    entry_source_sha256_before = _sha256_file(entry_path)
    entry_source_sha256_after = None
    entry_probe_copy_sha256 = None
    entry_actor_state_sha256_before = None
    entry_actor_state_sha256_after_reload = None
    entry_source_unchanged = False
    independent_copy_verified = False
    process = _process_evidence()
    probe_tempdir = tempfile.TemporaryDirectory(prefix='s2-probe-entry-')
    try:
        # The probe gets its own on-disk copy of the approved entry.  The actor
        # loaded by the probe can then never share a mutable training file.
        probe_entry = Path(probe_tempdir.name) / 'entry.pt'
        shutil.copyfile(entry_path, probe_entry)
        entry_probe_copy_sha256 = _sha256_file(probe_entry)
        actor = s2_physics.load_s1_actor(
            path=probe_entry, expected_sha256=entry_source_sha256_before,
            device=device)
        entry_actor_state_sha256_before = _sha256_state(actor['model'])
        env = s2_train_real.S2TrainEnv(num_envs=num_envs, device=device)
    except Exception:
        probe_tempdir.cleanup()
        raise
    trainer = s2_train.S2Trainer(model=actor['model'], env=env, device=device,
                                 attempt_id=f'probe-{num_envs}')
    timings, eval_timing, video_timing = [], None, None
    oom = nonfinite = contact_overflow = False
    peak_mib = 0.0
    restore_ok = False
    evaluation_envs = 0
    evaluation_steps = 0
    video_envs = 0
    video_steps = 0
    failure_reasons = []
    error_detail = None
    admitted = False
    try:
        torch.cuda.reset_peak_memory_stats()
        for _ in range(REQUIREMENTS['warmups']):
            trainer.train_rollout()
        for _ in range(REQUIREMENTS['measurements']):
            started = time.perf_counter()
            trainer.train_rollout()
            timings.append(time.perf_counter() - started)
            peak_mib = max(peak_mib, torch.cuda.max_memory_allocated() / 2 ** 20)
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'probe.pt'
            trainer.save(path)
            env.close(); env = None
            # Evaluation: 100 envs, fixed worst-timeout window.
            eval_env = s2_train_real.S2TrainEnv(num_envs=100, device=device)
            eval_started = time.perf_counter()
            obs = eval_env.reset()
            eval_steps = int(PROBE_EVAL_TIMEOUT_STEPS)
            with torch.no_grad():
                for _ in range(eval_steps):
                    action = actor['model'].action_distribution(obs).sample()
                    obs, _reward, _done, _info = eval_env.step(action)
            eval_timing = time.perf_counter() - eval_started
            evaluation_envs, evaluation_steps = 100, eval_steps
            peak_mib = max(peak_mib, torch.cuda.max_memory_allocated() / 2 ** 20)
            eval_env.close()
            # Video: one env, short deterministic segment.
            video_env = s2_train_real.S2TrainEnv(num_envs=1, device=device)
            video_started = time.perf_counter()
            obs = video_env.reset()
            video_steps = min(200, eval_steps)
            with torch.no_grad():
                for _ in range(video_steps):
                    action = actor['model'].action_distribution(obs).sample()
                    obs, _reward, _done, _info = video_env.step(action)
            video_timing = time.perf_counter() - video_started
            video_envs = 1
            peak_mib = max(peak_mib, torch.cuda.max_memory_allocated() / 2 ** 20)
            video_env.close()
            # Save/load/restore and one full continuation rollout.
            restored = s2_train.S2Trainer(
                model=s2_physics.load_s1_actor(
                    path=probe_entry,
                    expected_sha256=entry_source_sha256_before,
                    device=device)['model'],
                env=s2_train_real.S2TrainEnv(num_envs=num_envs, device=device),
                device=device, attempt_id=f'probe-{num_envs}-restore')
            restored.resume(path, device=device)
            restored.train_rollout()
            restore_ok = True
            peak_mib = max(peak_mib, torch.cuda.max_memory_allocated() / 2 ** 20)
            restored.env.close()
        if peak_mib > REQUIREMENTS['gpu_peak_mib']:
            failure_reasons.append('gpu_peak_mib')
        if host_free is not None and host_free < REQUIREMENTS['host_free_mib']:
            failure_reasons.append('host_free_mib')
        if evaluation_envs != REQUIREMENTS['evaluation_envs']:
            failure_reasons.append('evaluation_envs')
        if video_envs != 1:
            failure_reasons.append('video_envs')
        if not restore_ok:
            failure_reasons.append('restore')
    except RuntimeError as error:
        import traceback
        error_detail = traceback.format_exc()
        oom = 'out of memory' in str(error).lower()
        nonfinite = 'nonfinite' in str(error).lower()
        contact_overflow = 'contact' in str(error).lower()
        if oom:
            failure_reasons.append('oom')
        if nonfinite:
            failure_reasons.append('nonfinite')
        if contact_overflow:
            failure_reasons.append('contact_overflow')
        if not failure_reasons:
            failure_reasons.append('runtime_error')
        peak_mib = (torch.cuda.max_memory_allocated() / 2 ** 20
                    if torch.cuda.is_available() else 0.0)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    # Independent-copy proof is evaluated after the probe, never inferred from
    # the caller's flag alone.  The source must be unchanged and the probe must
    # have used a byte-identical on-disk copy.
    try:
        entry_source_sha256_after = _sha256_file(entry_path)
        entry_source_unchanged = (
            entry_source_sha256_before is not None
            and entry_source_sha256_before == entry_source_sha256_after)
        entry_copy_identical = (
            entry_probe_copy_sha256 is not None
            and entry_probe_copy_sha256 == entry_source_sha256_before)
        if entry_source_unchanged:
            reloaded = s2_physics.load_s1_actor(
                path=entry_path, expected_sha256=entry_source_sha256_before,
                device='cpu')
            entry_actor_state_sha256_after_reload = _sha256_state(reloaded['model'])
        entry_state_match = (
            entry_actor_state_sha256_before is not None
            and entry_actor_state_sha256_before == entry_actor_state_sha256_after_reload)
        independent_copy_verified = bool(
            independent_copy and entry_source_unchanged and entry_copy_identical
            and entry_state_match and process.get('independent_process') is True)
    except Exception as error:
        error_detail = (error_detail or '') + '\nindependent-copy: ' + repr(error)
        failure_reasons.append('independent_copy_evidence')
        independent_copy_verified = False
    finally:
        probe_tempdir.cleanup()

    independent_copy_requested = bool(independent_copy)
    if not independent_copy_requested:
        failure_reasons.append('independent_copy')
    if not independent_copy_verified and 'independent_copy' not in failure_reasons:
        failure_reasons.append('independent_copy')
    if not entry_source_unchanged:
        failure_reasons.append('entry_source_unchanged')
    if entry_probe_copy_sha256 != entry_source_sha256_before:
        failure_reasons.append('entry_probe_copy_sha256')
    if entry_actor_state_sha256_before != entry_actor_state_sha256_after_reload:
        failure_reasons.append('entry_actor_state_sha256')
    admitted = not failure_reasons
    report = dict(
        num_envs=num_envs, admitted=bool(admitted), peak_mib=peak_mib,
        host_free_mib=host_free, oom=oom, nonfinite=nonfinite,
        contact_overflow=contact_overflow,
        warmups=REQUIREMENTS['warmups'], measurements=len(timings),
        restore_checks=1 if restore_ok else 0, restore_ok=bool(restore_ok),
        independent_copy=bool(independent_copy_verified),
        independent_copy_requested=independent_copy_requested,
        entry_weight=dict(
            path=str(entry_path),
            source_sha256_before=entry_source_sha256_before,
            source_sha256_after=entry_source_sha256_after,
            probe_copy_sha256=entry_probe_copy_sha256,
            source_unchanged=bool(entry_source_unchanged)),
        entry_state=dict(
            sha256_before=entry_actor_state_sha256_before,
            sha256_after_reload=entry_actor_state_sha256_after_reload,
            match=bool(entry_actor_state_sha256_before
                       == entry_actor_state_sha256_after_reload)),
        process=process,
        evaluation_envs=evaluation_envs, evaluation_steps=evaluation_steps,
        video_envs=video_envs, video_steps=video_steps,
        elapsed_s=float(sum(timings)), timings_s=timings,
        evaluation_s=eval_timing, video_s=video_timing,
        failure_reasons=failure_reasons, requirements=dict(REQUIREMENTS),
        error=str(error_detail).splitlines()[-1] if error_detail else None,
        error_detail=error_detail,
        note='real S2TrainEnv probe including evaluation 100 and video 1; '
             'independent on-disk entry copy and source/state SHA evidence')
    if output_dir is not None:
        path = Path(output_dir) / 'probe_report.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


ADMISSION_SCHEMA = 'a2_v09_s2_1_admission_v1'


def write_admission(path, *, probe_report, probe_path, num_envs,
                    independent_review_sha256, independent_review_path=None):
    """Write the explicit training-admission document; caller supplies review."""
    import hashlib
    import json
    from datetime import datetime, timezone
    from pathlib import Path
    failures = probe_report_failures(probe_report, num_envs=num_envs)
    if failures:
        raise ValueError('cannot write admission: probe requirements failed: '
                         + ', '.join(failures))
    digest = str(independent_review_sha256)
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('independent review SHA256 must be 64 hex chars')
    probe_path = Path(probe_path)
    doc = dict(
        schema=ADMISSION_SCHEMA, admitted=True, num_envs=int(num_envs),
        probe_report=dict(path=str(probe_path),
                          sha256=hashlib.sha256(probe_path.read_bytes()).hexdigest(),
                          admitted=True),
        independent_review=dict(sha256=digest,
                                path=(str(independent_review_path)
                                      if independent_review_path else None)),
        entry_weight_sha256=s2_spec.S1_ENTRY_WEIGHT_SHA256,
        created_at_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n')
    return doc


def validate_admission(doc, *, probe_path, num_envs):
    import hashlib
    import json
    from pathlib import Path
    if not isinstance(doc, dict) or doc.get('schema') != ADMISSION_SCHEMA:
        raise ValueError('admission schema mismatch')
    if doc.get('admitted') is not True:
        raise ValueError('admission is not true')
    if int(doc.get('num_envs', -1)) != int(num_envs):
        raise ValueError('admission num_envs differs')
    probe_meta = doc.get('probe_report', {})
    if probe_meta.get('admitted') is not True:
        raise ValueError('admission probe is not admitted')
    path = Path(probe_path)
    if not path.exists():
        raise ValueError('probe report file missing')
    probe = json.loads(path.read_text())
    if probe_meta.get('sha256') != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError('admission probe SHA mismatch')
    failures = probe_report_failures(probe, num_envs=num_envs)
    if failures:
        raise ValueError('admission probe requirements failed: '
                         + ', '.join(failures))
    review = doc.get('independent_review', {})
    digest = str(review.get('sha256', ''))
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('admission independent review SHA missing')
    if doc.get('entry_weight_sha256') != s2_spec.S1_ENTRY_WEIGHT_SHA256:
        raise ValueError('admission entry weight SHA differs')
    return True
