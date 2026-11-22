"""S2 resource admission: 1024 first, documented fallback to 512 (§2.3)."""
from __future__ import annotations

from dataclasses import dataclass

from . import s2_spec

PROBE_EVAL_TIMEOUT_STEPS = 9472  # batch hard-timeout inclusive; 9471 is last index
REQUIREMENTS = dict(gpu_peak_mib=7168, host_free_mib=4096,
                    warmups=5, measurements=50, restore_checks=1,
                    rollout_steps=48, adam_updates_per_rollout=20,
                    independent_copy=True, allow_oom=False,
                    allow_nonfinite=False, allow_contact_overflow=False,
                    evaluation_envs=100)


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
    import time
    from pathlib import Path
    import torch
    try:
        import psutil
        host_free = psutil.virtual_memory().available / 2 ** 20
    except Exception:
        host_free = None
    from . import s2_physics, s2_train, s2_train_real
    env = s2_train_real.S2TrainEnv(num_envs=num_envs, device=device)
    actor = s2_physics.load_s1_actor(device=device)
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
                model=s2_physics.load_s1_actor(device=device)['model'],
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
        admitted = not failure_reasons
    except RuntimeError as error:
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
        admitted = False
        peak_mib = (torch.cuda.max_memory_allocated() / 2 ** 20
                    if torch.cuda.is_available() else 0.0)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    report = dict(
        num_envs=num_envs, admitted=bool(admitted), peak_mib=peak_mib,
        host_free_mib=host_free, oom=oom, nonfinite=nonfinite,
        contact_overflow=contact_overflow,
        warmups=REQUIREMENTS['warmups'], measurements=len(timings),
        restore_checks=1 if restore_ok else 0, restore_ok=bool(restore_ok),
        independent_copy=bool(independent_copy),
        evaluation_envs=evaluation_envs, evaluation_steps=evaluation_steps,
        video_envs=video_envs, video_steps=video_steps,
        elapsed_s=float(sum(timings)), timings_s=timings,
        evaluation_s=eval_timing, video_s=video_timing,
        failure_reasons=failure_reasons, requirements=dict(REQUIREMENTS),
        note='real S2TrainEnv probe including evaluation 100 and video 1')
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
    if probe_report.get('admitted') is not True:
        raise ValueError('cannot write admission: probe not admitted')
    if probe_report.get('num_envs') != int(num_envs):
        raise ValueError('probe num_envs differs')
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
        entry_weight_sha256=__import__(
            'src.tasks.velocity.v09.s2_spec', fromlist=['S1_ENTRY_WEIGHT_SHA256']
        ).S1_ENTRY_WEIGHT_SHA256,
        created_at_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n')
    return doc


def validate_admission(doc, *, probe_path, num_envs):
    import hashlib
    from pathlib import Path
    if not isinstance(doc, dict) or doc.get('schema') != ADMISSION_SCHEMA:
        raise ValueError('admission schema mismatch')
    if doc.get('admitted') is not True:
        raise ValueError('admission is not true')
    if int(doc.get('num_envs', -1)) != int(num_envs):
        raise ValueError('admission num_envs differs')
    probe = doc.get('probe_report', {})
    if probe.get('admitted') is not True:
        raise ValueError('admission probe is not admitted')
    path = Path(probe_path)
    if not path.exists():
        raise ValueError('probe report file missing')
    if probe.get('sha256') != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError('admission probe SHA mismatch')
    review = doc.get('independent_review', {})
    digest = str(review.get('sha256', ''))
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('admission independent review SHA missing')
    return True
