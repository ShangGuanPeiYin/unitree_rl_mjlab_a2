"""S2 resource admission: 1024 first, documented fallback to 512 (§2.3)."""
from __future__ import annotations

from dataclasses import dataclass

from . import s2_spec

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


def probe_plan(num_envs, *, evaluation_timeout_steps=9472):
    if num_envs not in s2_spec.S2_ENV_CANDIDATES:
        raise ValueError('S2 probe is 1024 then 512 only')
    samples = num_envs * REQUIREMENTS['warmups'] * REQUIREMENTS['rollout_steps']
    samples += num_envs * REQUIREMENTS['measurements'] * REQUIREMENTS['rollout_steps']
    samples += num_envs * REQUIREMENTS['rollout_steps']  # restore rollout
    samples += REQUIREMENTS['evaluation_envs'] * int(evaluation_timeout_steps)
    return dict(num_envs=num_envs, requirements=dict(REQUIREMENTS),
                control_samples=samples, physics_substeps=4 * samples,
                warmups=REQUIREMENTS['warmups'],
                measurements=REQUIREMENTS['measurements'],
                restore_checks=REQUIREMENTS['restore_checks'])


def choose_resource(probes):
    """Return the first admitted 1024→512, carrying failure evidence."""
    evidence = []
    for result in probes:
        evidence.append(dict(num_envs=result.num_envs, admitted=result.admitted(),
                             failures=result.failures(),
                             evidence_sha256=result.evidence_sha256))
        if result.admitted():
            return dict(selected=result.num_envs, fallback_evidence=evidence)
    return dict(selected=None, fallback_evidence=evidence,
                reason='both 1024 and 512 failed; stop and report')
