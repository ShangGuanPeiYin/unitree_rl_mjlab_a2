"""S2 §6.1.1 gate matrix and §6.5 acceptance-quality checks (pure CPU)."""
from __future__ import annotations

GATE_LIMITS = {
    'S2.1': dict(groups=36, cases=720, threshold=18),
    'S2.2': dict(groups=33, cases=660, threshold=18),
    'S2.3': dict(groups=51, cases=1020, threshold=18),
}
FINAL_GROUPS = 27
FINAL_CASES = 100
FINAL_THRESHOLD = 95
QUALITY = dict(pitch_rms_deg=5.0, pitch_p99_deg=10.0, pitch_over15_seconds=0.2,
               speed_rmse=0.10, stand_mean_speed=0.05, stand_peak_speed=0.10,
               stand_displacement_m=0.10, stand_support_fraction=0.90,
               recovery_seconds=3.0)


def quality_reasons(metrics, *, stop=False):
    reasons = []
    if metrics.get('pitch_rms_deg') is None or metrics['pitch_rms_deg'] > QUALITY['pitch_rms_deg']:
        reasons.append('pitch_rms')
    if metrics.get('pitch_p99_deg') is None or metrics['pitch_p99_deg'] > QUALITY['pitch_p99_deg']:
        reasons.append('pitch_p99')
    if metrics.get('pitch_over15_max_frames', 0) * 0.02 >= QUALITY['pitch_over15_seconds']:
        reasons.append('pitch_over15')
    if metrics.get('speed_rmse') is None or metrics['speed_rmse'] > QUALITY['speed_rmse']:
        reasons.append('speed_rmse')
    if metrics.get('nonfoot_contact', False):
        reasons.append('nonfoot_contact')
    if metrics.get('cruise_stall_latched', False):
        reasons.append('cruise_stall')
    if stop:
        if metrics.get('recovery_seconds') is None or metrics['recovery_seconds'] > QUALITY['recovery_seconds']:
            reasons.append('recovery')
        if metrics.get('stand_mean_speed') is None or metrics['stand_mean_speed'] > QUALITY['stand_mean_speed']:
            reasons.append('stand_mean_speed')
        if metrics.get('stand_peak_speed') is None or metrics['stand_peak_speed'] > QUALITY['stand_peak_speed']:
            reasons.append('stand_peak_speed')
        if metrics.get('stand_max_displacement') is None or metrics['stand_max_displacement'] > QUALITY['stand_displacement_m']:
            reasons.append('stand_displacement')
        if metrics.get('stand_support_fraction') is None or metrics['stand_support_fraction'] < QUALITY['stand_support_fraction']:
            reasons.append('stand_support')
    return reasons


def group_result(successes, total, *, threshold, engineering_invalid=False):
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError('invalid group counts')
    reasons = []
    if engineering_invalid:
        reasons.append('engineering_invalid')
    if successes < threshold:
        reasons.append('below_threshold')
    return dict(successes=successes, total=total, threshold=threshold,
                passed=not reasons, reasons=reasons)


def gate_matrix():
    return {'S2.1': dict(GATE_LIMITS['S2.1'], suites=('gate_A', 'gate_B')),
            'S2.2': dict(GATE_LIMITS['S2.2'], suites=('gate_A', 'gate_B')),
            'S2.3': dict(GATE_LIMITS['S2.3'], suites=('gate_A', 'gate_B'))}


def final_acceptance(records):
    """27 groups ×100 cases; engineering errors invalidate the whole set."""
    if len(records) != FINAL_GROUPS * FINAL_CASES:
        raise ValueError('final acceptance requires 2700 cases')
    by_group = {}
    for row in records:
        by_group.setdefault(row['group'], []).append(row)
    groups = {}
    invalid = any(row.get('engineering_error') for row in records)
    for group, rows in sorted(by_group.items()):
        passed = sum(bool(row.get('success')) and not quality_reasons(
            row.get('metrics', {}), stop=row.get('stop', False)) for row in rows)
        groups[group] = group_result(passed, len(rows),
                                     threshold=FINAL_THRESHOLD,
                                     engineering_invalid=invalid)
    return dict(groups=groups, group_count=len(groups),
                passed=all(row['passed'] for row in groups.values()) and not invalid,
                engineering_invalid=invalid)
