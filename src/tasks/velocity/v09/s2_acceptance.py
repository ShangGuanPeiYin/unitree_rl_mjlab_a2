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
    if metrics.get('valid_cruise_window') is not True:
        reasons.append('missing_cruise_window')
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
        for key in ('stand_roll_abs_max_deg', 'stand_pitch_abs_max_deg'):
            if metrics.get(key) is None or metrics[key] > 10.0:
                reasons.append('stand_attitude')
        if metrics.get('stand_height_min') is None or metrics['stand_height_min'] < 0.30:
            reasons.append('stand_height')
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


def _flat_retention(candidate, reference):
    tolerances = dict(
        speed_rmse=max(float(reference.get('speed_rmse', 0.0)) * 0.05, 0.02),
        pitch_rms_deg=1.0,
        slip=max(float(reference.get('slip', 0.0)) * 0.10, 1e-8),
        action_change=max(float(reference.get('action_change', 0.0)) * 0.10, 1e-8),
        drift=0.02,
    )
    reasons = []
    for key, tolerance in tolerances.items():
        if candidate.get(key) is None or reference.get(key) is None:
            reasons.append(f'{key}_missing')
        elif float(candidate[key]) > float(reference[key]) + tolerance:
            reasons.append(key)
    for key in ('stride', 'frequency'):
        if candidate.get(key) is None or reference.get(key) is None:
            reasons.append(f'{key}_missing')
        elif abs(float(candidate[key]) - float(reference[key])) > abs(float(reference[key])) * 0.10:
            reasons.append(key)
    return reasons


def final_acceptance(records, *, original_6400=None):
    """27 groups ×100; 95/100, full quality and 6400 flat retention."""
    if len(records) != FINAL_GROUPS * FINAL_CASES:
        raise ValueError('final acceptance requires exactly 2700 cases')
    by_group = {}
    for row in records:
        by_group.setdefault(row['group'], []).append(row)
    if len(by_group) != FINAL_GROUPS:
        raise ValueError('final acceptance requires exactly 27 groups')
    for group, rows in by_group.items():
        if len(rows) != FINAL_CASES:
            raise ValueError(f'group {group} has {len(rows)} cases, requires 100')
    invalid = any(row.get('engineering_error') for row in records)
    groups = {}
    flat_groups = 0
    flat_reasons = []
    for group, rows in sorted(by_group.items()):
        passed = sum(bool(row.get('success')) and not quality_reasons(
            row.get('metrics', {}), stop=row.get('stop', False)) for row in rows)
        result = group_result(passed, len(rows), threshold=FINAL_THRESHOLD,
                              engineering_invalid=invalid)
        task = rows[0].get('task', '')
        if task.startswith('flat_'):
            flat_groups += 1
            if original_6400 is None or group not in original_6400:
                result['passed'] = False
                result['reasons'].append('missing_6400_reference')
                flat_reasons.append('missing_6400_reference')
            else:
                aggregate = {key: sum(float(row['metrics'][key])
                                      for row in rows if row['metrics'].get(key) is not None)
                             / max(1, sum(1 for row in rows
                                          if row['metrics'].get(key) is not None))
                             for key in ('speed_rmse', 'pitch_rms_deg', 'slip',
                                         'action_change', 'drift', 'stride', 'frequency')}
                retention = _flat_retention(aggregate, original_6400[group])
                if retention:
                    result['passed'] = False
                    result['reasons'].extend(retention)
                    flat_reasons.extend(retention)
        groups[group] = result
    if flat_groups != 9:
        flat_reasons.append('flat_group_count')
    return dict(groups=groups, group_count=len(groups),
                flat_groups=flat_groups,
                flat_retention_reasons=sorted(set(flat_reasons)),
                passed=(all(row['passed'] for row in groups.values())
                        and not invalid and flat_groups == 9
                        and not flat_reasons))
