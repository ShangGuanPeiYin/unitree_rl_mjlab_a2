"""S1 flat paired retention according to planning ruling Q001."""
from __future__ import annotations

import math
import statistics

METRICS = ('speed_rmse', 'pitch_rms_deg', 'slip', 'action_change', 'stride', 'frequency')


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _metric_comparison(name, reference, candidate):
    if name in ('stride', 'frequency'):
        return abs(candidate - reference) <= abs(reference) * .1
    tolerance = {
        'speed_rmse': max(reference * .05, .02),
        'pitch_rms_deg': 1.,
        'slip': reference * .1 if reference else 1e-8,
        'action_change': reference * .1 if reference else 1e-8,
        'drift': .02,
    }[name]
    return candidate <= reference + tolerance


def compare_flat_group(teacher, candidate, *, stop=False):
    """Compare full fixed groups while preserving each metric's valid pairing.

    Each record carries case_id, success, metrics and optional missing_reasons.
    None denotes a missing complete window or insufficient gait cycles.
    """
    count = len(teacher)
    if count not in (20, 100) or len(candidate) != count:
        raise ValueError('a fixed complete 20/100-case group is required')
    teachers = {record['case_id']: record for record in teacher}
    candidates = {record['case_id']: record for record in candidate}
    if len(teachers) != count or len(candidates) != count or teachers.keys() != candidates.keys():
        raise ValueError('paired cases must have identical unique case IDs')
    invalid = any(record.get('engineering_error') for record in teacher + candidate)
    teacher_success = sum(bool(record['success']) for record in teacher)
    candidate_success = sum(bool(record['success']) for record in candidate)
    result = dict(N=count, teacher_success=teacher_success, candidate_success=candidate_success,
                  engineering_invalid=invalid, planning_required=False, metrics={},
                  passed=not invalid and candidate_success >= teacher_success)
    result['teacher_wilson95'] = wilson_interval(teacher_success, count)
    result['candidate_wilson95'] = wilson_interval(candidate_success, count)
    for name in METRICS + (('drift',) if stop else ()):
        valid_ids = [identity for identity, record in teachers.items()
                     if _finite(record['metrics'].get(name))]
        missing_ids = [identity for identity in valid_ids
                       if not _finite(candidates[identity]['metrics'].get(name))]
        outside = [identity for identity in candidates if identity not in valid_ids
                   and _finite(candidates[identity]['metrics'].get(name))]
        pairs = [dict(case_id=identity,
                      teacher=teachers[identity]['metrics'].get(name),
                      candidate=candidates[identity]['metrics'].get(name),
                      candidate_missing_reason=candidates[identity].get('missing_reasons', {}).get(name))
                 for identity in valid_ids]
        metric = dict(teacher_valid_count=len(valid_ids),
                      candidate_valid_count=sum(_finite(row['metrics'].get(name)) for row in candidate),
                      candidate_missing_teacher_ids=missing_ids,
                      candidate_outside_teacher_ids=outside, pairs=pairs,
                      teacher_missing=[dict(case_id=identity,
                          reason=record.get('missing_reasons', {}).get(name))
                          for identity, record in teachers.items() if identity not in valid_ids],
                      passed=False, teacher_aggregate=None, candidate_aggregate=None)
        if not valid_ids:
            result['planning_required'] = True
        elif not missing_ids:
            aggregate = statistics.median if name in ('stride', 'frequency') else statistics.mean
            reference = aggregate(teachers[identity]['metrics'][name] for identity in valid_ids)
            current = aggregate(candidates[identity]['metrics'][name] for identity in valid_ids)
            metric.update(teacher_aggregate=reference, candidate_aggregate=current,
                          passed=_metric_comparison(name, reference, current))
        result['metrics'][name] = metric
        result['passed'] &= metric['passed']
    return result


def compare_stair_group(teacher, candidate):
    """All physical failures and signed terminal distances remain in the group."""
    import numpy as np
    count = len(teacher)
    if count not in (20, 100) or len(candidate) != count:
        raise ValueError('a fixed complete 20/100-case group is required')
    teachers = {row['case_id']: row for row in teacher}
    candidates = {row['case_id']: row for row in candidate}
    if len(teachers) != count or len(candidates) != count or teachers.keys() != candidates.keys():
        raise ValueError('paired cases must have identical unique case IDs')
    ids = list(teachers)
    reference = [teachers[identity]['forward_distance'] for identity in ids]
    current = [candidates[identity]['forward_distance'] for identity in ids]
    if not all(_finite(value) for value in reference + current):
        raise ValueError('every stair case requires its signed terminal distance')
    invalid = any(row.get('engineering_error') for row in teacher + candidate)
    teacher_success = sum(bool(row['success']) for row in teacher)
    candidate_success = sum(bool(row['success']) for row in candidate)
    result = dict(N=count, teacher_success=teacher_success, candidate_success=candidate_success,
                  engineering_invalid=invalid, quantiles={}, failure_classes={},
                  passed=not invalid and candidate_success >= teacher_success)
    result['teacher_wilson95'] = wilson_interval(teacher_success, count)
    result['candidate_wilson95'] = wilson_interval(candidate_success, count)
    for percentile in (10, 50, 90):
        before, after = float(np.percentile(reference, percentile)), float(np.percentile(current, percentile))
        tolerance = max(abs(before) * .05, .05)
        passed = after >= before - tolerance
        result['quantiles'][f'P{percentile}'] = dict(teacher=before, candidate=after,
                                                   tolerance=tolerance, passed=passed)
        result['passed'] &= passed
    difference = statistics.mean(after - before for before, after in zip(reference, current))
    tolerance = max(statistics.mean(abs(value) for value in reference) * .05, .05)
    result['paired_mean'] = dict(difference=difference, tolerance=tolerance,
                                 passed=difference >= -tolerance)
    result['passed'] &= result['paired_mean']['passed']
    classes = set().union(*(row['failure_masks'].keys() for row in teacher + candidate))
    for name in sorted(classes):
        before = sum(bool(row['failure_masks'].get(name, False)) for row in teacher)
        after = sum(bool(row['failure_masks'].get(name, False)) for row in candidate)
        passed = after <= before
        result['failure_classes'][name] = dict(teacher=before, candidate=after, passed=passed)
        result['passed'] &= passed
    result['pairs'] = [dict(case_id=identity, teacher=teachers[identity]['forward_distance'],
                           candidate=candidates[identity]['forward_distance']) for identity in ids]
    return result


def wilson_interval(successes, total):
    """Wilson 95% interval accompanies empirical counts, with no gate effect."""
    if (isinstance(successes, bool) or isinstance(total, bool)
            or not isinstance(successes, int) or not isinstance(total, int)
            or total < 1 or not 0 <= successes <= total):
        raise ValueError('valid integer success count and positive denominator required')
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total
                           + z * z / (4 * total * total)) / denominator
    return dict(successes=successes, total=total, empirical_rate=proportion,
                confidence=.95, lower=max(0., center - margin), upper=min(1., center + margin))
