"""Planning Q001 full-denominator and missing-window counterexamples."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from a2_v07_bootstrap import activate
activate()
from src.tasks.velocity.v09.acceptance import compare_flat_group


def records():
    return [dict(case_id=i, success=True, metrics=dict(speed_rmse=.04,
        pitch_rms_deg=2., slip=.1, action_change=.2, stride=.2, frequency=2.))
        for i in range(20)]


def test_candidate_missing_teacher_valid_window_fails():
    teacher, student = records(), records()
    student[0]['metrics']['stride'] = None
    assert not compare_flat_group(teacher, student)['passed']


def test_teacher_entirely_missing_metric_requests_planning():
    teacher, student = records(), records()
    for record in teacher:
        record['metrics']['frequency'] = None
    result = compare_flat_group(teacher, student)
    assert not result['passed']
    assert result['planning_required']


def test_paired_mean_uses_teacher_valid_ids_and_reports_outside():
    teacher, student = records(), records()
    teacher[0]['metrics']['speed_rmse'] = None
    student[0]['metrics']['speed_rmse'] = 100.
    result = compare_flat_group(teacher, student)
    assert result['passed']
    metric = result['metrics']['speed_rmse']
    assert metric['teacher_valid_count'] == 19
    assert metric['candidate_outside_teacher_ids'] == [0]


def test_denominator_cannot_shrink():
    with pytest.raises(ValueError):
        compare_flat_group(records()[:-1], records()[:-1])


def test_zero_success_does_not_approve_lost_stair_progress():
    from src.tasks.velocity.v09.acceptance import compare_stair_group
    teacher = [dict(case_id=i, success=False, forward_distance=0. if i < 4 else 8.,
                    failure_masks={'nonfoot': True}) for i in range(20)]
    student = [dict(row, forward_distance=0.) for row in teacher]
    result = compare_stair_group(teacher, student)
    assert not result['passed']
    assert not result['quantiles']['P50']['passed']


def test_stair_failure_class_cannot_increase():
    from src.tasks.velocity.v09.acceptance import compare_stair_group
    teacher = [dict(case_id=i, success=False, forward_distance=4.,
                    failure_masks={'nonfoot': True, 'fall': False}) for i in range(20)]
    student = [dict(row, failure_masks={'nonfoot': False, 'fall': True}) for row in teacher]
    assert not compare_stair_group(teacher, student)['passed']


def test_wilson_interval_reports_uncertainty_without_changing_gates():
    from src.tasks.velocity.v09.acceptance import wilson_interval
    zero = wilson_interval(0, 20)
    full = wilson_interval(20, 20)
    assert zero['lower'] == pytest.approx(0., abs=1e-12)
    assert zero['upper'] == pytest.approx(1 - full['lower'])
    assert full['upper'] == pytest.approx(1., abs=1e-12)
    assert zero['upper'] == pytest.approx(.161125158052819)
