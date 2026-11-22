"""S2.0 event and segment metrics for the top-entry decomposition (§4.3).

Pure functions over one complete, explicitly timestamped per-control-step trace.
They import no simulator and never invent a contact or gait threshold:

* ``foot_on_platform`` is supplied by the caller and must already encode the
  §5.3 top-support contract (upward force, exact surface, two-frame debounce).
* ``foot_contact`` is the caller's contact signal used only for landing cycles.
* ``pitch`` sign follows the upstream ``euler_heading`` convention; this module
  reports signed extrema and states that convention rather than redefining it.

Missing windows or fewer than two complete landing cycles are returned as
``None`` with an explicit reason, never as a zero.  A failure on the same frame
as a target crossing wins, matching §4.2 precedence.
"""
from __future__ import annotations

import math

import numpy as np

from . import s2_spec

DT = 0.02
RAD_TO_DEG = 180.0 / math.pi
EPS = 1e-9
FRONT_FEET = (0, 1)
SEGMENT_NAMES = ('front_entry_to_four_entry', 'first_meter',
                 'post_first_meter', 'after_stop_request')
REQUIRED_GAIT_CYCLES = 2
SIGN_CONVENTION = ('raw upstream pitch; positive and negative extrema are both '
                   'reported, no forward-lean sign is assumed')

EVENT_NAMES = ('front_foot_entry', 'four_feet_platform', 'g_level_positive',
               'g_level_full', 'top_arrived', 'failure', 'stop_request')


def _array(trace, key, *, shape_tail=None, dtype=float, required=True):
    if key not in trace:
        if required:
            raise ValueError(f'trace missing required field: {key}')
        return None
    value = np.asarray(trace[key], dtype=dtype)
    if shape_tail is not None and value.shape[1:] != shape_tail:
        if value.ndim == 0:
            raise ValueError(f'{key} has scalar shape, expected trailing {shape_tail}')
        raise ValueError(f'{key} has trailing shape {value.shape[1:]}, expected {shape_tail}')
    return value


def _validate_trace(trace):
    step = np.asarray(trace.get('step'), dtype=np.int64)
    if step.ndim != 1 or step.size == 0:
        raise ValueError('step must be a nonempty 1-D array')
    if step.size > 1 and not bool(np.all(np.diff(step) > 0)):
        raise ValueError('step must be strictly increasing')
    count = step.size
    fields = dict(
        step=step,
        route_x=_array(trace, 'route_x'),
        base_xy=_array(trace, 'base_xy', shape_tail=(2,)),
        feet_xy=_array(trace, 'feet_xy', shape_tail=(4, 2)),
        feet_route_x=_array(trace, 'feet_route_x', shape_tail=(4,), required=False),
        pitch=_array(trace, 'pitch'),
        foot_on_platform=_array(trace, 'foot_on_platform', shape_tail=(4,), dtype=bool),
        foot_contact=_array(trace, 'foot_contact', shape_tail=(4,), dtype=bool),
        g_level=_array(trace, 'g_level'),
        top_arrived=_array(trace, 'top_arrived', dtype=bool, required=False),
        failure=_array(trace, 'failure', dtype=bool, required=False),
        stop_request=_array(trace, 'stop_request', dtype=bool, required=False),
        terminal=_array(trace, 'terminal', dtype=bool, required=False),
        c_exec=_array(trace, 'c_exec', required=False),
        velocity_route=_array(trace, 'velocity_route', required=False),
        pitch_rate=_array(trace, 'pitch_rate', required=False),
        foot_normal_force=_array(trace, 'foot_normal_force', shape_tail=(4,), required=False),
    )
    for key, value in fields.items():
        if value is not None and key not in ('step',) and value.shape[0] != count:
            raise ValueError(f'{key} length differs from step')
    if not np.all(np.isfinite(fields['route_x'])):
        raise ValueError('route_x must be finite')
    if not np.all(np.isfinite(fields['pitch'])):
        raise ValueError('pitch must be finite')
    if np.any(fields['g_level'] < -EPS) or np.any(fields['g_level'] > 1 + EPS):
        raise ValueError('g_level must lie in [0, 1]')
    if fields['feet_route_x'] is None:
        fields['feet_route_x'] = fields['feet_xy'][..., 0]
    return fields


def _validate_platform(platform):
    if not isinstance(platform, dict):
        raise ValueError('platform must be a mapping')
    try:
        front_x = float(platform['front_x'])
        length = float(platform['length'])
        half_width = float(platform['y_half_width'])
    except KeyError as error:
        raise ValueError(f'platform missing {error.args[0]}') from None
    if not (math.isfinite(front_x) and math.isfinite(length) and math.isfinite(half_width)):
        raise ValueError('platform geometry must be finite')
    if length <= 0 or half_width <= 0:
        raise ValueError('platform length and half width must be positive')
    return dict(front_x=front_x, length=length, y_half_width=half_width)


def _first(mask):
    hits = np.flatnonzero(np.asarray(mask, dtype=bool))
    return int(hits[0]) if hits.size else None


def _event(index, step):
    return dict(index=index, step=None if index is None else int(step[index]),
                present=index is not None)


def case_events(trace, *, platform, front_feet=FRONT_FEET):
    """Latch the §4.3 event times on a complete trace (plain Python indexes)."""
    fields = _validate_trace(trace)
    geometry = _validate_platform(platform)
    step = fields['step']
    count = step.size
    front = tuple(int(i) for i in front_feet)
    if not front or min(front) < 0 or max(front) > 3:
        raise ValueError('front_feet must be a nonempty subset of 0..3')
    x, y = fields['feet_xy'][..., 0], fields['feet_xy'][..., 1]
    inside = ((x >= geometry['front_x'] - EPS)
              & (x <= geometry['front_x'] + geometry['length'] + EPS)
              & (np.abs(y) <= geometry['y_half_width'] + EPS))
    front_supported = fields['foot_on_platform'][:, list(front)] & inside[:, list(front)]
    four_inside = inside.all(axis=1)
    indices = dict(
        front_foot_entry=_first(front_supported.any(axis=1)),
        four_feet_platform=_first(four_inside),
        g_level_positive=_first(fields['g_level'] > 0),
        g_level_full=_first(fields['g_level'] >= 1.0 - EPS),
        top_arrived=(_first(fields['top_arrived'])
                     if fields['top_arrived'] is not None else None),
        failure=_first(fields['failure']) if fields['failure'] is not None else None,
        stop_request=(_first(fields['stop_request'])
                      if fields['stop_request'] is not None else None),
        terminal=(_first(fields['terminal']) if fields['terminal'] is not None
                  else count - 1),
    )
    if indices['terminal'] is None:
        indices['terminal'] = count - 1
    events = {name: _event(index, step) for name, index in indices.items()}
    # Q019: the first real front-foot top support is the same latched event.
    events['front_foot_top_support'] = dict(events['front_foot_entry'])
    return events


def _limit(*indices):
    values = [index for index in indices if index is not None]
    return min(values) if values else None


def _reason(end, failure, stop, terminal):
    if end is None:
        return 'not_entered'
    if failure is not None and end == failure:
        return 'failure'          # §4.2 same-frame failure precedence
    if stop is not None and end == stop:
        return 'stop_request'
    if terminal is not None and end == terminal and end != failure:
        return 'terminal'
    return 'target'


def segment_bounds(trace, events, *, zero_anchor='four_feet_platform', route_distance=1.0):
    """Segment indexes; shared boundaries are documented, failure wins ties."""
    fields = _validate_trace(trace)
    step = fields['step']
    route_x = fields['route_x']
    count = step.size
    if zero_anchor not in ('four_feet_platform', 'spawn'):
        raise ValueError('zero_anchor must be four_feet_platform or spawn')
    if isinstance(route_distance, bool) or not isinstance(route_distance, (int, float)):
        raise ValueError('route_distance must be a number')
    route_distance = float(route_distance)
    if not math.isfinite(route_distance) or route_distance <= 0:
        raise ValueError('route_distance must be positive and finite')
    failure = events['failure']['index']
    stop = events['stop_request']['index']
    terminal = events['terminal']['index']
    terminal = count - 1 if terminal is None else terminal
    front = events['front_foot_entry']['index']
    four = events['four_feet_platform']['index']
    zero = four if zero_anchor == 'four_feet_platform' else 0
    if zero_anchor == 'four_feet_platform' and four is None:
        zero = None
    # A zero point latched on or after the failure frame is not a valid segment
    # start: §4.2 gives the same-frame failure precedence.
    if zero is not None and failure is not None and zero >= failure:
        zero = None

    one_meter = None
    if zero is not None:
        targets = np.flatnonzero(route_x[zero:] >= route_x[zero] + route_distance)
        one_meter = zero + int(targets[0]) if targets.size else None

    bounds = {}

    if front is None:
        bounds['front_entry_to_four_entry'] = dict(
            start=None, end=None, entered=False, completed=False, end_reason='not_entered')
    else:
        reached = (four is not None and (failure is None or four < failure)
                   and four <= terminal)
        end = four if reached else _limit(failure, terminal)
        bounds['front_entry_to_four_entry'] = dict(
            start=front, end=end, entered=True, completed=bool(reached),
            end_reason='target' if reached else _reason(end, failure, stop, terminal))

    if zero is None:
        bounds['first_meter'] = dict(start=None, end=None, entered=False,
                                     completed=False, end_reason='not_entered')
    else:
        reached = (one_meter is not None and (failure is None or one_meter < failure)
                   and (stop is None or one_meter <= stop) and one_meter <= terminal)
        end = one_meter if reached else _limit(failure, stop, terminal)
        bounds['first_meter'] = dict(
            start=zero, end=end, entered=True, completed=bool(reached),
            end_reason='target' if reached else _reason(end, failure, stop, terminal))

    if not bounds['first_meter']['completed']:
        bounds['post_first_meter'] = dict(start=None, end=None, entered=False,
                                          completed=False, end_reason='not_entered')
    else:
        start = one_meter
        end = _limit(stop, failure, terminal)
        end = max(start, end) if end is not None else start
        bounds['post_first_meter'] = dict(
            start=start, end=end, entered=True,
            completed=not (failure is not None and end == failure),
            end_reason=_reason(end, failure, stop, terminal))

    if stop is None or not bounds['first_meter']['entered']:
        bounds['after_stop_request'] = dict(start=None, end=None, entered=False,
                                            completed=False, end_reason='not_entered')
    else:
        end = _limit(failure, terminal)
        if end is None or end < stop:
            end = terminal
        bounds['after_stop_request'] = dict(
            start=stop, end=end, entered=True,
            completed=not (failure is not None and end == failure),
            end_reason=_reason(end, failure, stop, terminal))
    return bounds


def _median(values):
    return float(np.median(values)) if len(values) else None


def _gait_cycles(fields, start, end, *, foot_count=4):
    step = fields['step']
    contact = fields['foot_contact']
    feet_x = fields['feet_route_x']
    feet = []
    all_strides, all_frequencies = [], []
    complete = True
    for foot in range(foot_count):
        window = contact[start:end + 1, foot]
        landings = np.flatnonzero(window[1:] & ~window[:-1]) + 1
        cycles = []
        for first, second in zip(landings[:-1], landings[1:]):
            a, b = start + int(first), start + int(second)
            period = float((step[b] - step[a]) * DT)
            cycles.append(dict(start_step=int(step[a]), end_step=int(step[b]),
                               stride_m=float(feet_x[b, foot] - feet_x[a, foot]),
                               period_seconds=period,
                               frequency_hz=(1.0 / period) if period > 0 else None))
        valid = len(cycles) >= REQUIRED_GAIT_CYCLES
        complete &= valid
        strides = [row['stride_m'] for row in cycles]
        frequencies = [row['frequency_hz'] for row in cycles
                       if row['frequency_hz'] is not None]
        all_strides.extend(strides)
        all_frequencies.extend(frequencies)
        feet.append(dict(foot=foot, complete_cycles=len(cycles), valid=bool(valid),
                         stride_median_m=_median(strides),
                         frequency_median_hz=_median(frequencies),
                         missing_reason=(None if valid
                                         else 'insufficient_complete_landing_cycles'),
                         cycles=cycles))
    return dict(feet=feet, valid=bool(complete),
                complete_cycles=sum(row['complete_cycles'] for row in feet),
                stride_median_m=_median(all_strides) if complete else None,
                frequency_median_hz=_median(all_frequencies) if complete else None,
                missing_reason=None if complete else 'insufficient_complete_landing_cycles')


def segment_metrics(trace, start, end, *, front_feet=FRONT_FEET):
    """Per-segment signed/kinematic/gait statistics; None means missing."""
    fields = _validate_trace(trace)
    step = fields['step']
    front = list(int(i) for i in front_feet)
    base = dict(start_step=None, end_step=None, sample_count=0,
                signed_distance_m=None, pitch_max_deg_signed=None,
                pitch_min_deg_signed=None, pitch_abs_max_deg=None,
                pitch_rate_abs_max_deg_s=None, pitch_rate_max_deg_s=None,
                pitch_rate_min_deg_s=None, front_any_support_fraction=None,
                front_all_support_fraction=None, front_normal_force_mean=None,
                front_normal_force_min=None, front_normal_force_max=None,
                actual_forward_speed_mean=None, c_exec_mean=None, c_exec_min=None,
                c_exec_max=None, gait=None, entered=False,
                missing={})
    if start is None or end is None or end < start:
        base['missing']['segment'] = 'segment_not_entered'
        return base
    base.update(entered=True, start_step=int(step[start]), end_step=int(step[end]),
                sample_count=int(end - start + 1),
                signed_distance_m=float(fields['route_x'][end] - fields['route_x'][start]))
    pitch = fields['pitch'][start:end + 1] * RAD_TO_DEG
    base['pitch_max_deg_signed'] = float(np.max(pitch))
    base['pitch_min_deg_signed'] = float(np.min(pitch))
    base['pitch_abs_max_deg'] = float(np.max(np.abs(pitch)))
    if fields['pitch_rate'] is not None:
        rate = fields['pitch_rate'][start:end + 1] * RAD_TO_DEG
        base['pitch_rate_abs_max_deg_s'] = float(np.max(np.abs(rate)))
        base['pitch_rate_max_deg_s'] = float(np.max(rate))
        base['pitch_rate_min_deg_s'] = float(np.min(rate))
    elif end > start:
        rate = np.diff(fields['pitch'][start:end + 1]) / DT * RAD_TO_DEG
        base['pitch_rate_abs_max_deg_s'] = float(np.max(np.abs(rate)))
        base['pitch_rate_max_deg_s'] = float(np.max(rate))
        base['pitch_rate_min_deg_s'] = float(np.min(rate))
    else:
        base['missing']['pitch_rate'] = 'single_sample_segment'

    support = fields['foot_on_platform'][start:end + 1][:, front]
    base['front_any_support_fraction'] = float(support.any(axis=1).mean())
    base['front_all_support_fraction'] = float(support.all(axis=1).mean())
    normal = fields['foot_normal_force']
    if normal is not None:
        window = normal[start:end + 1][:, front]
        base['front_normal_force_mean'] = float(window.mean())
        base['front_normal_force_min'] = float(window.min())
        base['front_normal_force_max'] = float(window.max())
    else:
        base['missing']['front_normal_force'] = 'trace_lacks_foot_normal_force'

    if fields['velocity_route'] is not None:
        base['actual_forward_speed_mean'] = float(
            fields['velocity_route'][start:end + 1].mean())
    elif end > start:
        base['actual_forward_speed_mean'] = float(
            (fields['route_x'][end] - fields['route_x'][start]) / ((end - start) * DT))
    else:
        base['missing']['actual_forward_speed'] = 'single_sample_segment'

    command = fields['c_exec']
    if command is not None:
        window = command[start:end + 1]
        base['c_exec_mean'] = float(window.mean())
        base['c_exec_min'] = float(window.min())
        base['c_exec_max'] = float(window.max())
    else:
        base['missing']['c_exec'] = 'trace_lacks_c_exec'

    base['gait'] = _gait_cycles(fields, start, end)
    return base


def failure_context(trace, events, *, platform):
    """Q018/Q019: classify where the terminal happened, never assume platform."""
    fields = _validate_trace(trace)
    geometry = _validate_platform(platform)
    index = events['failure']['index']
    terminal = events['terminal']['index']
    four = events['four_feet_platform']['index']
    if index is None:
        return dict(present=False, index=None, step=None, route_x=None,
                    surface=None, classification=None,
                    after_four_foot_entry=None,
                    signed_from_platform_front_m=None,
                    onto_platform_m=None, before_platform_m=None)
    route_x = float(fields['route_x'][index])
    front = geometry['front_x']
    if route_x >= front:
        surface = 'platform'
    elif route_x >= 0.0:
        surface = 'stairs'
    else:
        surface = 'approach'
    after_four = four is not None and index >= four
    if surface == 'platform' and after_four:
        classification = 'platform_cruise_failure'
    elif surface == 'platform':
        classification = 'platform_failure_before_four_feet'
    elif surface == 'stairs':
        classification = 'stair_failure'
    else:
        classification = 'approach_failure'
    feet_support = fields['foot_on_platform'][index]
    return dict(
        present=True, index=index, step=int(events['failure']['step']),
        route_x=route_x, body_terminal_surface=surface, surface=surface,
        classification=classification,
        after_four_foot_entry=bool(after_four),
        four_feet_on_platform_at_failure=bool(feet_support.all()),
        feet_on_platform_at_failure=int(feet_support.sum()),
        signed_from_platform_front_m=route_x - front,
        onto_platform_m=max(0.0, route_x - front),
        before_platform_m=max(0.0, front - route_x),
        terminal_step=int(events['terminal']['step']),
        terminal_before_failure=bool(terminal is not None and terminal < index),
    )


def original_failure_coverage(case_result, original, *, tolerance_m=0.05):
    """Q017/Q018: did the re-test reach the original terminal route position?"""
    if isinstance(tolerance_m, bool) or not isinstance(tolerance_m, (int, float)):
        raise ValueError('tolerance_m must be a number')
    tolerance_m = float(tolerance_m)
    if not math.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError('tolerance_m must be nonnegative and finite')
    if 'route_x' not in original:
        raise ValueError('original failure position requires route_x')
    target = float(original['route_x'])
    reached = float(case_result['max_route_x'])
    covered = reached >= target - tolerance_m
    context = case_result.get('failure_context') or {}
    surface_match = (None if original.get('surface') is None or not context
                     else context.get('surface') == original.get('surface'))
    return dict(
        original_route_x=target, original_surface=original.get('surface'),
        reached_route_x=reached, tolerance_m=tolerance_m,
        covered=bool(covered),
        status='covered' if covered else 'not_covered',
        surface_match=surface_match,
        new_failure_surface=(context.get('surface') if context.get('present') else None),
        note=('not_verified_before_original_failure_position' if not covered else None),
    )


def summarize_case(trace, *, platform, zero_anchor='four_feet_platform',
                   route_distance=1.0, front_feet=FRONT_FEET, map_evidence=None):
    """One case: events, segment bounds, per-segment metrics and gate delays."""
    fields = _validate_trace(trace)
    events = case_events(trace, platform=platform, front_feet=front_feet)
    bounds = segment_bounds(trace, events, zero_anchor=zero_anchor,
                            route_distance=route_distance)
    segments = {}
    target_segments = ('front_entry_to_four_entry', 'first_meter')
    for name in SEGMENT_NAMES:
        row = segment_metrics(trace, bounds[name]['start'], bounds[name]['end'],
                              front_feet=front_feet)
        for key in ('completed', 'end_reason'):
            row[key] = (bounds[name]['completed'] if key == 'completed'
                        else bounds[name]['end_reason'])
        row['failed'] = bool(row['entered'] and row['end_reason'] == 'failure')
        if name in target_segments:
            row['not_reached'] = bool((not row['entered'])
                                      or (not row['completed'] and not row['failed']))
        else:
            row['not_reached'] = not row['entered']
        row['valid_gait_samples'] = bool(row['entered'] and row['gait']['valid'])
        segments[name] = row
    zero = (events['four_feet_platform']['index']
            if zero_anchor == 'four_feet_platform' else 0)
    if zero is not None and events['failure']['index'] is not None \
            and zero >= events['failure']['index']:
        zero = None
    delays = {}
    for name in ('g_level_positive', 'g_level_full'):
        index = events[name]['index']
        delays[name + '_seconds_from_zero'] = (
            None if index is None or zero is None
            else float((events[name]['step'] - int(fields['step'][zero])) * DT))
    route_x = fields['route_x']
    top = events['top_arrived']['index']
    # Q019: a platform start has no top event; D_post starts at spawn local_x.
    d_post_zero = 0 if zero_anchor == 'spawn' else top
    if d_post_zero is not None and events['failure']['index'] is not None \
            and d_post_zero > events['failure']['index']:
        d_post_zero = None
    d_post = None if d_post_zero is None else float(route_x[-1] - route_x[d_post_zero])
    d_post_segments = {}
    for name in SEGMENT_NAMES:
        row = segments[name]
        if row['entered'] and d_post_zero is not None and row['end_step'] is not None:
            end_index = bounds[name]['end']
            d_post_segments[name] = float(route_x[end_index] - route_x[d_post_zero])
        else:
            d_post_segments[name] = None
    context = failure_context(trace, events, platform=platform)
    # Q025: quality window starts one metre after the relevant zero point.
    failure = events['failure']['index']
    stop = events['stop_request']['index']
    terminal = events['terminal']['index']
    base = 0 if zero_anchor == 'spawn' else zero
    quality = dict(start_index=None, end_index=None, entered=False,
                   completed=False, start_m=None, end_m=None,
                   nominal_length_m=None, metrics=None)
    if base is not None:
        start_target = float(route_x[base]) + s2_spec.PLATFORM_QUALITY_WINDOW_START_M
        starts = np.flatnonzero(route_x[base:] >= start_target)
        if starts.size:
            start_index = base + int(starts[0])
            if zero_anchor == 'spawn':
                end_target = (float(route_x[base])
                              + s2_spec.PLATFORM_QUALITY_WINDOW_END_M)
                ends = np.flatnonzero(route_x[start_index:] >= end_target)
                end_index = start_index + int(ends[0]) if ends.size else None
                limit = _limit(stop, failure, terminal)
                if limit is not None and (end_index is None or limit < end_index):
                    end_index = limit
            else:
                end_index = _limit(stop, failure, terminal)
            quality.update(
                start_index=start_index, end_index=end_index, entered=True,
                completed=bool(end_index is not None
                               and (failure is None or end_index < failure)
                               and (stop is None or end_index <= stop)),
                start_m=float(route_x[start_index] - route_x[base]),
                end_m=(None if end_index is None
                       else float(route_x[end_index] - route_x[base])),
                nominal_length_m=(None if end_index is None
                                  else float(route_x[end_index] - route_x[start_index])),
            )
            quality['metrics'] = segment_metrics(trace, start_index, end_index,
                                                 front_feet=front_feet)
            quality['metrics']['failed'] = bool(
                failure is not None and end_index == failure)
    result = dict(events=events, bounds=bounds, segments=segments,
                quality_window=quality,
                zero_anchor=zero_anchor, zero_index=zero,
                first_meter_zero_index=zero,
                d_post_zero_index=d_post_zero,
                d_post_terminal_m=d_post,
                d_post_segments_m=d_post_segments,
                g_level_delay_seconds=delays,
                first_meter_reached=bool(segments['first_meter']['completed']),
                post_first_meter_reached=bool(segments['post_first_meter']['entered']),
                terminal_step=int(fields['step'][-1]),
                terminal_route_x=float(route_x[-1]),
                max_route_x=float(np.max(route_x)),
                failure_step=events['failure']['step'],
                failure_context=context,
                stop_request_step=events['stop_request']['step'],
                sign_convention=SIGN_CONVENTION)
    if map_evidence is not None:
        attach_map_evidence(result, map_evidence)
    return result


def summarize_group(cases, *, speed=None):
    """Aggregate entered/failed/not-reached/valid-gait counts per segment."""
    if not isinstance(cases, list) or not cases:
        raise ValueError('nonempty case result list required')
    contexts = [row.get('failure_context') or {} for row in cases]
    evidence = [row['map_evidence'] for row in cases if 'map_evidence' in row]
    result = dict(case_count=len(cases), speed=speed,
                  physical_failures=sum(row['failure_step'] is not None for row in cases),
                  failure_on_platform=sum(1 for row in contexts
                                          if row.get('surface') == 'platform'),
                  failure_on_stairs=sum(1 for row in contexts
                                        if row.get('surface') == 'stairs'),
                  failure_on_approach=sum(1 for row in contexts
                                          if row.get('surface') == 'approach'),
                  first_meter_reached=sum(bool(row['first_meter_reached']) for row in cases),
                  post_first_meter_reached=sum(bool(row['post_first_meter_reached'])
                                               for row in cases),
                  segments={})
    if evidence:
        result['map_evidence'] = summarize_map_evidence(evidence)
    for name in SEGMENT_NAMES:
        rows = [row['segments'][name] for row in cases]
        entered = [row for row in rows if row['entered']]
        valid = [row for row in entered if row['gait'] and row['gait']['valid']]
        result['segments'][name] = dict(
            entered=len(entered), failed=sum(row['failed'] for row in rows),
            not_reached=sum(row['not_reached'] for row in rows),
            completed=sum(row['completed'] for row in rows),
            valid_gait_samples=len(valid),
            sample_count=sum(row['sample_count'] for row in entered),
            signed_distance_m_mean=(float(np.mean([row['signed_distance_m']
                                                   for row in entered]))
                                    if entered else None),
            pitch_abs_max_deg_max=(float(np.max([row['pitch_abs_max_deg']
                                                 for row in entered]))
                                   if entered else None),
            stride_median_m=_median([row['gait']['stride_median_m'] for row in valid]),
            frequency_median_hz=_median([row['gait']['frequency_median_hz']
                                         for row in valid]),
        )
    return result


def compare_gait_to_flat(reference, candidate, *, segment='first_meter'):
    """Ratio of a segment's gait medians to the same-speed flat reference."""
    if segment not in SEGMENT_NAMES:
        raise ValueError('unknown segment')
    if reference.get('speed') is not None and candidate.get('speed') is not None:
        if float(reference['speed']) != float(candidate['speed']):
            raise ValueError('flat comparison requires the same cruise speed')
    left = reference['segments'][segment]
    right = candidate['segments'][segment]
    result = dict(segment=segment, speed=reference.get('speed'), metrics={})
    for key in ('stride_median_m', 'frequency_median_hz'):
        before, after = left[key], right[key]
        result['metrics'][key] = dict(
            flat=before, segment=after,
            ratio=(None if before in (None, 0) or after is None else after / before),
            available=before is not None and after is not None,
            missing_reason=(None if before is not None and after is not None
                            else 'flat_or_segment_cycles_insufficient'),
        )
    return result


MAP_EVIDENCE_FIELDS = (
    'branch', 'frames_consumed', 'valid_fraction_numerator',
    'valid_fraction_denominator', 'valid_fraction', 'saturation_numerator',
    'saturation_denominator', 'saturation', 'exact_clip_numerator',
    'exact_clip_denominator', 'exact_clip_fraction', 'event_step',
    'native_sample_id', 'first_replacement_actor_step', 'first_replacement_frame',
    's_minus_t', 'exposed', 'replacement_applied', 'replacement_frames_consumed',
    'sample_reference_count', 'sample_references_sha256',
    'first_sample_reference', 'last_sample_reference',
    'exact_clip_missing_reason', 'missing_reason')


def normalize_map_evidence(evidence):
    """Q023/Q024 record: keep raw numerators/denominators, NA never zero."""
    if not isinstance(evidence, dict):
        raise ValueError('map evidence must be a mapping')
    optional = {'exact_clip_missing_reason', 'missing_reason'}
    missing = [key for key in MAP_EVIDENCE_FIELDS
               if key not in evidence and key not in optional]
    if missing:
        raise ValueError(f'map evidence missing fields: {missing}')
    result = {key: evidence.get(key) for key in MAP_EVIDENCE_FIELDS}
    if result['missing_reason'] is None and not result.get('exposed', False):
        result['missing_reason'] = ('event_not_reached' if result.get('event_step') is None
                                    else 'no_valid_refresh')
    for key in ('valid_fraction_numerator', 'valid_fraction_denominator',
                'saturation_numerator', 'saturation_denominator',
                'exact_clip_numerator', 'frames_consumed',
                'replacement_frames_consumed', 'sample_reference_count'):
        value = result[key]
        if value is not None and (isinstance(value, bool)
                                  or not isinstance(value, int) or value < 0):
            raise ValueError(f'map evidence {key} must be a nonnegative integer or None')
    if result['valid_fraction_denominator'] in (None, 0):
        result['valid_fraction'] = None
    elif result['valid_fraction'] is None:
        result['valid_fraction'] = (result['valid_fraction_numerator']
                                    / result['valid_fraction_denominator'])
    if result['saturation_denominator'] in (None, 0):
        result['saturation'] = None
        result['exact_clip_fraction'] = None
    else:
        if result['saturation'] is None:
            result['saturation'] = (result['saturation_numerator']
                                    / result['saturation_denominator'])
        if result['exact_clip_denominator'] in (None, 0):
            result['exact_clip_fraction'] = None
        elif result['exact_clip_fraction'] is None:
            result['exact_clip_fraction'] = (result['exact_clip_numerator']
                                             / result['exact_clip_denominator'])
    result['zero_valid_denominator'] = result['valid_fraction_denominator'] in (None, 0)
    result['zero_saturation_denominator'] = result['saturation_denominator'] in (None, 0)
    result['not_exposed'] = not bool(result['exposed'])
    native = result['native_sample_id']
    if native is not None:
        from .s2_map_ablation import validate_native_sample_id
        validate_native_sample_id(native)
    return result


def attach_map_evidence(case_result, evidence):
    """Attach a normalized map-evidence record to one case result."""
    case_result['map_evidence'] = normalize_map_evidence(evidence)
    return case_result


def summarize_map_evidence(records):
    """Group sums of raw Q023 numerators/denominators; NA when denominator 0."""
    rows = [normalize_map_evidence(row) for row in records]
    if not rows:
        raise ValueError('nonempty map evidence list required')
    valid_num = sum(row['valid_fraction_numerator'] for row in rows)
    valid_den = sum(row['valid_fraction_denominator'] for row in rows)
    sat_num = sum(row['saturation_numerator'] for row in rows)
    sat_den = sum(row['saturation_denominator'] for row in rows)
    clip_rows = [row for row in rows if row['exact_clip_fraction'] is not None]
    clip_den = sum(row['saturation_denominator'] for row in clip_rows)
    clip_num = sum(row['exact_clip_numerator'] for row in clip_rows)
    return dict(
        cases=len(rows), frames_consumed=sum(row['frames_consumed'] for row in rows),
        valid_fraction_numerator=valid_num,
        valid_fraction_denominator=valid_den,
        valid_fraction=(valid_num / valid_den if valid_den else None),
        saturation_numerator=sat_num, saturation_denominator=sat_den,
        saturation=(sat_num / sat_den if sat_den else None),
        exact_clip_numerator=clip_num,
        exact_clip_denominator=clip_den if clip_rows else None,
        exact_clip_fraction=(clip_num / clip_den if clip_den else None),
        exact_clip_available_cases=len(clip_rows),
        missing_reason_counts={reason: sum(1 for row in rows
                                           if row['missing_reason'] == reason)
                               for reason in ('event_not_reached', 'no_valid_refresh')},
        replacement_frames_consumed=sum(row['replacement_frames_consumed'] for row in rows),
        sample_reference_count=sum(row['sample_reference_count'] for row in rows),
        not_exposed=sum(1 for row in rows if row['not_exposed']),
        zero_valid_denominator=sum(1 for row in rows
                                   if row['zero_valid_denominator']),
        zero_saturation_denominator=sum(1 for row in rows
                                        if row['zero_saturation_denominator']),
    )
