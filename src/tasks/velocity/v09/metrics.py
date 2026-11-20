"""S1 per-episode metrics from complete, explicitly timestamped raw evidence.

Missing gait cycles/windows are reported as missing, never as zero error. These
are absolute case-quality checks; teacher-relative group gates live elsewhere.
"""
from __future__ import annotations

import math
import torch
from . import spec

DT = .02
RAD_TO_DEG = 180 / math.pi


def flat_command(task, stop_mode, cruise, actor_index):
    """Command for zero-based Actor call index (the first call is index 0)."""
    if task == 'flat_walk' or actor_index < 400:
        return cruise
    if task != 'flat_stop':
        raise ValueError('unknown flat task')
    if stop_mode == 'instant':
        return 0.0
    if stop_mode != 'ramp_2s':
        raise ValueError('unknown stop mode')
    return cruise * max(0.0, 1 - (actor_index - 400 + 1) / 100)


def flat_schedule(task, stop_mode):
    if task == 'flat_walk':
        return dict(completion_steps=1000, hard_timeout_steps=1001, hold_actor_index=None)
    if task != 'flat_stop' or stop_mode not in ('instant', 'ramp_2s'):
        raise ValueError('invalid flat task/stop mode')
    if stop_mode == 'ramp_2s':
        # Planning Q002/B: k100 at index499 is zero but still DECEL;
        # formal HOLD is indices500..1499, recovery starts at index499.
        decision = spec.FLAT_RAMP_TIMING_DECISION
        hold, completion, first_zero = (decision['hold_actor_index'], decision['completion_steps'],
                                        decision['first_zero_actor_index'])
    else:
        hold, completion, first_zero = 400, 1400, 400
    return dict(completion_steps=completion, hard_timeout_steps=completion + 1,
                hold_actor_index=hold, first_zero_actor_index=first_zero)


def support_from_geometry(feet_xy, top_forces, geom_bounds, geom_tile, tile_ids,
                          previous_count, inset=.042):
    """Top upward force plus exact inset XY eligibility; two-frame debounce."""
    x, y = feet_xy.unbind(-1)
    x0, x1, y0, y1 = geom_bounds.unbind(-1)
    safe = ((x[..., None] >= x0 + inset) & (x[..., None] <= x1 - inset)
            & (y[..., None] >= y0 + inset) & (y[..., None] <= y1 - inset)
            & (geom_tile[None, None] == tile_ids[:, None, None]))
    # Different top faces are not summed into fictitious support.
    upward = torch.where(safe, top_forces, -torch.inf).max(-1).values
    raw = upward > 5
    counts = torch.where(raw, previous_count + 1, 0)
    return counts >= 2, counts


def _median(values):
    return torch.quantile(values.float(), .5).item() if values.numel() else None


class EpisodeTraceBuffer:
    """Device-resident raw evidence, bounded by the frozen maximum hard timeout.

    Allocation and terminal CPU export are explicit so resource admission includes
    the actual metric cost. No per-step GPU-to-CPU trajectory transfer is used.
    """
    def __init__(self, num_envs, max_steps, device='cpu'):
        self.num_envs, self.max_steps = int(num_envs), int(max_steps)
        self.device = torch.device(device)
        self.fields = {}
        self.counts = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.initial_position = torch.zeros(num_envs, 3, device=device)

    @property
    def storage_bytes(self):
        tensors = [*self.fields.values(), self.counts, self.initial_position]
        return sum(t.numel() * t.element_size() for t in tensors)

    @torch.no_grad()
    def append(self, frame):
        if bool((self.counts >= self.max_steps).any()):
            raise RuntimeError('episode evidence exceeded frozen maximum hard timeout')
        ids = torch.arange(self.num_envs, device=self.device)
        for key, value in frame.items():
            if key == 'previous_action':
                continue  # reconstructed exactly from raw tokens plus reset zero
            value = value.detach()
            if value.shape[0] != self.num_envs:
                raise ValueError('evidence batch size differs')
            if key not in self.fields:
                self.fields[key] = torch.empty((self.num_envs, self.max_steps, *value.shape[1:]),
                                               dtype=value.dtype, device=self.device)
            self.fields[key][ids, self.counts] = value
        self.counts += 1

    def reset(self, ids):
        self.counts[ids] = 0

    def evidence(self, env_id):
        n = int(self.counts[env_id])
        result = {key: value[env_id, :n].detach().cpu().clone() for key, value in self.fields.items()}
        result['initial_position'] = self.initial_position[env_id].detach().cpu().clone()
        action = result['action']
        result['previous_action'] = torch.cat((torch.zeros_like(action[:1]), action[:-1]), 0)
        return result


def cruise_stall(evidence, cruise):
    """Planning Q003: exact 51-position window including true segment x0."""
    e = evidence
    if 'initial_position' not in e:
        raise ValueError('stall evidence must include actual reset/segment initial position')
    x = e['position'][:, 0].float()
    command = e.get('command', torch.full_like(x, cruise))
    enabled = e.get('is_cruise', torch.ones_like(x, dtype=torch.bool)).bool() & (command > .1)
    previous_x = float(e['initial_position'][0])
    positions, highwaters = [], []
    count = maximum = 0
    latched_step = None
    for i in range(len(x)):
        current = float(x[i])
        if not bool(enabled[i]):
            positions, highwaters, count = [], [], 0
        else:
            if not positions:
                positions, highwaters = [previous_x], [previous_x]
            positions.append(current)
            highwaters.append(max(highwaters[-1], current))
            eligible = len(positions) >= 51
            qualifies = (eligible and positions[-1] - positions[-51] < .15 * float(command[i])
                         and highwaters[-1] - highwaters[-51] < .03)
            count = count + 1 if qualifies else 0
            maximum = max(maximum, count)
            if count >= 50 and latched_step is None:
                latched_step = int(e['step'][i])
            positions, highwaters = positions[-51:], highwaters[-51:]
        previous_x = current
    return dict(stall_max_consecutive_frames=maximum, stall_latched_step=latched_step,
                cruise_stall_latched=latched_step is not None)


def summarize_trajectory(evidence, *, task, stop_mode, cruise, physical_failure=False):
    """All arrays are pre-reset samples at the ends of physical control steps.

    Required fields: step, position[3], roll, pitch, velocity_b[3], velocity_w[3],
    omega[3], foot_x[4], foot_velocity_xy[4,2], raw_contact[4], filtered[4],
    support[4], action[12], previous_action[12], height, nonfoot_contact.
    The evidence mapping itself remains available to callers for independent
    recomputation; returned statistics include exact denominators and cycles.
    """
    e = {key: torch.as_tensor(value).detach().cpu() for key, value in evidence.items()}
    step = e['step'].long()
    if not step.numel() or bool((step[1:] != step[:-1] + 1).any()):
        raise ValueError('complete contiguous control-step evidence is required')
    cruise_end = 1000 if task == 'flat_walk' else 400 if task == 'flat_stop' else int(step[-1])
    window = (step > 100) & (step <= cruise_end) if task.startswith('flat_') else torch.ones_like(step, dtype=torch.bool)
    n = int(window.sum())
    required_cruise_samples = 900 if task == 'flat_walk' else 300 if task == 'flat_stop' else 50
    valid_window = (n == required_cruise_samples if task.startswith('flat_') else n >= required_cruise_samples)
    result = dict(sample_count=len(step), cruise_sample_count=n,
                  cruise_required_sample_count=required_cruise_samples, cruise_window_valid=valid_window,
                  speed_rmse=None, pitch_rms_degrees=None, pitch_p99_degrees=None,
                  roll_rms_degrees=None, base_height_mean=float(e['height'].float().mean()),
                  slip=None, action_change=None, stride=None, frequency=None,
                  gait_valid=False, gait_cycles=[], physical_failure=bool(physical_failure))
    quality_reasons = []
    result.update(cruise_stall(e, cruise))
    if result['cruise_stall_latched'] and task.startswith('flat_'):
        quality_reasons.append('cruise_stall')
    if valid_window:
        velocity = e['velocity_b'][window].float()
        # The original 2D speed-tracking contract includes lateral velocity.
        result['speed_rmse'] = float(((velocity[:, 0] - cruise).square() + velocity[:, 1].square()).mean().sqrt())
        pitch = e['pitch'][window].float() * RAD_TO_DEG
        roll = e['roll'][window].float() * RAD_TO_DEG
        result['pitch_rms_degrees'] = float(pitch.square().mean().sqrt())
        result['pitch_p99_degrees'] = float(torch.quantile(pitch.abs(), .99))
        result['roll_rms_degrees'] = float(roll.square().mean().sqrt())
        slip = (e['filtered'][window].float() * e['foot_velocity_xy'][window].float().square().sum(-1)).sum(-1)
        delta = (e['action'][window] - e['previous_action'][window]).float().square().sum(-1)
        result['slip'], result['action_change'] = float(slip.mean()), float(delta.mean())
        over_count = maximum_over = 0
        for over in (pitch.abs() > 15).tolist():
            over_count = over_count + 1 if over else 0
            maximum_over = max(maximum_over, over_count)
        result['pitch_over_15_max_frames'] = maximum_over
        for failed, name in ((result['pitch_rms_degrees'] > 5, 'pitch_rms'),
                             (result['pitch_p99_degrees'] > 10, 'pitch_p99'),
                             (maximum_over >= 10, 'pitch_continuous'),
                             (result['speed_rmse'] > .10 + 1e-8, 'speed_rmse'),
                             (bool(e['nonfoot_contact'][window].any()), 'nonfoot_contact')):
            if failed:
                quality_reasons.append(name)
    else:
        quality_reasons.append('missing_cruise_window')
    per_foot_stride, per_foot_frequency = [], []
    cycles_valid = True
    contact = e['raw_contact'].bool()
    for foot in range(4):
        # A reset's initial contact is not a synthetic landing event.
        landings = ((contact[1:, foot] & ~contact[:-1, foot]).nonzero().flatten() + 1)
        landings = landings[window[landings]]
        cycles = []
        for start, end in zip(landings[:-1].tolist(), landings[1:].tolist()):
            period = float((step[end] - step[start]) * DT)
            stride = float(e['foot_x'][end, foot] - e['foot_x'][start, foot])
            cycles.append(dict(start_step=int(step[start]), end_step=int(step[end]),
                               forward_distance=stride, period_seconds=period,
                               frequency_hz=1 / period))
        result['gait_cycles'].append(cycles)
        cycles_valid &= len(cycles) >= 2
        per_foot_stride.extend(row['forward_distance'] for row in cycles)
        per_foot_frequency.extend(row['frequency_hz'] for row in cycles)
    cycles_valid &= valid_window
    result['gait_valid'] = bool(cycles_valid)
    if cycles_valid:
        result['stride'] = _median(torch.tensor(per_foot_stride))
        result['frequency'] = _median(torch.tensor(per_foot_frequency))
    if task == 'flat_stop':
        schedule = flat_schedule(task, stop_mode)
        hold_actor = schedule['hold_actor_index']
        # A step-end sample belongs to the command used at Actor index step-1.
        hold = (step - 1 >= hold_actor)
        stable = ((e['velocity_w'][:, :2].norm(dim=-1) <= .05)
                  & (e['roll'].abs() <= 10 / RAD_TO_DEG)
                  & (e['pitch'].abs() <= 10 / RAD_TO_DEG)
                  & (e['omega'].norm(dim=-1) <= .25) & (e['height'] >= .30)
                  & e.get('height_valid', torch.ones_like(step, dtype=torch.bool)).bool()
                  & e['support'].bool().all(-1)
                  & (e['foot_velocity_xy'].norm(dim=-1).mean(-1) <= .05))
        consecutive, recovery = 0, None
        first_zero = schedule['first_zero_actor_index']
        recovery_window = step - 1 >= first_zero
        for i in recovery_window.nonzero().flatten().tolist():
            consecutive = consecutive + 1 if bool(stable[i]) else 0
            elapsed = (int(step[i]) - first_zero) * DT
            if consecutive >= 25:
                recovery = elapsed
                break
        last = (step > schedule['completion_steps'] - 500) & (step <= schedule['completion_steps'])
        result['hold_sample_count'] = int(hold.sum())
        result['stand_sample_count'] = int(last.sum())
        result['recovery_seconds'] = recovery
        result['stand_window_valid'] = int(last.sum()) == 500
        result['stand_max_displacement'] = None
        if recovery is None or recovery > 3 + 1e-9:
            quality_reasons.append('recovery')
        if int(hold.sum()) < 1000 or not result['stand_window_valid']:
            quality_reasons.append('missing_hold_window')
        if result['stand_window_valid']:
            indices = last.nonzero().flatten()
            # Position at the *start* of the interval preceding the first sample.
            origin_index = int(indices[0]) - 1
            if origin_index < 0:
                raise ValueError('standing window lacks its true start position')
            displacement = (e['position'][last, :2] - e['position'][origin_index, :2]).norm(dim=-1)
            speed = e['velocity_w'][last, :2].norm(dim=-1)
            result.update(stand_mean_speed=float(speed.mean()), stand_peak_speed=float(speed.max()),
                          stand_max_displacement=float(displacement.max()),
                          stand_support_fraction=float(e['support'][last].bool().all(-1).float().mean()))
            for failed, name in ((result['stand_mean_speed'] > .05 + 1e-8, 'stand_mean_speed'),
                                 (result['stand_peak_speed'] > .10 + 1e-8, 'stand_peak_speed'),
                                 (result['stand_max_displacement'] > .10 + 1e-8, 'stand_displacement'),
                                 (bool((e['roll'][last].abs() > 10 / RAD_TO_DEG).any()), 'stand_roll'),
                                 (bool((e['pitch'][last].abs() > 10 / RAD_TO_DEG).any()), 'stand_pitch'),
                                 (bool((e['height'][last] < .30).any()), 'stand_height'),
                                 (not bool(e.get('height_valid', torch.ones_like(step, dtype=torch.bool))[last].all()), 'stand_height_missing'),
                                 (result['stand_support_fraction'] < .90 - 1e-8, 'stand_support')):
                if failed:
                    quality_reasons.append(name)
    result['quality_reasons'] = quality_reasons
    result['quality_pass'] = not quality_reasons and not physical_failure
    # Gait absence separately blocks relative group acceptance, rather than
    # changing the SDS absolute per-case quality/reward definition.
    result['relative_metrics_complete'] = valid_window and bool(cycles_valid)
    # Q001: group gates select each metric's teacher-valid ID set independently.
    result['pitch_rms_deg'] = result['pitch_rms_degrees']
    result['drift'] = result.get('stand_max_displacement')
    result['metric_validity'] = {}
    for key in ('speed_rmse', 'pitch_rms_deg', 'slip', 'action_change', 'drift', 'stride', 'frequency'):
        valid_metric = result.get(key) is not None
        reason = (None if valid_metric else 'not_applicable' if key == 'drift' and task != 'flat_stop'
                  else 'insufficient_landing_cycles' if key in ('stride', 'frequency')
                  else 'incomplete_standing_window' if key == 'drift' else 'incomplete_cruise_window')
        result['metric_validity'][key] = dict(valid=valid_metric, missing_reason=reason)
    result['group_metrics'] = {key: dict(value=result.get(key), **status)
                               for key, status in result['metric_validity'].items()}
    return result
