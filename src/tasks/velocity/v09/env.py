"""S1 adapter around unchanged v07 physics/termination for all 22 stair tasks.

Flat tasks use complete quality windows. Their rewards are legacy diagnostics;
S1 learning must use only supervised teacher labels. This module creates no
simulation at import time, and synthetic tests never construct the environment.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math

import torch

from src.tasks.velocity.v07.env import V07Env, StallDiagnostics
from src.tasks.velocity.v07.events import euler_heading
from src.tasks.velocity.v07 import spec as original_spec
from src.tasks.velocity.v07 import terrain
from . import spec
from .cases import stream_id
from .map import PairedHeightMaps
from .metrics import (EpisodeTraceBuffer, flat_command, flat_schedule,
                      summarize_trajectory, support_from_geometry)

QUALITY_FAILURE_CODE = 11
FLAT_TASK_TIMEOUT_CODE = 12


class EpisodeIds:
    """External per-role monotonic IDs; retain this object when N changes."""
    def __init__(self, next_by_role=None):
        self.next_by_role = dict(next_by_role or {})

    def reserve(self, role):
        current = int(self.next_by_role.get(role, 0))
        if current < 0:
            raise ValueError('negative episode id')
        self.next_by_role[role] = current + 1
        return current

    def state_dict(self):
        return dict(self.next_by_role)


class S1StallDiagnostics(StallDiagnostics):
    """Preserve old stair diagnostics, disable old F0's fixed-command assumption."""
    def __init__(self, num_envs, device, flat_mask, cruise):
        super().__init__(num_envs, device)
        self.flat_mask, self.cruise = flat_mask, cruise

    def update(self, local_x, executed_c, climbed, first_step_acquired, enabled):
        # Flat walk/stop quality is measured independently on the executed
        # command. The obsolete F0 stall diagnostic is never used for flat.
        c = torch.where(self.flat_mask, self.cruise, executed_c)
        return super().update(local_x, c, climbed, first_step_acquired,
                              enabled & ~self.flat_mask)


class V09S1Env(V07Env):
    """Explicit manifest/provider cases; no slot-dependent spawn/command RNG.

    ``case_provider(role, episode_id)`` must return a frozen-rule case using its
    independent named streams. Fixed evaluation cases can omit the provider.
    All actual simulator steps, including done-slot padding, remain visible to
    the caller's external budget ledger. No step is silently skipped.
    """
    def __init__(self, cases, *, device='cuda:0', seed=42, render_mode=None,
                 roles=None, case_provider=None, episode_ids=None, max_trace_steps=None,
                 course='S1.1'):
        cases = [deepcopy(case) for case in cases]
        if not cases or seed != 42:
            raise ValueError('S1 requires explicit nonempty cases and training seed 42')
        for case in cases:
            self._validate_case(case)
        self._v09_ready = False
        self.v09_cases = cases
        self.v09_roles = list(roles or [case['task'] for case in cases])
        if len(self.v09_roles) != len(cases):
            raise ValueError('roles/cases length differs')
        self.case_provider = case_provider
        if course not in ('S1.1', 'S1.2'):
            raise ValueError('only S1.1/S1.2 course is supported')
        self.course = course
        self.episode_ids = episode_ids if episode_ids is not None else EpisodeIds()
        self.case_episode_ids = [-1] * len(cases)
        self._completed_evidence, self._completed_metrics = {}, {}
        initial = [self._pose_row(case) for case in cases]
        super().__init__(num_envs=len(cases), device=device, seed=seed, phase='L',
                         substage='L40', initial_cases=initial, render_mode=render_mode)
        self.recipe = spec.RECIPE_VERSION
        self.protocol = 'a2-v09-s1-native-map-distillation'
        self.roles = list(self.v09_roles)
        self.role_ids = self.roles
        self.paired_maps = PairedHeightMaps(len(cases), device)
        self.map_cache = self.paired_maps.student
        self.flat_mask = torch.zeros(len(cases), dtype=torch.bool, device=device)
        self.flat_stop_mask = torch.zeros_like(self.flat_mask)
        self.ramp_mask = torch.zeros_like(self.flat_mask)
        self.completion_steps = torch.zeros(len(cases), dtype=torch.long, device=device)
        self.hard_timeout_steps = torch.zeros_like(self.completion_steps)
        self.teacher_control = torch.zeros_like(self.flat_mask)
        self.controller_streams = [''] * len(cases)
        self.support_count = torch.zeros(len(cases), 4, dtype=torch.long, device=device)
        # Include every old stair task, even if the first training allocation
        # happens not to contain the longest episode.
        largest = max(original_spec.episode_step_limit(tile.tile_id, speed)
                      for tile in original_spec.TERRAIN_TABLE for speed in spec.SPEEDS)
        largest = max(largest, 1501)
        trace_steps = largest if max_trace_steps is None else int(max_trace_steps)
        if trace_steps < max(case['hard_timeout_steps'] for case in cases):
            raise ValueError('trace capacity cannot truncate current case evidence')
        self.trace = EpisodeTraceBuffer(len(cases), trace_steps, device)
        self.stall = S1StallDiagnostics(len(cases), device, self.flat_mask, self.start_vx)
        self._v09_ready = True
        self.reset()

    @staticmethod
    def _validate_case(case):
        if (case.get('stage') != 'S1' or case.get('task') not in ('flat_walk', 'flat_stop', 'stairs')
                or case.get('speed') not in spec.SPEEDS):
            raise ValueError('only frozen S1 tasks/speeds are supported')
        if case['task'].startswith('flat_'):
            schedule = flat_schedule(case['task'], case['stop_mode'])
            if case['hard_timeout_steps'] != schedule['hard_timeout_steps']:
                raise ValueError('flat timeout differs from frozen timing contract')
        if not case.get('streams') or not case.get('case_sha256'):
            raise ValueError('case requires named streams and immutable identity')

    @staticmethod
    def _pose_row(case):
        return [*case['spawn']['local_position_xy'], case['spawn']['local_yaw_radians'], case['speed']]

    def _reset_ids(self, ids, *, resample_tiles=True):
        if not self._v09_ready:
            return super()._reset_ids(ids, resample_tiles=resample_tiles)
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long).flatten()
        for i in ids.tolist():
            episode_id = self.episode_ids.reserve(self.v09_roles[i])
            self.case_episode_ids[i] = episode_id
            if self.case_provider is not None:
                case = deepcopy(self.case_provider(self.v09_roles[i], episode_id))
                self._validate_case(case)
                self.v09_cases[i] = case
            case = self.v09_cases[i]
            row = self._pose_row(case)
            self.initial_cases[i] = torch.tensor(row[:3], device=self.device)
            self.initial_command[i] = row[3]
            self.tile_ids[i] = int(case['geometry']['tile_id'])
            self.flat_mask[i] = case['task'] != 'stairs'
            self.flat_stop_mask[i] = case['task'] == 'flat_stop'
            self.ramp_mask[i] = case['stop_mode'] == 'ramp_2s'
            if case['task'] != 'stairs':
                self.completion_steps[i] = flat_schedule(case['task'], case['stop_mode'])['completion_steps']
            self.hard_timeout_steps[i] = case['hard_timeout_steps']
            controller = stream_id(role=self.v09_roles[i], episode_id=episode_id,
                                   channel='teacher_controller', suite='training')
            self.controller_streams[i] = controller
            fraction = (int(controller[:16], 16) >> 11) / float(1 << 53)
            probability = .50 if self.course == 'S1.1' else .10
            self.teacher_control[i] = self.case_provider is not None and fraction < probability
        self.trace.reset(ids)
        self.support_count[ids] = 0
        # Every random quantity was already frozen by case streams; parent
        # writes default q0/zero velocity and retains all physical parameters.
        super()._reset_ids(ids, resample_tiles=False)
        self.trace.initial_position[ids] = self.robot.data.root_link_pos_w[ids]
        self.paired_maps.reset(self.robot.data.root_link_pos_w,
                               euler_heading(self.robot.data.root_link_quat_w)[2], self.tile_ids, ids)

    def _observations(self, update=True):
        if not self._v09_ready:
            return super()._observations(update=update)
        # Parent assembles the exact original privileged35 and current actual
        # action token; update=False prevents a second history or map write.
        shared = super()._observations(update=False)
        if update:
            self.history_cache.update(shared['proprio'])
            self.paired_maps.update(self.robot.data.root_link_pos_w,
                                    euler_heading(self.robot.data.root_link_quat_w)[2], self.tile_ids)
        shared['history'] = self.history_cache.history.clone()
        teacher, student = self.paired_maps.observations(shared)
        self.teacher_observation = {k: v.clone() for k, v in teacher.items()}
        if update:
            self._terminal_teacher = {k: v.clone() for k, v in teacher.items()}
        self.student_observation = {k: v.clone() for k, v in student.items()}
        return self.student_observation

    def observation_pair(self):
        """Read the already assembled same-state snapshots, without FIFO writes."""
        return self.teacher_observation, self.student_observation

    def set_course(self, course):
        if course not in ('S1.1', 'S1.2'):
            raise ValueError('only S1.1/S1.2 course is supported')
        self.course = course

    def _metric_frame(self, state):
        data = self.robot.data
        support, self.support_count = support_from_geometry(
            data.site_pos_w[:, self.foot_ids, :2], self.contacts.top_forces,
            self.geom_step_bounds, self.geom_tile, self.tile_ids, self.support_count)
        height, valid = terrain.query_surface(data.root_link_pos_w[:, :2], self.tile_ids)
        frame = dict(step=self.episode_length_buf.clone(), position=data.root_link_pos_w.clone(),
                     roll=state.roll, pitch=state.pitch, velocity_b=data.root_link_lin_vel_b.clone(),
                     velocity_w=data.root_link_lin_vel_w.clone(), omega=data.root_link_ang_vel_b.clone(),
                     foot_x=data.site_pos_w[:, self.foot_ids, 0].clone(),
                     foot_velocity_xy=data.site_lin_vel_w[:, self.foot_ids, :2].clone(),
                     raw_contact=self.contacts.raw.clone(), filtered=self.contacts.filtered.clone(),
                     support=support, action=self.action_manager.action.clone(),
                     height=torch.where(valid, data.root_link_pos_w[:, 2] - height, 0.),
                     height_valid=valid,
                     nonfoot_contact=self.control_body_hit | self.control_long_hit,
                     is_cruise=~self.flat_stop_mask | (self.episode_length_buf - 1 < 400),
                     command=self.sample_command[:, 0].clone())
        return frame

    def _task_state(self):
        state = super()._task_state()
        if not self._v09_ready:
            return state
        self.trace.append(self._metric_frame(state))
        self._physical_failure = state.failure.clone()
        success, failure, timeout, reason = (state.success.clone(), state.failure.clone(),
                                             state.timeout.clone(), state.reason.clone())
        self._flat_outcome = {}
        success[self.flat_mask], timeout[self.flat_mask] = False, False
        complete = self.flat_mask & (self.episode_length_buf == self.completion_steps) & ~state.failure
        hard_timeout = self.flat_mask & (self.episode_length_buf >= self.hard_timeout_steps) & ~state.failure
        failure |= hard_timeout
        reason = torch.where(hard_timeout, FLAT_TASK_TIMEOUT_CODE, reason)
        flat_terminal = self.flat_mask & (state.failure | complete | hard_timeout)
        for i in flat_terminal.nonzero().flatten().tolist():
            case = self.v09_cases[i]
            if bool(state.failure[i]):
                self._flat_outcome[i] = 'physical_failure'
            elif bool(complete[i]):
                evidence = self.trace.evidence(i)
                metrics = summarize_trajectory(evidence, task=case['task'], stop_mode=case['stop_mode'], cruise=case['speed'])
                self._completed_evidence[i], self._completed_metrics[i] = evidence, metrics
                success[i], failure[i] = metrics['quality_pass'], not metrics['quality_pass']
                reason[i] = 2 if metrics['quality_pass'] else QUALITY_FAILURE_CODE
                self._flat_outcome[i] = 'success' if metrics['quality_pass'] else 'quality_failure'
            else:
                self._flat_outcome[i] = 'task_timeout'
        # At step end, prepare command for next Actor index=completed steps.
        stop = self.flat_stop_mask & (self.episode_length_buf >= 400)
        next_v = torch.where(self.ramp_mask,
                              self.start_vx * (1 - (self.episode_length_buf - 399) / 100).clamp(0, 1), 0.)
        self.command[:, 0] = torch.where(stop, next_v, self.command[:, 0])
        self.command[self.flat_mask, 1:] = 0
        return replace(state, success=success, failure=failure, timeout=timeout, reason=reason)

    def _flat_pass(self, state):
        return self.flat_mask & state.success if self._v09_ready else super()._flat_pass(state)

    def _episode_records(self, state, done):
        legacy_state = replace(state, reason=torch.where(state.reason > 10, 1, state.reason))
        records = super()._episode_records(legacy_state, done)
        if not self._v09_ready:
            return records
        for record in records:
            i, case = record['env_id'], self.v09_cases[record['env_id']]
            if i not in self._completed_evidence:
                evidence = self.trace.evidence(i)
                metrics = summarize_trajectory(evidence, task=case['task'], stop_mode=case['stop_mode'],
                                               cruise=case['speed'], physical_failure=bool(self._physical_failure[i]))
                self._completed_evidence[i], self._completed_metrics[i] = evidence, metrics
            if case['task'] != 'stairs':
                record['original_physical_reason'] = record['reason'] if bool(self._physical_failure[i]) else None
                record['reason'] = self._flat_outcome[i]
                record['timeout'] = record['reason'] == 'task_timeout'
                record['truncated'] = False
            record.update(stage='S1', adapter_course=self.course, task=case['task'], stop_mode=case['stop_mode'], case_sha256=case['case_sha256'],
                          streams=deepcopy(case['streams']), episode_id=self.case_episode_ids[i],
                          reason_code=int(state.reason[i]),
                          controller_stream=self.controller_streams[i], teacher_control=bool(self.teacher_control[i]),
                          metrics=deepcopy(self._completed_metrics[i]), code_version=spec.CODE_VERSION,
                          recipe=spec.RECIPE_VERSION, sds_sha256=spec.SDS_SHA256)
        return records

    def _outcomes(self, state, done):
        legacy_state = replace(state, reason=torch.where(state.reason > 10, 1, state.reason))
        rows = super()._outcomes(legacy_state, done)
        if self._v09_ready:
            for row in rows:
                i = row['env_id']
                if i in self._flat_outcome:
                    row['reason'] = self._flat_outcome[i]
                    row['timeout'] = row['reason'] == 'task_timeout'
        return rows

    def step(self, action):
        self._completed_evidence, self._completed_metrics = {}, {}
        obs, reward, terminated, truncated, info = super().step(action)
        # Original stair timeout/termination stays unchanged. New flat task
        # timeout and quality failure are already true terminals in state.
        info['terminal_teacher_observation'] = self._terminal_teacher
        info['teacher_observation'] = self.teacher_observation
        info['v09_evidence'] = self._completed_evidence
        info['v09_flat_outcomes'] = dict(self._flat_outcome)
        info['metrics_storage_bytes'] = self.trace.storage_bytes
        info['sds_sha256'] = spec.SDS_SHA256
        info['code_version'], info['recipe_version'] = spec.CODE_VERSION, spec.RECIPE_VERSION
        return obs, reward, terminated, truncated, info
