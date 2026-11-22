"""SDS 0.4 T3: environment equivalence, sums and bounded diagnostics.

The fixtures construct only source objects (``object.__new__``) and mocked
simulator I/O; no simulator, GPU, W&B or training is started.  Every test runs
under the v04 bootstrap so the legacy ``src.tasks`` scanner is not executed.
"""
import dataclasses
import math
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / 'scripts')):
    if entry not in sys.path:
        sys.path.insert(0, entry)
import a2_v04_bootstrap  # noqa: E402

if not a2_v04_bootstrap.is_active(str(ROOT)):
    a2_v04_bootstrap.activate(str(ROOT))

from src.tasks.velocity.v02.rewards import (  # noqa: E402
    FirstFootfallState, ParkingState, ProgressState)
from src.tasks.velocity.v02.terrain import atlas_metadata  # noqa: E402
# The frozen v02 environment config imports the A2 registry.  Load the registry
# first (the same order as a normal src.tasks start) so the bootstrap namespace
# does not expose the legacy config <-> v02.env_cfg cycle.
import src.tasks.velocity.config.a2  # noqa: E402,F401
from src.tasks.velocity.v02_formal.env import FormalEnv as FrozenEnv  # noqa: E402
from src.tasks.velocity.v02_formal.success import ClimbState, ParkingDiagnosticState  # noqa: E402
from src.tasks.velocity.v02_formal.spec import recipe_config  # noqa: E402
from src.tasks.velocity.v03.speed import SegmentSpeedState  # noqa: E402
from src.tasks.velocity.v04.env import (  # noqa: E402
    ContactWindowRing, GATE_REASON_NAMES, V04Env)
from src.tasks.velocity.v04.rewards import (  # noqa: E402
    SwingClearanceState, event_reward, nonfoot_reward, orientation_reward,
    speed_reward)
from src.tasks.velocity.v04.spec import RECIPE_VERSION, reward_config_sha256  # noqa: E402


def _v04(cls):
    return isinstance(cls, type) and issubclass(cls, V04Env)


def fixture(cls, batch=2):
    """Physical sample fixture shared by frozen-v02, v03 and v04 classes."""
    e = object.__new__(cls)
    stem = torch.tensor([[[1.7, .1, .352], [1.7, -.1, .352],
                          [1.2, .1, .352], [1.2, -.1, .352]]] * batch)
    d = S(root_link_pos_w=torch.tensor([[1.6, 0., .8]] * batch),
          root_link_quat_w=torch.tensor([[1., 0., 0., 0.]] * batch),
          site_pos_w=stem,
          root_link_lin_vel_b=torch.tensor([[.4, 0., 0.]] * batch),
          root_link_lin_vel_w=torch.tensor([[.35, 0., 0.]] * batch),
          root_link_ang_vel_b=torch.zeros(batch, 3), root_link_ang_vel_w=torch.zeros(batch, 3),
          joint_pos=torch.zeros(batch, 12), joint_vel=torch.zeros(batch, 12))
    e.robot = S(data=d)
    e.foot_ids = list(range(4))
    e.origins = torch.zeros(10, 3)
    e.tile_ids = torch.full((batch,), 5, dtype=torch.long)
    e.command = torch.tensor([[.4, 0., 0.]] * batch)
    e.geom_top = torch.tensor([.32, .32, .32])
    e.geom_step = torch.tensor([4, 4, 4])
    e.geom_tile = torch.tensor([5, 5, 6])
    e.bounds = torch.tensor([[.6, .8, -.4, .4], [.8, 2.8, -.4, .4], [.8, 2.8, -.4, .4]])
    e.geom_step_bounds = e.bounds.clone()
    tops = torch.zeros(batch, 4, 3)
    tops[:, :, 1] = 6.
    e.contacts = S(top_forces=tops, link_force=torch.zeros(batch, 1, 3),
                   body_link_mask=torch.tensor([True]), nonfoot_duration=torch.zeros(batch, 1),
                   found=torch.ones(batch, 4, dtype=torch.bool))
    e.control_body_hit = torch.zeros(batch, dtype=torch.bool)
    e.control_long_hit = e.control_body_hit.clone()
    e.climb = ClimbState(batch)
    e.parking_diagnostics = ParkingDiagnosticState(batch)
    e.parking_diagnostic = False
    e.stop_steps = torch.zeros(batch, dtype=torch.long)
    e.episode_length_buf = torch.zeros(batch, dtype=torch.long)
    e.start_vx = torch.full((batch,), .4)
    e.scene = S(num_envs=batch)
    e.initial_pose = torch.tensor([[-1., 0., 0.]] * batch)
    e.slot_classes = torch.ones(batch, dtype=torch.long)
    e.evaluation = False
    e.run_seed = 42
    e.checkpoint_identity = 'fixture'
    e.stage = 'S1'
    e.stage_iteration = 0
    e.previous_stage = None
    e._executed_command = None
    e.sim = S(device='cpu', data=S(qpos=torch.zeros(batch, 19), qvel=torch.zeros(batch, 18)))
    e.cfg = S(sim=S(mujoco=S(timestep=.005)), decimation=4)
    if _v04(cls):
        V04Env._allocate_task_state(e, batch, 'cpu')
        e.speed_state = SegmentSpeedState(batch)
    return e


def task_step(e):
    e._executed_command = e.command.clone()
    e.episode_length_buf += 1
    return e._task_state()


def fake_swing(batch, *, gate=True, reward=0.):
    zeros = torch.zeros(batch, 4)
    return S(positive_gate=torch.full((batch,), bool(gate)),
             reward=torch.full((batch,), float(reward)),
             target_step=torch.full((batch, 4), -1, dtype=torch.long),
             selected_step=torch.full((batch, 4), -1, dtype=torch.long),
             phi=zeros.clone(), increment=zeros.clone(), cumulative_paid=zeros.clone(),
             consumed_fraction=zeros.clone(), active=torch.zeros(batch, 4, dtype=torch.bool),
             safe_support=torch.zeros(batch, 4, dtype=torch.bool),
             gate_details=dict(reason=torch.zeros(batch, dtype=torch.long)))


def reward_fixture(cls, batch=3):
    """Frozen reward inputs shared by v02 and v04; no complete step machinery."""
    e = fixture(cls, batch)
    d = e.robot.data
    d.default_joint_pos = torch.zeros(batch, 12)
    d.joint_acc = torch.zeros(batch, 12)
    d.site_lin_vel_w = torch.zeros(batch, 4, 3)
    d.soft_joint_pos_limits = torch.tensor([[[-1., 1.]] * 12] * batch)
    e.scene = {'robot/root_angmom': S(data=torch.full((batch, 3), .1))}
    e.joint_ids = list(range(12))
    e.action_manager = S(action=torch.full((batch, 12), .02))
    e.sample_command = e.command.clone()
    e.recipe_options = recipe_config()
    e.recipe, e.mode = 'C', 'formal'
    e.footfall = FirstFootfallState(batch, 40)
    e.progress = ProgressState(batch)
    e.progress.reset(torch.arange(batch), torch.zeros(batch))
    e.stop = torch.zeros(batch, dtype=torch.bool)
    e.last_contact = torch.zeros(batch, 4, dtype=torch.bool)
    e.episode_reward_sums = {}
    e.slip_sum = torch.zeros(batch)
    e.nonfoot_sum = torch.zeros(batch)
    e.limit_sum = torch.zeros(batch)
    e.contacts.support_geom = torch.zeros(batch, 4, dtype=torch.long)
    e.contacts.support = torch.ones(batch, 4, dtype=torch.bool)
    e.contacts.contact_duration = torch.full((batch, 4), .02)
    e.contacts.foot_force = torch.full((batch, 4, 3), 6.)
    e.contacts.link_names = ['base_link']
    e.climb_event = torch.tensor([True, False, False])[:batch]
    state = (torch.tensor([[.1, .01, .5]] * batch), torch.full((batch,), .02),
             torch.full((batch,), .03), torch.full((batch,), .04),
             torch.ones(batch, dtype=torch.bool), e.climb_event.clone(),
             torch.tensor([False, False, True])[:batch],
             torch.zeros(batch, dtype=torch.bool), torch.zeros(batch, dtype=torch.long))
    return e, state, torch.zeros(batch, 12)


def complete_fixture(cls, batch=2):
    """Only simulator I/O is faked; frozen reset/step/reward/safety are real."""
    e = fixture(cls, batch)
    d = e.robot.data
    e.speed_state = SegmentSpeedState(batch)

    class Scene(dict):
        num_envs = batch

        def reset(self, ids):
            pass

        def write_data_to_sim(self):
            pass

        def update(self, dt):
            pass

    e.scene = Scene({'robot/root_angmom': S(data=torch.zeros(batch, 3))})
    e.sim.reset = lambda ids: None
    e.sim.step = lambda: None
    e.sim.forward = lambda: None
    e.sim.sense = lambda: None
    d.default_root_state = torch.zeros(batch, 13)
    d.default_root_state[:, 2] = .5
    d.default_root_state[:, 3] = 1
    d.default_joint_pos = torch.zeros(batch, 12)
    d.joint_acc = torch.zeros(batch, 12)
    d.default_root_state[:, 7:13] = 0
    d.root_link_ang_vel_b = torch.zeros(batch, 3)
    d.root_link_ang_vel_w = torch.zeros(batch, 3)
    d.projected_gravity_b = torch.tensor([[0., 0., -1.]] * batch)
    d.site_lin_vel_w = torch.zeros(batch, 4, 3)
    d.soft_joint_pos_limits = torch.tensor([[[-1., 1.]] * 12] * batch)
    e.robot.indexing = S(joint_v_adr=list(range(12)))
    e.sim.data.qfrc_actuator = torch.zeros(batch, 12)
    e.robot.write_root_state_to_sim = lambda state, ids: (
        d.root_link_pos_w.__setitem__(ids, state[:, :3]),
        d.root_link_quat_w.__setitem__(ids, state[:, 3:7]),
        d.root_link_lin_vel_b.__setitem__(ids, state[:, 7:10]))
    e.robot.write_joint_state_to_sim = lambda pos, vel, env_ids: (
        d.joint_pos.__setitem__(env_ids, pos), d.joint_vel.__setitem__(env_ids, vel))
    e.joint_ids = list(range(12))
    e.action_manager = S(action=torch.zeros(batch, 12))
    e.action_manager.process_action = lambda a: setattr(e.action_manager, 'action', a.clone())
    e.action_manager.apply_action = lambda: None
    e.action_manager.reset = lambda ids: e.action_manager.action.__setitem__(ids, 0.)
    e.contacts.sample = lambda *args, **kwargs: None
    e.contacts.reset = lambda ids: None
    e.contacts.link_force = torch.zeros(batch, 2, 3)
    e.contacts.body_link_mask = torch.tensor([False, True])
    e.contacts.nonfoot_duration = torch.zeros(batch, 2)
    e.contacts.support_geom = torch.ones(batch, 4, dtype=torch.long)
    e.contacts.found = torch.ones(batch, 4, dtype=torch.bool)
    e.contacts.support = torch.ones(batch, 4, dtype=torch.bool)
    e.contacts.foot_force = torch.zeros(batch, 4, 3)
    e.contacts.foot_force[:, :, 2] = 6.
    e.contacts.contact_duration = torch.ones(batch, 4)
    e.contacts.link_names = ['thigh_link', 'base_link']
    e.recipe_options = recipe_config()
    e.recipe, e.mode = 'C', 'formal'
    e.parking = ParkingState(batch)
    e.footfall = FirstFootfallState(batch, 40)
    e.progress = ProgressState(batch)
    e.initial_cases = None
    e.fixed_vx = None
    e.rng = torch.Generator().manual_seed(42)
    meta = atlas_metadata()
    e.spawn_x = torch.tensor([m.terrain_origin_world[0] - m.frame_origin_world[0] for m in meta])
    e.initial_noise = torch.tensor([[(m.initial_distance_range[1] - m.initial_distance_range[0]) / 2,
                                     (m.initial_y_range[1] - m.initial_y_range[0]) / 2,
                                     (m.initial_yaw_range[1] - m.initial_yaw_range[0]) / 2] for m in meta])
    e.perturb_ramp = torch.tensor([m.perturb_ramp_updates for m in meta])
    e.history_cache = S(reset=lambda obs, ids: None)
    e.map_cache = S(reset=lambda *args: None)
    e.stop = torch.zeros(batch, dtype=torch.bool)
    e.last_contact = torch.zeros(batch, 4, dtype=torch.bool)
    e.episode_reward_sums = {}
    for name in ('sum_vx', 'sum_speed', 'max_lateral', 'highest', 'slip_sum', 'nonfoot_sum',
                 'limit_sum', 'peak_torque', 'height_defaults', 'first_step_support_time',
                 'first_nonfoot_time', 'max_body_x', 'stagnation_total', 'stagnation_current',
                 'stagnation_longest', 'max_heading', 'sum_vx_world',
                 'sum_horizontal_speed_world', 'moving_sum_vx', 'moving_sum_speed'):
        setattr(e, name, torch.zeros(batch))
    e.moving_count = torch.zeros(batch, dtype=torch.long)
    e.first_nonfoot_link = torch.full((batch,), -1, dtype=torch.long)
    e.first_nonfoot_force = torch.zeros(batch, 3)
    e.first_foot_times = torch.full((batch, 4), -1.)
    e.foot_step_times = torch.full((batch, 4, 40), -1.)
    e.before_first_foot_height = torch.zeros(batch, 2)
    e.body_crossed_first = torch.zeros(batch, dtype=torch.bool)
    e.all_feet_on_platform = torch.zeros(batch, dtype=torch.bool)
    e._sim_step_counter = 0
    e.common_step_counter = 0
    e._terminal_speed = {}
    e._segment_samples = None
    e._observations = lambda update=True: {
        'proprio': d.root_link_pos_w.clone(), 'privileged': torch.zeros(batch, 35)}
    e._foot_heights = lambda: (torch.zeros(batch, 4), torch.zeros(batch, 4, dtype=torch.bool))
    return e


def common_episode_fields(info):
    return [{key: record.get(key) for key in
             ('env_id', 'success', 'climb_success', 'reason', 'post_climb_failure',
              'parking_success', 'first_step_support_time')} for record in info['episodes']]


class FootGeometryTests(unittest.TestCase):
    """SDS 6.1 fail-closed sphere/site geometry check, CPU objects only."""

    @staticmethod
    def _fixture():
        import mujoco

        class Model:
            ngeom = 5
            geom_type = [mujoco.mjtGeom.mjGEOM_SPHERE] * 4 + [mujoco.mjtGeom.mjGEOM_BOX]
            geom_size = [[.032, 0., 0.]] * 4 + [[.1, .1, .1]]
            geom_bodyid = [0, 1, 2, 3, 4]
            geom_pos = [[0., 0., -.02], [0., 0., -.02], [0., 0., -.02], [0., 0., -.02],
                        [0., 0., 0.]]
            # Entity-local foot site indices 0..3 map to global model sites 11..14.
            # Sites 0..3 are decoys with wrong bodies/positions: if the validator
            # used `foot_ids` directly it would reject the valid fixture.
            site_bodyid = [7, 8, 9, 10] + [7] * 7 + [0, 1, 2, 3]
            site_pos = [[9., 9., 9.]] * 4 + [[0., 0., 0.]] * 7 + [[0., 0., -.02]] * 4
            _names = ['FL_foot_collision', 'FR_foot_collision', 'RL_foot_collision',
                      'RR_foot_collision', 'base_collision']

            def geom(self, index):
                return S(name=self._names[index])

        e = object.__new__(V04Env)
        e.sim = S(mj_model=Model())
        e.foot_ids = [0, 1, 2, 3]
        e.robot = S(indexing=S(site_ids=torch.tensor([11, 12, 13, 14])))
        return e

    def test_valid_four_sphere_sites_pass(self):
        self._fixture()._validate_foot_geometry()

    def test_local_foot_ids_are_mapped_through_robot_indexing(self):
        # Decoy sites 0..3 would fail; the valid fixture only passes because the
        # validator maps entity-local ids through robot.indexing.site_ids.
        self._fixture()._validate_foot_geometry()
        e = self._fixture()
        e.robot.indexing.site_ids[0] = 0  # decoy body 7 instead of global foot site 11
        with self.assertRaisesRegex(ValueError, 'FL.*bodies'):
            e._validate_foot_geometry()

    def test_wrong_radius_is_rejected_with_leg_name(self):
        e = self._fixture()
        e.sim.mj_model.geom_size[0] = [.04, 0., 0.]
        with self.assertRaisesRegex(ValueError, 'FL.*radius'):
            e._validate_foot_geometry()

    def test_offset_centre_is_rejected_with_leg_name(self):
        e = self._fixture()
        e.sim.mj_model.geom_pos[1] = [.001, 0., -.02]
        with self.assertRaisesRegex(ValueError, 'FR.*centre'):
            e._validate_foot_geometry()

    def test_wrong_body_is_rejected_with_leg_name(self):
        e = self._fixture()
        e.sim.mj_model.geom_bodyid[2] = 9
        with self.assertRaisesRegex(ValueError, 'RL.*bodies'):
            e._validate_foot_geometry()

    def test_wrong_type_is_rejected_with_leg_name(self):
        import mujoco
        e = self._fixture()
        e.sim.mj_model.geom_type[3] = mujoco.mjtGeom.mjGEOM_BOX
        with self.assertRaisesRegex(ValueError, 'RR.*sphere'):
            e._validate_foot_geometry()

    def test_missing_geom_is_rejected_with_leg_name(self):
        e = self._fixture()
        e.sim.mj_model._names[0] = 'not_a_foot'
        with self.assertRaisesRegex(ValueError, 'FL.*exactly one'):
            e._validate_foot_geometry()

    def test_duplicate_geom_is_rejected_with_leg_name(self):
        e = self._fixture()
        e.sim.mj_model._names[1] = 'FL_foot_collision'
        with self.assertRaisesRegex(ValueError, 'FL.*exactly one'):
            e._validate_foot_geometry()


class EquivalenceTests(unittest.TestCase):
    def test_unchanged_weighted_terms_bit_equal_frozen_v02_and_old_terms_diagnostic(self):
        old, state, previous = reward_fixture(FrozenEnv)
        new, state2, previous2 = reward_fixture(V04Env)
        new._swing_result = fake_swing(3, gate=True, reward=0.)
        FrozenEnv._reward(old, state, previous)
        V04Env._reward(new, state2, previous2)
        replaced = ('track_linear_velocity', 'orientation', 'nonfoot')
        compared = 0
        for key, value in old.step_reward_tensors.items():
            if key == 'reward_total':
                continue
            name = key.split('/', 1)[1] if '/' in key else key
            if key.startswith('reward_weighted/') and name in replaced:
                torch.testing.assert_close(
                    new.step_reward_tensors['diagnostic/old_reward_weighted_' + name],
                    value, rtol=0, atol=0)
                continue
            if key.startswith(('reward_raw/', 'reward_baseline/')) and name in replaced:
                prefix = key.split('/', 1)[0]
                torch.testing.assert_close(
                    new.step_reward_tensors['diagnostic/old_' + prefix + '_' + name],
                    value, rtol=0, atol=0)
                continue
            if key in ('reward_event/success', 'reward_event/failure', 'reward_progress'):
                continue
            if key == 'reward_event/footfall_count':
                torch.testing.assert_close(new.step_reward_tensors['diagnostic/footfall_count'],
                                           value, rtol=0, atol=0)
                continue
            torch.testing.assert_close(new.step_reward_tensors[key], value, rtol=0, atol=0)
            compared += 1
        self.assertGreater(compared, 10)
        # New replaced terms equal their pure v0.4 functions exactly.
        executed = new.sample_command[:, 0]
        want = speed_reward(new.robot.data.root_link_lin_vel_b[:, 0],
                            new.robot.data.root_link_lin_vel_b[:, 1], executed,
                            new._swing_result.positive_gate, dt=new.step_dt,
                            enabled=executed > 0)
        torch.testing.assert_close(new.step_reward_tensors['reward_weighted/track_linear_velocity'],
                                   want, rtol=0, atol=0)
        torch.testing.assert_close(
            new.step_reward_tensors['reward_weighted/orientation'],
            orientation_reward(state2[1], state2[2], dt=new.step_dt), rtol=0, atol=0)
        torch.testing.assert_close(
            new.step_reward_tensors['reward_weighted/nonfoot'],
            nonfoot_reward(new.contacts.link_force, dt=new.step_dt), rtol=0, atol=0)
        self.assertNotIn('reward_progress', new.episode_reward_sums)

    def test_total_only_rewarded_terms_and_sum_namespaces_never_pollute(self):
        e, state, previous = reward_fixture(V04Env, batch=3)
        e._swing_result = fake_swing(3, gate=True, reward=.75)
        total = V04Env._reward(e, state, previous)
        tensors = e.step_reward_tensors
        weighted = [value for key, value in tensors.items()
                    if key.startswith('reward_weighted/')]
        want = (torch.stack(weighted).sum(0) + tensors['reward_event/success']
                + tensors['reward_event/failure'] + tensors['reward_event/footfall']
                + tensors['reward_event/swing_clearance'])
        torch.testing.assert_close(total, want, rtol=0, atol=0)
        torch.testing.assert_close(tensors['reward_total'], want, rtol=0, atol=0)
        self.assertEqual(tensors['reward_event/success'].tolist(), [10., 0., 0.])
        self.assertEqual(tensors['reward_event/failure'].tolist(), [0., 0., -4.])
        self.assertEqual(tensors['reward_event/swing_clearance'].tolist(), [.75, .75, .75])
        self.assertTrue(tensors['reward_progress'].eq(0).all())
        # Every non-rewarded namespace is present but excluded from total and sums.
        self.assertTrue(any(key.startswith('reward_raw/') for key in tensors))
        self.assertTrue(any(key.startswith('reward_baseline/') for key in tensors))
        self.assertTrue(any(key.startswith('diagnostic/') for key in tensors))
        polluted = sum(value for key, value in tensors.items()
                       if key.startswith(('reward_raw/', 'reward_baseline/', 'diagnostic/')))
        self.assertGreater(float(polluted.abs().sum()), 0.)
        recomputed = torch.stack(weighted).sum(0) + event_reward(
            state[5] & ~state[6], state[6]) + tensors['reward_event/footfall'] + .75
        torch.testing.assert_close(want, recomputed, rtol=0, atol=0)
        for key in e.episode_reward_sums:
            self.assertTrue(key == 'reward_total' or key.startswith('reward_weighted/')
                            or key.startswith('reward_event/'), key)
        for key in e.episode_reward_sums:
            self.assertFalse(key.startswith(('diagnostic/', 'reward_raw/', 'reward_baseline/')))
        for key, value in e.episode_reward_sums.items():
            if key != 'reward_total':
                torch.testing.assert_close(value, tensors[key], rtol=0, atol=0)
        # Failure wins when success and failure arrive in the same control frame.
        e2, state2, previous2 = reward_fixture(V04Env, batch=2)
        e2._swing_result = fake_swing(2, gate=True, reward=0.)
        state2 = (state2[0], state2[1], state2[2], state2[3], state2[4],
                  torch.tensor([True, False]), torch.tensor([True, False]),
                  state2[7], state2[8])
        e2.climb_event = torch.tensor([True, False])
        same = V04Env._reward(e2, state2, previous2)
        self.assertEqual(e2.step_reward_tensors['reward_event/success'].tolist(), [0., 0.])
        self.assertEqual(e2.step_reward_tensors['reward_event/failure'].tolist(), [-4., 0.])
        self.assertAlmostEqual(float(same[0]), -4. + sum(
            float(v[0]) for k, v in e2.step_reward_tensors.items()
            if k.startswith('reward_weighted/')) + float(
            e2.step_reward_tensors['reward_event/footfall'][0]), places=5)
        # Parking-diagnostic state success is not the frozen one-time safe-arrival
        # event; only climb_event pays +10 (matching frozen v02 reward).
        e3, state3, previous3 = reward_fixture(V04Env, batch=1)
        e3._swing_result = fake_swing(1, gate=True, reward=0.)
        e3.climb_event = torch.zeros(1, dtype=torch.bool)
        state3 = (state3[0], state3[1], state3[2], state3[3], state3[4],
                  torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool),
                  state3[7], state3[8])
        V04Env._reward(e3, state3, previous3)
        self.assertEqual(e3.step_reward_tensors['reward_event/success'].tolist(), [0.])

    def test_old_body_progress_never_enters_total_or_episode_sums(self):
        e, state, previous = reward_fixture(V04Env, batch=1)
        e._swing_result = fake_swing(1, gate=True, reward=0.)
        # Body moves forward with a positive command and current support: the
        # frozen ProgressState would pay 2*delta, but v0.4 must pay zero.
        state = (torch.tensor([[.1, .0, .5]]),) + state[1:]
        total = V04Env._reward(e, state, previous)
        old_progress = e.step_reward_tensors['diagnostic/old_body_progress']
        self.assertGreater(float(old_progress[0]), 0.)
        self.assertTrue(e.step_reward_tensors['reward_progress'].eq(0).all())
        self.assertNotIn('reward_progress', e.episode_reward_sums)
        self.assertNotIn('diagnostic/old_body_progress', e.episode_reward_sums)
        weighted = [value for key, value in e.step_reward_tensors.items()
                    if key.startswith('reward_weighted/')]
        want = (torch.stack(weighted).sum(0) + e.step_reward_tensors['reward_event/success']
                + e.step_reward_tensors['reward_event/failure']
                + e.step_reward_tensors['reward_event/footfall']
                + e.step_reward_tensors['reward_event/swing_clearance'])
        torch.testing.assert_close(total, want, rtol=0, atol=0)
        # A repeated sample at the same body x has no high-water delta; the
        # progress diagnostic is zero and the total remains progress-free.
        state = (torch.tensor([[.1, .0, .5]]),) + state[1:]
        V04Env._reward(e, state, previous)
        self.assertEqual(float(e.step_reward_tensors['diagnostic/old_body_progress'][0]), 0.)
        self.assertIn('reward_total', e.episode_reward_sums)

    def test_static_front_foot_with_pitching_body_cannot_earn_footfall_or_progress(self):
        e, state, previous = reward_fixture(V04Env, batch=1)
        e._swing_result = fake_swing(1, gate=True, reward=0.)
        e.contacts.top_forces.zero_()  # no real first-tread contact
        q = torch.tensor([[math.cos(.25), 0., math.sin(.25), 0.]])  # nose-down pitch
        e.robot.data.root_link_quat_w = q
        e.robot.data.site_pos_w[0, :2] = torch.tensor([[.1, .1, .112], [.1, -.1, .112]])
        e.robot.data.root_link_pos_w[0, 0] = .1
        e.climb_event.zero_()
        state = state[:5] + (torch.zeros(1, dtype=torch.bool),) + state[6:]
        totals = []
        for k in range(12):
            state = (torch.tensor([[.1 + .01 * k, .0, .5]]),) + state[1:]
            totals.append(V04Env._reward(e, state, previous))
        for reward in totals:
            self.assertEqual(float(reward[0]), float(reward[0]))
        tensors = e.step_reward_tensors
        self.assertTrue(tensors['reward_event/footfall'].eq(0).all())
        self.assertTrue(tensors['reward_event/success'].eq(0).all())
        self.assertTrue(tensors['reward_progress'].eq(0).all())
        # Even though the frozen body-progress diagnostic can be positive, it is
        # excluded from every total returned above.
        weighted = [value for key, value in tensors.items() if key.startswith('reward_weighted/')]
        want = (torch.stack(weighted).sum(0) + tensors['reward_event/success']
                + tensors['reward_event/failure'] + tensors['reward_event/footfall']
                + tensors['reward_event/swing_clearance'])
        torch.testing.assert_close(totals[-1], want, rtol=0, atol=0)
        self.assertTrue(e.step_reward_tensors['reward_raw/orientation'].abs().sum() > 0.)


class SwingSampleWiringTests(unittest.TestCase):
    def test_sample_uses_control_end_truth_and_translates_before_inverse_rotation(self):
        from mjlab.utils.lab_api.math import quat_apply_inverse
        e = fixture(V04Env, 2)
        q = torch.tensor([[math.cos(.4), 0., math.sin(.4), 0.]] * 2)
        e.robot.data.root_link_quat_w = q
        e.robot.data.root_link_pos_w = torch.tensor([[.3, -.2, .7]] * 2)
        e.robot.data.site_pos_w = torch.tensor(
            [[[.4, -.1, .8], [.2, -.3, .9], [.1, -.1, .75], [.25, -.25, .85]]] * 2)
        e.contacts.link_force = torch.zeros(2, 1, 3)
        e.contacts.link_force[0, 0, 0] = 11.
        state = task_step(e)
        sample = e._build_swing_sample(state)
        torch.testing.assert_close(sample.tile_ids, e.tile_ids, rtol=0, atol=0)
        torch.testing.assert_close(sample.num_steps, torch.full((2,), 4, dtype=torch.long),
                                   rtol=0, atol=0)
        torch.testing.assert_close(sample.foot_pos_w, e.robot.data.site_pos_w,
                                   rtol=0, atol=0)
        torch.testing.assert_close(sample.contact_found, e.contacts.found, rtol=0, atol=0)
        torch.testing.assert_close(sample.top_forces, e.contacts.top_forces, rtol=0, atol=0)
        torch.testing.assert_close(sample.geom_union_bounds, e.geom_step_bounds, rtol=0, atol=0)
        torch.testing.assert_close(sample.geom_tile, e.geom_tile, rtol=0, atol=0)
        torch.testing.assert_close(sample.geom_step, e.geom_step, rtol=0, atol=0)
        torch.testing.assert_close(sample.executed_c, e.sample_command[:, 0], rtol=0, atol=0)
        self.assertEqual(sample.failure.tolist(), state[6].tolist())
        self.assertEqual(sample.roll.tolist(), state[1].tolist())
        self.assertEqual(sample.pitch.tolist(), state[2].tolist())
        self.assertEqual(sample.nonfoot_hit.tolist(), [True, False])
        relative = sample.foot_pos_w - e.robot.data.root_link_pos_w[:, None]
        quat = q[:, None].expand(-1, 4, -1).reshape(-1, 4)
        expected = quat_apply_inverse(quat, relative.reshape(-1, 3)).reshape(2, 4, 3)
        torch.testing.assert_close(sample.foot_pos_b, expected, rtol=0, atol=0)
        wrong = (quat_apply_inverse(quat, sample.foot_pos_w.reshape(-1, 3)).reshape(2, 4, 3)
                 - e.robot.data.root_link_pos_w[:, None])
        self.assertGreater(float((sample.foot_pos_b - wrong).abs().max()), 1e-4)
        self.assertTrue(torch.isfinite(sample.foot_pos_b).all())

    def test_parking_diagnostic_disables_swing_motion_gate(self):
        e = fixture(V04Env, 1)
        e.parking_diagnostic = True
        e.climb.climbed[:] = True
        e.climb.climb_step[:] = 0
        e.episode_length_buf[:] = 1
        e.command.zero_()
        state = task_step(e)
        sample = e._build_swing_sample(state)
        self.assertEqual(sample.executed_c.tolist(), [0.])
        self.assertFalse(bool(sample.caller_enabled.any()))
        self.assertFalse(bool(e._swing_result.positive_gate.any()))


class StepEquivalenceTests(unittest.TestCase):
    def test_terminal_observation_and_episode_info_are_pre_reset(self):
        e = complete_fixture(V04Env)
        from src.tasks.velocity.v02_formal.env import timeout_steps
        e.episode_length_buf[0] = int(timeout_steps(torch.tensor([4.]), torch.tensor([.08]),
                                                   e.start_vx[:1])[0]) - 1
        e.robot.data.root_link_pos_w[:, 0] = .2
        obs, reward, terminated, truncated, info = e.step(torch.zeros(2, 12))
        self.assertEqual(terminated.tolist(), [False, False])
        self.assertEqual(truncated.tolist(), [True, False])
        self.assertAlmostEqual(info['terminal_observation']['proprio'][0, 0].item(), .2, places=6)
        self.assertEqual(info['diagnostics']['step'],
                         info['segment_samples']['step'][0])
        self.assertEqual(info['episodes'][0]['terminal_step'],
                         int(info['segment_samples']['step'][0]))
        self.assertEqual(info['episodes'][0]['reason'], 'timeout')
        # Diagnostics describe the terminal control sample, not the reset state.
        self.assertEqual(obs['proprio'][0, 0].item(), -1.)
        self.assertAlmostEqual(float(info['swing']['executed_c'][0]), .4, places=6)
        torch.testing.assert_close(info['swing']['clearance_world'][0],
                                   e._swing_aux['clearance_world'][0], rtol=0, atol=0)
        self.assertIn('foot_dwell', info)
        self.assertEqual(info['foot_dwell']['count'].tolist(), [[1, 1, 1, 1], [1, 1, 1, 1]])
        self.assertEqual(info['reward_version'], RECIPE_VERSION)
        self.assertEqual(info['reward_config_sha256'], reward_config_sha256())
        self.assertIn('reward_gate', info)
        self.assertIn('reward_gate_close_reasons', info)
        for name in GATE_REASON_NAMES[1:]:
            self.assertIn(name, info['reward_gate_close_reasons'])
        diag = info['diagnostics']
        for key in ('swing_target_step', 'swing_phi', 'swing_increment',
                    'swing_consumed_fraction', 'swing_cumulative_paid',
                    'swing_safe_support', 'swing_highest_safe_step',
                    'swing_clearance_world', 'swing_common_highest_safe_step',
                    'foot_dwell_samples', 'foot_dwell_seconds',
                    'front_first_step_time', 'hind_first_step_time'):
            self.assertIn(key, diag, key)

    def test_terminal_swing_snapshot_survives_automatic_reset(self):
        from src.tasks.velocity.v02_formal.env import timeout_steps
        e = complete_fixture(V04Env)
        e.step(torch.zeros(2, 12))
        self.assertEqual(e.swing.highest_safe_step[0].tolist(), [-1, -1, -1, -1])
        e.episode_length_buf[0] = int(timeout_steps(torch.tensor([4.]), torch.tensor([.08]),
                                                   e.start_vx[:1])[0]) - 1
        _, _, terminated, truncated, info = e.step(torch.zeros(2, 12))
        self.assertTrue(truncated[0])
        self.assertEqual(info['swing']['highest_safe_step'][0].tolist(), [4, 4, 4, 4])
        self.assertEqual(info['swing']['common_highest_safe_step'][0].item(), 4)
        self.assertAlmostEqual(float(info['swing']['executed_c'][0]), .4, places=6)
        # The live state was reset after the terminal info was captured.
        self.assertEqual(e.swing.highest_safe_step[0].tolist(), [-1, -1, -1, -1])
        self.assertIn('swing_highest_safe_step', info['diagnostics'])
        self.assertEqual(info['diagnostics']['swing_highest_safe_step'].tolist(), [4, 4, 4, 4])

    def test_async_partial_reset_only_touches_selected_ids(self):
        e = complete_fixture(V04Env)
        for _ in range(3):
            e.step(torch.zeros(2, 12))
        swing_before = {name: getattr(e.swing, name).clone() for name in
                        ('ready', 'active', 'highest_safe_step', 'target_step', 'paid_fraction')}
        dwell_before = {name: getattr(e, name).clone() for name in
                        ('dwell_window', 'dwell_count', 'dwell_samples', 'dwell_seconds')}
        ring = ContactWindowRing(device='cpu', max_samples=4, max_contacts=2)
        ring.push(geom0=torch.tensor([0]), geom1=torch.tensor([1]), foot=torch.tensor([0]),
                  link=torch.tensor([-1]), robot_side=torch.tensor([0]),
                  point=torch.zeros(1, 3), normal=torch.zeros(1, 3), force=torch.zeros(1, 3),
                  duration=torch.zeros(1), substep=0, control_step=0, physics_counter=1)
        e._contact_ring = ring
        with patch.object(FrozenEnv, '_reset_ids', lambda self, ids: None):
            e._reset_ids(torch.tensor([0]))
        for name, before in swing_before.items():
            after = getattr(e.swing, name)
            if name in ('highest_safe_step', 'target_step'):
                expected = torch.full_like(after[0], -1)
            else:
                expected = torch.zeros_like(after[0])
            torch.testing.assert_close(after[0], expected, rtol=0, atol=0)
            torch.testing.assert_close(after[1], before[1], rtol=0, atol=0)
        for name, before in dwell_before.items():
            after = getattr(e, name)
            torch.testing.assert_close(after[0], torch.zeros_like(after[0]), rtol=0, atol=0)
            torch.testing.assert_close(after[1], before[1], rtol=0, atol=0)
        self.assertEqual(ring.size, 0)
        # Resetting only env1 must keep env0's swing/dwell and contact ring.
        refill = ContactWindowRing(device='cpu', max_samples=4, max_contacts=2)
        refill.push(geom0=torch.tensor([0]), geom1=torch.tensor([1]), foot=torch.tensor([0]),
                    link=torch.tensor([-1]), robot_side=torch.tensor([0]),
                    point=torch.zeros(1, 3), normal=torch.zeros(1, 3), force=torch.zeros(1, 3),
                    duration=torch.zeros(1), substep=0, control_step=0, physics_counter=1)
        e._contact_ring = refill
        for _ in range(2):
            e.step(torch.zeros(2, 12))
        snap = getattr(e.swing, 'phi').clone()
        with patch.object(FrozenEnv, '_reset_ids', lambda self, ids: None):
            e._reset_ids(torch.tensor([1]))
        torch.testing.assert_close(e.swing.phi[0], snap[0], rtol=0, atol=0)
        self.assertEqual(e._contact_ring.size, 1)

    def test_diagnostic_aggregates_highest_safe_step_and_first_step_times(self):
        e = complete_fixture(V04Env)
        e.step(torch.zeros(2, 12))
        e.foot_step_times[0, :, 0] = torch.tensor([-1., .4, .6, -1.])
        e._swing_aux['highest_safe_step'][0] = torch.tensor([1, 2, -1, 3])
        e._swing_aux['common_highest_safe_step'] = e._swing_aux['highest_safe_step'].amin(-1)
        diag = e.diagnostics_snapshot(0)
        self.assertAlmostEqual(diag['front_first_step_time'], .4, places=6)
        self.assertAlmostEqual(diag['hind_first_step_time'], .6, places=6)
        torch.testing.assert_close(diag['foot_first_step_time'],
                                   torch.tensor([-1., .4, .6, -1.]), rtol=0, atol=0)
        self.assertEqual(diag['swing_highest_safe_step'].tolist(), [1, 2, -1, 3])
        self.assertEqual(diag['swing_common_highest_safe_step'], -1)
        self.assertIn('reward_gate_reason_name', diag)
        self.assertIn(diag['reward_gate_reason_name'], GATE_REASON_NAMES)

    def test_swing_state_advances_once_per_control_sample(self):
        e = complete_fixture(V04Env)
        calls = []
        original = e.swing.update

        def counted(sample):
            calls.append(1)
            return original(sample)

        e.swing.update = counted
        e.step(torch.zeros(2, 12))
        self.assertEqual(len(calls), 1)
        e.step(torch.zeros(2, 12))
        self.assertEqual(len(calls), 2)
        # Frozen footfall is advanced exactly once per reward call; a repeated
        # call cannot claim the same event again.
        e2, state, previous = reward_fixture(V04Env, batch=2)
        e2._swing_result = fake_swing(2, gate=True, reward=0.)
        V04Env._reward(e2, state, previous)
        self.assertEqual(e2.step_reward_tensors['reward_event/footfall'].tolist(), [0., 0.])
        V04Env._reward(e2, state, previous)
        self.assertEqual(e2.step_reward_tensors['reward_event/footfall'].tolist(), [1., 1.])
        V04Env._reward(e2, state, previous)
        self.assertEqual(e2.step_reward_tensors['reward_event/footfall'].tolist(), [0., 0.])

    def test_obs_action_success_failure_timeout_equal_frozen_on_same_input(self):
        scenarios = {}
        for name in ('timeout', 'nonfoot_failure'):
            frozen = complete_fixture(FrozenEnv)
            new = complete_fixture(V04Env)
            for env in (frozen, new):
                env.robot.data.root_link_pos_w[:, 0] = .2
                if name == 'timeout':
                    from src.tasks.velocity.v02_formal.env import timeout_steps
                    env.episode_length_buf[0] = int(timeout_steps(
                        torch.tensor([4.]), torch.tensor([.08]), env.start_vx[:1])[0]) - 1
                else:
                    env.contacts.link_force[:, 0, 0] = 20.
                    env.contacts.nonfoot_duration = torch.full((2, 2), .2)
            frozen.slot_classes = torch.full((2,), 3, dtype=torch.long)
            frozen_obs, frozen_reward, frozen_term, frozen_trunc, frozen_info = frozen.step(torch.zeros(2, 12))
            new_obs, new_reward, new_term, new_trunc, new_info = new.step(torch.zeros(2, 12))
            torch.testing.assert_close(new_term, frozen_term, rtol=0, atol=0)
            torch.testing.assert_close(new_trunc, frozen_trunc, rtol=0, atol=0)
            for key in frozen_obs:
                torch.testing.assert_close(new_obs[key], frozen_obs[key], rtol=0, atol=0)
            for key in frozen_info['terminal_observation']:
                torch.testing.assert_close(new_info['terminal_observation'][key],
                                           frozen_info['terminal_observation'][key], rtol=0, atol=0)
            self.assertEqual(common_episode_fields(new_info), common_episode_fields(frozen_info))
            torch.testing.assert_close(new.action_manager.action, frozen.action_manager.action,
                                       rtol=0, atol=0)
            # Reward intentionally differs, but only reward does.
            if name == 'nonfoot_failure':
                self.assertEqual(new_info['episodes'][0]['reason'], 'nonfoot_persistent')
                self.assertTrue(new_term[0])
            else:
                self.assertEqual(new_info['episodes'][0]['reason'], 'timeout')
                self.assertTrue(new_trunc[0])
            scenarios[name] = (frozen, new)

    def test_reward_gate_never_controls_termination(self):
        from src.tasks.velocity.v02_formal.env import timeout_steps
        from src.tasks.velocity.v04.rewards import SwingResult

        def run(gate):
            e = complete_fixture(V04Env)
            e.episode_length_buf[0] = int(timeout_steps(torch.tensor([4.]), torch.tensor([.08]),
                                                       e.start_vx[:1])[0]) - 1
            e.robot.data.root_link_pos_w[:, 0] = .2
            real = SwingClearanceState.update

            def forced(state, sample, gate=gate):
                result = real(state, sample)
                return dataclasses.replace(result,
                                           positive_gate=torch.full_like(result.positive_gate, gate))

            with patch.object(SwingClearanceState, 'update', forced):
                return e, e.step(torch.zeros(2, 12))

        open_env, open_result = run(True)
        closed_env, closed_result = run(False)
        torch.testing.assert_close(open_result[2], closed_result[2], rtol=0, atol=0)
        torch.testing.assert_close(open_result[3], closed_result[3], rtol=0, atol=0)
        self.assertEqual(common_episode_fields(open_result[4]), common_episode_fields(closed_result[4]))
        for key in open_result[0]:
            torch.testing.assert_close(open_result[0][key], closed_result[0][key], rtol=0, atol=0)
        self.assertTrue(open_result[4]['reward_gate'].all())
        self.assertFalse(closed_result[4]['reward_gate'].any())
        self.assertNotEqual(float(open_result[1].sum()), float(closed_result[1].sum()))
        frozen = complete_fixture(FrozenEnv)
        frozen.slot_classes = torch.full((2,), 3, dtype=torch.long)
        frozen.episode_length_buf[0] = int(timeout_steps(torch.tensor([4.]), torch.tensor([.08]),
                                                         frozen.start_vx[:1])[0]) - 1
        frozen.robot.data.root_link_pos_w[:, 0] = .2
        _, _, term, trunc, _ = frozen.step(torch.zeros(2, 12))
        torch.testing.assert_close(closed_result[2], term, rtol=0, atol=0)
        torch.testing.assert_close(closed_result[3], trunc, rtol=0, atol=0)
        probe = fixture(V04Env, 3)
        probe._swing_result = fake_swing(3, gate=False, reward=0.)
        probe._swing_result.gate_details['reason'] = torch.tensor([0, 2, 5])
        payload = probe._gate_payload()
        self.assertEqual(sorted(payload['reward_gate_close_reasons']), sorted(
            name for name in GATE_REASON_NAMES[1:]))
        self.assertEqual(payload['reward_gate_close_reasons']['failure'].tolist(),
                         [False, True, False])
        self.assertEqual(payload['reward_gate_close_reasons']['insufficient_safe_support'].tolist(),
                         [False, False, True])


class FootDwellTests(unittest.TestCase):
    def _static_approach(self, batch=1):
        e = fixture(V04Env, batch)
        # No support yet: the next step is tread 1, whose world front edge is x=0.
        e.contacts.top_forces.zero_()
        e.contacts.found.zero_()
        e.robot.data.site_pos_w[:, :, 0] = -.1
        return e

    def test_window_needs_25_samples_and_only_qualifies_inside_edge_and_command(self):
        e = self._static_approach()
        for _ in range(24):
            task_step(e)
        self.assertFalse(e.dwell_near_static.any())
        self.assertTrue(e.dwell_samples.eq(0).all())
        task_step(e)
        self.assertTrue(e.dwell_near_static.all())
        self.assertTrue(e.dwell_qualifying.all())
        self.assertEqual(e.dwell_samples.tolist(), [[1, 1, 1, 1]])
        self.assertAlmostEqual(float(e.dwell_seconds[0, 0]), .02, places=6)
        # Moving one foot by >1 cm breaks only that foot's near-static window.
        e.robot.data.site_pos_w[0, 0, 0] = -.05
        task_step(e)
        self.assertFalse(bool(e.dwell_near_static[0, 0]))
        self.assertTrue(bool(e.dwell_near_static[0, 1]))
        self.assertEqual(e.dwell_samples[0].tolist(), [1, 2, 2, 2])
        # A single middle spike (endpoints still close) breaks near-static:
        # the rule is maximum displacement from the window's first sample.
        e4 = self._static_approach()
        for _ in range(25):
            task_step(e4)
        self.assertTrue(e4.dwell_near_static.all())
        e4.dwell_window[0, 0, 12, 0] += .02
        task_step(e4)
        self.assertFalse(bool(e4.dwell_near_static[0, 0]))
        self.assertTrue(bool(e4.dwell_near_static[0, 1]))
        # Outside the 0..0.30 m approach window nothing accumulates.
        e2 = self._static_approach()
        e2.robot.data.site_pos_w[:, :, 0] = -.5
        for _ in range(30):
            task_step(e2)
        self.assertTrue(e2.dwell_near_static.all())
        self.assertTrue(e2.dwell_samples.eq(0).all())
        # At or below the 0.1 m/s command threshold nothing accumulates.
        e3 = self._static_approach()
        e3.command[:, 0] = .05
        for _ in range(30):
            task_step(e3)
        self.assertTrue(e3.dwell_near_static.all())
        self.assertTrue(e3.dwell_samples.eq(0).all())

    def test_reset_clears_window_and_counters_only_for_selected_ids(self):
        e = self._static_approach(batch=2)
        for _ in range(26):
            task_step(e)
        before = {name: getattr(e, name).clone() for name in
                  ('dwell_window', 'dwell_count', 'dwell_samples', 'dwell_seconds',
                   'dwell_current_seconds', 'dwell_longest_seconds')}
        with patch.object(FrozenEnv, '_reset_ids', lambda self, ids: None):
            e._reset_ids(torch.tensor([0]))
        for name, value in before.items():
            torch.testing.assert_close(getattr(e, name)[0], torch.zeros_like(value[0]),
                                       rtol=0, atol=0)
            torch.testing.assert_close(getattr(e, name)[1], value[1], rtol=0, atol=0)
        self.assertNotIn('body_stagnation', ' '.join(name for name in dir(e) if 'dwell' in name))


class ContactWindowTests(unittest.TestCase):
    def _hook_fixture(self):
        e = object.__new__(V04Env)
        V04Env._allocate_task_state(e, 1, 'cpu')
        e.cfg = S(sim=S(mujoco=S(timestep=.005)), decimation=4)
        e.episode_length_buf = torch.zeros(1, dtype=torch.long)
        e._sim_step_counter = 0
        e.sim = S(device='cpu')  # no mj_model: names are optional evidence
        calls = []

        class Contacts:
            foot_lookup = torch.tensor([0, -1, -1, -1, -1, -1])
            nonfoot_lookup = torch.tensor([-1, -1, 0, -1, 0, -1])
            contact_duration = torch.tensor([[.1, .2, .3, .4]])
            nonfoot_duration = torch.tensor([[.5]])
            link_names = ['base_link']

            def sample(self, assigned_tiles, physics_dt=None, control=False):
                calls.append((physics_dt, control))
                return self

        e.contacts = Contacts()
        rows = dict(
            geom=torch.tensor([[0, 1], [2, 4], [0, 3], [1, 5], [0, 5]], dtype=torch.long),
            world=torch.tensor([0, 0, 0, 0, 1], dtype=torch.long),
            point=torch.arange(15, dtype=torch.float32).reshape(5, 3),
            normal=torch.ones(5, 3),
            force=torch.arange(15, dtype=torch.float32).reshape(5, 3),
        )
        e._read_physics_contact_rows = lambda: rows
        return e, calls

    def test_ring_is_bounded_chronological_and_exports_true_5ms_rows(self):
        ring = ContactWindowRing(device='cpu', max_samples=100, max_contacts=3)
        for sample in range(130):
            count = sample % 5
            ring.push(
                geom0=torch.zeros(count, dtype=torch.long),
                geom1=torch.ones(count, dtype=torch.long),
                foot=torch.full((count,), -1, dtype=torch.long),
                link=torch.zeros(count, dtype=torch.long),
                robot_side=torch.zeros(count, dtype=torch.long),
                point=torch.full((count, 3), float(sample)),
                normal=torch.ones(count, 3),
                force=torch.full((count, 3), float(sample)),
                duration=torch.full((count,), .005), substep=sample % 4,
                control_step=sample // 4, physics_counter=sample)
        self.assertEqual(ring.size, 100)
        self.assertEqual(ring.writes, 130)
        first_bytes = ring.nbytes
        empty = ContactWindowRing(device='cpu', max_samples=2, max_contacts=1)
        empty.push(geom0=torch.zeros(0, dtype=torch.long), geom1=torch.zeros(0, dtype=torch.long),
                   foot=torch.zeros(0, dtype=torch.long), link=torch.zeros(0, dtype=torch.long),
                   robot_side=torch.zeros(0, dtype=torch.long), point=torch.zeros(0, 3),
                   normal=torch.zeros(0, 3), force=torch.zeros(0, 3), duration=torch.zeros(0),
                   substep=0, control_step=0, physics_counter=0)
        self.assertEqual(empty.size, 1)
        window = ring.export(physics_dt=.005, control_dt=.02,
                             geom_names=['a', 'b'], link_names=['base_link'])
        self.assertEqual(window['retained_samples'], 100)
        self.assertEqual(window['physics_dt'], .005)
        self.assertEqual(window['max_samples'], 100)
        counters = window['samples']['physics_counter'].tolist()
        self.assertEqual(counters[0], 30)
        self.assertEqual(counters[-1], 129)
        counts = window['samples']['contact_count'].tolist()
        self.assertEqual(counts[:5], [0, 1, 2, 3, 3])
        self.assertEqual(max(counts), 3)
        self.assertEqual(ring.nbytes, first_bytes)
        self.assertGreater(window['dropped_contacts'], 0)
        self.assertEqual(window['contacts']['force'][0, 0].item(), 31.)
        # Every exported row carries its own true 5 ms substep and sample index.
        self.assertEqual(window['contacts']['substep'][:9].tolist(),
                         [3, 0, 0, 1, 1, 1, 2, 2, 2])
        self.assertEqual(window['contacts']['sample_index'][:2].tolist(), [1, 2])
        self.assertEqual(window['contacts']['geom0_name'][0], 'a')
        self.assertEqual(window['contacts']['link_name'][0], 'base_link')

    def test_hook_captures_only_physics_substeps_and_exports_to_cpu(self):
        e, calls = self._hook_fixture()
        V04Env._install_contact_window(e)
        e._geom_names = ['fl_foot', 'terrain1', 'base', 'terrain2', 'base2', 'terrain3']
        for counter in range(1, 5):
            e._sim_step_counter = counter
            e.episode_length_buf[0] = 7
            out = e.contacts.sample(None, physics_dt=.005)
            self.assertIs(out, e.contacts)
        # The single 20 ms control sample must not be disguised as 5 ms data.
        e.contacts.sample(None, control=True)
        self.assertEqual(e.contacts.sample is not None, True)
        self.assertEqual(len(calls), 5)
        self.assertEqual(e._contact_ring.size, 4)
        self.assertEqual(e._contact_ring.writes, 4)
        window = e._export_contact_window()
        self.assertEqual(window['physics_dt'], .005)
        self.assertEqual(window['control_dt'], .02)
        self.assertEqual(window['samples']['substep'].tolist(), [0, 1, 2, 3])
        self.assertEqual(window['samples']['control_step'].tolist(), [7, 7, 7, 7])
        self.assertEqual(window['samples']['physics_counter'].tolist(), [1, 2, 3, 4])
        self.assertEqual(window['samples']['contact_count'].tolist(), [3, 3, 3, 3])
        # Rows are the actual captured 5 ms values (other env / non-robot dropped).
        self.assertEqual(window['contacts']['geom0'].tolist(), [0, 2, 0] * 4)
        self.assertEqual(window['contacts']['foot'].tolist(), [0, -1, 0] * 4)
        torch.testing.assert_close(window['contacts']['duration'],
                                   torch.tensor([.1, .5, .1] * 4), rtol=0, atol=0)
        self.assertEqual(window['contacts']['force'][0].tolist(), [0., -1., -2.])
        self.assertEqual(window['contacts']['geom0_name'], ['fl_foot', 'base', 'fl_foot'] * 4)
        self.assertEqual(window['contacts']['link_name'], [None, 'base_link', None] * 4)

    def test_env0_failure_exports_prefailure_window_in_info_and_diagnostics(self):
        e = complete_fixture(V04Env)
        ring = ContactWindowRing(device='cpu', max_samples=4, max_contacts=2)
        for counter, force in ((98, 1.), (99, 2.), (100, 3.)):
            ring.push(geom0=torch.tensor([0]), geom1=torch.tensor([1]),
                      foot=torch.tensor([0]), link=torch.tensor([-1]),
                      robot_side=torch.tensor([0]), point=torch.zeros(1, 3),
                      normal=torch.zeros(1, 3), force=torch.full((1, 3), force),
                      duration=torch.full((1,), .1), substep=counter % 4,
                      control_step=counter // 4, physics_counter=counter)
        e._contact_ring = ring
        e._geom_names = ['fl_foot', 'terrain']
        e.contacts.link_force[:, 0, 0] = 20.
        e.contacts.nonfoot_duration = torch.full((2, 2), .2)
        _, _, terminated, _, info = e.step(torch.zeros(2, 12))
        self.assertTrue(terminated[0])
        self.assertEqual(info['episodes'][0]['reason'], 'nonfoot_persistent')
        window = info['contact_window']
        self.assertEqual(window['kind'], 'env0_true_5ms_physics_contact_window')
        self.assertEqual(window['retained_samples'], 3)
        self.assertEqual(window['physics_dt'], .005)
        self.assertEqual(window['control_dt'], .02)
        self.assertEqual(window['samples']['physics_counter'].tolist(), [98, 99, 100])
        self.assertEqual(window['contacts']['force'][:, 0].tolist(), [1., 2., 3.])
        self.assertEqual(window['contacts']['geom0_name'], ['fl_foot'] * 3)
        self.assertEqual(window['contacts']['substep'].tolist(), [2, 3, 0])
        self.assertEqual(info['diagnostics']['contact_window']['retained_samples'], 3)
        self.assertEqual(info['diagnostics']['contact_window_summary']['retained_samples'], 3)
        # The pre-reset export is preserved after the automatic env0 reset.
        self.assertEqual(ring.size, 0)
        # No failure, no full export.
        e2 = complete_fixture(V04Env)
        e2._contact_ring = ContactWindowRing(device='cpu', max_samples=4, max_contacts=2)
        _, _, terminated2, _, info2 = e2.step(torch.zeros(2, 12))
        self.assertFalse(terminated2.any())
        self.assertNotIn('contact_window', info2)

    def test_capture_failure_is_bounded_and_never_changes_physics_calls(self):
        e, calls = self._hook_fixture()
        V04Env._install_contact_window(e)
        e._read_physics_contact_rows = lambda: (_ for _ in ()).throw(RuntimeError('warp read failed'))
        e.contacts.sample(None, physics_dt=.005)
        self.assertEqual(len(calls), 1)
        self.assertEqual(e._contact_capture_errors, 1)
        self.assertIn('warp read failed', e._contact_capture_last_error)
        self.assertEqual(e._contact_ring.size, 0)

    def test_ring_push_overhead_is_measured_on_cpu(self):
        ring = ContactWindowRing(device='cpu', max_samples=100, max_contacts=64)
        count = 16
        args = dict(geom0=torch.zeros(count, dtype=torch.long),
                    geom1=torch.ones(count, dtype=torch.long),
                    foot=torch.zeros(count, dtype=torch.long),
                    link=torch.zeros(count, dtype=torch.long),
                    robot_side=torch.zeros(count, dtype=torch.long),
                    point=torch.zeros(count, 3), normal=torch.ones(count, 3),
                    force=torch.ones(count, 3), duration=torch.full((count,), .005),
                    substep=0, control_step=0, physics_counter=0)
        start = time.perf_counter()
        for sample in range(1000):
            args['physics_counter'] = sample
            ring.push(**args)
        elapsed = time.perf_counter() - start
        # Informational only: no wall-clock assertion, but keep the measured
        # value available to the implementation notes via the test log.
        print(f'\nContactWindowRing.push: {elapsed / 1000 * 1e6:.1f} us/sample, '
              f'{ring.nbytes} bytes storage', file=sys.stderr)
        self.assertEqual(ring.size, 100)


if __name__ == '__main__':
    unittest.main()
