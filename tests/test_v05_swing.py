"""Synthetic 0.5 swing-clearance and footfall state-machine tests (SDS 3.5/3.6).

Every case builds ``SwingSample``/``SwingClearanceState`` directly from small
synthetic geom metadata; no simulator, policy, training or GPU is involved and
the 20 ms update is driven by hand.  The v0.5 changes under test are the 0.15
payment, the 0.60 m target window and the legal ``N == 0`` flat path; the
physical recognition, anchor and anti-double-payment semantics stay at the v0.4
behaviour required by SDS 3.6.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a2_v05_bootstrap  # noqa: E402

a2_v05_bootstrap.activate(ROOT)

import math  # noqa: E402
import unittest  # noqa: E402

import torch  # noqa: E402

from src.tasks.velocity.v05 import rewards as r  # noqa: E402
from src.tasks.velocity.v05 import spec  # noqa: E402

FOOT_RADIUS = spec.FOOT_RADIUS
PAY = .15
WINDOW = .60


def api():
    return r


class Rig:
    """One mutable synthetic scene per environment, convertible to SwingSample."""

    def __init__(self, tiles=(5,), nums=(4,), step_height=.08):
        self.B = len(tiles)
        self.tiles = list(tiles)
        self.nums = list(nums)
        self.h = step_height
        self.specs = []  # (tile, step, top_z, (xmin, xmax, ymin, ymax))
        for tile, n in zip(self.tiles, self.nums):
            self.specs.append((tile, 0, 0., (-.5, .5, -.4, .4)))
            for k in range(1, n + 1):
                self.specs.append((tile, k, k * self.h, ((k - 1) * .2, k * .2, -.4, .4)))
        self.G = len(self.specs)
        self.foot_w = torch.zeros(self.B, 4, 3)
        self.foot_w[..., 2] = FOOT_RADIUS
        self.foot_b = torch.zeros(self.B, 4, 3)
        self.found = torch.zeros(self.B, 4, dtype=torch.bool)
        self.forces = torch.zeros(self.B, 4, self.G)
        self.executed_c = .4
        self.caller_enabled = True
        self.failure = False
        self.roll = 0.
        self.pitch = 0.
        self.nonfoot_hit = False
        self.state = api().SwingClearanceState(self.B)

    # ------------------------------------------------------------- geometry
    def _index(self, tile, step):
        for i, (t, s, _, _) in enumerate(self.specs):
            if t == tile and s == step:
                return i
        raise KeyError((tile, step))

    def _env_index(self, env, step, tile=None):
        return self._index(self.tiles[env] if tile is None else tile, step)

    def set_geom(self, tile, step, *, top=None, bounds=None):
        i = self._index(tile, step)
        old = self.specs[i]
        self.specs[i] = (old[0], old[1], old[2] if top is None else top,
                         old[3] if bounds is None else tuple(bounds))

    def move_geom(self, tile, step, new_tile):
        i = self._index(tile, step)
        old = self.specs[i]
        self.specs[i] = (new_tile, old[1], old[2], old[3])

    def add_geom(self, tile, step, top, bounds):
        self.specs.append((tile, step, top, tuple(bounds)))
        self.forces = torch.cat((self.forces, torch.zeros(self.B, 4, 1)), dim=2)
        self.G += 1
        return self.G - 1

    # --------------------------------------------------------------- frames
    def support(self, env, foot, step, force=6., x=None, y=.1, tile=None):
        i = self._env_index(env, step, tile)
        top, bounds = self.specs[i][2], self.specs[i][3]
        self.found[env, foot] = True
        self.forces[env, foot, :] = 0.
        self.forces[env, foot, i] = force
        self.foot_w[env, foot, 0] = (bounds[0] + bounds[1]) / 2. if x is None else x
        self.foot_w[env, foot, 1] = y
        self.foot_w[env, foot, 2] = top + FOOT_RADIUS
        self.foot_b[env, foot, 2] = top

    def air(self, env, foot, *, x=None, y=None, z=None, bz=None, h_world=0., h_body=0.):
        self.found[env, foot] = False
        self.forces[env, foot, :] = 0.
        if x is not None:
            self.foot_w[env, foot, 0] = x
        if y is not None:
            self.foot_w[env, foot, 1] = y
        self.foot_w[env, foot, 2] = FOOT_RADIUS + h_world if z is None else z
        self.foot_b[env, foot, 2] = h_body if bz is None else bz

    def lift(self, env, foot, *, x=None, y=None):
        """Break contact while keeping the last support pose (phi starts at 0)."""
        z = float(self.foot_w[env, foot, 2])
        bz = float(self.foot_b[env, foot, 2])
        self.air(env, foot, x=x, y=y, z=z, bz=bz)

    def sample(self):
        geom_tile = torch.tensor([s[0] for s in self.specs], dtype=torch.long)
        geom_step = torch.tensor([s[1] for s in self.specs], dtype=torch.long)
        geom_top = torch.tensor([s[2] for s in self.specs], dtype=torch.float32)
        geom_bounds = torch.tensor([s[3] for s in self.specs], dtype=torch.float32)
        return api().SwingSample(
            tile_ids=torch.tensor(self.tiles, dtype=torch.long),
            num_steps=torch.tensor(self.nums, dtype=torch.long),
            foot_pos_w=self.foot_w.clone(), foot_pos_b=self.foot_b.clone(),
            contact_found=self.found.clone(), top_forces=self.forces.clone(),
            geom_tile=geom_tile, geom_step=geom_step, geom_top=geom_top,
            geom_union_bounds=geom_bounds,
            executed_c=torch.full((self.B,), self.executed_c),
            caller_enabled=torch.full((self.B,), self.caller_enabled),
            failure=torch.full((self.B,), self.failure),
            roll=torch.full((self.B,), self.roll),
            pitch=torch.full((self.B,), self.pitch),
            nonfoot_hit=torch.full((self.B,), self.nonfoot_hit))

    def step(self):
        return self.state.update(self.sample())

    # ------------------------------------------------------------- scenario
    def warmup(self, step=0, frames=2, feet=(0, 1, 2, 3), y=.1):
        last = None
        for _ in range(frames):
            for env in range(self.B):
                for foot in feet:
                    self.support(env, foot, step, y=y)
            last = self.step()
        return last

    def swing(self, env=0, foot=0, *, x=-.1, y=.1, frames=2):
        last = None
        for _ in range(frames):
            self.lift(env, foot, x=x, y=y)
            last = self.step()
        return last

    def frame(self, phi, env=0, feet=(0,), x=None, y=None):
        for foot in feet:
            if not bool(self.state.target_locked[env, foot]):
                raise AssertionError(f'foot {foot} has no locked target for phi frame')
            height = (float(self.state.target_top_z[env, foot])
                      - float(self.state.takeoff_support_z[env, foot]) + .04)
            z = float(self.state.takeoff_bottom_z[env, foot]) + FOOT_RADIUS + phi * height
            bz = float(self.state.takeoff_body_z[env, foot]) + phi * .04
            self.air(env, foot, x=x, y=y, z=z, bz=bz)
        return self.step()

    def contact(self, phi, env=0, foot=0, step=0, force=6., x=None, y=.1):
        """Contact at a locked-target height; used to test the close-frame rule."""
        if not bool(self.state.target_locked[env, foot]):
            raise AssertionError(f'foot {foot} has no locked target for contact frame')
        height = (float(self.state.target_top_z[env, foot])
                  - float(self.state.takeoff_support_z[env, foot]) + .04)
        z = float(self.state.takeoff_bottom_z[env, foot]) + FOOT_RADIUS + phi * height
        bz = float(self.state.takeoff_body_z[env, foot]) + phi * .04
        self.found[env, foot] = True
        self.forces[env, foot, :] = 0.
        self.forces[env, foot, self._env_index(env, step)] = force
        if x is not None:
            self.foot_w[env, foot, 0] = x
        self.foot_w[env, foot, 1] = y
        self.foot_w[env, foot, 2] = z
        self.foot_b[env, foot, 2] = bz


class SwingTests(unittest.TestCase):
    def assertPaid(self, value, expected, delta=1e-6):
        self.assertAlmostEqual(float(value), expected, delta=delta)

    def test_initial_air_has_no_reward_and_no_anchor(self):
        rig = Rig()
        for k in range(1, 6):
            for foot in range(4):
                rig.air(0, foot, h_world=k * .01)
            result = rig.step()
            self.assertEqual(float(result.reward[0]), 0.)
            self.assertFalse(bool(result.active.any()))
            self.assertFalse(bool(result.positive_gate[0]))
        self.assertFalse(bool(rig.state.ready.any()))
        self.assertTrue(bool((rig.state.highest_safe_step == -1).all()))

    def test_two_safe_support_samples_then_two_contact_free_samples_start(self):
        rig = Rig()
        for foot in range(4):
            rig.support(0, foot, 0)
        first = rig.step()
        self.assertFalse(bool(first.safe_support.any()))
        for foot in range(4):
            rig.support(0, foot, 0)
        second = rig.step()
        self.assertTrue(bool(second.safe_support.all()))
        self.assertTrue(bool(rig.state.ready.all()))
        self.assertTrue(bool(second.positive_gate[0]))
        self.assertEqual(rig.state.selected_step.tolist(), [[0, 0, 0, 0]])
        rig.lift(0, 0, x=-.1)
        one_air = rig.step()
        self.assertFalse(bool(one_air.active[0, 0]))
        self.assertEqual(int(rig.state.no_contact_count[0, 0]), 1)
        rig.lift(0, 0, x=-.1)
        started = rig.step()
        self.assertTrue(bool(started.active[0, 0]))
        self.assertEqual(int(started.target_step[0, 0]), 1)
        self.assertEqual(float(started.phi[0, 0]), 0.)
        self.assertEqual(float(started.reward[0]), 0.)
        # Anchors come from the last confirmed support sample, never the air frame.
        self.assertEqual(float(rig.state.takeoff_bottom_z[0, 0]), 0.)
        self.assertEqual(float(rig.state.takeoff_body_z[0, 0]), 0.)
        self.assertEqual(float(rig.state.takeoff_support_z[0, 0]), 0.)
        self.assertEqual(int(rig.state.highest_safe_step[0, 0]), 0)

    def test_single_frame_contact_jitter_and_side_contact_do_not_anchor(self):
        rig = Rig()
        rig.support(0, 0, 0)
        one = rig.step()
        self.assertFalse(bool(one.safe_support.any()))
        self.assertFalse(bool(rig.state.ready[0, 0]))
        for k in (1, 2, 3):
            rig.air(0, 0, h_world=.01 * k)
            self.assertFalse(bool(rig.step().active.any()))
        for _ in range(2):
            for foot in range(4):
                rig.support(0, foot, 0)
            rig.step()
        rig.lift(0, 0, x=-.1)
        rig.step()
        rig.lift(0, 0, x=-.1)
        self.assertTrue(bool(rig.step().active[0, 0]))

        side = Rig()
        for _ in range(2):
            side.found[0, 0] = True
            side.foot_w[0, 0, 2] = FOOT_RADIUS
            side.forces[0, 0, :] = 0.
            result = side.step()
        self.assertFalse(bool(result.safe_support.any()))
        self.assertFalse(bool(side.state.ready[0, 0]))
        for _ in range(2):
            side.air(0, 0, h_world=.02)
            self.assertFalse(bool(side.step().active.any()))

        ready = Rig()
        ready.warmup()
        ready.lift(0, 0, x=-.1)
        ready.step()
        ready.support(0, 0, 0)
        ready.step()
        ready.lift(0, 0, x=-.1)
        ready.step()
        ready.lift(0, 0, x=-.1)
        self.assertFalse(bool(ready.step().active[0, 0]))

    def test_height_reference_frames_and_foot_radius(self):
        rig = Rig()
        rig.warmup()
        rig.swing()
        self.assertEqual(int(rig.state.target_step[0, 0]), 1)
        # H = 0.08 - 0 + 0.04 = 0.12 in both the world and body frames.
        self.assertEqual(float(rig.frame(0.).phi[0, 0]), 0.)
        rig.air(0, 0, x=-.1, z=.062, bz=.01)
        half = rig.step()
        self.assertAlmostEqual(float(half.phi[0, 0]), .25, delta=1e-6)
        self.assertPaid(half.reward[0], .25 * PAY)
        rig.air(0, 0, x=-.1, z=.062, bz=.04)
        body_only = rig.step()
        self.assertAlmostEqual(float(body_only.phi[0, 0]), .25, delta=1e-6)
        self.assertPaid(body_only.reward[0], 0.)
        rig.air(0, 0, x=-.1, z=.152, bz=0.)
        world_only = rig.step()
        self.assertEqual(float(world_only.phi[0, 0]), 0.)
        self.assertPaid(world_only.reward[0], 0.)
        rig.air(0, 0, x=-.1, z=.092, bz=.04)
        self.assertAlmostEqual(float(rig.step().phi[0, 0]), .5, delta=1e-6)

    def test_only_body_or_only_world_lift_cannot_full_reward(self):
        rig = Rig()
        rig.warmup()
        rig.swing()
        rig.air(0, 0, x=-.1, z=.032, bz=.04)
        body = rig.step()
        self.assertEqual(float(body.phi[0, 0]), 0.)
        self.assertPaid(body.reward[0], 0.)
        rig.air(0, 0, x=-.1, z=.152, bz=0.)
        world = rig.step()
        self.assertEqual(float(world.phi[0, 0]), 0.)
        rig.air(0, 0, x=-.1, z=.152, bz=.02)
        split_a = rig.step()
        self.assertAlmostEqual(float(split_a.phi[0, 0]), .5, delta=1e-6)
        self.assertPaid(split_a.reward[0], .5 * PAY)
        rig.air(0, 0, x=-.1, z=.092, bz=.04)
        split_b = rig.step()
        self.assertAlmostEqual(float(split_b.phi[0, 0]), .5, delta=1e-6)
        self.assertPaid(split_b.reward[0], 0.)
        self.assertLess(float(rig.state.actual_paid_amount[0, 0, 0]), PAY)

    def test_phi_progression_pays_at_most_015(self):
        rig = Rig()
        rig.warmup()
        rig.swing()
        first = rig.frame(.5)
        second = rig.frame(1.)
        repeat = rig.frame(1.)
        self.assertPaid(first.reward[0], .5 * PAY)
        self.assertPaid(second.reward[0], .5 * PAY)
        self.assertPaid(repeat.reward[0], 0.)
        self.assertAlmostEqual(float(rig.state.actual_paid_amount[0, 0, 0]), PAY, delta=1e-6)
        self.assertGreaterEqual(float(rig.state.paid_fraction[0, 0, 0]), 1. - 1e-6)
        self.assertEqual(spec.SWING_AMOUNT_PER_TARGET, .15)

    def test_descend_reascend_and_repeat_same_target_never_exceed_budget(self):
        rig = Rig()
        rig.warmup()
        rig.swing()
        self.assertPaid(rig.frame(1.).reward[0], PAY)
        self.assertPaid(rig.frame(.2).reward[0], 0.)
        self.assertPaid(rig.frame(1.).reward[0], 0.)
        rig.roll = math.radians(31.)
        self.assertPaid(rig.frame(1.).reward[0], 0.)
        rig.roll = 0.
        self.assertPaid(rig.frame(1.).reward[0], 0.)
        self.assertAlmostEqual(float(rig.state.actual_paid_amount[0, 0, 0]), PAY, delta=1e-6)

    def test_gate_off_consumes_fraction_without_paying(self):
        rig = Rig()
        rig.warmup()
        rig.swing()
        self.assertPaid(rig.frame(.5).reward[0], .5 * PAY)
        rig.roll = math.radians(31.)
        blocked = rig.frame(1.)
        self.assertFalse(bool(blocked.positive_gate[0]))
        self.assertPaid(blocked.reward[0], 0.)
        self.assertAlmostEqual(float(blocked.consumed_fraction[0, 0]), 1., delta=1e-6)
        self.assertAlmostEqual(float(rig.state.paid_fraction[0, 0, 0]), 1., delta=1e-6)
        self.assertAlmostEqual(float(rig.state.actual_paid_amount[0, 0, 0]), .5 * PAY,
                               delta=1e-6)
        rig.roll = 0.
        recovered = rig.frame(1.)
        self.assertTrue(bool(recovered.positive_gate[0]))
        self.assertPaid(recovered.reward[0], 0.)
        self.assertPaid(rig.frame(.4).reward[0], 0.)

    def test_failure_frame_has_no_positive_reward_and_blocks_new_swings(self):
        rig = Rig()
        rig.warmup()
        rig.swing()
        self.assertPaid(rig.frame(.5).reward[0], .5 * PAY)
        rig.failure = True
        failed = rig.frame(1.)
        self.assertFalse(bool(failed.positive_gate[0]))
        self.assertPaid(failed.reward[0], 0.)
        self.assertAlmostEqual(float(failed.consumed_fraction[0, 0]), 1., delta=1e-6)
        rig.failure = False
        self.assertPaid(rig.frame(1.).reward[0], 0.)
        self.assertAlmostEqual(float(rig.state.actual_paid_amount[0, 0, 0]), .5 * PAY,
                               delta=1e-6)
        blocked = Rig()
        blocked.warmup()
        blocked.failure = True
        self.assertFalse(bool(blocked.swing().active[0, 0]))
        blocked.failure = False
        self.assertTrue(bool(blocked.swing(frames=1).active[0, 0]))

    def test_support_posture_nonfoot_suppression_still_consumes(self):
        def exercise(name, apply, restore):
            with self.subTest(name=name):
                rig = Rig()
                rig.warmup()
                rig.swing()
                self.assertPaid(rig.frame(.5).reward[0], .5 * PAY)
                apply(rig)
                blocked = rig.frame(1.)
                self.assertFalse(bool(blocked.positive_gate[0]), msg=name)
                self.assertPaid(blocked.reward[0], 0.)
                self.assertAlmostEqual(float(blocked.consumed_fraction[0, 0]), 1., delta=1e-6)
                restore(rig)
                recovered = rig.frame(1.)
                self.assertTrue(bool(recovered.positive_gate[0]), msg=name)
                self.assertPaid(recovered.reward[0], 0.)
                self.assertAlmostEqual(float(rig.state.actual_paid_amount[0, 0, 0]),
                                       .5 * PAY, delta=1e-6)

        exercise('roll', lambda rig: setattr(rig, 'roll', math.radians(31.)),
                 lambda rig: setattr(rig, 'roll', 0.))
        exercise('pitch forward', lambda rig: setattr(rig, 'pitch', math.radians(-15.)),
                 lambda rig: setattr(rig, 'pitch', 0.))
        exercise('pitch up', lambda rig: setattr(rig, 'pitch', math.radians(56.)),
                 lambda rig: setattr(rig, 'pitch', 0.))
        exercise('nonfoot', lambda rig: setattr(rig, 'nonfoot_hit', True),
                 lambda rig: setattr(rig, 'nonfoot_hit', False))
        exercise('caller disabled', lambda rig: setattr(rig, 'caller_enabled', False),
                 lambda rig: setattr(rig, 'caller_enabled', True))

        rig = Rig()
        rig.warmup()
        rig.swing()
        self.assertPaid(rig.frame(.5).reward[0], .5 * PAY)
        for foot in (1, 2, 3):
            rig.air(0, foot, x=-.1)
        rig.step()
        blocked = rig.frame(1.)
        self.assertFalse(bool(blocked.positive_gate[0]))
        self.assertPaid(blocked.reward[0], 0.)
        self.assertAlmostEqual(float(blocked.consumed_fraction[0, 0]), 1., delta=1e-6)
        for _ in range(2):
            for foot in (1, 2, 3):
                rig.support(0, foot, 0)
            rig.step()
        self.assertPaid(rig.frame(1.).reward[0], 0.)
        self.assertAlmostEqual(float(rig.state.actual_paid_amount[0, 0, 0]), .5 * PAY,
                               delta=1e-6)

    def test_target_window_boundaries_and_lock_after_window_entry(self):
        def started(offset):
            rig = Rig()
            rig.warmup()
            rig.swing(x=-offset)
            return rig

        no_lock = started(WINDOW + 1e-4)
        result = no_lock.step()
        self.assertTrue(bool(result.active[0, 0]))
        self.assertEqual(int(result.target_step[0, 0]), -1)
        self.assertEqual(float(result.reward[0]), 0.)
        for offset, target in ((WINDOW, 1), (0., 1), (-1e-4, -1)):
            rig = started(offset)
            self.assertEqual(int(rig.step().target_step[0, 0]), target, msg=str(offset))
        # A target that appears beyond the window locks on the first sample inside.
        rig = Rig()
        rig.warmup()
        rig.swing(x=-(WINDOW + .10))
        self.assertEqual(int(rig.step().target_step[0, 0]), -1)
        rig.lift(0, 0, x=-(WINDOW - .10))
        self.assertEqual(int(rig.step().target_step[0, 0]), 1)
        # Once locked, crossing the front edge does not cancel the target.
        crossing = Rig()
        crossing.warmup()
        crossing.swing(x=-.1)
        self.assertEqual(int(crossing.state.target_step[0, 0]), 1)
        paid = crossing.frame(1., x=.05)
        self.assertEqual(int(paid.target_step[0, 0]), 1)
        self.assertPaid(paid.reward[0], PAY)
        self.assertEqual(spec.SWING_TARGET_WINDOW_M, WINDOW)

    def test_target_never_moves_to_a_different_step_after_lock(self):
        rig = Rig(nums=(4,))
        rig.warmup()
        rig.swing(x=-.1)
        self.assertEqual(int(rig.state.target_step[0, 0]), 1)
        # Land on step 1 twice so the foot has a new higher safe support, then lift
        # again: the new swing may target step 2, but the locked step-1 target is
        # never rewritten inside the same swing.
        for _ in range(2):
            rig.support(0, 0, 1, x=.1)
            rig.step()
        self.assertFalse(bool(rig.state.target_locked[0, 0]))
        for _ in range(2):
            rig.air(0, 0, z=.112, bz=.08, x=.1)
            result = rig.step()
        self.assertEqual(int(result.target_step[0, 0]), 2)

    def test_wrong_tile_metadata_is_an_error_and_decoy_is_ignored(self):
        wrong = Rig()
        wrong.warmup()
        wrong.move_geom(5, 1, 6)
        with self.assertRaisesRegex(ValueError, 'target 1'):
            for _ in range(2):
                wrong.lift(0, 0, x=-.1)
                wrong.step()
        decoy = Rig()
        decoy.add_geom(6, 1, .5, (0., .2, -.4, .4))
        decoy.warmup()
        decoy.swing()
        paid = decoy.frame(.5)
        self.assertEqual(int(paid.target_step[0, 0]), 1)
        self.assertAlmostEqual(float(paid.phi[0, 0]), .5, delta=1e-6)
        self.assertPaid(paid.reward[0], .5 * PAY)

    def test_target_lateral_42mm_boundary(self):
        rig = Rig()
        rig.warmup()
        rig.swing(x=-.1, y=.358)
        self.assertEqual(int(rig.state.target_step[0, 0]), 1)
        too_wide = Rig()
        too_wide.warmup()
        too_wide.swing(x=-.1, y=.3581)
        self.assertEqual(int(too_wide.state.target_step[0, 0]), -1)
        too_wide.lift(0, 0, x=-.1, y=.358)
        self.assertEqual(int(too_wide.step().target_step[0, 0]), 1)

    def test_last_step_has_no_n_plus_one_target(self):
        rig = Rig(nums=(2,))
        rig.warmup(step=2)
        self.assertEqual(int(rig.state.highest_safe_step[0, 0]), 2)
        result = rig.swing()
        self.assertTrue(bool(result.active[0, 0]))
        self.assertEqual(int(result.target_step[0, 0]), -1)
        self.assertEqual(float(result.reward[0]), 0.)
        self.assertEqual(float(rig.state.paid_fraction.abs().sum()), 0.)

    def test_s5_last_step_at_max_steps_does_not_wrap_onto_step_40(self):
        rig = Rig(nums=(40,), step_height=.2)
        self.assertEqual(rig.state.max_steps, spec.MAX_SWING_STEPS)
        rig.warmup()
        state = rig.state
        state.highest_safe_step[0, 0] = 40
        state.ready[0, 0] = True
        state.qualified_step[0, 0] = 40
        state.last_contact_bottom_z[0, 0] = 7.8
        state.last_contact_body_z[0, 0] = 7.8
        state.last_contact_support_z[0, 0] = 7.8
        state.no_contact_count[0, 0] = 0
        for _ in range(2):
            rig.air(0, 0, x=7.7, y=.1, z=7.8 + FOOT_RADIUS, bz=7.8)
            result = rig.step()
        self.assertTrue(bool(result.active[0, 0]))
        self.assertEqual(int(result.target_step[0, 0]), -1)
        self.assertFalse(bool(state.target_locked[0, 0]))
        self.assertEqual(float(result.reward[0]), 0.)
        self.assertEqual(float(state.paid_fraction.abs().sum()), 0.)

    def test_target_not_above_takeoff_support_is_not_valid(self):
        for top in (0., -.02):
            with self.subTest(top=top):
                rig = Rig(nums=(1,))
                rig.set_geom(5, 1, top=top)
                rig.warmup()
                result = rig.swing()
                self.assertTrue(bool(result.active[0, 0]))
                self.assertEqual(int(result.target_step[0, 0]), -1)
                self.assertEqual(float(result.reward[0]), 0.)

    def test_invalid_target_height_raises(self):
        rig = Rig()
        rig.warmup()
        rig.swing()
        self.assertEqual(int(rig.state.target_step[0, 0]), 1)
        rig.state.target_top_z[0, 0] = rig.state.takeoff_support_z[0, 0] - .05
        with self.assertRaisesRegex(ValueError, 'height'):
            rig.step()
        nan_rig = Rig()
        nan_rig.warmup()
        nan_rig.swing()
        nan_rig.state.target_top_z[0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'height'):
            nan_rig.step()

    def test_close_frame_consumes_phi_before_clearing_target(self):
        rig = Rig()
        rig.warmup()
        rig.swing()
        self.assertEqual(int(rig.state.target_step[0, 0]), 1)
        rig.contact(.5, step=0, x=-.1)
        first = rig.step()
        self.assertTrue(bool(first.active[0, 0]))
        self.assertPaid(first.reward[0], 0.)
        self.assertAlmostEqual(float(rig.state.paid_fraction[0, 0, 0]), .5, delta=1e-6)
        self.assertAlmostEqual(float(rig.state.actual_paid_amount[0, 0, 0]), 0., delta=1e-9)
        rig.contact(1., step=0, x=-.1)
        second = rig.step()
        self.assertFalse(bool(second.active[0, 0]))
        self.assertFalse(bool(rig.state.target_locked[0, 0]))
        self.assertEqual(int(second.target_step[0, 0]), -1)
        self.assertPaid(second.reward[0], 0.)
        self.assertAlmostEqual(float(rig.state.paid_fraction[0, 0, 0]), 1., delta=1e-6)
        self.assertAlmostEqual(float(rig.state.actual_paid_amount[0, 0, 0]), 0., delta=1e-9)
        for _ in range(2):
            rig.support(0, 0, 0, x=-.1)
            rig.step()
        rig.swing(x=-.1)
        self.assertEqual(int(rig.state.target_step[0, 0]), 1)
        self.assertPaid(rig.frame(1., x=-.1).reward[0], 0.)
        self.assertPaid(rig.state.actual_paid_amount[0, 0, 0], 0.)

    def test_last_step_union_bounds_are_used_for_target_and_support(self):
        rig = Rig(nums=(2,))
        rig.set_geom(5, 2, bounds=(.2, .4, -.8, .8))
        rig.warmup(step=1)
        rig.swing(x=.15, y=.6)
        self.assertEqual(int(rig.state.target_step[0, 0]), 2)
        self.assertPaid(rig.frame(1., x=.15, y=.6).reward[0], PAY)
        support = Rig(nums=(2,))
        support.set_geom(5, 2, bounds=(.2, .4, -.8, .8))
        support.warmup(step=2, y=.6)
        self.assertTrue(bool(support.state.safe_support.all()))
        self.assertEqual(support.state.selected_step.tolist(), [[2, 2, 2, 2]])

    def test_ground_step_zero_real_top_contact_and_selection(self):
        ground = Rig()
        ground.warmup()
        self.assertTrue(bool(ground.state.safe_support.all()))
        self.assertEqual(ground.state.selected_step.tolist(), [[0, 0, 0, 0]])
        self.assertEqual(int(ground.state.highest_safe_step[0, 0]), 0)
        top = Rig()
        top.warmup()
        for _ in range(2):
            top.support(0, 0, 1, x=.1)
            result = top.step()
        self.assertTrue(bool(result.safe_support[0, 0]))
        self.assertEqual(int(result.selected_step[0, 0]), 1)
        self.assertEqual(int(top.state.highest_safe_step[0, 0]), 1)
        higher = Rig()
        for _ in range(2):
            higher.found[0, 0] = True
            higher.foot_w[0, 0, 0] = .1
            higher.foot_w[0, 0, 1] = .1
            higher.foot_w[0, 0, 2] = FOOT_RADIUS
            higher.forces[0, 0, :] = 0.
            higher.forces[0, 0, higher._index(5, 0)] = 6.
            higher.forces[0, 0, higher._index(5, 1)] = 7.
            result = higher.step()
        self.assertEqual(int(result.selected_step[0, 0]), 1)
        tied = Rig()
        for _ in range(2):
            tied.found[0, 0] = True
            tied.foot_w[0, 0, 0] = .1
            tied.foot_w[0, 0, 1] = .1
            tied.foot_w[0, 0, 2] = FOOT_RADIUS
            tied.forces[0, 0, :] = 0.
            tied.forces[0, 0, tied._index(5, 0)] = 6.
            tied.forces[0, 0, tied._index(5, 1)] = 6.
            result = tied.step()
        self.assertEqual(int(result.selected_step[0, 0]), 0)

    def test_42mm_and_5n_support_boundaries_with_aggregation(self):
        for y, expected in ((.358, True), (.3581, False)):
            with self.subTest(kind='42mm', y=y):
                rig = Rig()
                for _ in range(2):
                    for foot in range(4):
                        rig.support(0, foot, 0, y=y if foot == 0 else .1)
                    result = rig.step()
                self.assertEqual(bool(result.safe_support[0, 0]), expected)
        for force, expected in ((5., False), (5.0001, True)):
            with self.subTest(kind='5N', force=force):
                rig = Rig()
                for _ in range(2):
                    for foot in range(4):
                        rig.support(0, foot, 0, force=force)
                    result = rig.step()
                self.assertEqual(bool(result.safe_support[0, 0]), expected)
        for first, second, expected in ((3., 3., True), (3., 2., False)):
            with self.subTest(kind='aggregate', force=(first, second)):
                rig = Rig()
                duplicate = rig.add_geom(5, 0, 0., (-.5, .5, -.4, .4))
                for _ in range(2):
                    for foot in range(4):
                        rig.support(0, foot, 0)
                    rig.forces[0, 0, :] = 0.
                    rig.forces[0, 0, rig._index(5, 0)] = first
                    rig.forces[0, 0, duplicate] = second
                    result = rig.step()
                self.assertEqual(bool(result.safe_support[0, 0]), expected)

    def test_same_surface_debounce_resets_on_step_or_tile_change(self):
        rig = Rig()
        for step in (0, 1, 0, 1):
            for foot in range(4):
                rig.support(0, foot, step)
            self.assertFalse(bool(rig.step().safe_support.any()), msg=f'step {step}')
        for _ in range(2):
            for foot in range(4):
                rig.support(0, foot, 1)
            result = rig.step()
        self.assertTrue(bool(result.safe_support.all()))
        self.assertEqual(result.selected_step.tolist(), [[1, 1, 1, 1]])
        foreign = Rig()
        foreign_index = foreign.add_geom(6, 0, 0., (-.5, .5, -.4, .4))
        for _ in range(2):
            for foot in range(4):
                foreign.found[0, foot] = True
                foreign.foot_w[0, foot, 0] = 0.
                foreign.foot_w[0, foot, 1] = .1
                foreign.foot_w[0, foot, 2] = FOOT_RADIUS
                foreign.forces[0, foot, :] = 0.
                foreign.forces[0, foot, foreign_index] = 6.
            result = foreign.step()
        self.assertFalse(bool(result.safe_support.any()))
        self.assertEqual(result.selected_step.tolist(), [[-1, -1, -1, -1]])

    def test_reset_clears_only_selected_environments(self):
        rig = Rig(tiles=(5, 6, 7), nums=(4, 4, 4))
        state = rig.state
        state.paid_fraction[:] = .3
        state.actual_paid_amount[:] = .07
        state.target_step[:] = 2
        state.target_locked[:] = True
        state.ready[:] = True
        state.active[:] = True
        state.safe_support[:] = True
        state.phi[:] = .4
        state.increment[:] = .01
        state.qualified_step[:] = 1
        state.highest_safe_step[:] = 1
        state._last_tile[:] = 5
        preserved = {name: getattr(state, name)[2].clone() for name in
                     ('paid_fraction', 'actual_paid_amount', 'target_step', 'target_locked',
                      'ready', 'active', 'safe_support', 'phi', 'qualified_step',
                      'highest_safe_step')}
        state.reset([0])
        self.assertEqual(float(state.paid_fraction[0].abs().sum()), 0.)
        self.assertEqual(float(state.actual_paid_amount[0].abs().sum()), 0.)
        self.assertTrue(bool((state.target_step[0] == -1).all()))
        self.assertFalse(bool(state.target_locked[0].any()))
        self.assertFalse(bool(state.ready[0].any()))
        self.assertFalse(bool(state.active[0].any()))
        self.assertEqual(float(state.phi[0].abs().sum()), 0.)
        self.assertAlmostEqual(float(state.paid_fraction[1].abs().sum()),
                               .3 * 4 * spec.MAX_SWING_STEPS, delta=1e-4)
        for name, value in preserved.items():
            self.assertTrue(torch.equal(getattr(state, name)[2], value), name)
        state.reset(torch.tensor([False, True, False]))
        self.assertEqual(float(state.paid_fraction[1].abs().sum()), 0.)
        for name, value in preserved.items():
            self.assertTrue(torch.equal(getattr(state, name)[2], value), name)
        state.reset()
        self.assertEqual(float(state.paid_fraction.abs().sum()), 0.)
        with self.assertRaises(ValueError):
            state.reset([3])
        with self.assertRaises(TypeError):
            state.reset(torch.tensor([0, 1, 2], dtype=torch.int32))
        with self.assertRaises(ValueError):
            state.reset(torch.tensor([True, False]))

    def test_adjacent_foot_and_target_budgets_are_isolated(self):
        rig = Rig()
        rig.warmup()
        for _ in range(2):
            for foot in (0, 1):
                rig.lift(0, foot, x=-.1)
            result = rig.step()
        self.assertEqual(result.target_step.tolist()[0][:2], [1, 1])
        both = rig.frame(1., feet=(0, 1))
        self.assertPaid(both.reward[0], 2 * PAY)
        self.assertPaid(rig.state.actual_paid_amount[0, 0, 0], PAY)
        self.assertPaid(rig.state.actual_paid_amount[0, 1, 0], PAY)
        for _ in range(2):
            rig.support(0, 0, 1, x=.1)
            rig.step()
        self.assertFalse(bool(rig.state.active[0, 0]))
        for _ in range(2):
            rig.air(0, 0, z=.112, bz=.08, x=.1)
            result = rig.step()
        self.assertEqual(int(result.target_step[0, 0]), 2)
        paid = rig.frame(1., feet=(0,), x=.1)
        self.assertPaid(paid.reward[0], PAY)
        self.assertPaid(rig.state.actual_paid_amount[0, 0, 0], PAY)
        self.assertPaid(rig.state.actual_paid_amount[0, 0, 1], PAY)
        self.assertPaid(rig.state.actual_paid_amount[0, 1, 0], PAY)
        self.assertEqual(float(rig.state.actual_paid_amount[0, 2:].abs().sum()), 0.)

    def test_adjacent_environment_budgets_are_isolated(self):
        rig = Rig(tiles=(5, 6), nums=(4, 8))
        rig.warmup()
        rig.swing(env=0, x=-.1)
        rig.swing(env=1, x=-.1)
        self.assertEqual(rig.state.target_step[:, 0].tolist(), [1, 1])
        first = rig.frame(.5, env=0)
        self.assertPaid(first.reward[0], .5 * PAY)
        second = rig.frame(.5, env=1)
        self.assertPaid(second.reward[1], .5 * PAY)
        third = rig.frame(1., env=0)
        self.assertPaid(third.reward[0], .5 * PAY)
        self.assertPaid(third.reward[1], 0.)
        fourth = rig.frame(1., env=1)
        self.assertPaid(fourth.reward[0], 0.)
        self.assertPaid(fourth.reward[1], .5 * PAY)
        self.assertPaid(rig.state.actual_paid_amount[0, 0, 0], PAY)
        self.assertPaid(rig.state.actual_paid_amount[1, 0, 0], PAY)

    def test_flat_n0_has_no_target_no_event_and_opens_the_gate(self):
        flat = Rig(tiles=(0,), nums=(0,))
        result = flat.warmup()
        self.assertTrue(bool(result.safe_support.all()))
        self.assertTrue(bool(result.positive_gate[0]))
        self.assertEqual(flat.state.selected_step.tolist(), [[0, 0, 0, 0]])
        self.assertEqual(int(flat.state.highest_safe_step[0, 0]), 0)
        started = flat.swing()
        self.assertTrue(bool(started.active[0, 0]))
        self.assertEqual(int(started.target_step[0, 0]), -1)
        self.assertFalse(bool(flat.state.target_locked[0, 0]))
        self.assertEqual(float(started.reward[0]), 0.)
        # Only the ground geom exists, so any step-1 query would raise; leaving
        # the loop clean proves the flat path never asks for a fake target.
        for _ in range(5):
            result = flat.step()
            self.assertEqual(float(result.reward[0]), 0.)
        self.assertEqual(float(flat.state.paid_fraction.abs().sum()), 0.)
        self.assertEqual(float(flat.state.actual_paid_amount.abs().sum()), 0.)

    def test_mixed_flat_and_stair_batch_never_validates_flat_rows(self):
        """Regression: only ``required`` rows may be checked against real geometry.

        A mixed-slot batch (SDS 5.1: e.g. P1 = flat + current stairs) always has
        flat ``N == 0`` environments whose placeholder candidate has no target
        geom.  The v0.4-era ``_target_geom`` asserted ``finite.any(-1).all()`` over
        the whole batch, so one flat slot aborted the whole control step with
        ``target top metadata is not finite`` as soon as any stair row queried a
        candidate.  Flat rows must not participate in the match/finiteness check.
        """
        rig = Rig(tiles=(0, 5), nums=(0, 4))
        for _ in range(2):
            for env in range(rig.B):
                for foot in range(4):
                    rig.support(env, foot, 0)
            rig.step()
        # Only the stair environment swings; the flat environment stays N=0.
        for _ in range(2):
            rig.lift(1, 0, x=-.1)
            result = rig.step()
        self.assertTrue(bool(result.active[1, 0]))
        self.assertEqual(int(result.target_step[1, 0]), 1)
        self.assertEqual(int(result.target_step[0, 0]), -1)
        self.assertFalse(bool(result.active[0].any()))
        self.assertFalse(bool(rig.state.target_locked[0].any()))
        self.assertEqual(float(result.reward[0]), 0.)
        paid = rig.frame(.5, env=1)
        self.assertAlmostEqual(float(paid.reward[1]), .5 * PAY, delta=1e-6)
        self.assertEqual(float(paid.reward[0]), 0.)
        self.assertTrue(bool(paid.safe_support[0].all()))
        self.assertTrue(bool(paid.positive_gate[0]))
        # The batch survives many more mixed steps without a spurious metadata error.
        for _ in range(5):
            rig.lift(1, 0, x=-.1)
            rig.step()
        # A required row with no matching geom still fails closed in the same
        # mixed batch (the fix must not turn the metadata error off).
        broken = Rig(tiles=(0, 5), nums=(0, 4))
        for _ in range(2):
            for env in range(broken.B):
                for foot in range(4):
                    broken.support(env, foot, 0)
            broken.step()
        broken.move_geom(5, 1, 6)
        with self.assertRaisesRegex(ValueError, 'target 1'):
            for _ in range(2):
                broken.lift(1, 0, x=-.1)
                broken.step()

    def test_invalid_swing_inputs_raise(self):
        state = api().SwingClearanceState(1)
        rig = Rig()
        sample = rig.sample()
        with self.assertRaises(TypeError):
            state.update(None)
        for field, value in (
                ('num_steps', torch.tensor([-1], dtype=torch.long)),
                ('num_steps', torch.tensor([41], dtype=torch.long)),
                ('tile_ids', torch.tensor([5], dtype=torch.int32)),
                ('foot_pos_w', torch.zeros(1, 3, 3)),
                ('foot_pos_b', torch.zeros(1, 4, 2)),
                ('contact_found', torch.zeros(1, 4, dtype=torch.long)),
                ('top_forces', torch.zeros(1, 4)),
                ('geom_tile', torch.tensor([5], dtype=torch.long)[:0]),
                ('roll', torch.tensor([float('nan')])),
                ('executed_c', torch.tensor([float('inf')]))):
            with self.subTest(field=field, value=str(value)):
                changed = dict(sample.__dict__)
                changed[field] = value
                with self.assertRaises(ValueError):
                    state.update(api().SwingSample(**changed))
        for bad in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                api().SwingClearanceState(bad)
        with self.assertRaises(ValueError):
            api().SwingClearanceState(1, max_steps=0)
        # num_steps == 0 is legal for the flat slot.
        changed = dict(sample.__dict__)
        changed['num_steps'] = torch.tensor([0], dtype=torch.long)
        state.update(api().SwingSample(**changed))


class FootfallTests(unittest.TestCase):
    """SDS 3.5: two-frame debounce, 0.50 per foot/step/tile, no platform double pay."""

    def build(self, tiles=(5,), nums=(4,)):
        specs = []
        for tile, n in zip(tiles, nums):
            specs.append((tile, 0, 0., (-.5, .5, -.4, .4)))
            for k in range(1, n + 1):
                specs.append((tile, k, k * .08, ((k - 1) * .2, k * .2, -.4, .4)))
        forces = torch.zeros(len(tiles), 4, len(specs))
        return specs, forces

    def charge(self, specs, forces, env, foot, step, tile, force=6., x=None, y=.1):
        index = next(i for i, spec in enumerate(specs) if spec[0] == tile and spec[1] == step)
        bounds = specs[index][3]
        forces[env, foot, :] = 0.
        forces[env, foot, index] = force
        return torch.tensor((bounds[0] + bounds[1]) / 2. if x is None else x,
                            dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def test_two_frame_debounce_pays_fifty_cents_per_foot(self):
        specs, forces = self.build()
        geom_tile = torch.tensor([s[0] for s in specs], dtype=torch.long)
        geom_step = torch.tensor([s[1] for s in specs], dtype=torch.long)
        geom_bounds = torch.tensor([s[3] for s in specs], dtype=torch.float32)
        state = r.FootfallState(1)
        state.reset(torch.tensor([0]))
        foot_xy = torch.full((1, 4, 2), .1)
        failure = torch.tensor([False])
        tile = torch.tensor([5], dtype=torch.long)
        for foot in range(4):
            foot_xy[0, foot] = torch.stack(self.charge(specs, forces, 0, foot, 1, 5))
        first = state.update(tile, foot_xy, forces, geom_tile, geom_step, geom_bounds, failure)
        self.assertEqual(int(first.events.sum()), 0)
        self.assertEqual(float(first.reward[0]), 0.)
        self.assertEqual(first.qualified_count.tolist(), [[1, 1, 1, 1]])
        second = state.update(tile, foot_xy, forces, geom_tile, geom_step, geom_bounds, failure)
        self.assertEqual(int(second.events.sum()), 4)
        self.assertEqual(float(second.reward[0]), 2.0)
        self.assertAlmostEqual(float(r.footfall_reward(second.events)[0]), 2.0, delta=1e-9)
        third = state.update(tile, foot_xy, forces, geom_tile, geom_step, geom_bounds, failure)
        self.assertEqual(int(third.events.sum()), 0)
        self.assertEqual(float(third.reward[0]), 0.)

    def test_ground_does_not_pay_and_platform_shares_the_last_step(self):
        specs, forces = self.build(nums=(2,))
        geom_tile = torch.tensor([s[0] for s in specs], dtype=torch.long)
        geom_step = torch.tensor([s[1] for s in specs], dtype=torch.long)
        geom_bounds = torch.tensor([s[3] for s in specs], dtype=torch.float32)
        state = r.FootfallState(1)
        state.reset(torch.tensor([0]))
        failure = torch.tensor([False])
        tile = torch.tensor([5], dtype=torch.long)
        foot_xy = torch.full((1, 4, 2), .1)
        foot_xy[0, 0] = torch.stack(self.charge(specs, forces, 0, 0, 0, 5))
        for _ in range(3):
            result = state.update(tile, foot_xy, forces, geom_tile, geom_step, geom_bounds,
                                  failure)
        self.assertEqual(int(result.events.sum()), 0)
        self.assertEqual(float(result.reward[0]), 0.)
        # The coplanar platform re-uses the last tread's step id: paying once must
        # cover both geoms, including a second platform-only geom.
        platform = len(specs)
        specs = specs + [(5, 2, .16, (.4, 2.4, -.4, .4))]
        forces = torch.cat((forces, torch.zeros(1, 4, 1)), dim=2)
        geom_tile = torch.cat((geom_tile, torch.tensor([5], dtype=torch.long)))
        geom_step = torch.cat((geom_step, torch.tensor([2], dtype=torch.long)))
        geom_bounds = torch.cat((geom_bounds, torch.tensor([[.4, 2.4, -.4, .4]])))
        foot_xy[0, 1] = torch.stack(self.charge(specs, forces, 0, 1, 2, 5, force=6., x=2.0))
        for _ in range(2):
            result = state.update(tile, foot_xy, forces, geom_tile, geom_step, geom_bounds,
                                  failure)
        self.assertEqual(int(result.events[0, 1]), 1)
        self.assertAlmostEqual(float(result.reward[0]), .50, delta=1e-9)
        # A later platform frame at the same step cannot pay a second time.
        for _ in range(3):
            result = state.update(tile, foot_xy, forces, geom_tile, geom_step, geom_bounds,
                                  failure)
        self.assertEqual(int(result.events.sum()), 0)
        self.assertEqual(specs[platform][1], 2)

    def test_failure_frame_has_zero_events_and_disabled_does_not_advance(self):
        specs, forces = self.build()
        geom_tile = torch.tensor([s[0] for s in specs], dtype=torch.long)
        geom_step = torch.tensor([s[1] for s in specs], dtype=torch.long)
        geom_bounds = torch.tensor([s[3] for s in specs], dtype=torch.float32)
        state = r.FootfallState(1)
        state.reset(torch.tensor([0]))
        tile = torch.tensor([5], dtype=torch.long)
        foot_xy = torch.full((1, 4, 2), .1)
        for foot in range(4):
            foot_xy[0, foot] = torch.stack(self.charge(specs, forces, 0, foot, 1, 5))
        success = torch.tensor([False])
        state.update(tile, foot_xy, forces, geom_tile, geom_step, geom_bounds, success)
        failed = state.update(tile, foot_xy, forces, geom_tile, geom_step, geom_bounds,
                              torch.tensor([True]))
        self.assertEqual(int(failed.events.sum()), 0)
        self.assertEqual(float(failed.reward[0]), 0.)
        # After the failure frame the same two-frame debounce restarts and pays once.
        recovered = state.update(tile, foot_xy, forces, geom_tile, geom_step, geom_bounds,
                                 success)
        self.assertEqual(int(recovered.events.sum()), 4)
        disabled = state.update(tile, foot_xy, forces, geom_tile, geom_step, geom_bounds,
                                success, enabled=False)
        self.assertEqual(int(disabled.events.sum()), 0)
        self.assertEqual(float(disabled.reward[0]), 0.)
        # A disabled frame does not advance the support debounce.  The failure
        # frame did advance it (only payment is blocked), so the counter is 3.
        self.assertEqual(int(state.qualified_count[0, 0]), 3)
        with self.assertRaises(TypeError):
            state.update(tile, foot_xy, forces, geom_tile, geom_step, geom_bounds, success,
                         enabled=1)

    def test_reset_clears_the_per_episode_payment_flags(self):
        specs, forces = self.build()
        geom_tile = torch.tensor([s[0] for s in specs], dtype=torch.long)
        geom_step = torch.tensor([s[1] for s in specs], dtype=torch.long)
        geom_bounds = torch.tensor([s[3] for s in specs], dtype=torch.float32)
        state = r.FootfallState(2)
        state.reset(torch.arange(2))
        tile = torch.tensor([5, 5], dtype=torch.long)
        foot_xy = torch.full((2, 4, 2), .1)
        forces = forces.expand(2, 4, -1).clone()
        for env in range(2):
            for foot in range(4):
                foot_xy[env, foot] = torch.stack(self.charge(specs, forces, env, foot, 1, 5))
        failure = torch.tensor([False, False])
        state.update(tile, foot_xy, forces, geom_tile, geom_step, geom_bounds, failure)
        paid = state.update(tile, foot_xy, forces, geom_tile, geom_step, geom_bounds, failure)
        self.assertEqual(int(paid.events.sum()), 8)
        state.reset(torch.tensor([0]))
        self.assertFalse(bool(state.foot_step_rewarded[0].any()))
        self.assertTrue(bool(state.foot_step_rewarded[1, :, 0].all()))
        self.assertFalse(bool(state.foot_step_rewarded[1, :, 1:].any()))
        again = state.update(tile, foot_xy, forces, geom_tile, geom_step, geom_bounds, failure)
        self.assertEqual(int(again.events[0].sum()), 0)
        self.assertEqual(int(again.events[1].sum()), 0)

    def test_events_recompute_is_fifty_cents(self):
        events = torch.tensor([[True, False, True, False],
                               [False, False, False, False]]).reshape(2, 4)
        self.assertAlmostEqual(float(r.footfall_reward(events)[0]), 1.0, delta=1e-12)
        self.assertEqual(float(r.footfall_reward(events)[1]), 0.)
        floating = torch.tensor([[1., 0., 0., 0.]])
        self.assertAlmostEqual(float(r.footfall_reward(floating)[0]), .5, delta=1e-12)
        with self.assertRaises(TypeError):
            r.footfall_reward(torch.tensor([1, 0]))
        with self.assertRaises(ValueError):
            r.footfall_reward(torch.tensor([[float('nan'), 0., 0., 0.]]))

    def test_step_candidates_semantics(self):
        rig = Rig(nums=(4,))
        selected, maximum = r.step_candidates(
            torch.tensor([5], dtype=torch.long), rig.foot_w[:, :, :2].clone(),
            rig.forces.clone(), torch.tensor([s[0] for s in rig.specs], dtype=torch.long),
            torch.tensor([s[1] for s in rig.specs], dtype=torch.long),
            torch.tensor([s[3] for s in rig.specs], dtype=torch.float32), 40)
        self.assertEqual(selected.tolist(), [[-1, -1, -1, -1]])
        self.assertTrue((~torch.isfinite(maximum)).all())
        with self.assertRaises(ValueError):
            r.step_candidates(torch.tensor([5], dtype=torch.long), rig.foot_w[:, :, :2],
                              rig.forces[:, :, :0].clone(),
                              torch.tensor([s[0] for s in rig.specs], dtype=torch.long),
                              torch.tensor([s[1] for s in rig.specs], dtype=torch.long),
                              torch.tensor([s[3] for s in rig.specs], dtype=torch.float32), 40)


if __name__ == '__main__':
    unittest.main()
