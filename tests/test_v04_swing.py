"""Synthetic v0.4 swing-clearance state-machine tests (T2).

Every case builds ``SwingSample``/``SwingClearanceState`` directly from small
synthetic geom metadata.  No simulator, policy, training or GPU is involved;
the 20 ms update is driven by hand.  Expected values come from SDS 0.4 §6.
"""
import importlib
import math
import unittest

import torch

FOOT_RADIUS = .032
PAY = .10
DT = .02


def api():
    try:
        return importlib.import_module('src.tasks.velocity.v04.rewards')
    except ModuleNotFoundError:
        raise AssertionError('V04 swing implementation is missing') from None


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

    def top(self, env, step, tile=None):
        return self.specs[self._env_index(env, step, tile)][2]

    def bounds(self, env, step, tile=None):
        return self.specs[self._env_index(env, step, tile)][3]

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
    def support(self, env, foot, step, force=6., x=None, y=.1, bz=None, tile=None):
        i = self._env_index(env, step, tile)
        top, bounds = self.specs[i][2], self.specs[i][3]
        self.found[env, foot] = True
        self.forces[env, foot, :] = 0.
        self.forces[env, foot, i] = force
        self.foot_w[env, foot, 0] = (bounds[0] + bounds[1]) / 2. if x is None else x
        self.foot_w[env, foot, 1] = y
        self.foot_w[env, foot, 2] = top + FOOT_RADIUS
        self.foot_b[env, foot, 2] = top if bz is None else bz

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
        rewards = api()
        return rewards.SwingSample(
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

    def land(self, env=0, foot=0, step=1, frames=2, x=None, y=.1):
        last = None
        for _ in range(frames):
            self.support(env, foot, step, x=x, y=y)
            last = self.step()
        return last


class SwingTests(unittest.TestCase):
    def assertPaid(self, value, expected, delta=1e-6):
        self.assertAlmostEqual(float(value), expected, delta=delta)

    # ------------------------------------------------------------- T2 basic
    def test_initial_air_has_no_reward_and_no_anchor(self):
        rig = Rig()
        for k in range(1, 6):
            for foot in range(4):
                rig.air(0, foot, h_world=k * .01)
            result = rig.step()
            self.assertEqual(float(result.reward[0]), 0.)
            self.assertFalse(bool(result.active.any()))
            self.assertTrue(bool((result.target_step == -1).all()))
            self.assertEqual(float(result.phi.abs().sum()), 0.)
        self.assertFalse(bool(rig.state.ready.any()))
        self.assertTrue(bool((rig.state.highest_safe_step == -1).all()))
        self.assertFalse(bool(result.positive_gate[0]))

    def test_two_safe_support_samples_then_two_contact_free_samples_start(self):
        rig = Rig()
        for foot in range(4):
            rig.support(0, foot, 0)
        first = rig.step()
        self.assertFalse(bool(first.safe_support.any()))  # one sample is not support
        self.assertFalse(bool(first.active.any()))
        for foot in range(4):
            rig.support(0, foot, 0)
        second = rig.step()
        self.assertTrue(bool(second.safe_support.all()))
        self.assertTrue(bool(rig.state.ready.all()))
        self.assertTrue(bool(second.positive_gate[0]))
        self.assertEqual(rig.state.selected_step.tolist(), [[0, 0, 0, 0]])

        rig.lift(0, 0, x=-.1)
        one_air = rig.step()
        self.assertFalse(bool(one_air.active[0, 0]))  # one airborne sample is not enough
        self.assertEqual(int(rig.state.no_contact_count[0, 0]), 1)
        rig.lift(0, 0, x=-.1)
        started = rig.step()
        self.assertTrue(bool(started.active[0, 0]))
        self.assertEqual(int(started.target_step[0, 0]), 1)
        self.assertEqual(float(started.phi[0, 0]), 0.)
        self.assertEqual(float(started.reward[0]), 0.)
        # Anchors are copied from the last confirmed support sample, not from
        # the already-airborne first frame.
        self.assertEqual(float(rig.state.takeoff_bottom_z[0, 0]), 0.)
        self.assertEqual(float(rig.state.takeoff_body_z[0, 0]), 0.)
        self.assertEqual(float(rig.state.takeoff_support_z[0, 0]), 0.)
        self.assertEqual(int(rig.state.highest_safe_step[0, 0]), 0)

    def test_single_frame_contact_jitter_and_side_contact_do_not_anchor(self):
        # A single real support frame followed by air never starts a swing.
        rig = Rig()
        rig.support(0, 0, 0)
        one_contact = rig.step()
        self.assertFalse(bool(one_contact.safe_support.any()))
        self.assertFalse(bool(rig.state.ready[0, 0]))
        for k in (1, 2, 3):
            rig.air(0, 0, h_world=.01 * k)
            result = rig.step()
        self.assertFalse(bool(result.active.any()))
        # Two real support samples do anchor, which proves the jitter case was
        # not rejected for an unrelated reason.
        for _ in range(2):
            for foot in range(4):
                rig.support(0, foot, 0)
            rig.step()
        rig.lift(0, 0, x=-.1)
        rig.step()
        rig.lift(0, 0, x=-.1)
        self.assertTrue(bool(rig.step().active[0, 0]))

        # A side/underside contact is found=true but contributes no qualifying
        # top force; it cannot become an anchor.
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
            result = side.step()
        self.assertFalse(bool(result.active.any()))

        # A one-frame contact while ready, then two contact-free samples, also
        # does not start a swing because ready is cleared by the unsafe contact.
        ready = Rig()
        ready.warmup()
        ready.lift(0, 0, x=-.1)
        ready.step()
        ready.support(0, 0, 0)
        result = ready.step()
        self.assertFalse(bool(rig.state.ready[0, 0]))
        ready.lift(0, 0, x=-.1)
        ready.step()
        ready.lift(0, 0, x=-.1)
        result = ready.step()
        self.assertFalse(bool(result.active[0, 0]))

    def test_height_reference_frames_and_foot_radius(self):
        rig = Rig()
        rig.warmup()
        rig.swing()
        self.assertEqual(int(rig.state.target_step[0, 0]), 1)
        # H = target_top(0.08) - takeoff_support(0) + 0.04 = 0.12.
        zero = rig.frame(0.)
        self.assertEqual(float(zero.phi[0, 0]), 0.)
        # bottom_z_world = foot_center - 0.032, and body height is its own
        # frame: world 0.03 / H 0.12 = 0.25, body 0.01 / 0.04 = 0.25.
        rig.air(0, 0, x=-.1, z=.062, bz=.01)
        half = rig.step()
        self.assertAlmostEqual(float(half.phi[0, 0]), .25, delta=1e-6)
        self.assertPaid(half.reward[0], .025)
        # Raising only body height cannot exceed the world-limited 0.25.
        rig.air(0, 0, x=-.1, z=.062, bz=.04)
        body_only = rig.step()
        self.assertAlmostEqual(float(body_only.phi[0, 0]), .25, delta=1e-6)
        self.assertPaid(body_only.reward[0], 0.)
        # Raising only world height with zero body lift cannot pay at all.
        rig.air(0, 0, x=-.1, z=.152, bz=0.)
        world_only = rig.step()
        self.assertEqual(float(world_only.phi[0, 0]), 0.)
        self.assertPaid(world_only.reward[0], 0.)
        rig.air(0, 0, x=-.1, z=.092, bz=.04)
        self.assertAlmostEqual(float(rig.step().phi[0, 0]), .5, delta=1e-6)

    def test_only_body_or_only_world_lift_and_split_peaks_cannot_full_reward(self):
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
        self.assertPaid(world.reward[0], 0.)
        # World peak with half body lift pays 0.05.  A later body peak with the
        # world at half height is still phi=0.5, so the split-frame peaks can
        # never combine into a full 0.10 payment.
        rig.air(0, 0, x=-.1, z=.152, bz=.02)
        split_a = rig.step()
        self.assertAlmostEqual(float(split_a.phi[0, 0]), .5, delta=1e-6)
        self.assertPaid(split_a.reward[0], .05)
        rig.air(0, 0, x=-.1, z=.092, bz=.04)
        split_b = rig.step()
        self.assertAlmostEqual(float(split_b.phi[0, 0]), .5, delta=1e-6)
        self.assertPaid(split_b.reward[0], 0.)
        self.assertLessEqual(float(rig.state.actual_paid_amount[0, 0, 0]), PAY + 1e-6)
        self.assertLess(float(rig.state.actual_paid_amount[0, 0, 0]), PAY)

    def test_phi_progression_pays_at_most_five_plus_five(self):
        rig = Rig()
        rig.warmup()
        rig.swing()
        first = rig.frame(.5)
        second = rig.frame(1.)
        repeat = rig.frame(1.)
        self.assertPaid(first.reward[0], .05)
        self.assertPaid(second.reward[0], .05)
        self.assertPaid(repeat.reward[0], 0.)
        self.assertAlmostEqual(float(rig.state.actual_paid_amount[0, 0, 0]), PAY, delta=1e-6)
        self.assertLessEqual(float(rig.state.actual_paid_amount[0, 0, 0]), PAY + 1e-6)
        self.assertGreaterEqual(float(rig.state.paid_fraction[0, 0, 0]), 1. - 1e-6)

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
        self.assertPaid(rig.frame(.5).reward[0], .05)
        rig.roll = math.radians(31.)
        blocked = rig.frame(1.)
        self.assertFalse(bool(blocked.positive_gate[0]))
        self.assertPaid(blocked.reward[0], 0.)
        self.assertAlmostEqual(float(blocked.consumed_fraction[0, 0]), 1., delta=1e-6)
        self.assertAlmostEqual(float(rig.state.paid_fraction[0, 0, 0]), 1., delta=1e-6)
        self.assertAlmostEqual(float(rig.state.actual_paid_amount[0, 0, 0]), .05, delta=1e-6)
        rig.roll = 0.
        recovered = rig.frame(1.)
        self.assertTrue(bool(recovered.positive_gate[0]))
        self.assertPaid(recovered.reward[0], 0.)
        self.assertPaid(rig.frame(.4).reward[0], 0.)

    def test_failure_frame_has_no_positive_reward(self):
        rig = Rig()
        rig.warmup()
        rig.swing()
        self.assertPaid(rig.frame(.5).reward[0], .05)
        rig.failure = True
        failed = rig.frame(1.)
        self.assertFalse(bool(failed.positive_gate[0]))
        self.assertPaid(failed.reward[0], 0.)
        self.assertAlmostEqual(float(failed.consumed_fraction[0, 0]), 1., delta=1e-6)
        rig.failure = False
        self.assertPaid(rig.frame(1.).reward[0], 0.)
        self.assertAlmostEqual(float(rig.state.actual_paid_amount[0, 0, 0]), .05, delta=1e-6)

        # Failure also blocks the start of a new swing, but the two contact-free
        # samples still count, so a later healthy frame can start.
        blocked = Rig()
        blocked.warmup()
        blocked.failure = True
        result = blocked.swing()
        self.assertFalse(bool(result.active[0, 0]))
        blocked.failure = False
        result = blocked.swing(frames=1)
        self.assertTrue(bool(result.active[0, 0]))

    def test_support_roll_pitch_nonfoot_suppression_still_consumes(self):
        def exercise(name, apply, restore):
            with self.subTest(name=name):
                rig = Rig()
                rig.warmup()
                rig.swing()
                self.assertPaid(rig.frame(.5).reward[0], .05)
                apply(rig)
                blocked = rig.frame(1.)
                self.assertFalse(bool(blocked.positive_gate[0]), msg=name)
                self.assertPaid(blocked.reward[0], 0., delta=1e-6)
                self.assertAlmostEqual(float(blocked.consumed_fraction[0, 0]), 1., delta=1e-6)
                restore(rig)
                recovered = rig.frame(1.)
                self.assertTrue(bool(recovered.positive_gate[0]), msg=name)
                self.assertPaid(recovered.reward[0], 0.)
                self.assertAlmostEqual(float(rig.state.actual_paid_amount[0, 0, 0]),
                                       .05, delta=1e-6)

        exercise('roll', lambda r: setattr(r, 'roll', math.radians(31.)),
                 lambda r: setattr(r, 'roll', 0.))
        exercise('pitch forward', lambda r: setattr(r, 'pitch', math.radians(-15.)),
                 lambda r: setattr(r, 'pitch', 0.))
        exercise('pitch up', lambda r: setattr(r, 'pitch', math.radians(56.)),
                 lambda r: setattr(r, 'pitch', 0.))
        exercise('nonfoot', lambda r: setattr(r, 'nonfoot_hit', True),
                 lambda r: setattr(r, 'nonfoot_hit', False))
        exercise('caller disabled', lambda r: setattr(r, 'caller_enabled', False),
                 lambda r: setattr(r, 'caller_enabled', True))

        # Fewer than two same-frame safe supports also consumes without paying.
        rig = Rig()
        rig.warmup()
        rig.swing()
        self.assertPaid(rig.frame(.5).reward[0], .05)
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
        self.assertAlmostEqual(float(rig.state.actual_paid_amount[0, 0, 0]), .05, delta=1e-6)

    # ------------------------------------------------------------ targeting
    def test_target_window_boundaries_and_lock_after_window_entry(self):
        def started(offset):
            rig = Rig()
            rig.warmup()
            rig.swing(x=-offset)
            return rig

        no_lock = started(.3001)
        result = no_lock.step()
        self.assertTrue(bool(result.active[0, 0]))
        self.assertEqual(int(result.target_step[0, 0]), -1)
        self.assertEqual(float(result.reward[0]), 0.)
        for offset, target in ((.30, 1), (0., 1), (-.0001, -1)):
            rig = started(offset)
            result = rig.step()
            self.assertEqual(int(result.target_step[0, 0]), target, msg=str(offset))

        # Started outside 0.30 m and locked on the first active sample that
        # enters the window.
        rig = Rig()
        rig.warmup()
        rig.swing(x=-.40)
        first = rig.step()
        self.assertEqual(int(first.target_step[0, 0]), -1)
        rig.lift(0, 0, x=-.20)
        entered = rig.step()
        self.assertEqual(int(entered.target_step[0, 0]), 1)

        # Once locked, crossing the front edge does not cancel the target.
        crossing = Rig()
        crossing.warmup()
        crossing.swing(x=-.1)
        self.assertEqual(int(crossing.state.target_step[0, 0]), 1)
        paid = crossing.frame(1., x=.05)
        self.assertEqual(int(paid.target_step[0, 0]), 1)
        self.assertPaid(paid.reward[0], PAY)

    def test_wrong_tile_metadata_is_an_error_and_decoy_is_ignored(self):
        wrong = Rig()
        wrong.warmup()
        wrong.move_geom(5, 1, 6)  # target step 1 now exists only under tile 6
        with self.assertRaisesRegex(ValueError, 'target 1'):
            for _ in range(2):
                wrong.lift(0, 0, x=-.1)
                wrong.step()
        decoy = Rig()
        decoy.add_geom(6, 1, .5, (0., .2, -.4, .4))
        decoy.warmup()
        decoy.swing()
        paid = decoy.frame(.5)
        # H uses the correct tile-5 top 0.08, so phi is 0.5; the tile-6 decoy
        # (top 0.5) must not enter the same-frame height.
        self.assertEqual(int(paid.target_step[0, 0]), 1)
        self.assertAlmostEqual(float(paid.phi[0, 0]), .5, delta=1e-6)
        self.assertPaid(paid.reward[0], .05)

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
        result = too_wide.step()
        self.assertEqual(int(result.target_step[0, 0]), 1)

    def test_last_step_has_no_n_plus_one_target(self):
        rig = Rig(nums=(2,))
        rig.warmup(step=2)
        self.assertEqual(int(rig.state.highest_safe_step[0, 0]), 2)
        result = rig.swing()
        self.assertTrue(bool(result.active[0, 0]))
        self.assertEqual(int(result.target_step[0, 0]), -1)
        self.assertEqual(float(result.reward[0]), 0.)
        self.assertEqual(float(rig.state.paid_fraction.abs().sum()), 0.)

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
        # The public lock path rejects a target top that is not above takeoff,
        # so the defensive H>0 guard is exercised by forcing the already-locked
        # target height; it must fail closed instead of paying a constant.
        rig.state.target_top_z[0, 0] = rig.state.takeoff_support_z[0, 0] - .05
        with self.assertRaisesRegex(ValueError, 'height'):
            rig.step()
        nan_rig = Rig()
        nan_rig.warmup()
        nan_rig.swing()
        nan_rig.state.target_top_z[0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'height'):
            nan_rig.step()

    def test_last_step_at_max_steps_does_not_wrap_onto_step_40(self):
        # 41 geoms: ground plus the 40 steps of S5.  A confirmed last support on
        # step 40 makes the only candidate 41; the upper clamp must not turn it
        # back into the already-reached step 40.
        rig = Rig(nums=(40,), step_height=.2)
        self.assertEqual(rig.state.max_steps, 40)
        self.assertEqual(rig.nums[0], 40)
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

    def test_close_frame_consumes_phi_before_clearing_target(self):
        rig = Rig()
        rig.warmup()
        rig.swing()
        self.assertEqual(int(rig.state.target_step[0, 0]), 1)
        # First of two consecutive actual contacts: no payment, but phi 0.5 is
        # consumed and the swing stays active for one more frame.
        rig.contact(.5, step=0, x=-.1)
        first = rig.step()
        self.assertTrue(bool(first.active[0, 0]))
        self.assertPaid(first.reward[0], 0.)
        self.assertAlmostEqual(float(rig.state.paid_fraction[0, 0, 0]), .5, delta=1e-6)
        self.assertPaid(rig.state.actual_paid_amount[0, 0, 0], 0.)
        # The second contact frame closes the swing.  Its phi=1 must be consumed
        # before active/target are cleared, otherwise the same target can be
        # re-locked and reclaim another 0.05.
        rig.contact(1., step=0, x=-.1)
        second = rig.step()
        self.assertFalse(bool(second.active[0, 0]))
        self.assertFalse(bool(rig.state.target_locked[0, 0]))
        self.assertEqual(int(second.target_step[0, 0]), -1)
        self.assertPaid(second.reward[0], 0.)
        self.assertAlmostEqual(float(rig.state.paid_fraction[0, 0, 0]), 1., delta=1e-6)
        self.assertPaid(rig.state.actual_paid_amount[0, 0, 0], 0.)
        # Rebuild real support, re-lock the same target and reach phi=1 again.
        for _ in range(2):
            rig.support(0, 0, 0, x=-.1)
            rig.step()
        rig.swing(x=-.1)
        self.assertEqual(int(rig.state.target_step[0, 0]), 1)
        reclaimed = rig.frame(1., x=-.1)
        self.assertPaid(reclaimed.reward[0], 0.)
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
        self.assertEqual(int(support.state.highest_safe_step[0, 0]), 2)

    # ---------------------------------------------------- support geometry
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

        # Larger upward force wins; an exact tie takes the smaller step.
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

        # Two geoms for the same (tile, step) aggregate their upward force.
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
            result = rig.step()
            self.assertFalse(bool(result.safe_support.any()), msg=f'step {step}')
        for _ in range(2):
            for foot in range(4):
                rig.support(0, foot, 1)
            result = rig.step()
        self.assertTrue(bool(result.safe_support.all()))
        self.assertEqual(result.selected_step.tolist(), [[1, 1, 1, 1]])

        # A qualifying force on another tile cannot support the assigned tile.
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
        for _ in range(2):
            for foot in range(4):
                foreign.support(0, foot, 0)
            result = foreign.step()
        self.assertTrue(bool(result.safe_support.all()))

    # ------------------------------------------------------- reset/budgets
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
        env2 = {name: getattr(state, name)[2].clone() for name in
                ('paid_fraction', 'actual_paid_amount', 'target_step', 'target_locked',
                 'ready', 'active', 'safe_support', 'phi', 'qualified_step', 'highest_safe_step')}
        state.reset([0])
        self.assertEqual(float(state.paid_fraction[0].abs().sum()), 0.)
        self.assertEqual(float(state.actual_paid_amount[0].abs().sum()), 0.)
        self.assertTrue(bool((state.target_step[0] == -1).all()))
        self.assertFalse(bool(state.target_locked[0].any()))
        self.assertFalse(bool(state.ready[0].any()))
        self.assertFalse(bool(state.active[0].any()))
        self.assertEqual(float(state.phi[0].abs().sum()), 0.)
        self.assertAlmostEqual(float(state.paid_fraction[1].abs().sum()), .3 * 4 * 40, delta=1e-4)
        for name, value in env2.items():
            self.assertTrue(torch.equal(getattr(state, name)[2], value), msg=name)
        state.reset(torch.tensor([False, True, False]))
        self.assertEqual(float(state.paid_fraction[1].abs().sum()), 0.)
        for name, value in env2.items():
            self.assertTrue(torch.equal(getattr(state, name)[2], value), msg=name)
        state.reset()
        self.assertEqual(float(state.paid_fraction.abs().sum()), 0.)
        self.assertEqual(float(state.actual_paid_amount.abs().sum()), 0.)
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
        self.assertPaid(both.reward[0], .2)
        self.assertPaid(rig.state.actual_paid_amount[0, 0, 0], .1)
        self.assertPaid(rig.state.actual_paid_amount[0, 1, 0], .1)

        # Foot 0 lands, closes its target, and starts a new swing to step 2.
        # Foot 1 keeps its already-paid target-1 slot untouched.
        for _ in range(2):
            rig.support(0, 0, 1, x=.1)
            rig.step()
        self.assertFalse(bool(rig.state.active[0, 0]))
        for _ in range(2):
            rig.air(0, 0, z=.112, bz=.08, x=.1)
            result = rig.step()
        self.assertEqual(int(result.target_step[0, 0]), 2)
        paid = rig.frame(1., feet=(0,), x=.1)
        self.assertPaid(paid.reward[0], .1)
        self.assertPaid(rig.state.actual_paid_amount[0, 0, 0], .1)
        self.assertPaid(rig.state.actual_paid_amount[0, 0, 1], .1)
        self.assertPaid(rig.state.actual_paid_amount[0, 1, 0], .1)
        self.assertEqual(float(rig.state.actual_paid_amount[0, 2:].abs().sum()), 0.)

    def test_adjacent_environment_budgets_are_isolated(self):
        # B=2 is a required contract; only line 331/334/341/388 multi-env
        # broadcasting can currently reject this sample.
        rig = Rig(tiles=(5, 6), nums=(4, 8))
        rig.warmup()
        rig.swing(env=0, x=-.1)
        rig.swing(env=1, x=-.1)
        self.assertEqual(rig.state.target_step[:, 0].tolist(), [1, 1])
        first = rig.frame(.5, env=0)
        self.assertPaid(first.reward[0], .05)
        second = rig.frame(.5, env=1)
        self.assertPaid(second.reward[1], .05)
        self.assertPaid(rig.state.actual_paid_amount[0, 0, 0], .05)
        self.assertPaid(rig.state.actual_paid_amount[1, 0, 0], .05)
        third = rig.frame(1., env=0)
        self.assertPaid(third.reward[0], .05)
        self.assertPaid(third.reward[1], 0.)
        fourth = rig.frame(1., env=1)
        self.assertPaid(fourth.reward[0], 0.)
        self.assertPaid(fourth.reward[1], .05)
        self.assertPaid(rig.state.actual_paid_amount[0, 0, 0], .1)
        self.assertPaid(rig.state.actual_paid_amount[1, 0, 0], .1)

    def test_invalid_swing_inputs_raise(self):
        rewards = api()
        state = rewards.SwingClearanceState(1)
        rig = Rig()
        sample = rig.sample()
        with self.assertRaises(TypeError):
            state.update(None)
        for field, value in (
                ('num_steps', torch.tensor([0], dtype=torch.long)),
                ('tile_ids', torch.tensor([5], dtype=torch.int32)),
                ('foot_pos_w', torch.zeros(1, 3, 3)),
                ('foot_pos_b', torch.zeros(1, 4, 2)),
                ('contact_found', torch.zeros(1, 4, dtype=torch.long)),
                ('top_forces', torch.zeros(1, 4)),
                ('geom_tile', torch.tensor([5], dtype=torch.long)[:0]),
                ('roll', torch.tensor([float('nan')])),
                ('executed_c', torch.tensor([float('inf')]))):
            with self.subTest(field=field):
                changed = dict(sample.__dict__)
                changed[field] = value
                with self.assertRaises(ValueError):
                    state.update(rewards.SwingSample(**changed))
        for bad in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                rewards.SwingClearanceState(bad)
        with self.assertRaises(ValueError):
            rewards.SwingClearanceState(1, max_steps=0)


if __name__ == '__main__':
    unittest.main()
