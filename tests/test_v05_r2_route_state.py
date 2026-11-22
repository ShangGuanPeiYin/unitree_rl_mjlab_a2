"""CPU evidence for the r2 ``route_error`` state contract (SDS 0.5-r2 4.2, 9.3).

Analytic tests only -- no simulator is constructed here:

* the exact formula ``[clip(local_y/0.20,-4,4), sin(e_world), cos(e_world)]`` for
  the SDS pose grid, dtype/shape/finiteness and saturation;
* ``e_world = wrap_to_pi(yaw_world - yaw_axis)`` with the frozen ``yaw_axis = 0``
  and the explicit "atlas is translation-only" premise;
* consistency between the route vector and ``local_y``/``yaw`` including the
  per-environment tile reference used by the environment (the tile origin only
  shifts ``local_y``, it never rotates or scales it).

The real-environment half of the same contract (reset/step/terminal semantics,
flat spawn, per-tile routing) lives in ``tests/test_v05_r2_environment.py``.
"""
import math
import unittest
from dataclasses import dataclass, field

import torch

from src.tasks.velocity.v05_r2 import spec
from src.tasks.velocity.v05_r2 import speed as speed_module
from src.tasks.velocity.v05_r2.env import (ROUTE_LATERAL_SATURATION_M,
                                           _check_translation_only_atlas,
                                           route_components, route_error_vector,
                                           wrap_to_pi)
from src.tasks.velocity.v05_r2.speed import (ROUTE_CLIP, ROUTE_LATERAL_SCALE,
                                            ROUTE_YAW_AXIS, _route_contract,
                                            route_state_contract)

TOL = 1e-7


class FormulaTests(unittest.TestCase):
    def test_exact_formula_grid(self):
        lateral = [-.8, -.4, -.2, 0., .2, .4, .8, .8001]
        yaw_deg = [0., 5., -5., 10., -10., 90., -90., 180.]
        yaw = [math.radians(value) for value in yaw_deg] + [math.pi, -math.pi]
        y = torch.tensor([[value] for value in lateral], dtype=torch.float32)
        route, e_world, saturated = route_error_vector(y[:, 0], torch.zeros(len(lateral)))
        self.assertEqual(route.shape, (len(lateral), 3))
        self.assertEqual(route.dtype, torch.float32)
        self.assertEqual(route.device, torch.device('cpu'))
        self.assertTrue(torch.isfinite(route).all())
        expected_lateral = torch.tensor(
            [max(-4., min(4., value / .20)) for value in lateral], dtype=torch.float32)
        self.assertLessEqual(float((route[:, 0] - expected_lateral).abs().max()), TOL)
        self.assertEqual(bool(saturated[6]), True)   # exactly 0.80 m saturates
        self.assertEqual(bool(saturated[7]), True)   # 0.8001 m saturates
        self.assertEqual(bool(saturated[5]), False)  # 0.40 m does not
        # The first component is 5x the F0 lateral gate scale.
        self.assertAlmostEqual(float(route[2, 0]), -1.0, places=6)  # -0.20 m -> -1.0
        self.assertAlmostEqual(float(route[4, 0]), 1.0, places=6)   # +0.20 m -> +1.0
        self.assertAlmostEqual(float(route[5, 0]), 2.0, places=6)   # +0.40 m -> +2.0
        self.assertAlmostEqual(float(route[6, 0]), 4.0, places=6)   # +0.80 m -> +4.0

        yaw_tensor = torch.tensor(yaw, dtype=torch.float32)
        route_yaw, e_world, _ = route_error_vector(torch.zeros(len(yaw)), yaw_tensor)
        self.assertTrue(torch.allclose(route_yaw[:, 1], torch.sin(yaw_tensor), atol=TOL))
        self.assertTrue(torch.allclose(route_yaw[:, 2], torch.cos(yaw_tensor), atol=TOL))
        # ``e_world`` is the wrapped angle, so compare it modulo 2*pi (pi maps to
        # -pi through atan2) and through its sine/cosine.
        self.assertTrue(torch.allclose(wrap_to_pi(e_world), wrap_to_pi(yaw_tensor), atol=TOL))
        self.assertTrue(torch.allclose(torch.sin(e_world), torch.sin(yaw_tensor), atol=TOL))
        # The unit vector stays on the unit circle and is (0,1) when aligned.
        self.assertLessEqual(float((route_yaw[:, 1:].norm(dim=-1) - 1.).abs().max()), TOL)
        aligned, _, _ = route_error_vector(torch.zeros(1), torch.zeros(1))
        self.assertLessEqual(float((aligned - torch.tensor([[0., 0., 1.]])).abs().max()), TOL)

    def test_saturation_clamps_and_never_grows(self):
        y = torch.tensor([-.79, .79, .8, .9, 2., 100., -100.], dtype=torch.float32)
        route, _, saturated = route_error_vector(y, torch.zeros(7))
        # 0.80 m is exactly the saturation point; 0.79 m is not.
        self.assertEqual(saturated.tolist(), [False, False, True, True, True, True, True])
        self.assertLessEqual(float(route[:, 0].abs().max()), ROUTE_CLIP)
        self.assertAlmostEqual(float(route[1, 0]), .79 / .20, places=5)
        self.assertTrue(torch.equal(route[2:, 0],
                                    torch.tensor([4., 4., 4., 4., -4.])))

    def test_flat_zero_pose_is_exactly_the_aligned_vector(self):
        route, e_world, _ = route_error_vector(torch.zeros(2), torch.zeros(2))
        self.assertTrue(torch.equal(route[:, 0], torch.zeros(2)))
        self.assertTrue(torch.equal(route[:, 1], torch.zeros(2)))
        self.assertTrue(torch.equal(route[:, 2], torch.ones(2)))
        self.assertTrue(torch.equal(e_world, torch.zeros(2)))

    def test_route_is_a_unit_heading_vector_not_a_degree(self):
        # SDS 4.2: components 2/3 are sin/cos of a radian angle, never the angle
        # itself and never degrees.
        yaw = torch.tensor([math.radians(90.)], dtype=torch.float32)
        route, e_world, _ = route_error_vector(torch.zeros(1), yaw)
        self.assertAlmostEqual(float(route[0, 1]), 1.0, places=6)
        self.assertAlmostEqual(float(route[0, 2]), 0.0, places=6)
        self.assertAlmostEqual(float(e_world[0]), math.pi / 2, places=6)


class SingleDecompositionTests(unittest.TestCase):
    """One helper supplies route vector, |local_y|, ratio and saturation."""

    def test_components_and_vector_are_the_same_decomposition(self):
        y = torch.tensor([-.9, -.8, -.7999, -.2, 0., .2, .7999, .8, .8001, 4.], dtype=torch.float32)
        yaw = torch.tensor([0., .1, -.2, math.pi, -3. * math.pi / 2, .05, -.05, .3, -.3, 1.],
                           dtype=torch.float32)
        components = route_components(y, yaw)
        route, e_world, saturated = route_error_vector(y, yaw)
        self.assertTrue(torch.equal(components.route_error, route))
        self.assertTrue(torch.equal(components.e_world, e_world))
        self.assertTrue(torch.equal(components.saturated, saturated))
        self.assertEqual(components.route_error.dtype, torch.float32)
        self.assertEqual(tuple(components.route_error.shape), (10, 3))

    def test_saturation_boundary_is_ge_080_metres_and_never_grows(self):
        self.assertEqual(ROUTE_LATERAL_SATURATION_M, .8)
        y = torch.tensor([0., .79, .7999, .8, .8001, .9, 100., -100.], dtype=torch.float32)
        components = route_components(y, torch.zeros(len(y)))
        self.assertEqual(components.saturated.tolist(),
                         [False, False, False, True, True, True, True, True])
        # abs_y / ratio / clipped component are the same numbers, not re-derived.
        self.assertTrue(torch.equal(components.abs_y, y.abs()))
        self.assertTrue(torch.allclose(components.normalized_abs_y_raw,
                                       y.abs() / ROUTE_LATERAL_SCALE, atol=1e-7))
        self.assertTrue(torch.equal(components.normalized_abs_y,
                                    components.route_error[:, 0].abs()))
        self.assertTrue(torch.equal(components.route_error[:, 0],
                                    (y / ROUTE_LATERAL_SCALE).clamp(-ROUTE_CLIP, ROUTE_CLIP)))
        self.assertLessEqual(float(components.route_error[:, 0].abs().max()), ROUTE_CLIP)
        self.assertAlmostEqual(float(components.route_error[3, 0]), 4.0, places=6)
        self.assertAlmostEqual(float(components.route_error[4, 0]), 4.0, places=6)
        # The unclipped ratio still exceeds the clip past the saturation point.
        self.assertGreater(float(components.normalized_abs_y_raw[4]), ROUTE_CLIP)

    def test_components_reject_bad_inputs(self):
        with self.assertRaises(TypeError):
            route_components([0.], torch.zeros(1))
        with self.assertRaises(ValueError):
            route_components(torch.zeros(2), torch.zeros(3))


class YawAxisTests(unittest.TestCase):
    def test_yaw_axis_is_zero_and_contract_is_explicit(self):
        self.assertEqual(ROUTE_YAW_AXIS, 0.)
        contract = route_state_contract()
        self.assertEqual(contract['yaw_axis'], 0.0)
        self.assertIn('yaw_axis_premise', contract)
        self.assertIn('translation', contract['yaw_axis_premise'])
        self.assertEqual(contract['lateral_scale'], .20)
        self.assertEqual(contract['clip'], 4.)
        self.assertEqual(contract['units'], 'radians')
        self.assertFalse(contract['normalized'])
        self.assertFalse(contract['noise'])
        self.assertTrue(contract['per_environment_tile'])
        self.assertEqual(contract['deployment_label'], 'ideal_route_state')

    def test_wrap_to_pi_wraps_around(self):
        values = torch.tensor([0., math.pi, -math.pi, 3. * math.pi / 2,
                               -3. * math.pi + .1, 10., -10.], dtype=torch.float32)
        wrapped = wrap_to_pi(values)
        self.assertTrue(torch.isfinite(wrapped).all())
        self.assertLessEqual(float(wrapped.abs().max()), math.pi + 1e-6)
        self.assertTrue(torch.allclose(torch.sin(wrapped), torch.sin(values), atol=TOL))
        self.assertTrue(torch.allclose(torch.cos(wrapped), torch.cos(values), atol=TOL))
        self.assertAlmostEqual(float(wrap_to_pi(torch.tensor(3. * math.pi / 2))),
                               -math.pi / 2, places=6)

    def test_nonzero_yaw_axis_is_rejected_not_silently_used(self):
        original = spec.ROUTE_STATE
        try:
            spec.ROUTE_STATE = dict(original, yaw_axis=1.0)
            with self.assertRaises(RuntimeError):
                _route_contract()
            spec.ROUTE_STATE = dict(original, yaw_axis=0.0)
            self.assertEqual(_route_contract()['yaw_axis'], 0.0)
            missing = {key: value for key, value in original.items()
                       if key not in ('lateral_scale_m', 'lateral_clip', 'yaw_axis')}
            spec.ROUTE_STATE = missing
            original_scale = spec.ROUTE_LATERAL_SCALE
            try:
                del spec.ROUTE_LATERAL_SCALE
                with self.assertRaises(RuntimeError):
                    _route_contract()
            finally:
                spec.ROUTE_LATERAL_SCALE = original_scale
        finally:
            spec.ROUTE_STATE = original

    def test_translation_only_atlas_premise_is_asserted(self):
        from src.tasks.velocity.v02.terrain import atlas_metadata
        metadata = atlas_metadata()
        self.assertIsNone(_check_translation_only_atlas(metadata))

        @dataclass(frozen=True)
        class Rotated:
            tile_id: int = 0
            stage: str = 'F0'
            N: int = 0
            h: float = 0.
            frame_origin_world: tuple = (0., 0., 0.)
            yaw: float = .3

        with self.assertRaises(RuntimeError):
            _check_translation_only_atlas((Rotated(),))

        @dataclass(frozen=True)
        class Broken:
            frame_origin_world: tuple = (0., float('nan'), 0.)

        with self.assertRaises(RuntimeError):
            _check_translation_only_atlas((Broken(),))


class TileReferenceTests(unittest.TestCase):
    """The tile reference only translates ``local_y`` (SDS 4.2 per-tile rule)."""

    def test_tile_origin_translation_is_consistent(self):
        origins = torch.tensor([[0., 0., 0.], [0., 8., 0.], [0., -8., 0.]])
        positions = torch.tensor([[1., .25, .4], [1., 8.25, .4], [1., -7.75, .4]])
        routes = []
        for tile in range(3):
            local = positions - origins[tile]
            route, _, _ = route_error_vector(local[:, 1], torch.zeros(3))
            routes.append(route)
        # With each environment on its *own* tile every row is the same local pose.
        self.assertAlmostEqual(float(routes[0][0, 0]), 1.25, places=6)
        self.assertAlmostEqual(float(routes[1][1, 0]), 1.25, places=6)
        self.assertAlmostEqual(float(routes[2][2, 0]), 1.25, places=6)
        # A whole-batch stage tile reads the wrong origin for the other rows.
        for tile in range(3):
            for row in range(3):
                if row == tile:
                    continue
                self.assertGreater(abs(float(routes[tile][row, 0]) - 1.25), 1e-3)

    def test_same_pose_different_tile_gives_a_different_local_y(self):
        origin = torch.tensor([[0., 0., 0.], [0., 8., 0.]])
        position = torch.tensor([[0., .1, .2], [0., .1, .2]])
        routes = []
        for tile in range(2):
            local = position - origin[tile]
            route, _, _ = route_error_vector(local[:, 1], torch.zeros(2))
            routes.append(route)
        self.assertTrue(torch.allclose(routes[0][:, 0], torch.tensor([.5, .5]), atol=1e-6))
        # Second tile origin is 8 m away, so the same world pose is out of range.
        self.assertTrue(torch.allclose(routes[1][:, 0], torch.tensor([-4., -4.]), atol=1e-6))

    def test_route_helper_rejects_bad_shapes(self):
        with self.assertRaises(TypeError):
            route_error_vector([0.], torch.zeros(1))
        with self.assertRaises(ValueError):
            route_error_vector(torch.zeros(2), torch.zeros(3))

    def test_scale_and_clip_are_the_only_numbers_in_the_formula(self):
        # A route vector built from a 1.0 m lateral offset must equal clip(5) = 4.
        route, _, _ = route_error_vector(torch.tensor([1.0]), torch.zeros(1))
        self.assertAlmostEqual(float(route[0, 0]), 4.0, places=6)
        self.assertAlmostEqual(float(route[0, 0]), ROUTE_CLIP, places=6)
        route, _, _ = route_error_vector(torch.tensor([ROUTE_LATERAL_SCALE]), torch.zeros(1))
        self.assertAlmostEqual(float(route[0, 0]), 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
