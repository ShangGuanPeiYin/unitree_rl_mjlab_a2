"""Geometry, open-space ray and contact checks, independent of an RL policy."""
import unittest

import mujoco
import numpy as np
from src.terrains import OpenStairsTerrainCfg, build_open_stairs_spec
from mjlab.terrains.terrain_generator import TerrainGenerator, TerrainGeneratorCfg


class OpenStairsTests(unittest.TestCase):
    def setUp(self):
        self.cfg = OpenStairsTerrainCfg()
        self.spec, self.spawn = build_open_stairs_spec(self.cfg)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def ray(self, position, direction):
        geom_id = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(self.model, self.data,
                                np.array(position, dtype=float), np.array(direction, dtype=float),
                                None, True, -1, geom_id)
        return distance, int(geom_id[0])

    def test_dimensions_and_spawn(self):
        self.assertEqual(self.model.ngeom, 42)
        np.testing.assert_allclose(self.spawn, (-1, 0, 0))
        for i in range(1, 41):
            geom = self.model.geom(f'tread_{i:02d}')
            np.testing.assert_allclose(geom.size, (.1, .4, .005))
            np.testing.assert_allclose(geom.pos, ((i - .5) * .2, 0, i * .2 - .005))
            self.assertEqual(int(geom.contype[0]), 1)
            self.assertEqual(int(geom.conaffinity[0]), 1)
        np.testing.assert_allclose(self.model.geom('top_landing').pos, (9, 0, 7.995))
        np.testing.assert_allclose(self.model.geom('top_landing').size, (1, .4, .005))
        np.testing.assert_allclose(self.model.geom('ground').pos, (4, 0, -.05))

    def test_tread_tops_and_landing_rays(self):
        for i in range(1, 41):
            distance, geom_id = self.ray(((i - .5)*.2, 0, i*.2+.1), (0, 0, -1))
            self.assertAlmostEqual(distance, .1, places=6)
            self.assertEqual(geom_id, self.model.geom(f'tread_{i:02d}').id)
        distance, geom_id = self.ray((9, 0, 8.1), (0, 0, -1))
        self.assertAlmostEqual(distance, .1)
        self.assertEqual(geom_id, self.model.geom('top_landing').id)

    def test_open_risers_underside_and_sides(self):
        distance, _ = self.ray((-.1, 0, .1), (1, 0, 0))
        self.assertEqual(distance, -1)  # no filled risers anywhere
        for i in range(1, 41):
            distance, _ = self.ray(((i-.5)*.2, 0, i*.2-.1), (0, 1, 0))
            self.assertEqual(distance, -1)  # no side walls
        distance, geom_id = self.ray((.1, 0, .1), (0, 0, 1))
        self.assertAlmostEqual(distance, .09)
        self.assertEqual(geom_id, self.model.geom('tread_01').id)
        distance, geom_id = self.ray((-.1, 0, .195), (1, 0, 0))
        self.assertAlmostEqual(distance, .1)
        self.assertEqual(geom_id, self.model.geom('tread_01').id)
        distance, geom_id = self.ray((.1, .5, .3), (0, 0, -1))
        self.assertAlmostEqual(distance, .3)
        self.assertEqual(geom_id, self.model.geom('ground').id)

    def test_real_contacts_and_open_space(self):
        probe = self.spec.worldbody.add_body(name='probe', pos=(.1, 0, .205))
        probe.add_freejoint()
        probe.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(.01, 0, 0), mass=.1)
        model = self.spec.compile()
        data = mujoco.MjData(model)
        for z, expected_contact in ((.205, True), (.185, True), (.195, True), (.10, False)):
            data.qpos[:3] = (.1, 0, z)
            data.qpos[3:7] = (1, 0, 0, 0)
            mujoco.mj_forward(model, data)
            self.assertEqual(data.ncon > 0, expected_contact)
        # A sphere falling onto a thin tread must remain on it, not pass through.
        mujoco.mj_resetData(model, data)
        data.qpos[2] = .25
        for _ in range(200):
            mujoco.mj_step(model, data)
        self.assertTrue(np.isfinite(data.qpos).all())
        self.assertGreater(data.qpos[2], .20)
        self.assertLess(data.qpos[2], .23)

    def test_native_grid_extension(self):
        spec = mujoco.MjSpec()
        generator = TerrainGenerator(TerrainGeneratorCfg(
            seed=42, size=self.cfg.size, num_rows=2, num_cols=1,
            curriculum=False, border_width=0,
            sub_terrains={'stairs': self.cfg},
        ))
        generator.compile(spec)
        model = spec.compile()
        self.assertEqual(model.ngeom, 84)
        self.assertAlmostEqual(generator.terrain_origins[1,0,0]-generator.terrain_origins[0,0,0], 12.)

    def test_invalid_dimensions(self):
        for kwargs in ({'step_count': 0}, {'step_height': float('nan')},
                       {'tread_thickness': .2}, {'ground_width': .5}):
            with self.assertRaises(ValueError):
                OpenStairsTerrainCfg(**kwargs)


if __name__ == '__main__':
    unittest.main()
