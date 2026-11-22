"""Compare analytic ideal maps against actual MuJoCo collision geometry."""
import unittest
import mujoco
import numpy as np
import torch
from mjlab.terrains.terrain_generator import TerrainGenerator
from src.terrains.ideal_stairs_atlas import atlas_generator_cfg,atlas_metadata,query_top_surface,allocate_stage_slots
from src.tasks.velocity.mdp.ideal_heightmap import IdealHeightMap,ProprioHistory

class IdealHeightmapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=mujoco.MjSpec()
        cls.generator=TerrainGenerator(atlas_generator_cfg())
        cls.generator.compile(cls.spec)
        cls.model=cls.spec.compile()
        cls.data=mujoco.MjData(cls.model)
        mujoco.mj_forward(cls.model,cls.data)
    def test_real_geometry_all_stages(self):
        self.assertEqual(self.model.ngeom,95)
        for meta in atlas_metadata():
            np.testing.assert_allclose(self.generator.terrain_origins[0,meta.tile_id],meta.terrain_origin_world)
            frame=np.array(meta.frame_origin_world)
            if meta.stage:
                self.assertAlmostEqual(meta.world_to_local(np.array(meta.terrain_origin_world))[0],-1.)
            samples=[(-1.,0.),(.1,.7),(meta.step_count*.2+1.,0.)]
            samples += [((i-.5)*.2,0.) for i in range(1,meta.step_count+1)]
            for x,y in samples:
                point=np.array([x,y])+frame[:2]
                height,valid=query_top_surface(torch.tensor(point),torch.tensor(meta.tile_id))
                gid=np.array([-1],dtype=np.int32)
                distance=mujoco.mj_ray(self.model,self.data,np.r_[point,10.],np.array([0.,0.,-1.]),None,True,-1,gid)
                self.assertTrue(valid.item())
                self.assertLess(abs(height.item()-(10.-distance)),.001)
    def test_isolation_bounds_yaw_and_above_robot(self):
        z,valid=query_top_surface(torch.tensor([[-9.9,-12.],[-9.9,-20.],[-9.9,20.]]),torch.tensor([0,1,5]))
        self.assertFalse(valid[0])  # physically over S1 but assigned S0
        self.assertFalse(valid[1])
        self.assertAlmostEqual(z[2].item(),.2,places=5)
        cache=IdealHeightMap(1)
        pos=torch.tensor([[-10.,20.,0.]])
        cache.reset(pos,torch.tensor([torch.pi/2]),torch.tensor([5]))
        torch.testing.assert_close(cache.sample_points_world[0,0,0],torch.tensor([-9.225,19.225]))
        self.assertGreater(cache.map[0,0].max().item(),0.)
    def test_motion_cache_and_independent_reset(self):
        cache=IdealHeightMap(2)
        pos=torch.tensor([[-10.1,20.,.6],[-10.1,-12.,.6]])
        yaw=torch.zeros(2); tiles=torch.tensor([5,1])
        old=cache.reset(pos,yaw,tiles).clone()
        sampled=cache.sample_points_world.clone()
        errors=[]
        for step in range(1,5):
            current=pos.clone(); current[:,0] += step*.012;current[:,2]+=step*.01
            angles=yaw+step*.01
            cached=cache.update(current,angles,tiles).clone()
            torch.testing.assert_close(cached[:,:2],old[:,:2])
            torch.testing.assert_close(cache.sample_points_world,sampled)
            valid=cached[:,1].bool()
            torch.testing.assert_close(cached[:,2][valid],torch.full_like(cached[:,2][valid],step*.04))
            fresh=IdealHeightMap(2).reset(current,angles,tiles)
            errors.append(float((cached[:,0]-fresh[:,0]).abs().max()))
        self.assertGreater(max(errors),0.)
        self.assertAlmostEqual(float((current-pos)[0,0]),.048,places=5)
        unchanged=cache.map[1].clone()
        cache.reset(current,angles,tiles,env_ids=[0])
        torch.testing.assert_close(cache.map[1],unchanged)
        torch.testing.assert_close(cache.sample_position[0],current[0])
        cache.update(current,angles,tiles)
        torch.testing.assert_close(cache.sample_position[1],current[1])
    def test_no_hit_and_map_matches_real_rays(self):
        cache=IdealHeightMap(1)
        pos=torch.tensor([[-9.9,20.,.4]])
        image=cache.reset(pos,torch.zeros(1),torch.tensor([5]))
        for i,j in ((0,0),(17,15),(30,16),(47,31)):
            xy=cache.sample_points_world[0,i,j].numpy().astype(float)
            gid=np.array([-1],dtype=np.int32)
            distance=mujoco.mj_ray(self.model,self.data,np.r_[xy,10.],np.array([0.,0.,-1.]),None,True,-1,gid)
            self.assertGreaterEqual(distance,0.)
            actual=(10.-distance-.4)/1.5
            self.assertLess(abs(float(image[0,0,i,j])-np.clip(actual,-1.,1.)),.001/1.5)
        outside=torch.tensor([[100.,100.,.4]])
        invalid=cache.reset(outside,torch.zeros(1),torch.tensor([5]))
        self.assertEqual(float(invalid[:,0].sum()),0.)
        self.assertEqual(float(invalid[:,1].sum()),0.)
        self.assertEqual(float(invalid[:,2].min()),1.)

    def test_largest_remainder_and_history(self):
        for count in (1,258,512,1024):
            for stage in range(6):
                slots=allocate_stage_slots(count,stage)
                self.assertEqual(len(slots),count)
        slots=allocate_stage_slots(258,5)
        self.assertEqual([int((slots==i).sum()) for i in (0,4,5)],[26,52,180])
        h=ProprioHistory(2);obs=torch.ones(2,45)
        h.update(obs);h.update(obs*2)
        h.reset(obs*3,[0])
        torch.testing.assert_close(h.history[0],torch.full((5,45),3.))
        torch.testing.assert_close(h.history[1,-1],torch.full((45,),2.))

if __name__=='__main__': unittest.main()
