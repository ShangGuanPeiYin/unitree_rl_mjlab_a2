import unittest
import torch
from src.tasks.pointcloud.perception import DualLidarPattern, PointCloudHistory, preprocess_dual, hits_to_body

class PerceptionTests(unittest.TestCase):
    def test_groups_and_geometry(self):
        groups=[DualLidarPattern(group=i).generate_rays()[1] for i in range(5)]
        self.assertEqual(sum(map(len,groups)),8192)
        self.assertLessEqual(max(map(len,groups))-min(map(len,groups)),1)
        rays=torch.cat(groups)
        self.assertEqual(len(torch.unique(rays,dim=0)),8192)
        torch.testing.assert_close(rays.norm(dim=-1),torch.ones(8192))
        torch.testing.assert_close(groups[0],DualLidarPattern().generate_rays()[1])
    def test_empty_and_fill(self):
        p=torch.zeros(2,50,3); m=torch.zeros(2,50,dtype=torch.bool)
        p[0,:,0]=torch.linspace(-1.1,1.1,50); m[0]=True
        out,mask=preprocess_dual(p,m,p,m&False)
        self.assertGreater(mask[0].sum(),0)
        self.assertFalse(mask[1].any())
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(len(torch.unique(out[0,mask[0]],dim=0)),int(mask[0].sum()))
    def test_reset_delay_and_no_registration(self):
        h=PointCloudHistory(2); h.reset([1],delay_steps=2)
        for n in range(4):
            h.advance(torch.full((2,70,3),float(n)),torch.ones(2,70,dtype=torch.bool))
        p,m,a=h.read()
        self.assertEqual(p[0,0,0,0],3); self.assertEqual(p[1,0,0,0],1)
        self.assertAlmostEqual(float(a[1,0]),.04,places=6)
        h.reset([0]); p,m,a=h.read()
        self.assertFalse(m[0].any()); self.assertTrue(m[1,0].all())
        self.assertEqual(h.scan_group[0],0)
    def test_balanced_quota_transfer_voxel_mean_and_chunks(self):
        x,y=torch.meshgrid(torch.linspace(-1.15,1.15,24),torch.linspace(-.55,.55,12),indexing='ij')
        f=torch.stack((x.flatten(),y.flatten(),torch.full((288,),-.5)),-1)[None].repeat(3,1,1)
        r=f.clone(); r[:,:,2]=.2
        fm=torch.ones(3,288,dtype=torch.bool); rm=fm.clone()
        rm[1,10:]=False; rm[2]=False; fm[2]=False
        a,m=preprocess_dual(f,fm,r,rm,chunk_size=1)
        b,n=preprocess_dual(f,fm,r,rm,chunk_size=128)
        torch.testing.assert_close(a,b); self.assertTrue(torch.equal(m,n))
        self.assertEqual((a[0,:,2]<0).sum(),35)
        self.assertEqual((a[1,:,2]<0).sum(),60)
        self.assertEqual(m.sum(1).tolist(),[70,70,0])
        q=torch.tensor([[[.01,.01,.01],[.03,.03,.03],[float('nan'),0,0]]])
        out,mask=preprocess_dual(q,torch.ones(1,3,dtype=torch.bool),q,torch.zeros(1,3,dtype=torch.bool))
        self.assertEqual(mask.sum(),1)
        torch.testing.assert_close(out[0,0],torch.tensor([.02,.02,.02]))

    def test_all_delay_boundaries_and_invalid_slots(self):
        h=PointCloudHistory(4)
        h.reset(torch.arange(4),torch.arange(4))
        for step in range(8):
            h.advance(torch.full((4,70,3),float(step)),torch.ones(4,70,dtype=torch.bool))
        p,m,a=h.read()
        torch.testing.assert_close(p[:,0,0,0],torch.tensor([7.,6.,5.,4.]))
        self.assertTrue(m.all())
        torch.testing.assert_close(a[:,0],torch.tensor([0.,.02,.04,.06]))
        h.advance(torch.full((4,70,3),float('nan')),torch.ones(4,70,dtype=torch.bool))
        p,m,a=h.read(); self.assertTrue(torch.isfinite(p).all()); self.assertFalse(m[0,0].any())
        with self.assertRaises(ValueError): h.reset([0],4)

    def test_occluder_not_visible_through(self):
        p=torch.tensor([[[1.,0,0],[2.,0,0]]])
        q,m=hits_to_body(p,torch.tensor([[1.,2.]]),torch.zeros(1,3),torch.eye(3)[None],torch.tensor([[True,False]]))
        self.assertEqual(m.tolist(),[[False,True]])
        torch.testing.assert_close(p,q)

if __name__=='__main__': unittest.main()

class StaticMujocoVisibilityTests(unittest.TestCase):
    def test_real_robot_occlusion_and_four_locations(self):
        import mujoco
        import numpy as np
        from src.terrains import build_open_stairs_spec
        from src.assets.robots.unitree_a2.a2_constants import get_spec
        spec,_=build_open_stairs_spec()
        robot=get_spec()
        spec.attach(robot,prefix='robot/',frame=spec.worldbody.add_frame())
        model=spec.compile(); data=mujoco.MjData(model)
        for j in range(model.njnt):
            name=model.joint(j).name
            if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE:
                data.qpos[model.jnt_qposadr[j]]=.9 if 'thigh' in name else -1.8 if 'calf' in name else (-.1 if 'FL_' in name or 'RL_' in name else .1)
        free=next(j for j in range(model.njnt) if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE)
        adr=model.jnt_qposadr[free]
        robot_ids=np.array([g for g in range(model.ngeom) if model.geom(g).name.startswith('robot/')])
        group=np.array([1,0,0,1,0,0],dtype=np.uint8)
        counts={}
        for name,position in [('flat',(-1.,0,.4)),('first',(-.25,0,.4)),('middle',(3.9,0,4.4)),('top',(8.5,0,8.4))]:
            data.qpos[adr:adr+3]=position; data.qpos[adr+3:adr+7]=[1,0,0,0]
            mujoco.mj_forward(model,data)
            observed=[]
            for rear in [False,True]:
                origins,directions=DualLidarPattern(rear=rear).generate_rays()
                origin=origins[0].numpy()+np.array(position)
                for g in robot_ids:
                    if model.geom_group[g]!=3:
                        continue
                    local=(origin-data.geom_xpos[g]) @ data.geom_xmat[g].reshape(3,3)
                    size=model.geom_size[g]; kind=model.geom_type[g]
                    inside=False
                    if kind==mujoco.mjtGeom.mjGEOM_BOX:
                        inside=np.all(np.abs(local)<size)
                    elif kind==mujoco.mjtGeom.mjGEOM_SPHERE:
                        inside=np.linalg.norm(local)<size[0]
                    elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:
                        inside=np.linalg.norm(local[:2])<size[0] and abs(local[2])<size[1]
                    elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE:
                        offset=local.copy(); offset[2]-=np.clip(offset[2],-size[1],size[1])
                        inside=np.linalg.norm(offset)<size[0]
                    else:
                        self.fail('unhandled collision containment primitive')
                    self.assertFalse(inside,(name,rear,model.geom(g).name,'sensor inside collision'))
                self_hits=0; visible=0
                for direction in directions.numpy():
                    gid=np.array([-1],dtype=np.int32)
                    distance=mujoco.mj_ray(model,data,origin,direction,group,True,-1,gid)
                    if gid[0] in robot_ids:
                        self_hits+=1
                    elif .15<=distance<=10:
                        local=origin+distance*direction-np.array(position)
                        if np.all(local>=[-1.2,-.6,-1.]) and np.all(local<=[1.2,.6,.5]): visible+=1
                observed.append((self_hits,visible))
                self.assertGreater(visible,0,(name,rear,'no terrain returns in ROI'))
                self.assertGreater(self_hits,0,(name,rear,'robot does not occlude'))
            counts[name]=observed
        print('static visibility (self hits, ROI terrain hits), front/rear:',counts)
