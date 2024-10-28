import mitsuba as mi
import drjit as dr
import numpy as np
from PDE3D import PATH
import os

@dr.syntax
def redistance(name_scene, res = 512, offset = 1.05, method = False):
    folder_name = os.path.join(PATH, "scenes", name_scene)
    scene = mi.load_file(os.path.join(folder_name, "scene.xml"))

    bbox_ = scene.bbox()
    ext = dr.max((bbox_.max - bbox_.min))/2
    bbox_center = (bbox_.max + bbox_.min) / 2
    bbox = mi.BoundingBox3f(min = bbox_center - ext * offset, max = bbox_center + ext * offset)

    z,y,x = dr.meshgrid(*[dr.linspace(mi.Float, 0.5/res, 1 - 0.5/res, res)
                          for i in range(3)], indexing='ij')
    points = bbox.min + (bbox.max - bbox.min) * mi.Point3f(x,y,z)

    sampler = mi.PCG32(size = dr.width(points))
    mask1 = mi.Bool(False)
    mask2 = mi.Bool(False)
    mask3 = mi.Bool(False)
    mask = mi.Bool(False)
    i = mi.UInt32(0)

    while(i < 2500):
        if dr.hint(method == 0, mode = "scalar"):
            ray1 = mi.Ray3f(points, dr.normalize(mi.Vector3f(sampler.next_float32()-0.5, 0, sampler.next_float32()-0.5)))
            ray2 = mi.Ray3f(points, dr.normalize(mi.Vector3f(sampler.next_float32()-0.5, sampler.next_float32()-0.5, 0)))
            ray3 = mi.Ray3f(points, dr.normalize(mi.Vector3f(0, sampler.next_float32()-0.5, sampler.next_float32()-0.5)))
            si1 = scene.ray_intersect(ray1)
            si2 = scene.ray_intersect(ray2)
            si3 = scene.ray_intersect(ray3)
            #mask1 = si1.n
            #mask2 |= ~si2.is_valid() 
            #mask3 |= ~si3.is_valid() 
            #mask = mask1 & mask2 & mask3
            
        elif dr.hint(method == 1, mode = "scalar"):
            ray1 = mi.Ray3f(points, dr.normalize(mi.Vector3f(sampler.next_float32()-0.5, 0, sampler.next_float32()-0.5)))
            ray2 = mi.Ray3f(points, dr.normalize(mi.Vector3f(sampler.next_float32()-0.5, sampler.next_float32()-0.5, 0)))
            ray3 = mi.Ray3f(points, dr.normalize(mi.Vector3f(0, sampler.next_float32()-0.5, sampler.next_float32()-0.5)))
            si1 = scene.ray_intersect(ray1)
            si2 = scene.ray_intersect(ray2)
            si3 = scene.ray_intersect(ray3)
            mask1 |= ~si1.is_valid()
            mask2 |= ~si2.is_valid() 
            mask3 |= ~si3.is_valid() 
            mask = mask1 & mask2 & mask3
        else:
            sample1 = dr.normalize(mi.Vector3f(sampler.next_float32()-0.5, 0, sampler.next_float32()-0.5))
            sample2 = dr.normalize(mi.Vector3f(sampler.next_float32()-0.5, sampler.next_float32()-0.5, 0))
            sample3 = dr.normalize(mi.Vector3f(0, sampler.next_float32()-0.5, sampler.next_float32()-0.5))
            transform1 = mi.Transform4f().rotate([1,0,0],45)
            transform2 = mi.Transform4f().rotate([1,0,0],45)
            transform3 = mi.Transform4f().rotate([0,1,0],45)
            sample1 = transform1 @ sample1
            sample2 = transform2 @ sample2
            sample3 = transform3 @ sample3
            ray1 = mi.Ray3f(points, sample1)
            ray2 = mi.Ray3f(points, sample2)
            ray3 = mi.Ray3f(points, sample3)
            si1 = scene.ray_intersect(ray1)
            si2 = scene.ray_intersect(ray2)
            si3 = scene.ray_intersect(ray3)
            mask1 |= ~si1.is_valid()
            mask2 |= ~si2.is_valid() 
            mask3 |= ~si3.is_valid() 
            mask = mask1 & mask2 & mask3

        i += 1
    
    values = dr.select(mask, 1.0, 0.0) - 0.5
    tensor = mi.TensorXf(values, (res, res, res)).numpy()

    import skfmm 
    redistanced = skfmm.distance(tensor,  dx=1 / np.array(tensor.shape))
    np.save(f"{folder_name}/sdf_high.npy", redistanced)
    np.save(f"{folder_name}/sdf.npy", redistanced[::2, ::2, ::2])


