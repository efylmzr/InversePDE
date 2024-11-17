import drjit as dr
import mitsuba as mi
from PDE3D.Coefficient import *
from .slice import Slice
from PDE3D import PATH
from PDE3D import ArrayXf
from PDE3D.utils.helpers import tea, get_rgb_from_colormap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


class BoundaryInfo:
    DRJIT_STRUCT = {
        "o" : mi.Point3f,
        "d" : mi.Vector3f,
        "r" : mi.Float, 
        "is_e" : mi.Bool,
        'bpoint' : mi.Point3f,
        'dirichlet' : ArrayXf
    }
    def __init__(self, origin = None, distance = None, radius = None, is_epsilon = None, bpoint = None, dirichlet = None):
        self.o = origin
        self.d = distance
        self.r = radius
        self.is_e = is_epsilon
        self.bpoint = bpoint
        self.dirichlet = dirichlet


class Shape():
    def __init__(self, single_closed_shape = True, epsilon=1e-5, inf_distance = 10, dirichlet : list[Coefficient] = []):
        self.single_closed = single_closed_shape
        self.epsilon = epsilon
        self.name = "boundary"
        self.inf_distance = inf_distance
        self.dirichlet = dirichlet
        self.num_conf_d = len(dirichlet)
        self.shape_mi = None


    def boundary_interaction(self, points : mi.Point3f, conf_numbers : list[mi.UInt32]) -> BoundaryInfo:
        pass

    def inside_closed_surface(self, points, L, conf_numbers):
        return mi.Bool(False), dr.zeros(ArrayXf, shape = (len(conf_numbers), dr.width(points)))
    
    def inside_closed_surface_mask(self, L):
        return mi.Bool(False)
    
    def visualize(self, image_res = [512, 512],  
                  cam_origin = mi.ScalarPoint3f(1,1,2), cam_target = mi.ScalarPoint3f(0, 0, 0), cam_up = mi.ScalarPoint3f([0,1,0]),
                  scale_cam = 2, slice : Slice = None, coeff = None,
                  slice_image = None, input_range = None, conf_number : mi.UInt32 = None, colormap : str = "viridis",
                  spp = 64, sym_colorbar = False):
        
        cam = mi.load_dict({'type': 'orthographic',
                            'to_world': mi.ScalarTransform4f().look_at(
                            origin=cam_origin,
                            target=cam_target,
                            up=cam_up,
                            )@ mi.ScalarTransform4f().scale([1 / scale_cam, 1 / scale_cam, 1])}) 
        
        #cam_dir = dr.normalize(-cam_origin)

        # Construct a grid of 2D coordinates
        x, y = dr.meshgrid((dr.arange(mi.Float, image_res[0]) + 0.5) / image_res[0],
                           (dr.arange(mi.Float, image_res[1]) + 0.5) / image_res[1])

    
        size = spp * image_res[0] * image_res[1]
        x = dr.repeat(x, spp)
        y = dr.repeat(y, spp)
        # Ray origin in local coordinates
        ray_origin_local = mi.Point2f(x, y)
        


        time = 0.
        wav_sample = [0.5, 0.33, 0.1]
        pos_sample = [[0.2, 0.1, 0.2], [0.6, 0.9, 0.2]]
        aperture_sample = 0 # Not being used

        ray, spec_weight = cam.sample_ray(0.0, wav_sample, ray_origin_local, aperture_sample)

        print(dr.width(ray))

        # Ray origin in world coordinates
        #ray_origin = mi.Frame3f(cam_dir).to_world(ray_origin_local) + cam_origin
        # Ray in world coordinates
        #ray = mi.Ray3f(o=ray_origin, d=cam_dir)
        
        #si_boundary = self.shape_mi.ray_intersect(ray)
        si_boundary = self.scene.ray_intersect(ray)

        visible_boundary_mask = si_boundary.is_valid()

        final_result = dr.ones(mi.Color3f, size)
        norm = None
        fnc_vals = None

        if slice is not None:
            if (slice_image is not None) and (coeff is not None):
                raise Exception("Only specify a solution or a coefficient.")
            elif (slice_image is None) and (coeff is None):
                coeff = ConstantCoefficient("", 0)
                input_range = [-0.5, 0.5]
            si_slice = slice.rectangle.ray_intersect(ray)

            visible_boundary_mask &=  (dr.dot(slice.transform.inverse() @ si_boundary.p, mi.Vector3f(0, 0, 1))<0)
            slice_mask = self.inside_closed_surface_mask(si_slice.p) & si_slice.is_valid()

            if coeff is not None:
                fnc_vals = dr.select(slice_mask ,coeff.get_value(si_slice.p), dr.nan)
            else:
                res_slice = [slice_image.shape[-2], slice_image.shape[-1]]
                image_points = slice.transform.inverse() @ si_slice.p
                image_points = mi.Point2f(image_points[0], image_points[1])
                
                indices1 =  res_slice[0] * (1 - (image_points[1] + 1) / 2)
                indices1 = dr.clip(mi.UInt32(indices1), 0, res_slice[0]-1)
                indices2 = res_slice[1] * (image_points[0] + 1) / 2
                indices2 = dr.clip(mi.UInt32(indices2), 0, res_slice[1]-1)
                fnc_vals = dr.gather(mi.Float, slice_image.array,  indices1 * res_slice[1] + indices2) 
                fnc_vals = dr.select(slice_mask, fnc_vals, dr.nan) 
            if input_range is None:
                if not sym_colorbar:
                    input_range = [dr.min(fnc_vals)[0], dr.max(fnc_vals)[0]]
                else:
                    r = max(-dr.min(fnc_vals)[0], dr.max(fnc_vals)[0])
                    input_range = [-r, r]

            if input_range[0] == input_range[1]:
                input_range[0] -= 0.1
                input_range[1] += 0.1
            if input_range[0] >= input_range[1]:
                input_range = [-0.1, 0.1]
            print(input_range)
            norm = mcolors.Normalize(vmin=input_range[0], vmax=input_range[1])
            normalized_value = norm(fnc_vals.numpy())

            color_coeff = get_rgb_from_colormap(normalized_value, colormap_name= colormap)
            color_val = mi.Color3f(color_coeff[0], color_coeff[1], color_coeff[2])
            final_result = dr.select(slice_mask, color_val, final_result)

            ray_last = si_slice.spawn_ray(ray.d)
            si_last = self.scene.ray_intersect(ray_last)
            mask_last = (~slice_mask &  (dr.dot(si_last.n, ray.d) < 0))
            si_boundary = dr.select(mask_last, si_last, si_boundary)
            visible_boundary_mask |= mask_last 

        
        # If boundary condition conf is not specified, then shade the boundary with an environment map.
        if conf_number is None:
            env_map = mi.load_dict({'type': 'envmap',
                       'filename': f'{PATH}/scenes/museum.exr'})
            seq = dr.arange(mi.UInt64, size)
            initstate, initseq = tea(mi.UInt64(seq), mi.UInt64(12))
            sampler = mi.PCG32()
            sampler.seed(initstate, initseq)

            sample = mi.Point2f(sampler.next_float32(), sampler.next_float32())
            
            dirSample, w_light = env_map.sample_direction(si_boundary, sample)
            ray_light = si_boundary.spawn_ray(dirSample.d)            
            # si_light = self.shape_mi.ray_intersect(ray_light)
            si_light = self.scene.ray_intersect(ray_light)

            sphere_color = mi.Color3f(0.3, 0.5, 0.8)
            result = dr.select(~si_light.is_valid(), w_light * sphere_color * dr.abs(dr.dot(si_boundary.n, ray_light.d)), 0.0)

            result_r = mi.math.linear_to_srgb(result[0])
            result_g = mi.math.linear_to_srgb(result[1])
            result_b = mi.math.linear_to_srgb(result[2])

            final_result = dr.select(visible_boundary_mask, mi.Color3f(result_r, result_g, result_b), final_result)
        else:
            raise NotImplementedError
            
        
        final_result = dr.block_sum(final_result, spp) / spp
        final_image = mi.TensorXf(final_result).numpy().T
        return np.reshape(final_image, [image_res[1], image_res[0], 3]), norm
                




            

            
