import mitsuba as mi
from .boundary_shape import *
from PDE3D.utils import create_bbox_points

class SDF(Shape):
    DRJIT_STRUCT = {
        'tensor' : mi.TensorXf,
        }
    def __init__(self, sdf_grid : np.array,  epsilon=1e-3, inf_distance = 10,  
                 dirichlet : list[Coefficient] = [], mesh_scene_xml = None, render_sdf = False, sdf_offset = 1.05, scale = 1):
        super().__init__(True, epsilon, inf_distance, dirichlet)
        grid = mi.TensorXf(sdf_grid[...,np.newaxis])
        self.grid_texture = mi.Texture3f(grid, wrap_mode=dr.WrapMode.Clamp, use_accel=False, migrate=False, 
                                    filter_mode=dr.FilterMode.Linear)

        if mesh_scene_xml == None:
            transform = mi.ScalarTransform4f().translate(-0.5 * scale).scale(scale)
            self.scale = dr.opaque(mi.Float, scale, shape = (1))
            self.mesh_scene = None
        else:
            self.mesh_scene = mi.load_file(mesh_scene_xml)
            max_range_mesh = dr.max(self.mesh_scene.bbox().max - self.mesh_scene.bbox().min) * sdf_offset
            center_mesh = (self.mesh_scene.bbox().max + self.mesh_scene.bbox().min) / 2
            transform = mi.ScalarTransform4f().scale(max_range_mesh).translate(mi.ScalarPoint3f(-0.5 + center_mesh/max_range_mesh))
            self.scale = dr.opaque(mi.Float, max_range_mesh, shape = (1))
        
        self.scene = mi.load_dict({
                            "type" : "scene",
                            'integrator_id': {
                                'type': 'path',
                                'max_depth': 5
                                },
                            "sdf": {
                                    "type" : "sdfgrid",
                                    "to_world" : transform,
                                    'grid': mi.TensorXf(grid)
                                     },
                            "emitter" : {
                                "type" : "constant"
                            }})
        
        self.render_sdf = render_sdf | (self.mesh_scene is None)
        self.transform = mi.Transform4f(transform)
        dr.make_opaque(self.transform)

        points = create_bbox_points(self.scene.bbox(), [256,256,256], spp = 1)
        mask = self.closest_dist(points) > 0
        points = dr.select(mask, points, dr.nan)
        min_bbox = mi.ScalarPoint3f(dr.min(points[0])[0], dr.min(points[1])[0], dr.min(points[2])[0])
        max_bbox = mi.ScalarPoint3f(dr.max(points[0])[0], dr.max(points[1])[0], dr.max(points[2])[0])
        self.bbox = mi.ScalarBoundingBox3f(min = min_bbox, max = max_bbox)


    def inside_closed_surface(self, points : mi.Point3f, L : mi.Float, conf_numbers : list[mi.UInt32] = None):
        points = self.transform.inverse() @ points
        return (self.grid_texture.eval(points)[0] < 0), L
    
    
    def inside_closed_surface_mask(self, points : mi.Point3f):
        points = self.transform.inverse() @ points
        return self.grid_texture.eval(points)[0] < 0
    
    def closest_dist(self, points : mi.Point3f):
        p = self.transform.inverse() @ points
        return -self.grid_texture.eval(p)[0] * self.scale
    @dr.syntax
    def boundary_interaction(self, points: mi.Point3f, conf_numbers : list[mi.UInt32] = [mi.UInt32(0)]) -> BoundaryInfo:
        p = self.transform.inverse() @ points
        dist_boundary = -self.grid_texture.eval(p)[0] * self.scale
        boundary_point = mi.Vector3f(points)
        
        num_conf = len(conf_numbers)
        dirichlet = dr.zeros(ArrayXf, shape = (num_conf, dr.width(points)))

        if dr.hint(self.num_conf_d == 1, mode = 'scalar'):
            dirichletval = self.dirichlet[0].get_value(boundary_point)
            for i in range(num_conf):
                dirichlet[i] = dirichletval
        else:
            for i in range(self.num_conf_d): 
                for j, conf in enumerate(conf_numbers):
                    dirichlet[j] = dr.select(i == conf, 
                                            self.dirichlet[i].get_value(boundary_point), 
                                            dirichlet[j])
        return BoundaryInfo(points, dist_boundary, mi.Float(dist_boundary), dist_boundary < self.epsilon,  mi.Point3f(boundary_point), dirichlet)
    

    def visualize(self, image_res = [512, 512],  
                  cam_origin = mi.ScalarPoint3f(1,1,2), cam_target = mi.ScalarPoint3f(0, 0, 0), cam_up = mi.ScalarPoint3f([0,1,0]),
                  scale_cam = 2, slice : Slice = None, coeff = None,
                  slice_image = None, input_range = None, conf_number : mi.UInt32 = None, colormap : str = "viridis",
                  spp = 64, sym_colorbar = False):
        
        if (conf_number is not None) and (slice is not None):
            raise Exception("Either specify a boundary condition or a slice.")
        
        cam = mi.load_dict({'type': 'orthographic',
                            'to_world': mi.ScalarTransform4f().look_at(
                            origin=cam_origin,
                            target=cam_target,
                            up=cam_up,
                            )@ mi.ScalarTransform4f().scale([1 / scale_cam, 1 / scale_cam, 1]), 
                            'film' : {'type': 'hdrfilm',
                            'pixel_format': 'rgba',
                            'width': 512,
                            'height': 512}}) 
        
        # Construct a grid of 2D coordinates
        x, y = dr.meshgrid((dr.arange(mi.Float, image_res[0]) + 0.5) / image_res[0],
                           (dr.arange(mi.Float, image_res[1]) + 0.5) / image_res[1])

    
        size = image_res[0] * image_res[1]
        #x = dr.repeat(x, spp)
        #y = dr.repeat(y, spp)
        # Ray origin in local coordinates
        ray_origin_local = mi.Vector2f(x, y)
        
        time = 0.
        wav_sample = [0.5, 0.33, 0.1] # Not being used
        aperture_sample = 0 # Not being used

        ray, spec_weight = cam.sample_ray(0.0, wav_sample, ray_origin_local, aperture_sample)
        direction = mi.Vector3f(ray.d)

        active = dr.ones(mi.Bool, size)
        final_result = dr.ones(mi.Color3f, size)
        norm = None
        fnc_vals = None

        scene = self.scene if self.render_sdf else self.mesh_scene

        if slice is not None:
            if (slice_image is not None) and (coeff is not None):
                raise Exception("Only specify a solution or a coefficient.")
            elif (slice_image is None) and (coeff is None):
                coeff = ConstantCoefficient("", 0)
                input_range = [-0.5, 0.5]
            si_slice = slice.rectangle.ray_intersect(ray)

            #visible_boundary_mask &=  (dr.dot(slice.transform.inverse() @ si_boundary.p, mi.Vector3f(0, 0, 1))<0)
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
            norm = mcolors.Normalize(vmin=input_range[0], vmax=input_range[1])
            normalized_value = norm(fnc_vals.numpy())

            color_coeff = get_rgb_from_colormap(normalized_value, colormap_name= colormap)
            color_val = mi.Color3f(color_coeff[0], color_coeff[1], color_coeff[2])
            final_result = dr.select(slice_mask, color_val, final_result)

            active &= ~slice_mask 
            ray = dr.select(si_slice.is_valid(), si_slice.spawn_ray(ray.d), ray)

        white_mask = ~scene.ray_intersect(ray).is_valid()
        active &= ~white_mask

        
        # If boundary condition conf is not specified, then shade the boundary with an environment map. 
        if conf_number is None:
            ray_d = mi.RayDifferential3f(o = dr.repeat(ray.o, spp), d = direction)
            sampler = mi.load_dict({'type': 'independent'})
            sampler.seed(1, size * spp)
            L_, _ ,_ = self.scene.integrator().sample(scene, sampler, ray_d, active = dr.repeat(active, spp))
            L_ = dr.select(dr.isfinite(L_), L_, 0)
            L = dr.block_sum(L_, spp) / spp
            result_r = mi.math.linear_to_srgb(L[0])
            result_g = mi.math.linear_to_srgb(L[1])
            result_b = mi.math.linear_to_srgb(L[2])
            final_result = dr.select(active, mi.Color3f(result_r, result_g, result_b), final_result)
        else:
            si = scene.ray_intersect(ray)
            boundary_val = self.dirichlet[conf_number].get_value(si.p)
            boundary_val = dr.select(active, boundary_val, dr.nan)
            
            if input_range is None:
                if not sym_colorbar:
                    input_range = [dr.min(boundary_val)[0], dr.max(boundary_val)[0]]
                else:
                    r = max(-dr.min(boundary_val)[0], dr.max(boundary_val)[0])
                    input_range = [-r, r]
            if input_range[0] == input_range[1]:
                input_range[0] -= 0.1
                input_range[1] += 0.1
            if input_range[0] >= input_range[1]:
                input_range = [-0.1, 0.1]
            
            norm = mcolors.Normalize(vmin=input_range[0], vmax=input_range[1])
            normalized_value = norm(boundary_val.numpy())
            color_coeff = get_rgb_from_colormap(normalized_value, colormap_name= colormap)
            color_val = mi.Color3f(color_coeff[0], color_coeff[1], color_coeff[2])
            final_result = dr.select(active, color_val, final_result)
        

        #final_result = dr.select(white_mask, mi.Color3f(1,1,1), final_result)
        final_image = mi.TensorXf(final_result).numpy().T
        return np.reshape(final_image, [image_res[1], image_res[0], 3]), norm