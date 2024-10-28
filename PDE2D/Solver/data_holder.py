import drjit as dr
import mitsuba as mi
import numpy as np
from PDE2D.Coefficient import *
from PDE2D.BoundaryShape import *
from PDE2D.utils import *
from PDE2D.Sampling import *
from mitsuba import Float, Point2f, TensorXf, Texture2f,Bool, UInt
from PDE2D import  DIM
from enum import IntEnum

class RegularizationType(IntEnum):
    none = 0,
    L2 = 1,
    tensorL2 = 2,
    L1 = 3,
    tensorL1 = 4,
    TV = 5,
    gradL1 = 6,
    gradL2 = 7,
    screeningL1 = 8,
    screeningL2 = 9


class DataHolder(object):
    def __init__(self, shape: Shape = Shape(), bbox_center: list = [0,0],
                 bbox_length = 2.1, max_window_grid = 8, 
                 max_mipmap_res = 1024, min_mipmap_res = 1,
                 max_z = 4, dist_texture_res = 512,
                 α : Coefficient = ConstantCoefficient("diffusion", 1), 
                 σ : Coefficient = ConstantCoefficient("screening", 0), 
                 f : Coefficient = ConstantCoefficient("source", 0), 
                 α_split : Coefficient = None,
                 σ_split : Coefficient = None, 
                 opt_param_shape: list = [], opt_param_α: list = [], 
                 opt_param_σ: list = [], opt_param_f: list = [], 
                 majorant_safety_low: float = 1.02, 
                 majorant_safety_high : float = 1.02,
                 default_majorant : float = None, 
                 verbose = False):
        self.shape = shape
        self.bbox_center = Point2f(bbox_center)
        self.bbox_length = Float(bbox_length)
        self.bbox = [[bbox_center[0] - bbox_length/2, bbox_center[1] - bbox_length/2],
                     [bbox_center[0] + bbox_length/2, bbox_center[1] + bbox_length/2]]
        self.max_mipmap_res = max_mipmap_res
        self.min_mipmap_res = min_mipmap_res
        self.max_window_grid = UInt32(max_window_grid)
        self.max_radius = bbox_length / min_mipmap_res * (max_window_grid - 1)
        self.verbose = verbose
        self.α = α
        self.σ = σ
        self.f = f
        # These are defined for fd computations.
        # When we deviate the coefficients, path splitting weights change 
        # We want fd forward computations to follow the same exact path.
        self.α_split = α_split if (α_split is not None) else α
        self.σ_split = σ_split if (σ_split is not None) else σ
        self.params_shape = opt_param_shape
        self.params_f = opt_param_f
        self.params_σ = opt_param_σ
        self.params_α = opt_param_α
        self.majorant_safety_high = majorant_safety_high
        self.majorant_safety_low = majorant_safety_low
        self.default_majorant = default_majorant
        self.has_continuous_neumann = self.shape.has_continuous_neumann
        self.has_delta = self.shape.has_delta
        self.NEE = self.shape.NEE
        self.Rscale = [Float(0), self.shape.max_distance]
        self.σscale = [Float(0.01), Float(10000)]
        self.meanfree_res = [256, 256] 
        self.dist_tex_res = dist_texture_res
        self.max_z = Float(max_z)
        self.effective_σ = self.calculate_effective_screening(res = self.max_mipmap_res)
        # We are multiplying the negative part with a safety factor as it might increase the througput too much.
        self.majorant = dr.maximum(self.effective_σ * self.majorant_safety_high, -self.effective_σ * self.majorant_safety_low) 
        self.σ_bar =dr.max(self.majorant.array) if self.default_majorant is None else Float(self.default_majorant)
        self.σ_bar = dr.maximum(1e-3, self.σ_bar)
        #self.create_opt_parameters()
            
    def σ_(self, σ, α, grad_α, laplacian_α):   # Equation 21 (2nd paper)
        return σ / α + 1/2 * (laplacian_α / α - dr.squared_norm(grad_α)/(2 * (α ** 2)))
            
    #def create_opt_parameters(self):
    #    self.opt_params = {}
    #    self.shape.get_opt_params(self.opt_params, self.params_shape)
    #    self.α.get_opt_params(self.opt_params, self.params_α)
    #    self.σ.get_opt_params(self.opt_params, self.params_σ)
    #    self.f.get_opt_params(self.opt_params, self.params_f)

    def get_opt_params(self, param_dict: dict, opt_params: list):
        self.shape.get_opt_params_shape(param_dict, opt_params)
        self.α.get_opt_params(param_dict, opt_params)
        self.σ.get_opt_params(param_dict, opt_params)
        self.f.get_opt_params(param_dict, opt_params)

    def update(self, opt):
        self.shape.update(opt)
        self.f.update(opt)
        self.σ.update(opt)
        self.α.update(opt)
        self.α_split = self.α
        self.σ_split = self.σ
        #self.create_accelaration()
                
    def create_accelaration(self):
        self.effective_σ = self.calculate_effective_screening(res = self.max_mipmap_res)
        self.majorant = dr.maximum(self.effective_σ * self.majorant_safety_high, -self.effective_σ * self.majorant_safety_low) 
        self.σ_bar =dr.max(self.majorant.array) if self.default_majorant is None else self.default_majorant
        self.σ_bar = dr.maximum(1e-3, self.σ_bar)
        self.majorant = (dr.maximum(1e-3, self.majorant))
        self.majorant_tex = TextureCoefficient("effective_screening", self.bbox, self.majorant.numpy(), interpolation = "linear")
        self.σ_mipmap = self.create_mipmap(self.majorant, min_res = self.min_mipmap_res, type = "max") 

        self.meanfree_tex = self.get_mean_free_image()
        self.r_best_tex, self.σ_best_tex, self.σ_begin_tex = self.get_Rσ_domain(res = self.dist_tex_res, n_bisection=5, n_grid_search=10)

    def get_mean_free_image(self, spp = 2**8, resolution = [256, 256]):
        R = self.Rscale[0] + (self.Rscale[1] - self.Rscale[0]) * dr.arange(Float, resolution[0]) / (resolution[0] - 1) 
        σ = self.σscale[0] * 2 ** (dr.arange(Float, resolution[1]) / (resolution[1] - 1) * dr.log2(self.σscale[1] / self.σscale[0]))
        RR, σσ = dr.meshgrid(R, σ, indexing = 'ij')
        RR = dr.repeat(RR, spp)
        σσ = dr.repeat(σσ, spp)
        
        z = Float(RR * dr.sqrt(σσ))
        sample = dr.arange(Float, spp) / spp + 1/(2 * spp)
        sample = dr.tile(sample, (resolution[0]) * resolution[1]) 
        green = GreensFunctionAnalytic(dim = DIM.Two, newton_steps = 8, grad = False)
        r, normG = green.sample(sample, RR, σσ)
        prob_boundary = 1 - σσ * normG
        result = r * (1-prob_boundary) + RR * prob_boundary
        result = dr.select(RR == 0, 0, result)
        result = TensorXf(dr.block_sum(result, spp) / spp)
        result = dr.reshape(TensorXf, result, shape = [resolution[0], resolution[1], 1])
        result_tex = Texture2f(result)
        return result_tex
    
    def get_mean_free_path(self, R, σ):
        Rgrid = 1 / self.meanfree_res[0]
        σgrid = 1 / self.meanfree_res[1]
        ind_R =  Rgrid / 2 + (R - self.Rscale[0]) / (self.Rscale[1] - self.Rscale[0]) * (1.0-Rgrid) 
        ind_σ = σgrid/2 + dr.log2(σ / self.σscale[0]) / dr.log2(self.σscale[1] / self.σscale[0]) * (1.0 - σgrid)
        return self.meanfree_tex.eval(Point2f(ind_σ, ind_R))[0]
        
    def calculate_effective_screening(self, res = 1024):
        with dr.suspend_grad():
            resolution  = [res, res]
            points = create_image_points(self.bbox, resolution, 1,  centered = True)
            active = Bool(True)
            if (self.shape.single_closed):
                active = self.shape.inside_closed_surface_mask(points)
            # Calculate the textures
            α_vals = self.α_split.get_value(points)
            grad_α, laplacian_α = self.α_split.get_grad_laplacian(points)
            σ_vals = self.σ_split.get_value(points)
            # Equation 21 (2nd paper)
            σ_new = self.σ_(σ_vals, α_vals, grad_α, laplacian_α)
            # Eliminate the calculations outside the boundary (if the given shape 
            # is single closed boundary)
            σ_new = dr.select(active, σ_new, 0)
            numpy_σ, tensor_σ  = create_image_from_result(σ_new, resolution)
            self.eff_screening_tex = TextureCoefficient("effective_screening", self.bbox, numpy_σ[0], interpolation = "linear")
            return tensor_σ[0]
            
    def create_mipmap(self, tensor, min_res, type = "max"):
        # Now create the mipmap hierarchy
        res = tensor.shape[0]
        num_iter = int(dr.floor(dr.log2(res // min_res)))
        n = res * res
        array = dr.zeros(Float, n * (num_iter + 1))
        current_res = res
        current_array = Float(tensor.array)
        dr.eval(current_array)
        dr.scatter(array, current_array, dr.arange(UInt, n))

        for k in range(num_iter):
            current_res //= 2
            i = dr.arange(UInt, current_res) * 2
            j = dr.arange(UInt, current_res) * 2
            ii, jj = dr.meshgrid(i, j, indexing = "ij")
            
            index00 = ii * current_res * 2 + jj
            index01 = ii * current_res * 2 + jj + 1
            index10 = (ii + 1) * current_res * 2 + jj
            index11 = (ii + 1) * current_res * 2 + jj + 1
    
            dr.eval(index00, index01, index10, index11)
            val00 = dr.gather(Float, current_array, index00)
            val01 = dr.gather(Float, current_array, index01)
            val10 = dr.gather(Float, current_array, index10)
            val11 = dr.gather(Float, current_array, index11)
            if type == "max":
                max0 = dr.maximum(val00, val01)
                max1 = dr.maximum(val10, val11)
                current_array = dr.maximum(max0, max1)
            elif type == "min":
                min0 = dr.minimum(val00, val01)
                min1 = dr.minimum(val10, val11)
                current_array = dr.minimum(min0, min1)
            elif type == "mean":
                current_array = (val00 + val01 + val10 + val11) / 4
            else:
                raise Exception("There is no such mipmap creation type.")
            current_tensor = TensorXf(current_array)
            current_tensor = dr.reshape(TensorXf, value = current_tensor, shape = [current_res, current_res])
            u_factor = res // current_res
            current_upsampled = upsample(current_tensor, scale_factor = [u_factor, u_factor])
            #current_upsampled = dr.upsample(current_tensor, scale_factor=[res//current_res, res//current_res])
            dr.scatter(array, current_upsampled.array, dr.arange(UInt, n) + (k+1) * n)
        tensor = TensorXf(array)
        tensor = dr.reshape(TensorXf, value = tensor, shape = [num_iter + 1, res, res])
        #return TensorXf(array, shape = [num_iter + 1, res, res])
        return tensor
    @dr.syntax
    def get_sphere_screening(self, points, radius):
        x = (points[0] - self.bbox[0][0]) / self.bbox_length
        y = 1.0 - (points[1] - self.bbox[0][1]) / self.bbox_length
        k_max, res_all,_ = self.σ_mipmap.shape
        #mask = mi.TensorXf(mi.Float(0) ,shape = [res_all, res_all])
        
        k_max -= 1
        # which mipmap level to select
        k =   UInt32(dr.ceil(dr.log2(2 * radius * res_all / ((self.max_window_grid - 1) * self.bbox_length))))
        
        k = dr.select(k > k_max, k_max, k)
        if k < 0:
            k = UInt32(0)
        #dr.select(k < 0, 0, k)
        # resolution of the selected grid
        res_decrease = UInt32(dr.round(Float(2)**Float(k)))
        
        #res_decrease = mi.UInt32(4)
        res = res_all // res_decrease
        
        n1_point = UInt32(dr.floor(y * res))
        n2_point = UInt32(dr.floor(x * res))
        
        # get the center grid val of sphere
        if self.max_window_grid % 2 == 0:
            n1 = UInt32(dr.round(y * res))
            n2 = UInt32(dr.round(x * res))
        else:
            n1 = n1_point
            n2 = n2_point
        
        # get the index of the window 
        n1_start = n1 - self.max_window_grid//2
        n2_start = n2 - self.max_window_grid//2
        #v = 0
        v = UInt32(0)
        # We start the majorant with the correspoinding grid where the point is inside
        index_point = k * res_all * res_all + n1_point * res_decrease * res_all + n2_point * res_decrease
        majorant = dr.gather(Float, self.σ_mipmap.array, index_point)
        
        #i = dr.arange(mi.UInt, res_decrease[0])
        #j = dr.arange(mi.UInt, res_decrease[0])
        #ii, jj = dr.meshgrid(i, j, indexing = "ij")
        #mask_indices = (ii + n1_point * res_decrease)  * res_all + jj + n2_point * res_decrease
        #dr.scatter(mask.array, mi.Float(1), mask_indices)
        grid_length = self.bbox_length / res
        
        #loop = mi.Loop("Iterate over grids and get the max majorant if it touches the sphere!", state= lambda : (v, majorant))
        while (v < self.max_window_grid**2):
            n1_iter = v // self.max_window_grid + n1_start
            n2_iter = v % self.max_window_grid + n2_start
            
            n1_iter = dr.select(n1_iter<0, 0, n1_iter)
            n1_iter = dr.select(n1_iter>=res, res-1, n1_iter)
            n2_iter = dr.select(n2_iter<0, 0, n2_iter)
            n2_iter = dr.select(n2_iter>=res, res-1, n2_iter)
  
            square_corner_x = self.bbox[0][0] + n2_iter * grid_length
            square_corner_y = self.bbox[0][1] + (res - n1_iter - 1) * grid_length
            corner = Point2f(square_corner_x, square_corner_y)
            dist = self.get_distance_to_square(points, corner, grid_length)
            
            #if dist[0] < radius:
            #    i = dr.arange(mi.UInt, res_decrease[0])
            #    j = dr.arange(mi.UInt, res_decrease[0])
            #    ii, jj = dr.meshgrid(i, j, indexing = "ij")
            #    mask_indices = (ii + n1_iter * res_decrease)  * res_all + jj + n2_iter * res_decrease
            #    dr.scatter(mask.array, mi.Float(1), mask_indices)
            index_point = k * res_all * res_all + n1_iter * res_decrease * res_all + n2_iter * res_decrease
            majorant_iter = dr.gather(Float, self.σ_mipmap.array, index_point)
            majorant = dr.select(dist < radius, dr.maximum(majorant_iter, majorant), majorant)
            v += 1
        #mask_tex = TextureCoefficient("mask", self.bbox, mask.numpy(), interpolation = "nearest")   
        return majorant
    
    def compute_regularization(self, λ : float, type : RegularizationType, 
                                resolution = [256, 256], spp = 1, coeff_str = "diffusion"):
        out_val = 0
        coeff = self.get_coefficient(coeff_str)
        if coeff.out_val is not None:
            out_val = coeff.out_val
        with dr.suspend_grad():
            points = self.shape.create_volume_points(resolution, spp)
            dL = dr.ones(Float, dr.width(points)) * dr.rcp(dr.width(points))
        if type == RegularizationType.none:
            reg = Float(0)

        elif type == RegularizationType.L2:
            vals = coeff.get_value(points)
            reg = dr.square(vals - out_val)

        elif type == RegularizationType.tensorL2:
            resolution = coeff.tensor.shape[0:2]
            reg = Float(0)
            dL = Float(1)
            for i in range(resolution[0]):
                for j in range(resolution[1]):
                    index = i * resolution[1] + j 
                    val = dr.gather(Float, self.α.tensor.array, index)
                    reg += dr.square(val - out_val)
        elif (type == RegularizationType.L1):
            vals = coeff.get_value(points) 
            reg = dr.abs(vals - out_val) 

        elif (type == RegularizationType.tensorL1):
            resolution = coeff.tensor.shape[0:2]
            reg = Float(0)
            dL = Float(1)
            for i in range(resolution[0]):
                for j in range(resolution[1]):
                    index = i * resolution[1] + j 
                    val = dr.gather(Float, self.α.tensor.array, index)
                    reg += dr.abs(val - out_val)
            reg /= ((resolution[0]) * resolution[1])

        elif (type == RegularizationType.TV):
            resolution = coeff.tensor.shape[0:2]
            reg = Float(0)
            dL = Float(1)
            for i in range(resolution[0]-1):
                for j in range(resolution[1]-1):
                    index = i * resolution[1] + j 
                    val = dr.gather(Float, self.α.tensor.array, index)
                    val1 = dr.gather(Float, self.α.tensor.array, index+1)
                    val2 = dr.gather(Float, self.α.tensor.array, index+resolution[1])
                    reg += dr.abs(val1 - val)
                    reg += dr.abs(val2 - val)
            reg /= ((resolution[0]-1) * resolution[1]-1)

        elif (type == RegularizationType.gradL1):
            grad = coeff.get_grad_laplacian(points)[0]
            reg =   dr.abs(grad[0]) + dr.abs(grad[1])

        elif(type == RegularizationType.gradL2):
            grad = coeff.get_grad_laplacian(points)[0]
            reg =   dr.squared_norm(grad)

        elif (type == RegularizationType.screeningL2) or (type == RegularizationType.screeningL1):
            σ = self.σ.get_value(points)
            α = self.α.get_value(points)
            grad_α, laplacian_α = self.α.get_grad_laplacian(points)
            σ_ = self.σ_(σ, α, grad_α, laplacian_α)
            reg = dr.square(σ_) if type == RegularizationType.screening_squared else dr.abs(σ_)

        else:
            raise Exception("There is no such regularization type.")
        return dL * reg * λ
    
    @dr.syntax
    def get_Rσ(self, points, radius, n_bisection = 10, n_grid_search = 10, screening_offset = Float(0)):
        σ_begin = self.get_sphere_screening(points, radius + 2 * screening_offset)
        σ = self.get_sphere_screening(points, radius + screening_offset)
        z = radius * dr.sqrt(σ)
        
        # We will shrink these radii for g
        r = Float(radius)
        # Here we shrink the radii of the spheres where z is high by bisection.
        # At each iter we shrink to the middle value of max and min z, and compute 
        # the corresponding z value by also querying the correct majorant value.
        # If we found something close enough to z_high, we end the iteration.
        if z > self.max_z:
            r_high = Float(radius)
            r_low  = self.max_z / dr.sqrt(σ)
            i = UInt32(0)
            while i < n_bisection:
                r_sep = (r_high + r_low) / 2
                σ_sep = self.get_sphere_screening(points, r_sep + screening_offset)
                z_sep = r_sep * dr.sqrt(σ_sep)
                if z_sep < self.max_z:
                    r_low = Float(r_sep)
                else:
                    r_high = Float(r_sep)
                i += 1
            r = Float(r_low)
            σ = self.get_sphere_screening(points, r + screening_offset)
            z = r * dr.sqrt(σ)
        
        # Now all z vals should be in the correct range that we can sample from.
        # We will compute the best radius value in terms of the mean free path 
        # by grid search.
        i = UInt32(0)
        meanfree_best = Float(0) 
        r_best = Float(0)
        while i < n_grid_search:
            r_iter = r * Float(i + 1) / n_grid_search
            σ_iter = self.get_sphere_screening(points, r_iter + screening_offset)
            meanfree_iter = self.get_mean_free_path(r_iter, σ_iter)
            if meanfree_iter > meanfree_best:
                meanfree_best = meanfree_iter
                r_best = r_iter
                σ = σ_iter
            i += 1
        
        return r_best, σ, σ_begin
        
    def get_coefficient(self, name : str = "diffusion"):
        if name == "diffusion":
            return self.α
        elif name == "screening":
            return self.σ
        elif name == "source":
            return self.f
        else:
            raise Exception("There is no such coefficient.")

    def get_Rσ_domain(self, res, n_bisection = 10, n_grid_search = 10):
        points = create_image_points(self.bbox, resolution = [res, res], spp = 1, centered = True)
        bi = self.shape.boundary_interaction(points, star_generation=False)
        # We will always add these small offset value while computing the majorant to 
        # account for the grid size. 
        s_offset = self.bbox_length / res / dr.sqrt(2) * 1.01
        self.radius_threshold = s_offset * 5
        
        r_best, σ_best, σ_begin = self.get_Rσ(points, bi.r, n_bisection = n_bisection, n_grid_search=n_grid_search,
                                     screening_offset=s_offset)
        # We need to be careful while using the corresponding radii as it does not represent 
        # exactly the correct radius values.
        r_image, _ = create_image_from_result(r_best, resolution = [res, res])
        σ_image, _ = create_image_from_result(σ_best, resolution = [res, res])
        σ_begin_image, _ = create_image_from_result(σ_begin, resolution = [res, res])
        r_best_tex = TextureCoefficient("Best-radius", self.bbox, r_image[0], interpolation = "nearest")
        σ_best_tex = TextureCoefficient("Best-majorant", self.bbox, σ_image[0], interpolation = "nearest")
        σ_begin_tex = TextureCoefficient("Beginning-majorant", self.bbox, σ_begin_image[0], interpolation = "nearest")
        return r_best_tex, σ_best_tex, σ_begin_tex
    
    @dr.syntax
    def get_Rσz(self, points, radius):
        r = self.r_best_tex.get_value(points)
        σ = self.σ_best_tex.get_value(points)
        σ_begin = self.σ_begin_tex.get_value(points)

        # If we chose a greater best radius due to discretization of the domain or 
        # if the distance is too small, then select the original distance for taking a step!
        if (radius < r) | (radius < 20 * self.shape.epsilon) | (radius < self.radius_threshold):
            r = radius
            σ = σ_begin 

        σ = dr.maximum(1e-3, σ)
        z = r * dr.sqrt(σ)
        # For rare cases, now the z value might be larger than the max z. Especially if the majorant 
        # is super high near the boundary.
        if z >= self.max_z:
            r *= (self.max_z / z)
            z = self.max_z
        # return the selected parameters for sampling the next step.
        return r, σ, z
        
    @dr.syntax
    def get_distance_to_square(self, point, corner, length):
        i = UInt32(0)
        min1 = Float(dr.inf)
        min2 = Float(dr.inf)
        p1 = Point2f(dr.nan)
        p2 = Point2f(dr.nan)
        while i < 4:
            n1 = Float(i // 2)
            n2 = Float(i %  2)
            corner_ = corner + length * (Point2f(0,1) * n1 + 
                                         Point2f(1,0) * n2)
            dist = dr.norm(corner_ - point)
            mask1 = dist < min1
            mask2 = dist < min2
            min2 = dr.select(mask1, min1, min2)
            min1 = dr.select(mask1, dist, min1)
            min2 = dr.select(~mask1 & mask2, dist, min2)
            p2 = Point2f(dr.select(mask1, p1, p2))
            p1 = Point2f(dr.select(mask1, corner_, p1))
            p2 = Point2f(dr.select(~mask1 & mask2, corner_, p2))
            i += 1
        vec1 = dr.normalize(p2 - p1)
        vec2 = point - p1
        d = dr.dot(vec1,vec2)
        d = dr.select(d<0, 0, d)
        d = dr.select(d>length, length, d)
        closest_point = p1 + d * vec1
        return dr.norm(point - closest_point)
        
    def zero_grad(self):
        self.α.zero_grad()
        self.σ.zero_grad()
        self.f.zero_grad()
        self.shape.zero_grad()
        
    def visualize(self, ax1, ax2, ax3, ax4, resolution = [512, 512], spp = 4):
        self.f.visualize(ax1, self.bbox, resolution, spp)
        self.σ.visualize(ax2, self.bbox, resolution, spp)
        self.α.visualize(ax3, self.bbox, resolution, spp)
        image, tensor = self.get_effective_screening(resolution, spp)
        plot_image(image[0], ax4)
        ax1.set_title("Source (f)")
        ax2.set_title("Screening (σ)")
        ax3.set_title("Diffusion (α)")
        ax4.set_title("Effective Screening (σ)")
        
    def get_effective_screening(self, resolution = [512, 512], spp = 4):
        points = create_image_points(self.bbox, resolution, spp)
        σ = self.σ.get_value(points)
        α = self.α.get_value(points)
        grad_α, laplacian_α = self.α.get_grad_laplacian(points)
        effective_σ = σ / α + 1/2 * (laplacian_α / α - dr.squared_norm(grad_α)/(2 * (α ** 2)))
        return create_image_from_result(effective_σ, resolution)
    
    def get_point_neumann(self, bi : BoundaryInfo, conf_number : UInt32) -> tuple[list[Float], list[Float], list[Float], list[Point2f]]:
        return self.shape.get_point_neumann(bi, conf_number)
    
    def sampleNEE_special(self, bi:BoundaryInfo, sample : Float, conf_number : UInt32):
         # If we have a special sampling routine for getting NEE. (sampling only electrodes.)
        return self.shape.sampleNEE(bi, sample, conf_number)
    
    @dr.syntax
    def sampleNEE(self, bi : BoundaryInfo, sample : Float, conf_numbers : list[UInt32]) -> tuple[Float, Float, Float, Point2f]:
        d, pdf_n_r, sampled_p = (Float(0), Float(1), Point2f(0))
        n_val = dr.zeros(ArrayXf, shape = (len(conf_numbers), dr.width(bi.origin)))
        if dr.hint(self.NEE == NEE.Normal, mode = 'scalar'): # Sample uniformly to the star part of the sphere.
            # Sampled direction for getting the Neumann contribution.
            dir_n, pdf_n = bi.sample_neumann(sample, bi.on_boundary)
            # Check the selected sample hits to the boundary shape with neumann value.
            #d, sampled_p, normals_n = self.shape.ray_intersect(bi.origin, dir_n, bi.on_boundary)
            ri = self.shape.ray_intersect(bi, dir_n, conf_numbers =conf_numbers)
            d = ri.t
            sampled_p = ri.intersected
            # If we hit to the boundary, add the contribution.
            if bi.is_star & (ri.t < bi.r) & ~ri.is_dirichlet:
                for i in range(len(conf_numbers)):
                    n_val[i] = Float(ri.neumann[i])
                pdf_n_r = pdf_n * dr.abs(dr.dot(dir_n, ri.normal)) * 2 * dr.pi # pdf multiplied with 2 * pi * bi.r
        
        elif dr.hint(self.NEE == NEE.BruteForce, mode = 'scalar'):
            dir_n, pdf_n = bi.sample_brute_force(sample)
            ri = self.shape.ray_intersect(bi, dir_n, conf_numbers =conf_numbers)
            d = ri.t
            sampled_p = ri.intersected

            if bi.is_star & (ri.t < bi.r) & ~ri.is_dirichlet:
                for i in range(len(conf_numbers)):
                    n_val[i] = Float(ri.neumann[i])
                pdf_n_r = pdf_n * dr.abs(dr.dot(dir_n, ri.normal)) * 2 * dr.pi # pdf multiplied with 2 * pi * bi.r
        return d, n_val, pdf_n_r, sampled_p
    

    def compute_high_conductance_points(self, max_num_points = 3, cond_threshold = 2, grad_threshold = 1, merge_distance = 0.2):
        bbox = self.shape.bbox
        bbox_center = Point2f(bbox[0][0] + bbox[1][0],
                              bbox[0][1] + bbox[1][1])
        bbox_length = max(bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1])

        if isinstance(self.shape, BoundaryWithDirichlets):
            points = self.shape.out_boundary.create_volume_points(resolution = [1024, 1024])
        else:
            points = self.shape.create_volume_points(resolution = [1024, 1024])

        val = self.α.get_value(points)
        grad, laplacian = self.α.get_grad_laplacian(points)
        mask = (dr.norm(grad) < grad_threshold) & (val > cond_threshold) & (laplacian < 0)
        indices = dr.compress(mask)
        points = dr.gather(Point2f, points, indices)
        if np.size(points.numpy()) == 0:
            return bbox_center.numpy().T

        #means = create_circle_points(origin=bbox_center, radius = bbox_length, 
        #                             resolution = 20, spp = 1, discrete_points= True)
        means = self.shape.create_volume_points(resolution = [16,16])

        means, groups = k_means(points, means, num_iter = 3)
        mask = ~dr.isnan(means[0] + means[1])
        indices = dr.compress(mask)
        means = dr.gather(Point2f, means, indices)

        """
        # Merge close points
        nmeans = dr.width(means)
        ind = dr.arange(UInt32, nmeans)
        for i in range(nmeans):
            if ind[i] == i:
                for j in range(i + 1, nmeans):
                    means_i = dr.gather(Point2f, means, i)
                    means_j = dr.gather(Point2f, means, j)
                    if dr.norm(means_i - means_j)[0] < merge_distance * bbox_length:
                        dr.scatter(means, Point2f(dr.nan), j)
                        ind[j] = i
        """
        # Recompute the means once more.
        #mask = ~dr.isnan(means[0] + means[1])
        #indices = dr.compress(mask)
        #means = dr.gather(Point2f, means, indices)
        means, groups = k_means(points, means, num_iter = 1)

        # Get the highest conduction region.∂
        val = self.α.get_value(points)
        cond_sum = dr.zeros(Float, dr.width(means))
        counter_sum = dr.zeros(Float, dr.width(means))
        dr.scatter_add(cond_sum, val, groups)
        dr.scatter_add(counter_sum, Float(1), groups)
        mean_cond = (cond_sum / counter_sum)
        
        # Now we sort with numpy to get the biggest mean conductance regions.
        mean_cond_np = mean_cond.numpy()
        
        #sort_index = mean_cond_np.argsort()[::-1][:num_points]
        sort_index = mean_cond_np.argsort()[::-1]
        
        # means
        means = means.numpy()[:, sort_index].T

        # Now we eliminate the points that are very close to the region 
        # we are interested in.
        n = means.shape[0]
        i = 0
        while(i < n):
            deleted_indices = []
            for k in range(i+1, n):
                dist = np.linalg.norm(means[i] - means[k])
                if dist < merge_distance * bbox_length:
                    deleted_indices.append(k)
            means = np.delete(means, deleted_indices, axis = 0)
            n = means.shape[0]
            i += 1
        num_points = min(means.shape[0], max_num_points)
        means =  means[:num_points]
        # Apply one last k-means
        #means = k_means(points, Point2f(means.T), num_iter = 2)[0].numpy()
        #return means.T
        if means.shape[0] == 0:
            means = np.zeros([1,2])
        return means
    
    def upsample2(self, coefficient = "diffusion"):
        coeff = self.get_coefficient(coefficient)
        coeff.upsample2()
