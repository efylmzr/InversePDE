import numpy as np
import sys
import mitsuba as mi
from mitsuba import Bool, Float, Point2f, PCG32, UInt64, UInt32
from PDE2D import GreenSampling, DIM, ArrayXb, ArrayXf
from ..data_holder import DataHolder
from ...Coefficient import *
from ...Sampling import *
from PDE2D.BoundaryShape import *


class Particle:
    DRJIT_STRUCT = {
        'points' : Point2f,
        'w': Float,
        'sampler' : PCG32,
        'path_index' : UInt32,
        'path_length' : UInt32,
        'thrown' : Bool
    }
    def __init__(self, points=None, w=None, sampler = None, path_index = None, path_length = None):
        self.points = points
        self.w = w
        self.sampler = sampler
        self.path_index = path_index
        self.path_length = path_length
        self.thrown = dr.zeros(Bool, dr.width(self.points))


class WosConstant(object):
    def __init__(self, input : DataHolder, seed : int = 37, 
                 max_z : float = 4, green_sampling : GreenSampling = 0, 
                 newton_steps : int = 5, opt_params : list[str] = []) -> None:
        self.input = input
        if(type(self.input.α) is not ConstantCoefficient):
            raise Exception("Diffusion parameter needs to be constant coefficient!")
        if(type(self.input.σ) is not ConstantCoefficient):
            raise Exception("Screening parameter needs to be constant coefficient!")
        self.σ = Float(self.input.σ.get_value(Point2f(0)) / self.input.α.get_value(Point2f(0)))
        self.seed = UInt64(seed)
        dr.make_opaque(self.seed)
        self.input = input
        self.max_z = Float(max_z)
        dr.make_opaque(self.max_z)
        self.opt_params = {}
        self.get_opt_params(self.opt_params, opt_params)

        if green_sampling == GreenSampling.Polynomial:
            self.green = GreensFunctionPolynomial(dim = DIM.Two, newton_steps = newton_steps)
        else:
            self.green = GreensFunctionAnalytic(dim = DIM.Two, newton_steps = newton_steps)

    def change_seed(self, seed : int):
        self.seed = dr.opaque(UInt64, seed, shape = (1))

    def get_opt_params(self, param_dict: dict, opt_params: list):
        self.input.get_opt_params(param_dict, opt_params)

    def update(self, opt):
        self.input.update(opt)
        self.σ = Float(self.input.σ.get_value(Point2f(0)) / self.input.α.get_value(Point2f(0)))
        self.input.shape.update_shape(opt)
    
    def zero_grad(self):
        self.input.zero_grad()
        
    @dr.syntax(print_code = False)
    def solve(self, points_in = None, active_conf_in : ArrayXb = None, L_in : ArrayXf = None, initial_w : Float = Float(1), 
              mode : dr.ADMode = dr.ADMode.Primal, dL = ArrayXf(0), 
              derivative_dir : Point2f = None, conf_numbers : list[UInt32] = [UInt32(0)], all_inside = False, 
              max_step = dr.inf, normal_derivative_dist : float = None) -> list[Float, Particle]:
        
        if conf_numbers is not None:
            num_conf = len(conf_numbers)
        else:
            num_conf = 1
        
        L_res = dr.zeros(ArrayXf, (num_conf, dr.width(points_in))) if (mode != dr.ADMode.Backward) else L_in

        active_conf = dr.ones(ArrayXb, shape = L_res.shape) if active_conf_in is None else ArrayXb(active_conf_in)
        assert L_res.shape == active_conf.shape

        if (L_in is None) and (mode is dr.ADMode.Backward):
            raise Exception("The primal solution needs to be specified in the backward gradient computation!")

        if mode == dr.ADMode.Forward:
            dL = ArrayXf(0)
        
        active = Bool(True)
        if dr.hint(self.input.shape.single_closed and not all_inside, mode = "scalar"):
            active, L_res = self.input.shape.inside_closed_surface(points_in, L_res, conf_numbers)
        
        particle = Particle(Point2f(points_in), Float(initial_w), PCG32(), dr.arange(UInt32, dr.width(points_in)), UInt32(0))
        
        initstate, initseq = tea(UInt64(particle.path_index), UInt64(self.seed))
        particle.sampler.seed(initstate=UInt64(initstate), initseq=UInt64(initseq))

        with dr.suspend_grad():
            if dr.hint(derivative_dir is not None, mode = "scalar"):
                particle = self.take_derivative_step(derivative_dir, L_res, particle, mode, dL, active, active_conf)
            while active:
                particle = self.take_step(L_res, particle, mode, dL, active, active_conf,
                                          normal_derivative_dist, conf_numbers) 
                active &= particle.path_length < max_step
        return (dL, particle) if mode == dr.ADMode.Forward else (L_res, particle)
    
    @dr.syntax(print_code = False)
    def take_step(self, L : Float, p : Particle, mode : dr.ADMode, dL : Float, active : Bool, active_conf : ArrayXb, 
                  normal_derivative_dist : float, conf_numbers : list[UInt32] = None):
        primal = (mode == dr.ADMode.Primal)

        bi = self.input.shape.boundary_interaction(p.points, star_generation = False, conf_numbers = conf_numbers)
        z = bi.r * dr.sqrt(self.σ)
        if z > self.max_z:
            bi.r *= self.max_z / z
            z = self.max_z
        
        self.green.initialize(z)
        dirichlet_ending = (active & bi.is_e & bi.is_d) 
        
        # Add the dirichlet boundary contribution in epsilon-shell!
        added_near = dr.select(dirichlet_ending & active_conf, p.w * bi.dval, 0)
        
        # Add the result
        L += added_near if dr.hint(primal, mode = 'scalar') else -added_near
        
        # Remove the channels in which the walk is finished. 
        active &= ~dirichlet_ending

        
        p.thrown |= bi.is_far
        active &= ~bi.is_far
    
        # Volume Contribution
        # Source Sampling (self.σ is detached! It is used for pdf calculations.)
        
        normG = Float(0)
        if dr.hint(not self.input.f.is_zero, mode = 'scalar'):
            r, normG = self.green.sample(p.sampler.next_float32(), bi.r, self.σ)
            dir_vol, _ = sample_uniform_direction(p.sampler.next_float32())
            points_vol = p.points + r * dir_vol 

            with dr.resume_grad(when=not primal):
                α_vol = self.input.α.get_value(points_vol)
                f_vol = self.input.f.get_value(points_vol) / α_vol
                f_cont = dr.select(active, p.w * f_vol * normG, 0)
                #if dr.isnan(f_cont):
                #    f_cont = Float(0)
                
                if dr.hint(mode == dr.ADMode.Backward, mode = 'scalar'):
                    dr.backward(dr.sum(f_cont * dL))
                elif dr.hint(mode == dr.ADMode.Forward, mode = 'scalar'):
                    dL += dr.forward_to(dr.sum(f_cont))

            L += f_cont if primal else -f_cont
        else:
            normG = self.green.eval_norm(bi.r, self.σ)

        # Boundary Sampling
        p.points, _ = sample_uniform_boundary(p.sampler.next_float32(), p.points, bi.r)   
        
        # Poisson Kernel computation
        P = (1 - normG * self.σ) 
        p.w *= P   
        p.path_length += 1

        # Boundary and Volume Contribution
        return p
    
    @dr.syntax
    def take_derivative_step(self, derivative_dir : Point2f, L : Float, p : Particle, mode : dr.ADMode, dL : Float, 
                             active : Bool, active_conf : ArrayXb) -> Particle:
        
        primal = (mode == dr.ADMode.Primal)
        # There is no way to sample Green's function analytically. Use polynomial.
        greenGrad = GreensFunctionPolynomial(dim = DIM.Two, newton_steps=10, grad = True)
        
        # Create boundary interaction.
        bi = self.input.shape.boundary_interaction(p.points, star_generation = False)
        # We just create spheres.
        bi.r = bi.d
        # Decrease radius for max_z.
        z = bi.r * dr.sqrt(self.σ)
        if z > self.max_z:
            bi.r *= self.max_z / z
            z = self.max_z

        greenGrad.initialize(z)
        # Remove the channels in which the walk is finished. 
        active &= ~(bi.is_d & bi.is_e)

        # Get the contribution of the source term
        f_cont = Float(0)
        if dr.hint(not self.input.f.is_zero, mode = 'scalar'):
            # Sample norm of the Gradient with the Greens function.
            r, norm_dG = greenGrad.sample(p.sampler.next_float32(), bi.r, self.σ)
            dir_vol, _, sign_vol = sample_cosine_direction(p.sampler.next_float32(), derivative_dir)
            points_vol = p.points + r * dir_vol 
            α_vol = self.input.α.get_value(points_vol)
            
            with dr.resume_grad(when=not primal):
                f_vol = self.input.f.get_value(points_vol) / α_vol
                f_cont = dr.select(active, f_vol * norm_dG * sign_vol * 2 / dr.pi , 0.0)
                #if dr.isnan(f_cont):
                #    f_cont = Float(0)
                if dr.hint(mode == dr.ADMode.Backward, mode = 'scalar'):
                    dr.backward(f_cont * dL)
                elif dr.hint(mode == dr.ADMode.Forward, mode = 'scalar'):
                    dL += dr.forward_to(f_cont)
        
            f_cont = dr.select(active_conf, f_cont, 0)

        L += f_cont if primal else -f_cont
        
        p.points, _, boundary_sign = sample_cosine_boundary(p.sampler.next_float32(), p.points, bi.r, derivative_dir)

        p.w *= eval_dP_norm(bi.r, self.σ) * 4 * bi.r * boundary_sign
        p.path_length += 1
        
        return p
    

    def create_normal_derivative(self, res : int, spp : int, distance : float, conf_numbers : list[UInt32]):
        shape = self.input.shape
        assert isinstance(shape, BoundaryWithDirichlets)
        assert len(shape.in_boundaries) == 1
        in_shape = shape.in_boundaries[0]

        points_, s_points, normal_dir = in_shape.create_boundary_points(distance = dr.epsilon(mi.Float) * 20, res = res, spp = spp, discrete_points = True)
        bi = BoundaryInfo(points_)
        ri = shape.ray_intersect(bi, dr.normalize(normal_dir))

        distance = dr.minimum(distance, 0.3 * ri.t)
        points = points_ + distance * dr.normalize(normal_dir)
        normal_der, _ = self.solve(points, derivative_dir=normal_dir, conf_numbers=conf_numbers, all_inside = True)
        _, result_mi = in_shape.create_boundary_result(normal_der, s_points, res)
        dr.eval(result_mi)
        normal_der = in_shape.set_normal_derivative(result_mi)
        return result_mi, normal_der
            