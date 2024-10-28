import numpy as np
import sys
from .data_holder import DataHolder
from PDE3D.Coefficient import *
from PDE3D.Sampling import *
from PDE3D.BoundaryShape import *
from PDE3D import ArrayXf

class Particle:
    DRJIT_STRUCT = {
        'points' : mi.Point3f,
        'w': mi.Float,
        'sampler' : mi.PCG32,
        'path_index' : mi.UInt32,
        'path_length' : mi.UInt32,
        'thrown' : mi.Bool
    }
    def __init__(self, points=None, w=None, sampler = None, path_index = None, path_length = None):
        self.points = points
        self.w = w
        self.sampler = sampler
        self.path_index = path_index
        self.path_length = path_length
        self.thrown = dr.zeros(mi.Bool, dr.width(self.points))


class WosConstant(object):
    def __init__(self, input : DataHolder, seed : int = 37, 
                 max_z : float = 5, newton_steps : int = 5, opt_params : list[str] = []) -> None:
        self.input = input
        if(type(self.input.α) is not ConstantCoefficient):
            raise Exception("Diffusion parameter needs to be constant coefficient!")
        if(type(self.input.σ) is not ConstantCoefficient):
            raise Exception("Screening parameter needs to be constant coefficient!")
        self.σ = mi.Float(self.input.σ.get_value(mi.Point3f(0.)) / self.input.α.get_value(mi.Point3f(0.)))
        self.seed = mi.UInt64(seed)
        dr.make_opaque(self.seed)
        self.input = input
        self.max_z = mi.Float(max_z)
        dr.make_opaque(self.max_z)
        self.opt_params = {}
        self.get_opt_params(self.opt_params, opt_params)
        self.green = GreensFunction( newton_steps = newton_steps, grad = False)

    def change_seed(self, seed : int):
        self.seed = dr.opaque(mi.UInt64, seed, shape = (1))

    def get_opt_params(self, param_dict: dict, opt_params: list):
        self.input.get_opt_params(param_dict, opt_params)

    def update(self, opt):
        self.input.update(opt)
        self.σ = mi.Float(self.input.σ.get_value(mi.Point3f(0)) / self.input.α.get_value(mi.Point3f(0)))
        self.input.shape.update_shape(opt)
    
    def zero_grad(self):
        self.input.zero_grad()
        
    @dr.syntax(print_code = False)
    def solve(self, points_in = None, L_in : ArrayXf = None, initial_w : mi.Float = mi.Float(1), 
              mode : dr.ADMode = dr.ADMode.Primal, dL = ArrayXf(0), 
              derivative_dir : mi.Vector3f = None, conf_numbers : list[mi.UInt32] = [mi.UInt32(0)], all_inside = False, 
              max_step = dr.inf, normal_derivative_dist : float = None) -> list[mi.Float, Particle]:
        
        num_conf = len(conf_numbers)
        L_res = dr.zeros(ArrayXf, (num_conf, dr.width(points_in))) if (mode == dr.ADMode.Primal) else L_in

        if (L_in is None) and (mode is not dr.ADMode.Primal):
            raise Exception("The primal solution needs to be specified in the gradient computation!")

        if mode == dr.ADMode.Forward:
            dL = ArrayXf(0)
        
        active = mi.Bool(True)
        if dr.hint(self.input.shape.single_closed and not all_inside, mode = "scalar"):
            active, L_res = self.input.shape.inside_closed_surface(points_in, L_res, conf_numbers)
        
        particle = Particle(mi.Point3f(points_in), mi.Float(initial_w), mi.PCG32(), dr.arange(mi.UInt32, dr.width(points_in)), mi.UInt32(0))
        
        initstate, initseq = tea(mi.UInt64(particle.path_index), mi.UInt64(self.seed))
        particle.sampler.seed(initstate=mi.UInt64(initstate), initseq=mi.UInt64(initseq))

        with dr.suspend_grad():
            if dr.hint(derivative_dir is not None, mode = "scalar"):
                particle = self.take_derivative_step(derivative_dir, L_res, particle, mode, dL, active)
            while active:
                particle = self.take_step(L_res, particle, mode, dL, active, conf_numbers) 
                active &= particle.path_length < max_step
        return L_res, particle
    
    @dr.syntax(print_code = False)
    def take_step(self, L : ArrayXf, p : Particle, mode : dr.ADMode, dL : mi.Float, active : mi.Bool, conf_numbers : list[mi.UInt32] = [mi.UInt32(0)]):
        primal = (mode == dr.ADMode.Primal)
        bi = self.input.shape.boundary_interaction(p.points, conf_numbers = conf_numbers)
        z = bi.r * dr.sqrt(self.σ)
        if z > self.max_z:
            bi.r *= self.max_z / z
            z = self.max_z
        
        #self.green.initialize(z)
        dirichlet_ending = (active & bi.is_e) 
        
        # Add the dirichlet boundary contribution in epsilon-shell!
        added_near = dr.select(dirichlet_ending, p.w * bi.dirichlet, 0)
        
        # Add the result
        L += added_near if dr.hint(primal, mode = 'scalar') else -added_near
        
        # Remove the channels in which the walk is finished. 
        active &= ~dirichlet_ending

        #p.thrown |= bi.is_far
        #active &= ~bi.is_far
    
        # Volume Contribution
        # Source Sampling (self.σ is detached! It is used for pdf calculations.)
        
        normG = mi.Float(0)
        if dr.hint(not self.input.f.is_zero, mode = 'scalar'):
            r, normG = self.green.sample(p.sampler.next_float32(), bi.r, self.σ)
            dir_vol = mi.warp.square_to_uniform_sphere(mi.Point2f(p.sampler.next_float32(), p.sampler.next_float32()))
            points_vol = p.points + mi.Point3f(r * dir_vol) 

            with dr.resume_grad(when=not primal):
                α_vol = self.input.α.get_value(points_vol)
                f_vol = self.input.f.get_value(points_vol) / α_vol
                f_cont = dr.select(active, p.w * f_vol * normG, 0)
                #if dr.isnan(f_cont):
                #    f_cont = mi.Float(0)
                
                if dr.hint(mode == dr.ADMode.Backward, mode = 'scalar'):
                    dr.backward(dr.sum(f_cont * dL))
                elif dr.hint(mode == dr.ADMode.Forward, mode = 'scalar'):
                    dL += dr.forward_to(dr.sum(f_cont))

            L += f_cont if primal else -f_cont
        else:
            normG = self.green.eval_norm(bi.r, self.σ)

        # Boundary Sampling
        p.points +=  mi.Point3f(mi.warp.square_to_uniform_sphere(mi.Point2f(p.sampler.next_float32(), p.sampler.next_float32()))) * bi.r
        
        # Poisson Kernel computation
        P = (1 - normG * self.σ) 
        p.w *= P   
        p.path_length += 1

        # Boundary and Volume Contribution
        return p