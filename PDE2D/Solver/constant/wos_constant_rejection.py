import numpy as np
import sys
import mitsuba as mi 
from mitsuba import Bool, Float, Point2f, UInt32
from PDE2D import  GreenSampling, ArrayXb
from ..data_holder import DataHolder
from ...Coefficient import *
from ...Sampling import *
from PDE2D.BoundaryShape import *
from .wos_constant import WosConstant
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


class WosConstantRejection(WosConstant):
    def __init__(self, input : DataHolder, seed : int = 37, 
                 max_z : float = 4, green_sampling : GreenSampling = 0, 
                 newton_steps : int = 5, opt_params : list[str] = []) -> None:
        super().__init__(input, seed, max_z, green_sampling, newton_steps, opt_params)
    
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
            #r, normG = self.green.sample(p.sampler.next_float32(), bi.r, self.σ)
            r, normG = self.sampleGreenRejection(p, bi.r, self.σ)
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
    def sampleGreenRejection(self, p : Particle, R : Float, σ : Float):
        # We apply rejection sampling based on WosVariable paper.
        if R <= σ:
            upper_bound = dr.maximum(2.2 * dr.maximum(dr.rcp(R), dr.rcp(σ)), 0.6 * dr.maximum(dr.sqrt(R), dr.sqrt(σ)))
        else:
            upper_bound = dr.maximum(2.2 * dr.minimum(dr.rcp(R), dr.rcp(σ)), 0.6 * dr.minimum(dr.sqrt(R), dr.sqrt(σ)))

        sample1 = p.sampler.next_float32() * R
        sample2 = p.sampler.next_float32() 
        pdf = self.green.eval_pdf_only(sample1, R, σ)
        while(sample2 * upper_bound > pdf):
            sample1 = p.sampler.next_float32() * R
            sample2 = p.sampler.next_float32()
            pdf = self.green.eval_pdf_only(sample1, R, σ)
        return sample1, self.green.eval_norm(R, σ)
            