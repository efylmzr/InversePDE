import numpy as np
from ..data_holder import DataHolder
from ...Coefficient import *
from ...Sampling import *
from ...BoundaryShape.interaction import BoundaryInfo
from PDE2D.BoundaryShape import *
from .wos_variable import *

class WosVariableRejection(WosVariable):
    def __init__(self, input : DataHolder, seed : int = 37, weight_window = [0.5, 2],  max_z : float = 4, 
                 green_sampling : GreenSampling = 0, newton_steps : int = 5, use_accelaration : bool = True,
                 opt_params : list[str] = []):
        super().__init__(input, seed,  weight_window, max_z, 
                         green_sampling, newton_steps, use_accelaration, opt_params)
        
    @dr.syntax(print_code = False)
    def take_step(self, L : ArrayXf, p : Particle, mode : dr.ADMode, split : Split, dL : ArrayXf, active : Bool, active_conf : ArrayXb = ArrayXb(True),
                  conf_numbers : list[UInt32] = None, max_length : UInt32 = None, tput_kill : Float = Float(0.8), 
                  fd_forward : bool = False, illumination_mask : Bool = Bool(True)):
        if conf_numbers is not None:
            num_conf = len(conf_numbers)
        else:
            num_conf = 1
        
        primal = (mode == dr.ADMode.Primal)
        bi = self.input.shape.boundary_interaction(p.points, star_generation = False, conf_numbers = conf_numbers)
        
        if bi.is_far:
            p.thrown = Bool(True)
            active &= Bool(False)
        
        # Decrease radius if it is big. 
        σ_bar = self.input.σ_bar
        z = Float(0)
        if self.use_accel:
            bi.r, σ_bar, z = self.input.get_Rσz(p.points, bi.r)
        else:
            z = bi.r * dr.sqrt(σ_bar)
            if z > self.max_z:
                bi.r *= self.max_z / z
                z = self.max_z
        
        self.green.initialize(z)
        dirichlet_ending = (active & bi.is_e & bi.is_d) 
        
        # Add the dirichlet boundary contribution in epsilon-shell!
        added_near = dr.select(dirichlet_ending & active_conf, p.w * bi.dval, 0)

        L += added_near if primal else -added_near

        with dr.resume_grad(when=not primal):
            α = self.input.α.get_value(p.points)
            
        # Remove the channels in which the walk is finished. 
        active &= ~dirichlet_ending

        f_cont = Float(0)        
        # Add the source contribution.
        if dr.hint(not self.input.f.is_zero, mode = 'scalar'):
            sample_source = Point2f(p.sampler.next_float32(), p.sampler.next_float32())
            #if illumination_mask:
            #r_vol, normG = self.green.sample(sample_source[0], bi.r, σ_bar)
            r_vol, normG = self.sampleGreenRejection(p, bi.r, σ_bar)
            dir_vol, _ = sample_uniform_direction(sample_source[1])
            points_vol = p.points + r_vol * dir_vol 
            with dr.resume_grad(when=not primal):
                α_vol = self.input.α.get_value(points_vol)
                f_vol = self.input.f.get_value(points_vol) 
                f_cont = p.w * f_vol * normG / dr.sqrt(α * α_vol)
                if dr.isnan(f_cont) | ~illumination_mask:
                    f_cont = Float(0)

        f_cont = dr.select(active_conf, f_cont, 0)
        L += f_cont if primal else -f_cont

        # Now select between boundary or volume sampling (2nd paper, eqn 28)
        normG = self.green.eval_norm(bi.r, σ_bar)
        prob_vol =  σ_bar * normG
        sample_rec = Point2f(p.sampler.next_float32(), p.sampler.next_float32())
        sample_vol = active & (sample_rec[0] < prob_vol)
        sample_rec[0] = dr.select(sample_vol, sample_rec[0] / prob_vol, (sample_rec[0] - prob_vol) / (1-prob_vol))

        r_next = Float(bi.r)
        if sample_vol:
            #r_next = self.green.sample(sample_rec[0], bi.r, σ_bar)[0]
            r_next = self.sampleGreenRejection(p, bi.r, σ_bar)[0]
            
        dir_next, _ = sample_uniform_direction(sample_rec[1])
        points_next = p.points + r_next * dir_next
        

        with dr.resume_grad(when=not primal):
            α_next = self.input.α.get_value(points_next)
            grad_α_next, laplacian_α_next = self.input.α.get_grad_laplacian(points_next)
            σ_next = self.input.σ.get_value(points_next)
            σ_new = self.σ_(σ_next, α_next, grad_α_next, laplacian_α_next)
            w_ = dr.select(active, dr.sqrt(α_next / α), 1.0)
            w_s = dr.select(sample_vol, (1.0 - σ_new / σ_bar), 1.0)
            w_update = w_ * w_s
            # Boundary and Volume Contribution
            prb_cont = dr.select(dr.isfinite(w_update), L * w_update / dr.detach(w_update), 0.0)

            if dr.hint(mode == dr.ADMode.Backward, mode = 'scalar'):
                dr.backward(dr.sum((prb_cont + f_cont) * dL)) 
            elif dr.hint(mode == dr.ADMode.Forward, mode = 'scalar'):
                dL += dr.forward_to(dr.sum(prb_cont + f_cont))
            
        p.w *= w_update
        # If we are not doing fd computation, then just use the original coefficient.
        if dr.hint((not fd_forward), mode = 'scalar'):
            if dr.hint(split == Split.Agressive, mode = 'scalar'):
                p.w_split *= w_update
            elif dr.hint(split == Split.Normal, mode = 'scalar'):
                p.w_split *= w_s
        else:
            α = self.input.α_split.get_value(p.points) # We did not get this before if f is zero!
            α_next = self.input.α_split.get_value(points_next)
            grad_α_next, laplacian_α_next = self.input.α_split.get_grad_laplacian(points_next)
            σ_next = self.input.σ_split.get_value(points_next)
            σ_new = self.σ_(σ_next, α_next, grad_α_next, laplacian_α_next)
            w_ = dr.select(active, dr.sqrt(α_next / α), 1.0)
            w_s = dr.select(sample_vol, (1.0 - σ_new / σ_bar), 1.0)
            if dr.hint(split == Split.Agressive, mode = 'scalar'):
                p.w_split *= (w_ * w_s)
            elif dr.hint(split == Split.Normal, mode = 'scalar'):
                p.w_split *= w_s

        if dr.hint(max_length is not None, mode = 'scalar'):
            if p.path_length > max_length:
                p.w *= tput_kill
                p.w_split *= tput_kill

        active &= dr.isfinite(w_update)
        p.points = points_next
        p.path_length += 1
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