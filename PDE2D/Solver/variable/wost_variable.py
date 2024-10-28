import numpy as np
import sys
from ..data_holder import DataHolder
from ...Coefficient import *
from ...Sampling import *
from .wos_variable import *

class WostVariable(WosVariable):
    def __init__(self, input : DataHolder, seed : int = 37, weight_window = [0.5, 2], 
                 max_z : float = 4, green_sampling : GreenSampling = 0,
                 newton_steps : int = 5, use_accelaration : Bool = True, opt_params : list[str] = []):
        super().__init__(input, seed,  weight_window, max_z, 
                         green_sampling, newton_steps, use_accelaration, opt_params)
    
    @dr.syntax
    def take_step(self, L : ArrayXf, p : Particle, mode : dr.ADMode, split : Split, dL : ArrayXf, active : Bool, active_conf : ArrayXb = ArrayXb(True),
                  conf_numbers : list[UInt32] = None, max_length : UInt32 = None, tput_kill : Float = Float(0.8), fd_forward : bool = False, 
                  illumination_mask: Bool = Bool(True)) -> Particle:
        primal = (mode == dr.ADMode.Primal)

        if conf_numbers is not None:
            num_conf = len(conf_numbers)
        else:
            num_conf = 1
    
        # Apply boundary interaction.
        bi = self.input.shape.boundary_interaction(p.points, star_generation = False, conf_numbers = conf_numbers)

        if bi.is_far:
            p.thrown = Bool(True)
            active &= Bool(False)
        
        # Decrease radius to sample from a reasonable Green's function.
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
        # Generate stars.
        bi = self.input.shape.star_generation(bi)

        # End the paths if we are in the epsilon shell of a dirichlet boundary.
        dirichlet_ending = (active & bi.is_e & bi.is_d) 
        
        # Add the dirichlet boundary contribution in epsilon-shell!
        added_near = dr.select(dirichlet_ending & active_conf, p.w * bi.dval, 0)
        
        L += added_near if primal else -added_near
        
        # Remove the channels in which the walk is finished. 
        active &= ~dirichlet_ending

        #This is used throughout the integrator. So we compute it in the beginning.
        with dr.resume_grad(when = not primal):
            α = self.input.α.get_value(p.points)
        
        # Source term contribution.
        f_cont = Float(0)
        if dr.hint(not self.input.f.is_zero, mode = 'scalar'):
            sample_source = Point2f(p.sampler.next_float32(), p.sampler.next_float32())
            #if illumination_mask:
            r_f, normG = self.green.sample(sample_source[0], bi.r, σ_bar)
            dir_f, _ = sample_star_direction(sample_source[1], bi.is_star & bi.on_boundary, bi.bn)
            points_f = mi.Point2f(p.points + r_f * dir_f)
            # If we are on a star, The sampled point might be outside of the boundary.
            # We need to check this with a ray intersection.
            ri_f = self.input.shape.ray_intersect(bi, dir_f)
            with dr.resume_grad(when=not primal):
                α_f = self.input.α.get_value(points_f)
                f_f = self.input.f.get_value(points_f) 
                f_cont = p.w * f_f * normG / dr.sqrt(α * α_f)
                #if dr.isnan(f_cont) | (r_f > ri_f.t) | ~illumination_mask:
                #    f_cont = Float(0)
                f_cont = dr.select(active_conf, f_cont, 0)
        
        L += f_cont if primal else -f_cont

        
        # Neumann boundary contribution.
        # If we have a continous Neumann on the boundary, we need to sample it.
        n_cont_cont = dr.zeros(ArrayXf, shape = L.shape)

        if dr.hint(self.input.has_continuous_neumann, mode = 'scalar'):
            # If we have a special sampling scheme based on boundary values, then we need to get all of the values.
            if dr.hint(self.input.NEE == NEE.Special, mode = 'scalar'):
                for i in range(num_conf):
                    conf_number = None if conf_numbers is None else conf_numbers[i]
                    #==if illumination_mask:
                    sample_neumann = p.sampler.next_float32()
                    dist_n, n_val, pdf_n_r, p_n = self.input.sampleNEE_special(bi, sample_neumann, conf_number)
                    G_n_r = self.green.eval(dist_n, bi.r, σ_bar)
                    
                    if ((pdf_n_r > 0) & (dist_n < bi.r) & (dist_n > 0)):
                        n_cont_cont[i] =  -p.w * n_val * G_n_r / pdf_n_r

                    with dr.resume_grad(when = not primal):
                        α_n = self.input.α.get_value(p_n)
                        n_cont_cont[i] *= dr.sqrt(dr.rcp( α * α_n)) if dr.hint(self.input.shape.measured_current, mode = 'scalar') else dr.sqrt(α_n / α)
                    
                    if dr.isnan(n_cont_cont[i]) | ~illumination_mask:
                        n_cont_cont[i] = Float(0)
            else:
                # Here n_val is an ArrayXf.
                dist_n, n_val, pdf_n_r, p_n = self.input.sampleNEE(bi, p.sampler.next_float32(), conf_numbers)
                G_n_r = self.green.eval(dist_n, bi.r, σ_bar)
                
                n_cont_cont_ = Float(0)
                if ((pdf_n_r > 0) & (dist_n < bi.r) & (dist_n > 0)):
                    n_cont_cont_ = -p.w * G_n_r / pdf_n_r

                with dr.resume_grad(when = not primal):
                    α_n = self.input.α.get_value(p_n)
                    n_cont_cont_ *= dr.sqrt(dr.rcp( α * α_n)) if dr.hint(self.input.shape.measured_current, mode = 'scalar') else dr.sqrt(α_n / α)
                
                if dr.isnan(n_cont_cont_) | ~illumination_mask:
                    n_cont_cont_ = Float(0)
                n_cont_cont = n_val * n_cont_cont_
        
        
        # Now, we get the all necessary delta distributions on the boundary (a.k.a. point current injections).
        n_cont_delta =dr.zeros(ArrayXf, shape = L.shape)
    
        if dr.hint(self.input.has_delta, mode = 'scalar'):
            for i in range(num_conf):
                conf_number = None if conf_numbers is None else conf_numbers[i]
                #if dr.hint(illumination_mask, mode="evaluated"):
                dist_n, n_val, pdf_n_r, sampled_n = self.input.get_point_neumann(bi, conf_number)
                # We can have multiple relevant electrodes, add all the contribution.
                for d, n, pdf_r, p_n in zip(dist_n, n_val, pdf_n_r, sampled_n):
                    n_cont_delta_iter = Float(0)
                    G_n_r = self.green.eval(d, bi.r, σ_bar)
                    with dr.resume_grad(when = not primal):
                        α_n = self.input.α.get_value(p_n)
                        if (pdf_r > 0) & (d <= bi.r):
                            n_cont_delta_iter = -p.w * n * G_n_r / pdf_r
                        # Here, we need to apply a correction term if the given neumann boundary is a current value.
                        n_cont_delta_iter *= dr.sqrt(dr.rcp(α * α_n)) if dr.hint(self.input.shape.measured_current, mode = 'scalar') else dr.sqrt(α_n / α)
                        n_cont_delta[i] += n_cont_delta_iter
            
                if dr.isnan(n_cont_delta[i]) | ~illumination_mask:
                    n_cont_delta[i] = Float(0)
        
        # Compute the total neumann contribution.
        with dr.resume_grad(when = not primal):
            n_cont = n_cont_cont + n_cont_delta 
            # There is a factor of 2 for smooth neumann boundaries if we are exactly on the boundary. (Check WoSt paper.)
            if bi.on_boundary:
                n_cont *= 2
            n_cont = dr.select(active_conf, n_cont, 0)
            
        L += n_cont if primal else -n_cont
        
        # Sampling the recursive term.
        # Now select between boundary or volume sampling (2nd paper, eqn 28)
        sample_rec = Point2f(p.sampler.next_float32(), p.sampler.next_float32())
        normG = self.green.eval_norm(bi.r, σ_bar)
        prob_vol =  σ_bar * normG
        sample_vol = active & (sample_rec[0] < prob_vol)
        sample_rec[0] = dr.select(sample_vol, sample_rec[0] / prob_vol, (sample_rec[0] - prob_vol) / (1-prob_vol))
        # Sample direction
        dir_next, _, _ = bi.sample_recursive(sample_rec[1])
        # We will stamp the next sampled point in case it is sampled outside of the star.
        ri_next = self.input.shape.ray_intersect(bi, dir_next)
        # Radius sampling with the Green's function.
        r_next = Float(bi.r)
        if sample_vol:
            r_next = self.green.sample(sample_rec[0], bi.r, σ_bar)[0]
        
        # Stamping. Also we need to update the sample vol term for correct throughput update.
        on_boundary_next = (ri_next.t < r_next)
        sample_vol &= ~on_boundary_next
        if on_boundary_next:
            r_next = ri_next.t

        # Next iteration points.
        points_next = mi.Point2f(ri_next.origin + r_next * dir_next)

        with dr.resume_grad(when=not primal):
            α_next = self.input.α.get_value(points_next)
            grad_α_next, laplacian_α_next = self.input.α.get_grad_laplacian(points_next)
            σ_next = self.input.σ.get_value(points_next)
            σ_new = self.σ_(σ_next, α_next, grad_α_next, laplacian_α_next)
            w_ =  dr.sqrt(α_next / α)
            w_s = dr.select(sample_vol, (1.0 - σ_new / σ_bar), 1.0)
            w_update = w_ * w_s
            # Path replay gradient contribution.
            prb_cont = dr.select(dr.isfinite(w_update), L * w_update / dr.detach(w_update), 0.0)
            
            # Here, all the gradients from different contributions computed in single backward pass.
            grad_cont = prb_cont + f_cont + n_cont
            if dr.hint(mode == dr.ADMode.Backward, mode = 'scalar'):
                dr.backward(dr.sum(grad_cont * dL))
            elif dr.hint(mode == dr.ADMode.Forward, mode = 'scalar'):
                dL += dr.forward_to(dr.sum(grad_cont))
        
        active &= dr.isfinite(w_update)
        
        # If we are not doing fd computation, then just use the original coefficient.
        if dr.hint((not fd_forward), mode = 'scalar'):
            if dr.hint(split == Split.Agressive, mode = 'scalar'):
                p.w_split *= w_update
            elif dr.hint(split == Split.Normal, mode = 'scalar'):
                p.w_split *= w_s
        else: # Otherwise use the non-deviated coefficients for throughput update.
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

        # Update the points for the next iteration.
        p.w *= w_update
        p.points = points_next
        p.path_length += 1
        return p