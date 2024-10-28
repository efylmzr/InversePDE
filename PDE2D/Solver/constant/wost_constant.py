import numpy as np
from ..data_holder import DataHolder
from ...Coefficient import *
from ...Sampling import *
from ...BoundaryShape.interaction import BoundaryInfo
from .wos_constant import *


class WostConstant(WosConstant):
    def __init__(self, input : DataHolder, seed : int = 37, 
                 max_z : float = 4, green_sampling : GreenSampling = 0, newton_steps : int = 5, opt_params : list[str] = []) -> None:
        super().__init__(input, seed, max_z, green_sampling, newton_steps, opt_params)
    
    @dr.syntax(print_code = False)
    def take_step(self, L : ArrayXf, p : Particle, mode : dr.ADMode, dL : ArrayXf, active : Bool, 
                  active_conf : ArrayXb, normal_derivative_dist : float = None, conf_numbers : list[UInt32] = None) -> Particle:
        
        if conf_numbers is not None:
            num_conf = len(conf_numbers)
        else:
            num_conf = 1
        
        primal = (mode == dr.ADMode.Primal)
        
        # Apply boundary interaction.
        with dr.resume_grad(when = (not primal) & (normal_derivative_dist is not None)):
            bi = self.input.shape.boundary_interaction(p.points, star_generation = False, conf_numbers = conf_numbers)
        
        # Decrease radius to sample from a reasonable Green's function.
        z = bi.r * dr.sqrt(self.σ)
        if z > self.max_z:
            bi.r *= self.max_z / z
            z = self.max_z

        self.green.initialize(z)
        # Generate stars.
        bi = self.input.shape.star_generation(bi)

        # End the paths if we are in the epsilon shell of a dirichlet boundary.
        dirichlet_ending = (active & bi.is_e & bi.is_d) 

        # Final contribution
        added_near = dr.select(dirichlet_ending & active_conf, p.w * bi.dval, 0.0)
        #added_near = Float(0)
        # Accumulate the throughput to the corresponding shape. (Only done if multiple shapes is defined.)
        #dirichlet_grad = Float(0)
        if dr.hint(primal, mode = 'scalar'):
            L += added_near
        else:
            L -= added_near
            # This part is for computing the derivative for discrete EIT experiments. Only 
            # single shape optimization is supported.
            if dr.hint((normal_derivative_dist is not None), mode = 'scalar'):
                assert isinstance(self.input.shape, BoundaryWithDirichlets)
                assert len(self.input.shape.in_boundaries) == 1
                
                jacobian = self.input.shape.get_jacobian_factor(bi, normal_derivative_dist)
                with dr.resume_grad(when = not primal): # This is for optimizing the shape of the dirichlet boundaries. 
                    normalder = dr.select(dirichlet_ending & active_conf,
                                            p.w * self.input.shape.get_normal_derivative(dr.detach(bi.bpoint)),
                                            0)    
                    #dirichlet_grad = bi.d 
                    distance_correction = self.input.shape.in_boundaries[0].get_distance_correction(p.points)
                    bi2 = self.input.shape.in_boundaries[0].boundary_interaction(p.points, star_generation = False, conf_numbers = conf_numbers)
                    dirichlet_grad = dr.select(dirichlet_ending, bi2.d * jacobian * normalder / distance_correction, 0)
                    #dr.backward(dirichlet_grad * dL)

        
        # Remove the channels in which the walk is finished. 
        active &= ~dirichlet_ending
        
        active &= ~bi.is_far
        p.thrown |= bi.is_far
        
        # Source Contribution
        # Source Sampling (self.σ is detached! It is used for pdf calculations.)
        if dr.hint(not self.input.f.is_zero, mode = 'scalar'):
            r_vol, normG = self.green.sample(p.sampler.next_float32(), bi.r, self.σ)
            dir_vol, _ = sample_star_direction(p.sampler.next_float32(), bi.on_boundary & bi.is_star, bi.bn)
            points_vol = mi.Point2f(p.points + r_vol * dir_vol)
            
            # If we are on a star, The sampled point might be outside of the boundary.
            # We need to check this with a ray intersection.
            ri_f = self.input.shape.ray_intersect(bi, dir_vol)

            with dr.resume_grad(when=not primal):
                α_vol = self.input.α.get_value(points_vol)
                f_vol = self.input.f.get_value(points_vol) / α_vol
                f_cont = dr.select(active & (r_vol <= ri_f.t), p.w * f_vol * normG, 0)
                if dr.isnan(f_cont):
                    f_cont = Float(0)
                f_cont = dr.select(active_conf, f_cont, 0)
            L += f_cont if primal else -f_cont

        # Now compute the Neumann Contribution. (NEE Contribution.)
        # If we have a continous Neumann on the boundary, we need to sample it.
        n_cont_cont = dr.zeros(ArrayXf, shape = L.shape)
        
        if dr.hint(self.input.has_continuous_neumann, mode = 'scalar'):
            # If we have a special sampling scheme then we need to sample for each configuration.
            
            if dr.hint(self.input.NEE == NEE.Special, mode = 'scalar'):
                for i in range(num_conf):
                    conf_number = None if conf_numbers is None else conf_numbers[i]
                    # Here n_val is a Float.
                    dist_n, n_val, pdf_n_r, _ = self.input.sampleNEE_special(bi, p.sampler.next_float32(), conf_number)
                    G_n_r = self.green.eval(dist_n, bi.r, self.σ)
                    if ((pdf_n_r > 0) & (dist_n < bi.r) & (dist_n > 0)):
                        n_cont_cont[i] =  -p.w * n_val * G_n_r / pdf_n_r
                    if dr.isnan(n_cont_cont[i]):
                        n_cont_cont[i] = Float(0)
            else: # If not then, we only call the sample function once. The neumann values will be different.
                # Here n_val is an ArrayXf.
                dist_n, n_val, pdf_n_r, _ = self.input.sampleNEE(bi, p.sampler.next_float32(), conf_numbers)
                G_n_r = self.green.eval(dist_n, bi.r, self.σ)
                
                n_cont_cont_ = Float(0)
                if ((pdf_n_r > 0) & (dist_n < bi.r) & (dist_n > 0)):
                    n_cont_cont_ = -p.w * G_n_r / pdf_n_r
                
                if dr.isnan(n_cont_cont_):
                    n_cont_cont_ = Float(0)
                n_cont_cont = n_val * n_cont_cont_


                
        # Now, we get the all necessary delta distributions on the boundary (a.k.a. point current injections).
        n_cont_delta = dr.zeros(ArrayXf, shape = L.shape)
        if dr.hint(self.input.has_delta, mode = 'scalar'):
            for i in range(num_conf):
                conf_number = None if conf_numbers is None else conf_numbers[i]
                dist_n, n_val, pdf_n_r, _ = self.input.get_point_neumann(bi, conf_number)
                # We can have multiple relevant electrodes, add all the contribution.
                for d, n, pdf_r in zip(dist_n, n_val, pdf_n_r):
                    G_n_r = self.green.eval(d, bi.r, self.σ)
                    if pdf_r > 0:
                        n_cont_delta[i] += -p.w * n * G_n_r / pdf_r
                if dr.isnan(n_cont_delta[i]):
                    n_cont_delta[i] = Float(0)
        # Compute the total neumann contribution, we need to multiply by two if we are on the boundary. (Check WoSt paper.)
        
        n_cont = n_cont_cont + n_cont_delta 
        if bi.on_boundary:
            n_cont *= 2

        n_cont = dr.select(active_conf, n_cont, 0)

        # One last step is to correct neumann value if the given Neumann is a current value.
        if self.input.shape.measured_current:
            with dr.resume_grad(when = not primal):
                n_cont /= self.input.α.get_value(Point2f(0)) # Constant Conductance

        L += n_cont if primal else -n_cont

        # Now, we can accumulate the gradients as all necessary info is collected.
        with dr.resume_grad(when = not primal):
            if dr.hint(mode == dr.ADMode.Backward, mode = 'scalar'):
        #        dr.backward(dr.sum((f_cont + n_cont + dirichlet_grad) * dL))
                dr.backward(dirichlet_grad * dL)
            elif dr.hint(mode == dr.ADMode.Forward, mode = 'scalar'):
        #        dL += dr.forward_to(dr.sum(f_cont + n_cont + dirichlet_grad))
                dL += dr.forward_to(dr.sum(dirichlet_grad))

        # Recursive step
        # Sample direction to get the next point.
        dir_next, sphere_p, _ = bi.sample_recursive(p.sampler.next_float32())
        # Check if we hit to the boundary before sphere.
        ri = self.input.shape.ray_intersect(bi, dir_next)
        on_boundary_next = bi.is_star & (ri.t < bi.r)
        next_points = dr.select(bi.is_star & on_boundary_next, ri.intersected, sphere_p)
        distance_rec = dr.select(on_boundary_next & bi.is_star, ri.t, bi.r)

        first_mask = ~self.input.shape.inside_closed_surface_mask(next_points)
        if first_mask:
            p.bad_mask = Bool(True)
            p.on_boundary1 = Bool(bi.on_boundary)
            p.on_boundary2 = Bool(on_boundary_next)
            p.loc1 = Point2f(p.points)
            p.loc2 = Point2f(next_points)
            p.intersected = Point2f(ri.intersected)
            p.direction = Point2f(dir_next)
            p.dist = Float(ri.t)
            p.boundary_normal = Point2f(bi.bn)
             
        # Poisson Kernel computation  
        p.points = Point2f(next_points)
        p.w *= self.green.eval_poisson_kernel(distance_rec, bi.r, self.σ)
        p.path_length += 1

        return p