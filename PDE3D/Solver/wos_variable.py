import numpy as np
from .data_holder import DataHolder
from PDE3D.Coefficient import *
from PDE3D.Sampling import *
from PDE3D.BoundaryShape import *
from PDE3D import Array4u64
from enum import IntEnum

class Split(IntEnum):
    Naive = 0,
    Normal = 1,
    Agressive = 2

class Particle:
    DRJIT_STRUCT = {
        'points' : mi.Point3f,
        'w': mi.Float,
        'w_split' : mi.Float,
        'sampler' : mi.PCG32,
        'path_index' : mi.UInt32,
        'path_length' : mi.UInt32,
        'traverse_h' : Array4u64,
        'thrown' : mi.Bool
    }
    def __init__(self, points=None, w=None, w_split = None, 
                 sampler = None, path_index = None, path_length = None, 
                 traverse_h = None):
        self.points = points
        self.w = w
        self.w_split = w_split
        self.sampler = sampler
        self.path_index = path_index
        self.path_length = path_length
        self.traverse_h = traverse_h
        self.thrown = mi.Bool(False)

class WosVariable(object):
    def __init__(self, input : DataHolder, seed : int = 37, weight_window = [0.5, 2],  max_z : float = 4, 
                 newton_steps : int = 5, use_accelaration : bool = True,
                 opt_params : list[str] = []):
        self.input = input
        self.seed = mi.UInt64(seed)
        dr.make_opaque(self.seed)
        self.input = input
        self.w_window = weight_window
        self.use_accel = use_accelaration
        self.max_z = mi.Float(max_z)
        self.input.max_z = self.max_z
        self.green = GreensFunction( newton_steps = newton_steps, grad = False)

        self.opt_params = {}
        self.get_opt_params(self.opt_params, opt_params)

    def change_seed(self, seed : int):
        self.seed = dr.opaque(mi.UInt64, seed, shape = (1))


    def get_opt_params(self, param_dict: dict, opt_params: list):
        self.input.get_opt_params(param_dict, opt_params)
        
    def update(self, opt):
        self.input.update(opt)
    
    def zero_grad(self):
        self.input.zero_grad()

    def get_opt_params(self, param_dict: dict, opt_params: list):
        self.input.get_opt_params(param_dict, opt_params)
    
    
    def σ_(self, σ, α, grad_α, laplacian_α):   # Equation 21 (2nd paper)
        return σ / α  + 1/2 * (laplacian_α / α - dr.squared_norm(grad_α)/(2 * (α ** 2)))
        
        
    @dr.syntax(print_code = False)
    def solve(self, points_in = None, split : Split = Split.Normal, derivative_dir : mi.Point3f = None, initial_w = mi.Float(1),
                   conf_numbers : list[mi.UInt32] = [mi.UInt32(0)], max_length : mi.UInt32 = None, tput_kill : mi.Float = mi.Float(0.8), all_inside = False, 
                   fd_forward = False, max_depth_split = 100, verbose : bool = True):
        size = dr.width(points_in)

        # The channel size of the rendering.
        num_conf = len(conf_numbers)

        #L_res = dr.zeros(Float, size)
        L_res = dr.zeros(ArrayXf, (num_conf, size))

        active = mi.Bool(True)
        if dr.hint(self.input.shape.single_closed and (not all_inside), mode = 'scalar'):
            active, L_res = self.input.shape.inside_closed_surface(points_in, L_res, conf_numbers)

        seq = dr.arange(mi.UInt64, size)
        initstate, initseq = tea(mi.UInt64(seq), mi.UInt64(self.seed))
        pcg = mi.PCG32()
        pcg.seed(initstate, initseq)

        particle = Particle(points = mi.Point3f(points_in), w = mi.Float(initial_w), w_split = mi.Float(1.0), 
                            sampler = mi.PCG32(pcg), path_index = dr.arange(mi.UInt32, size),
                            path_length = mi.UInt32(0), traverse_h = Array4u64(1,0,0,0))
        
        
        with dr.suspend_grad():
            # If we apply no path splitting.    
            if dr.hint(split == Split.Naive, mode = 'scalar'):
                # Primal phase.
                # We take a derivative step in the beginning if the direction is specified.
                #if dr.hint(derivative_dir is not None, mode = 'scalar'):
                #    particle = self.take_derivative_step(derivative_dir, L_res, particle, dr.ADMode.Primal, ArrayXf(0), active, active_conf = active_conf)
                # Take other steps.
                while active:
                    particle = self.take_step(L_res, particle, dr.ADMode.Primal, split, ArrayXf(0), active,
                                              conf_numbers, max_length, tput_kill, fd_forward)
                    # Russian roulette
                    if (particle.w_split < self.w_window[0]) & active:
                        if particle.sampler.next_float32() >= particle.w:
                            active = mi.Bool(False)
                        else:
                            particle.w = mi.Float(1)
                return L_res, particle
            
            # Otherwise do the path splitting scheme.
            iter_num = 0
            while (size > 0) and (iter_num < (max_depth_split + 1)):
                queue_index = mi.UInt32(0)

                is_split = iter_num < max_depth_split 
                if dr.hint(is_split, mode = 'scalar'):
                    # Preallocate memory for the queue. The necessary amount of memory is
                    # task-dependent (how many splits there are)
                    queue_size = dr.maximum(50, int(2 * size))
                    queue_size_opaque = dr.opaque(mi.UInt32, queue_size)
                    queue = dr.empty(dtype=Particle, shape=queue_size)

                # Get the primal result of each iteration in the gradient computation for prb.
                L_iter = dr.zeros(ArrayXf, shape = (num_conf, size))

                # We again first take the derivative direction if it is specified.
                #if dr.hint((derivative_dir is not None) & (iter_num == 0), mode = 'scalar'):
                #    particle = self.take_derivative_step(derivative_dir, L_iter, particle, dr.ADMode.Primal, mi.Float(0), active)
                
                while active:
                    # This is the main part of the algorithm (WoS).
                    particle = self.take_step(L_iter, particle, dr.ADMode.Primal, split, mi.Float(0), active, 
                                              conf_numbers, max_length, tput_kill, fd_forward = fd_forward)

                    # Russian roulette
                    if (particle.w_split < self.w_window[0]) & active:
                        if particle.sampler.next_float32() >= particle.w_split:
                            active = mi.Bool(False)
                        else:
                            particle.w /= particle.w_split
                            particle.w_split = mi.Float(1)
                    
                    
                    # Splitting begins. #################################################
                    if (particle.w_split >= self.w_window[1]) & active:
                        particle, new_particle = split_particle(particle)

                        if dr.hint(is_split, mode = 'scalar'):
                            slot = dr.scatter_inc(queue_index, index=0)
                            
                            # Be careful not to write beyond the end of the queue
                            valid =  (slot < queue_size_opaque)

                            # Write 'new_state' into the reserved slot
                            dr.scatter(target=queue, value=new_particle, index=slot, active=valid)
                        
                dr.scatter_add(L_res, L_iter, particle.path_index)
                next_size = queue_index[0]
                if verbose:
                    print('%u : %u -> %u' % (iter_num, size, next_size))
                iter_num += 1 

                if dr.hint(is_split, mode = "scalar"):
                    if next_size > queue_size:
                        print('Warning: Preallocated queue was too small: tried to store '
                            f'{next_size} elements in a queue of size {queue_size}')
                        size = queue_size
                
                if dr.hint(iter_num == max_depth_split, mode = "scalar"):
                    print(f'Warning : The split tree depth exceeds the specified value {max_depth_split}. '
                          f'The rest of the particles ({size}, {size / dr.width(points_in) * 100 :.1f} %) will be'
                          'simulated without splitting.')
                
                size = next_size

                # Generate the varibles for the next step.
                if size > 0:
                    # Get the values from the queue for the next iter.
                    particle = dr.reshape(type(particle), value=queue, shape=next_size, shrink=True)
                    # Initially, all particles are active in the next iter.
                    active = dr.full(mi.Bool, True, size)

                    #active_conf = dr.gather(ArrayXb, active_conf_begin, particle.path_index)
        return L_res, particle
        
    
    @dr.syntax(print_code = False)
    def solve_grad(self, points_in : mi.Point3f = None, split : Split = Split.Normal, 
                   mode : dr.ADMode = dr.ADMode.Backward, dL : ArrayXf = ArrayXf(0), 
                   derivative_dir : mi.Vector3f = None, conf_numbers : list[mi.UInt32] = [mi.UInt32(0)], 
                   max_length : mi.UInt32 = None, tput_kill : mi.Float = mi.Float(0.8), all_inside = False, fd_forward = False, 
                   max_depth_split = 100, verbose = False):
        
        size = dr.width(points_in)
        if conf_numbers is not None:
            num_conf = len(conf_numbers)
        else:
            num_conf = 1
        #L_res = dr.zeros(Float, size)
        L_res = dr.zeros(ArrayXf, (num_conf, size))
        
        # Loss grad value splatted to the paths.
        dL_begin = ArrayXf(dL)

        if mode == dr.ADMode.Forward:
            dL = ArrayXf(0) 
        
        active = mi.Bool(True)

        if dr.hint(self.input.shape.single_closed and (not all_inside), mode = 'scalar'):
            active, L_res = self.input.shape.inside_closed_surface(points_in, L_res, conf_numbers)

        seq = dr.arange(mi.UInt64, size)
        initstate, initseq = tea(mi.UInt64(seq), mi.UInt64(self.seed))
        pcg = mi.PCG32()
        pcg.seed(initstate, initseq)

        particle = Particle(points = mi.Point3f(points_in), w = mi.Float(1.0), w_split = mi.Float(1.0), 
                            sampler = mi.PCG32(pcg), path_index = dr.arange(mi.UInt32, size),
                            path_length = mi.UInt32(0), traverse_h = Array4u64(1,0,0,0))
        
        particle_prb = Particle(points = mi.Point3f(points_in), w = mi.Float(1.0), w_split = mi.Float(1.0), 
                                sampler = mi.PCG32(pcg), path_index = dr.arange(mi.UInt32, size),
                                path_length = mi.UInt32(0), traverse_h = Array4u64(1,0,0,0))
        active_prb = mi.Bool(active)

        with dr.suspend_grad():
            # If we apply no path splitting.    
            if dr.hint(split == Split.Naive, mode = 'scalar'):
                # Primal phase.
                # We take a derivative step in the beginning if the direction is specified.
                #if dr.hint(derivative_dir is not None , mode = 'scalar'):
                #    particle = self.take_derivative_step(derivative_dir, L_res, particle, dr.ADMode.Primal, mi.Float(0), active)
                
                # Take other steps.
                while active:
                    particle = self.take_step(L_res, particle, dr.ADMode.Primal, split, mi.Float(0), active, 
                                              conf_numbers, max_length, tput_kill, fd_forward)
                    # Russian roulette
                    if active & (particle.w_split < self.w_window[0]):
                        if particle.sampler.next_float32() >= particle.w:
                            active = mi.Bool(False)
                        else:
                            particle.w = mi.Float(1)
                
                # Replay phase.
                L_replay = ArrayXf(L_res)
                # We do the same exact thing with different compuation mode. 
                if dr.hint(derivative_dir is not None, mode = 'scalar'):
                    particle_prb = self.take_derivative_step(derivative_dir, L_replay, particle_prb, mode, dL, active_prb)
                # Take other steps.
                while active_prb:
                    particle_prb = self.take_step(L_replay, particle_prb, mode, split, dL, active_prb, 
                                                  conf_numbers, max_length, tput_kill, fd_forward)
                    # Russian roulette
                    if active_prb & (particle_prb.w_split < self.w_window[0]):
                        if particle_prb.sampler.next_float32() >= particle_prb.w:
                            active_prb = mi.Bool(False)
                        else:
                            particle_prb.w = mi.Float(1)
                return L_res, particle
            
            # Otherwise do the path splitting scheme.
            iter_num = 0
            traverse_index = dr.zeros(Array4u64, size) # We start with the traverse index of the last splitted particle.
            traverse_index_prb = dr.zeros(Array4u64, size)
            traverse_index[0] = mi.UInt64(1)
            traverse_index_prb[0] = mi.UInt64(1) 

            while (size > 0) & (iter_num < (max_depth_split + 1)):
                queue_index = mi.UInt32(0)
                is_split = iter_num < max_depth_split 

                if dr.hint(is_split, mode = 'scalar'):
                    # Preallocate memory for the queue. The necessary amount of memory is
                    # task-dependent (how many splits there are)
                    queue_size = dr.maximum(50, int(2 * size))
                    queue_size_opaque = dr.opaque(mi.UInt32, queue_size)
                    queue = dr.empty(dtype=Particle, shape=queue_size)

                # Get the primal result of each iteration in the gradient computation for prb.
                L_iter = dr.zeros(ArrayXf, shape = (num_conf, size))
                # We again first take the derivative step if it is specified.
                #if dr.hint((derivative_dir is not None) & (iter_num == 0), mode = 'scalar'):
                #    first_traverse = is_one(traverse_index)
                #    particle = self.take_derivative_step(derivative_dir, L_iter, particle, dr.ADMode.Primal, ArrayXf(0), 
                #                                        active, illumination_mask= first_traverse)
                
                while active:
                    # This is the main part of the algorithm (WoS).
                    first_traverse = is_one(traverse_index)
                    particle = self.take_step(L_iter, particle, dr.ADMode.Primal, split, ArrayXf(0), active,
                                              conf_numbers, max_length, tput_kill, fd_forward = fd_forward,
                                              illumination_mask= first_traverse)
                    # Russian roulette
                    if active & (particle.w_split < self.w_window[0]):
                        if particle.sampler.next_float32() >= particle.w_split:
                            active = mi.Bool(False)
                        else:
                            particle.w /= particle.w_split
                            particle.w_split = mi.Float(1)
                    
                    
                    # Splitting begins. #################################################
                    if ((particle.w_split >= self.w_window[1]) & active):
                        particle, new_particle = split_particle(particle)
                        
                        if dr.hint(is_split, mode = 'scalar'):
                            slot = dr.scatter_inc(queue_index, index=0, active = first_traverse)
                        
                            # Be careful not to write beyond the end of the queue
                            valid = first_traverse & (slot < queue_size_opaque)

                            # Write 'new_state' into the reserved slot
                            dr.scatter(target=queue, value=new_particle, index=slot, active=valid)

                        if ~first_traverse:
                            msb2, traverse_index = MSB2(traverse_index)
                            if msb2 == 1:
                                particle = new_particle

                if dr.hint(mode != dr.ADMode.Primal, mode = 'scalar'):
                    # Start the replay phase.
                    L_replay = ArrayXf(L_iter)
                    # We again first take the derivative step if the direction is specified.
                    #if dr.hint((derivative_dir is not None) & (iter_num == 0), mode = 'scalar'):
                        #first_traverse_prb = is_one(traverse_index_prb)
                        #particle_prb = self.take_derivative_step(derivative_dir, L_replay, particle_prb, mode, dL, 
                        #                                       active_prb, illumination_mask=first_traverse_prb)
                    
                    while active_prb:
                        # This is the main part of the algorithm (WoS).
                        first_traverse_prb = is_one(traverse_index_prb)
                        particle_prb = self.take_step(L_replay, particle_prb, mode, split, dL, active_prb,
                                                      conf_numbers, max_length, tput_kill, fd_forward = fd_forward,
                                                      illumination_mask=first_traverse_prb)

                        # Russian roulette
                        if ((particle_prb.w_split < self.w_window[0]) & active_prb):
                            if particle_prb.sampler.next_float32() >= particle_prb.w_split:
                                active_prb = mi.Bool(False)
                            else:
                                particle_prb.w /= particle_prb.w_split
                                particle_prb.w_split = mi.Float(1)
                    

                        
                        if ((particle_prb.w_split >= self.w_window[1]) & active_prb):
                            # Split the particle in the same way.
                            particle_prb, new_particle_prb = split_particle(particle_prb)

                            if ~first_traverse_prb:
                                msb2_prb, traverse_index_prb = MSB2(traverse_index_prb)
                                if msb2_prb == 1:
                                    particle_prb = new_particle_prb
                        
                dr.scatter_add(L_res, L_iter, particle.path_index)
                next_size = queue_index[0]
                
            
                if verbose:
                    print('%u : %u -> %u' % (iter_num, size, next_size))
                iter_num += 1 

                if dr.hint(is_split, mode = "scalar"):
                    if next_size > queue_size:
                        print('Warning: Preallocated queue was too small: tried to store '
                            f'{next_size} elements in a queue of size {queue_size}')
                        size = queue_size
                
                if dr.hint(iter_num == max_depth_split, mode = "scalar"):
                    print(f'Warning : The split tree depth exceeds the specified value f{max_depth_split}. '
                          f'The rest of the particles ({size}, {size / dr.width(points_in) * 100 :.1f} %) will be'
                          'simulated without splitting.')
                
                size = next_size
                # Generate the varibles for the next step.
                if size > 0: 
                    # Get the values from the queue. 
                    particle_f = dr.reshape(type(particle), value=queue, shape=size, shrink=True) 
                    
                    # Initially, all particles are active in the next iter.
                    active = dr.full(mi.Bool, True, size)
                
                    # Set the traverse index to be the last traverse history.
                    traverse_index = Array4u64(particle_f.traverse_h)
                
                    # Get the initial points for the next run.
                    next_points = dr.gather(mi.Point3f, points_in, particle_f.path_index)

                    # Get the active configurations.
                    #active_conf = dr.gather(ArrayXb, active_conf_begin, particle_f.path_index)

                    # Get the loss grad value splatted to the paths.
                    if mode == dr.ADMode.Backward:
                        dL = dr.gather(ArrayXf, dL_begin, particle_f.path_index)

                    initseq, initstate = tea(mi.UInt64(particle_f.path_index), mi.UInt64(self.seed))
                    pcg_iter = mi.PCG32()
                    pcg_iter.seed(initseq, initstate)

                    particle = Particle(points=mi.Point3f(next_points),
                                        w = mi.Float(1),
                                        w_split = mi.Float(1),
                                        sampler = mi.PCG32(pcg_iter),
                                        path_index = mi.UInt32(particle_f.path_index),
                                        path_length = mi.UInt32(0),
                                        traverse_h= Array4u64(1,0,0,0))
                
                    # Generate the same for the replay stage.
                    active_prb = mi.Bool(active)
                    traverse_index_prb = Array4u64(traverse_index)

                    particle_prb = Particle(points=mi.Point3f(next_points),
                                            w = mi.Float(1.),
                                            w_split = mi.Float(1.),
                                            sampler = mi.PCG32(pcg_iter),
                                            path_index = mi.UInt32(particle_f.path_index),
                                            path_length = mi.UInt32(0),
                                            traverse_h= Array4u64(1,0,0,0))
        return L_res, particle
    
    
    @dr.syntax(print_code = False)
    def take_step(self, L : ArrayXf, p : Particle, mode : dr.ADMode, split : Split, dL : ArrayXf, active : mi.Bool, 
                  conf_numbers : list[mi.UInt32] = None, max_length : mi.UInt32 = None, tput_kill : mi.Float = mi.Float(0.8), 
                  fd_forward : bool = False, illumination_mask : mi.Bool = mi.Bool(True)):
        
        if conf_numbers is not None:
            num_conf = len(conf_numbers)
        else:
            num_conf = 1

        primal = (mode == dr.ADMode.Primal)
        bi = self.input.shape.boundary_interaction(p.points, conf_numbers = conf_numbers)
        
        #if bi.is_far:
        #    p.thrown = mi.Bool(True)
        #    active &= mi.Bool(False)
        
        # Decrease radius if it is big. 
        σ_bar = self.input.σ_bar
        z = mi.Float(0)
        #if self.use_accel:
        #    bi.r, σ_bar, z = self.input.get_Rσz(p.points, bi.r)
        #else:
        z = bi.r * dr.sqrt(σ_bar)
        if z > self.max_z:
            bi.r *= self.max_z / z
            z = self.max_z
        
        #self.green.initialize(z)
        dirichlet_ending = (active & bi.is_e) 
        
        # Add the dirichlet boundary contribution in epsilon-shell!
        added_near = dr.select(dirichlet_ending, p.w * bi.dirichlet, 0)

        L += added_near if primal else -added_near

        with dr.resume_grad(when=not primal):
            α = self.input.α.get_value(p.points)
            
        # Remove the channels in which the walk is finished. 
        active &= ~dirichlet_ending

        f_cont = mi.Float(0)        
        # Add the source contribution.
        if dr.hint(not self.input.f.is_zero, mode = 'scalar'):
            sample_source = mi.Point3f(p.sampler.next_float32(), p.sampler.next_float32(), p.sampler.next_float32())
            #if illumination_mask:
            r_vol, normG = self.green.sample(sample_source[0], bi.r, σ_bar)
            #dir_vol, _ = sample_uniform_direction(sample_source[1])
            dir_vol = mi.warp.square_to_uniform_sphere(mi.Point2f(sample_source[1], sample_source[2]))
            points_vol = p.points + mi.Point3f(r_vol * dir_vol) 
            with dr.resume_grad(when=not primal):
                α_vol = self.input.α.get_value(points_vol)
                f_vol = self.input.f.get_value(points_vol) 
                f_cont = p.w * f_vol * normG / dr.sqrt(α * α_vol)
                if dr.isnan(f_cont) | ~illumination_mask:
                    f_cont = mi.Float(0)

        #f_cont = dr.select(active_conf, f_cont, 0)
        L += f_cont if primal else -f_cont

        # Now select between boundary or volume sampling (2nd paper, eqn 28)
        normG = self.green.eval_norm(bi.r, σ_bar)
        prob_vol =  σ_bar * normG
        sample_rec = mi.Point3f(p.sampler.next_float32(), p.sampler.next_float32(), p.sampler.next_float32())
        sample_vol = active & (sample_rec[0] < prob_vol)
        sample_rec[0] = dr.select(sample_vol, sample_rec[0] / prob_vol, (sample_rec[0] - prob_vol) / (1-prob_vol))

        r_next = mi.Float(bi.r)
        if sample_vol:
            r_next = self.green.sample(sample_rec[0], bi.r, σ_bar)[0]
            
        dir_next = mi.warp.square_to_uniform_sphere(mi.Point2f(sample_rec[1], sample_rec[2]))
        points_next = p.points + mi.Point3f(r_next * dir_next)
        

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

def is_one(index : Array4u64) -> mi.Bool:
    return (index[0] == mi.UInt64(1)) & (index[1] == mi.UInt64(0)) & (index[2] == mi.UInt64(0)) & (index[3] == mi.UInt64(0))

@dr.syntax
def shift_left(index : Array4u64):
    index_new = Array4u64(index)
    for i in range(3, 0, -1):
        index_new[i]  = index[i] << 1
        if dr.lzcnt(index[i-1]) == 0:
            index_new[i] += 1
    index_new[0] = index[0] << 1
    return index_new

@dr.syntax
def MSB2(index : Array4u64):
    "Find the 2nd MSB and throw it out"
    index_residual = mi.UInt32(0)
    index_full = mi.UInt32(0)
    for i in range(3, -1, -1):
        if index_residual == 0:
            index_residual += (64 - mi.UInt32(dr.lzcnt(index[i])))
            index_full = mi.UInt32(i)

    if (index_residual == 0) & (index_full > 0):
        index_full -= 1
        index_residual = mi.UInt32(64)
    
    msb2 = mi.UInt64(0)
    thrown = Array4u64(index)
    for i in range(4):
        if index_full == i:
            if index_residual > 1:
                shift_num = (index_residual - 2)
                msb2 = (index[i] >> shift_num) & 1
                msb2e = mi.UInt64(1) << shift_num
                thrown[i]  = index[i] % msb2e + msb2e
            elif index_residual == 1:
                if i > 0:
                    msb2 = (index[i-1] >> 63) & 1
                    thrown[i] = mi.UInt64(0)
                    msb2e  = mi.UInt64(1)<<63
                    thrown[i-1] = index[i-1] % msb2e + msb2e
    return msb2, thrown


def split_particle(particle : Particle):
    new_particle_state = particle.sampler.next_uint64()
    shifted = shift_left(particle.traverse_h)
    new_particle = Particle(points = particle.points,
                            w=particle.w/2,
                            w_split = particle.w_split/2,
                            sampler = mi.PCG32(particle.sampler),
                            path_index = particle.path_index,
                            path_length = particle.path_length,
                            traverse_h = Array4u64(shifted))
    new_particle.traverse_h[0] += mi.UInt64(1)

    new_particle.sampler.state = new_particle_state
    particle.w /= 2
    particle.w_split /= 2
    particle.traverse_h = Array4u64(shifted) 
    return particle, new_particle

