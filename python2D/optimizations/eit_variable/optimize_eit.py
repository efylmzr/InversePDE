
import numpy as np
import mitsuba as mi 
mi.set_variant("cuda_ad_rgb")
import drjit as dr
from PDE2D.Coefficient import *
from PDE2D.utils import *
from PDE2D.BoundaryShape import *
from PDE2D.Solver import *
from python2D.optimizations.sketch import *
from mitsuba import Float, UInt32, UInt64, Point2f
from PDE2D import Split
import time

def compute_primal(wos : WostVariable, split : Split, spe : int, delete_injection : bool, 
                   conf_numbers : list[UInt32], max_split_depth : int, compute_std: bool, 
                   verbose : bool = False, kill_step = dr.inf, kill_rate = 0.99):
    points_el, active_conf, electrode_nums = wos.input.shape.out_boundary.create_electrode_points(spe = spe, delete_injection = delete_injection, conf_numbers = conf_numbers)
    L, _ = wos.solve(points_el, active_conf, split = split, max_depth_split = max_split_depth, 
                     conf_numbers=conf_numbers, all_inside = True, verbose = verbose, 
                     tput_kill = kill_rate, max_length=kill_step)
    el_std = None
    if compute_std:
        el_tensor, el_std = create_electrode_result(L, spe, electrode_nums, apply_normalization=True, compute_std = True)
        el_std = el_std.numpy()
    else:
        el_tensor = create_electrode_result(L, spe, electrode_nums, apply_normalization = True, compute_std = False)
    return L, el_tensor, el_std, electrode_nums

def compute_primals(wos : WostVariable, split : Split, spe : int, dirichlet_spe : int,  seed : int, delete_injection : bool, 
                    max_split_depth : int, compute_std : bool, confs_iter : int = 8, num_electrodes : int = 16, dirichlet_offset : int = 0,
                    kill_step = dr.inf, kill_rate = 0.99, wos_dummy : WostVariable = None, selected_point : int = 0, conf_numbers = []):
    num_conf = len(conf_numbers)
    results = np.zeros([num_conf, num_electrodes])
    std = np.zeros([num_conf, num_electrodes])
    electrode_nums = np.zeros([num_conf, num_electrodes - 2])
    
    num_iter = int(np.ceil(num_conf / confs_iter))
    #print(num_iter)
    initstate, initseq = tea(dr.arange(UInt64, num_iter), UInt64(seed))
    pcg = PCG32()
    pcg.seed(initstate, initseq)
    seeds = pcg.next_uint32_bounded(10000).numpy()
    #print(seeds)
    for i, seed in enumerate(seeds):
        begin= i * confs_iter
        end = min(num_conf, (i + 1) * confs_iter)
        conf_numbers_i = [conf_numbers[j] for j in range(begin, end)]
        dr.make_opaque(conf_numbers_i)
        wos.change_seed(seed)

        # Compute the voltages of the points inside.
        if (wos.input.shape.num_shapes > 1):
            origins = wos.input.shape.get_origins()
            origins = np.delete(origins, selected_point, axis = 0)
            in_points =  Point2f(origins.T)
            in_points = dr.repeat(in_points, dirichlet_spe)
            #dr.set_log_level(3)
            L_d, _ = wos_dummy.solve(in_points, split = split, max_depth_split = max_split_depth, 
                                     conf_numbers=conf_numbers_i, all_inside = True, max_length=kill_step, tput_kill = kill_rate, verbose = False)
            #dr.set_log_level(0)
            dirichlet_vals  = (dr.block_sum(L_d, dirichlet_spe) / dirichlet_spe).numpy().T
            dirichlet_vals = np.insert(dirichlet_vals, selected_point, dirichlet_offset, axis = 0)
            wos.input.shape.update_in_boundary_dirichlets(dirichlet_vals.tolist())
        
        _, signal, el_std, elnums = compute_primal(wos, split, spe, delete_injection, conf_numbers_i, max_split_depth, compute_std, 
                                                   kill_step = kill_step, kill_rate = kill_rate)
        results[begin : end] = signal.numpy()
        electrode_nums[begin : end] = elnums.numpy() 
        if el_std is not None:
            std[begin : end] = elnums.numpy()
    return results, std, electrode_nums

def optimize_eit(path : str, wos : WostVariable, wos_obj : WostVariable, wos_dummy : WostVariable, split : Split, 
                spe : int, primal_spe : int, dirichlet_spe : int, seed : int, conf_per_iter : int, max_split_depth : int,
                num_iter : int, learning_rate : float, λ_L1 : float, λ_TV : float, post_process : callable, 
                cond_threshold : float, grad_threshold : float, max_dirichlet : int, dirichlet_radius : float, dirichlet_offset : float, 
                merge_distance : float, normalize_grad :bool = False, plot : bool = True, bbox_plot : list[list[float]] = [[-1.05, -1.05],[1.05, 1.05]],  
                delete_injection : bool = True, compute_std : bool = True, verbose : bool = False, 
                max_range : list[float] = [0.1, 3], fileset : str = None, centered_dirichlet = False, kill_step = dr.inf, kill_rate = 0.99, measure_time = True, 
                vis_confs = []):
    set_matplotlib(9)
    selected_point = 0
    num_conf = wos.input.shape.out_boundary.num_confs
    print(num_conf)
    conf_per_iter = min(num_conf, conf_per_iter)
    num_electrodes = wos.input.shape.out_boundary.num_electrodes
    objectives = wos.input.shape.out_boundary.voltages
    
    # Compute the primals for getting the first loss values.
    conf_numbers_all = [dr.opaque(UInt32, i, shape=(1)) for i in range(num_conf)]
    primals, primals_std, electrode_nums = compute_primals(wos, split, primal_spe, dirichlet_spe, seed, delete_injection, max_split_depth, compute_std, 
                                                           confs_iter = num_electrodes, num_electrodes= num_electrodes, kill_step=kill_step, 
                                                           kill_rate=kill_rate, conf_numbers = conf_numbers_all)
    
    losses = MSE_numpy(primals, objectives)

    plot_coeff(wos.input.α, wos.input.shape, bbox_plot, path, "begin-scaled", coeff_obj = wos_obj.input.α, max_range = max_range) 
    plot_coeff(wos.input.α, wos.input.shape, bbox_plot, path, "begin", coeff_obj = wos_obj.input.α) 

    
    # Create the optimizer
    opt = Adam(lr= learning_rate, params = wos.opt_params)
    wos.update(opt)

    loss_list = []
    loss_reg_list = []
    # Record the objective optimization parameters
    record_dict = {}
    #if max_dirichlet > 0:
    dirichlet_points =  wos.input.shape.get_origins()
    record_dict[f"dirichletpoints-0"] = np.array(dirichlet_points)
    record_dict["primals-0"] = np.array(primals.squeeze())
    record_dict[f"objectives"] = objectives
    for key in opt.keys():
        record_dict[f"{key}-0"] = opt[key].numpy()

    if wos_obj is not None:
        dirichlet_obj = wos_obj.input.shape.get_origins()
        record_dict["dirichlet-objective"] = np.array(dirichlet_obj)
        record_dict["objective-tensor"] = wos_obj.input.α.tensor.numpy().squeeze()
    
    initstate, initseq = tea(dr.arange(UInt32, conf_per_iter), UInt64(seed))
    sampler = PCG32()
    sampler.seed(initstate, initseq)
    group_size = int(dr.ceil(num_conf / conf_per_iter))
    confs = dr.arange(UInt32, num_conf)

    print("Optimization Started!")
    # Begin optimization
    
    for i in range(num_iter):

        # Primal results of some confs are computed at each iteration for visualization purposes.
        if len(vis_confs) > 0:
            if measure_time:
                dr.sync_thread()
                t0 = time.time()
            primals_vis, _, _= compute_primals(wos, split, primal_spe, dirichlet_spe, seed + i, delete_injection, max_split_depth, compute_std, 
                                                           confs_iter = num_electrodes, num_electrodes= num_electrodes, kill_step=kill_step, 
                                                           kill_rate=kill_rate, conf_numbers = vis_confs, wos_dummy=wos_dummy,
                                                           dirichlet_offset = dirichlet_offset, selected_point=selected_point)
            record_dict[f"vis-{i}"] = np.array(primals_vis)
            if measure_time:
                dr.sync_thread()
                t1 = time.time()
                t_vis = t1 - t0
                print(f"Vis primal time: {t_vis}")
                record_dict[f"t-vis{i}"] = t_vis
        
        seed_iter = sampler.next_uint32()[0]       
        # Select some confs randomly.
        confs_iter = dr.gather(UInt32, confs, sampler.next_uint32_bounded(group_size) + dr.arange(UInt32, conf_per_iter) * group_size)
        confs_iter = dr.select(confs_iter >= num_conf, sampler.next_uint32_bounded(num_conf), confs_iter)
        confs_iter = confs_iter.numpy().squeeze()
        confs_opaque = [dr.opaque(UInt32, j, shape = (1)) for j in confs_iter.tolist()]
        obj_res_iter = objectives[confs_iter, :]
        wos.change_seed(seed_iter)
        
        if measure_time:
            dr.sync_thread()
            t0 = time.time()
        # Compute the voltages of the points inside.
        if (wos.input.shape.num_shapes > 1) and not centered_dirichlet:
            origins = wos.input.shape.get_origins()
            origins = np.delete(origins, selected_point, axis = 0)
            in_points =  Point2f(origins.T)
            in_points = dr.repeat(in_points, dirichlet_spe)
            L_d, _ = wos_dummy.solve(in_points, split = split, max_depth_split = max_split_depth, 
                                     conf_numbers=confs_opaque, all_inside = True, verbose = verbose, max_length=kill_step, tput_kill = kill_rate)
            dirichlet_vals  = (dr.block_sum(L_d, dirichlet_spe) / dirichlet_spe).numpy().T
            dirichlet_vals = np.insert(dirichlet_vals, selected_point, dirichlet_offset, axis = 0)
            wos.input.shape.update_in_boundary_dirichlets(dirichlet_vals.tolist())

        if measure_time:
            dr.sync_thread()
            t1 = time.time()
            t_dirichlet = t1 - t0
            print(f"Dirichlet time: {t_dirichlet}")
            record_dict[f"t-dirichlet{i}"] = t_dirichlet


        L, signal, signal_std, _ = compute_primal(wos, split, primal_spe, delete_injection, confs_opaque,  
                                                  max_split_depth, compute_std, verbose = verbose, 
                                                  kill_step = kill_step, kill_rate = kill_rate)
        dr.eval(signal)

        signal_np = signal.numpy()
        primals[confs_iter] = signal.numpy()
        losses[confs_iter] = MSE_numpy(signal_np, obj_res_iter)
        if compute_std:
            primals_std[confs_iter] = signal_std

        obj_opaque = ArrayXf(obj_res_iter.tolist())
        dr.make_opaque(obj_opaque)
        loss_grad = compute_loss_grad(signal, obj_opaque)
        if normalize_grad:
            L, signal, _, elnums = compute_primal(wos, split, spe, delete_injection, confs_opaque, max_split_depth, False, kill_step = kill_step, kill_rate = kill_rate)
        else:
            L = Float(1)
            elnums = None
        
        dL = compute_dL(L, loss_grad, spe, elnums, apply_normalization=normalize_grad)
        points_el, active_conf, _ = wos.input.shape.out_boundary.create_electrode_points(spe = spe, delete_injection = delete_injection, 
                                                                                         conf_numbers = confs_opaque)
        #print(f"Grad (num_shapes = {wos.input.shape.num_shapes})")
        #dr.set_log_level(3)

        if measure_time:
            dr.sync_thread()
            t2 = time.time()
            t_primal = t2 - t1
            print(f"Primal time: {t_primal}")
            record_dict[f"t-primal{i}"] = t_primal

        with dr.isolate_grad():
            _ = wos.solve_grad(points_in = points_el, active_conf_in = active_conf, split = split, dL = dL, max_depth_split = max_split_depth, 
                                conf_numbers=confs_opaque, all_inside = True, verbose = verbose, max_length=kill_step, tput_kill = kill_rate)
        #dr.set_log_level(0)

        if measure_time:
            dr.sync_thread()
            t3 = time.time()
            t_grad = t3 - t2
            print(f"Grad time: {t_grad}")
            record_dict[f"t-grad{i}"] = t_grad
        
        loss_reg = 0
        if λ_L1 > 0:
            reg_L1 = wos.input.compute_regularization(λ_L1, RegularizationType.tensorL1)
            dr.backward(reg_L1)
            loss_reg += dr.sum(reg_L1)[0]
        
        if λ_TV > 0:
            reg_TV = wos.input.compute_regularization(λ_TV, RegularizationType.TV)
            dr.backward(reg_TV)
            loss_reg += dr.sum(reg_TV)[0]

        if plot:
            iter_plot(wos, bbox_plot, path, f"{i}", primals, primals_std, electrode_nums, compute_std = compute_std, wos_obj = wos_obj, max_range = max_range)
        
        coeff = wos.input.get_coefficient("diffusion")
        grad_np = dr.grad(coeff.tensor).numpy()
        opt.step()
    
        post_process(opt)
        wos.update(opt)
        wos_dummy.update(opt)

        if not centered_dirichlet and (max_dirichlet > 0):
            dirichlet_points = wos.input.compute_high_conductance_points(max_num_points=max_dirichlet, cond_threshold= cond_threshold, 
                                                                        grad_threshold=grad_threshold, merge_distance = merge_distance)
            
            if dirichlet_points.shape[0] == 1:
                wos.input.shape.update_in_boundaries_circle(origins = dirichlet_points, radius = dirichlet_radius, dirichlet_values = [dirichlet_offset])
                wos_dummy.input.shape.update_in_boundaries_circle(origins = dirichlet_points, radius = dirichlet_radius, dirichlet_values = None)
            else:
                selected_point = sampler.next_uint32_bounded(dirichlet_points.shape[0])[0]
                wos_dummy.input.shape.update_in_boundaries_circle(origins = [dirichlet_points[selected_point]], 
                                                                radius = dirichlet_radius, dirichlet_values = [dirichlet_offset])
                wos.input.shape.update_in_boundaries_circle(origins = dirichlet_points, 
                                                            radius = dirichlet_radius, dirichlet_values = None)
            record_dict[f"dirichletpoints-{i+1}"] = np.array(dirichlet_points)

        if wos.use_accel:
            wos.input.create_accelaration()
            wos_dummy.input.create_accelaration()

        if measure_time:
            dr.sync_thread()
            t4 = time.time()
            t_accel = t4 - t3
            print(f"Accel time : {t_accel}")
            record_dict[f"t-accel{i}"] = t_accel

        record_dict[f"grad-{i+1}"] = np.array(grad_np.squeeze())
        record_dict[f"primals-{i+1}"] = np.array(primals)
        
        for key in opt.keys():
            record_dict[f"{key}-{i+1}"] = opt[key].numpy()
            
        
        
        loss_reg_list.append(loss_reg)
        loss_list.append(np.array(losses))  
        record_dict["loss"] = np.array(loss_list)  
        record_dict["loss-reg"] = np.array(loss_reg_list)
        print(f"Iteration {i} is finished. Loss = {np.array(losses).sum() + loss_reg}")
        #print(time.time() - t)
    
    print("Optimization Ended! Animations will be generated.")
    plot_summary(loss_list, loss_reg_list, path, log=False)
    plot_summary(loss_list, loss_reg_list, path, log=True)
    if wos_obj is not None:
        plot_coeff(wos.input.α, wos.input.shape, bbox_plot, path, "end-scaled", coeff_obj = wos_obj.input.α, max_range = max_range) 
        plot_coeff(wos.input.α, wos.input.shape, bbox_plot, path, "end", coeff_obj = wos_obj.input.α) 
    else:
        plot_coeff(wos.input.α, wos.input.shape, bbox_plot, path, "end-scaled", coeff_obj = None, max_range = max_range) 
        plot_coeff(wos.input.α, wos.input.shape, bbox_plot, path, "end", coeff_obj = None)

    if fileset is None:
        create_animation(record_dict, path, num_iter, bbox_plot, wos,
                        resolution = [1024, 1024], max_range = max_range, wos_obj = wos_obj, opt_param = "diffusion.texture.tensor")
        create_animation(record_dict, path, num_iter, bbox_plot, wos,
                        resolution = [1024, 1024], wos_obj = wos_obj, opt_param = "diffusion.texture.tensor")
    else:
        create_animation(record_dict, path, num_iter, bbox_plot, wos,
                        resolution = [1024, 1024], max_range = max_range, wos_obj = wos_obj, 
                        opt_param = "diffusion.texture.tensor", fileset = fileset)

    np.save(os.path.join(path, "record.npy"), record_dict)
    print("Animations are generated.")
