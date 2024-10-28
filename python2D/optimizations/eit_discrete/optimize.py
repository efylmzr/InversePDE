import numpy as np
import mitsuba as mi 
mi.set_variant("cuda_ad_rgb")
import drjit as dr
from PDE2D.Coefficient import *
from PDE2D.utils import *
from PDE2D.BoundaryShape import *
from PDE2D.Solver import *
from python2D.optimizations.eit_discrete.sketch import *
from mitsuba import Float, UInt32, UInt64, Point2f
import time

def compute_primal(wos : WostConstant, spe : int, delete_injection : bool, 
                   conf_numbers : list[UInt32]):
    points_el, active_conf, electrode_nums = wos.input.shape.out_boundary.create_electrode_points(spe = spe, delete_injection = delete_injection, conf_numbers = conf_numbers)
    L, _ = wos.solve(points_el, active_conf, conf_numbers=conf_numbers, all_inside = True)

    el_tensor = create_electrode_result(L, spe, electrode_nums, apply_normalization = True, compute_std = False)
    return L, el_tensor,electrode_nums

def compute_primals(wos : WostConstant, spe : int, seed : int, delete_injection : bool,  
                    confs_iter : int = 8, num_electrodes : int = 16, 
                    conf_numbers = []):
    num_conf = len(conf_numbers)
    results = np.zeros([num_conf, num_electrodes])
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
        
        _, signal,  elnums = compute_primal(wos, spe, delete_injection, conf_numbers_i)
        results[begin : end] = signal.numpy()
        electrode_nums[begin : end] = elnums.numpy() 
    return results, electrode_nums

def optimize_eit(path : str, wos : WostConstant, wos_obj : WostConstant,
                spe : int, primal_spe : int, seed : int, conf_per_iter : int,
                res_normalder : int, spp_normalder : int, dist_normalder :int,
                num_iter : int, learning_rate : float,
                post_process : callable,
                plot : bool = True, bbox_plot : list[list[float]] = [[-1.05, -1.05],[1.05, 1.05]],  
                delete_injection : bool = True,
                measure_time = True, 
                vis_confs = [], is_sdf = False):
    set_matplotlib(9)
    num_conf = wos.input.shape.out_boundary.num_confs
    conf_per_iter = min(num_conf, conf_per_iter)
    num_electrodes = wos.input.shape.out_boundary.num_electrodes
    objectives = wos.input.shape.out_boundary.voltages

    # Record the objective optimization parameters
    record_dict = {}
    if wos_obj is not None:
        if is_sdf:
            record_dict[f"tensor-obj"] = wos_obj.input.shape.in_boundaries[0].tensor.numpy()
            record_dict[f"tensor-0"] = wos.input.shape.in_boundaries[0].tensor.numpy()
        else:
            record_dict[f"origin-obj"] = wos_obj.input.shape.in_boundaries[0].origin.numpy()
            record_dict[f"radius-obj"] = wos_obj.input.shape.in_boundaries[0].radius.numpy()
            record_dict[f"origin-0"] = wos.input.shape.in_boundaries[0].origin.numpy()
            record_dict[f"radius-0"] = wos.input.shape.in_boundaries[0].radius.numpy()
    
    # Compute the primals for getting the first loss values.
    conf_numbers_all = [dr.opaque(UInt32, i, shape=(1)) for i in range(num_conf)]
    primals, electrode_nums = compute_primals(wos, primal_spe, seed, delete_injection,
                                                           confs_iter = num_electrodes, num_electrodes= num_electrodes, conf_numbers = conf_numbers_all)
    
    losses = MSE_numpy(primals, objectives)

    # Create the optimizer
    opt = mi.ad.Adam(lr= learning_rate, params = wos.opt_params)
    wos.update(opt)

    loss_list = []
    loss_reg_list = []
    #if max_dirichlet > 0:
    record_dict["primals-0"] = np.array(primals.squeeze())
    record_dict[f"objectives"] = objectives
    for key in opt.keys():
        record_dict[f"{key}-0"] = opt[key].numpy()
    
    initstate, initseq = tea(dr.arange(UInt32, conf_per_iter), UInt64(seed))
    sampler = PCG32()
    sampler.seed(initstate, initseq)
    group_size = int(dr.ceil(num_conf / conf_per_iter))
    confs = dr.arange(UInt32, num_conf)

    print("Optimization Started!")
    # Begin optimization
    
    for i in range(num_iter):
        seed_iter = sampler.next_uint32()[0]       
        # Select some confs randomly.
        confs_iter = dr.gather(UInt32, confs, sampler.next_uint32_bounded(group_size) + dr.arange(UInt32, conf_per_iter) * group_size)
        confs_iter = dr.select(confs_iter >= num_conf, sampler.next_uint32_bounded(num_conf), confs_iter)
        confs_iter = confs_iter.numpy().squeeze()
        confs_opaque = [dr.opaque(UInt32, j, shape = (1)) for j in confs_iter.tolist()]
        obj_res_iter = objectives[confs_iter, :]
        wos.change_seed(seed_iter)

        obj_opaque = ArrayXf(obj_res_iter.tolist())
        dr.make_opaque(obj_opaque)

        result, _ = wos.create_normal_derivative(res_normalder, spp_normalder, distance = dist_normalder, conf_numbers=confs_opaque)
        #dr.eval(result)
        wos.input.shape.in_boundaries[0].set_normal_derivative(result)
        points, active_conf, electrode_nums = wos.input.shape.out_boundary.create_electrode_points(spe, conf_numbers=confs_opaque)
        
        #dr.set_log_level(3)
        L, _ = wos.solve(points, active_conf, conf_numbers=confs_opaque, all_inside = True)
        el_tensor = create_electrode_result(L, spe, electrode_nums, apply_normalization=True)
        #dr.set_log_level(0)

        loss_grad = compute_loss_grad(result = el_tensor, result_ref = obj_opaque)
        dL = compute_dL(L = L, loss_grad=loss_grad, spe = spe, electrode_nums=electrode_nums, apply_normalization=True)

        signal_np = el_tensor.numpy()
        primals[confs_iter] = signal_np
        losses[confs_iter] = MSE_numpy(signal_np, obj_res_iter)

        _ = wos.solve(points, active_conf, L_in = L, conf_numbers=confs_opaque, dL = dL, 
                      all_inside = True, normal_derivative_dist=dist_normalder, 
                      mode = dr.ADMode.Backward)
        opt.step()
        post_process(opt)
        wos.update(opt)

        record_dict[f"primals-{i+1}"] = np.array(primals)

        
        
        for key in opt.keys():
            record_dict[f"{key}-{i+1}"] = opt[key].numpy()

        
        loss_list.append(np.array(losses))  
        record_dict["loss"] = np.array(loss_list) 
        if is_sdf:
            record_dict[f"tensor-{i+1}"] = opt["inboundary.dirichlet.tensor"].numpy()
        else:
            record_dict[f"origin-{i+1}"] = opt["inboundary.dirichlet.origin"].numpy()
            record_dict[f"radius-{i+1}"] = opt["inboundary.dirichlet.radius"].numpy()

        print(f"Iteration {i} is finished. Loss = {np.array(losses).sum()}")
        #print(time.time() - t)
    
    print("Optimization Ended! Animations will be generated.")
    plot_summary(loss_list, loss_reg_list, path, log=False)
    plot_summary(loss_list, loss_reg_list, path, log=True)
    create_animation_shape(record_dict, path, num_iter-1, bbox_plot, wos, wos_obj = wos_obj, resolution = [1024, 1024], 
                           type = "sdf" if is_sdf else "sphere", plot_center = False)


    np.save(os.path.join(path, "record.npy"), record_dict)
    print("Animations are generated.")
