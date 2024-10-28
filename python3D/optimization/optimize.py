
import numpy as np
import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
from PDE3D.Coefficient import *
from PDE3D.utils import *
from PDE3D.BoundaryShape import *
from PDE3D.Solver import *
from python3D.optimization.sketch import *
import time

def compute_primal(wos : WosVariable, split : Split, bbox : mi.ScalarBoundingBox3f, resolution : list[int], spp : int,
                   centered : bool, max_split_depth : int, conf_numbers : list[mi.UInt32], compute_std: bool, 
                   verbose : bool = False):
    points = create_bbox_points(bbox, resolution, spp, centered=centered)

    L, _ = wos.solve(points, split = split, max_depth_split = max_split_depth, 
                     conf_numbers=conf_numbers, all_inside = False, verbose = verbose)
    std = None
    if compute_std:
        _, tensor, var, _= create_volume_from_result(L, resolution, compute_std)
        std = np.sqrt(var)
    else:
        _, tensor = create_volume_from_result(L, resolution, compute_std)
    return L, tensor, std

def compute_primals(wos : WosVariable, split : Split, seed : int, bbox : mi.ScalarBoundingBox3f, resolution : list[int],
                    spp : int, centered : bool, max_split_depth : int, compute_std : bool, 
                    confs_iter : int = 8):
    num_conf = wos.input.shape.num_conf_d
    
    results = np.zeros([num_conf, resolution[0], resolution[1], resolution[2]])
    results_std = np.zeros([num_conf, resolution[0], resolution[1], resolution[2]])
    
    num_iter = int(np.ceil(num_conf / confs_iter))
    initstate, initseq = tea(dr.arange(mi.UInt64, num_iter), mi.UInt64(seed))
    pcg = mi.PCG32()
    pcg.seed(initstate, initseq)
    seeds = pcg.next_uint32_bounded(10000).numpy().squeeze()
    seeds = [seeds] if seeds.ndim == 0 else seeds
    for i, seed in enumerate(seeds):
        begin= i * confs_iter
        end = min(num_conf, (i + 1) * confs_iter)
        conf_numbers = [dr.opaque(mi.UInt32, j, shape = (1)) for j in range(begin, end)]
        wos.change_seed(seed)
        
        _, tensor, std = compute_primal(wos, split, bbox, resolution, spp, centered, max_split_depth, 
                                        conf_numbers, compute_std)
        results[begin : end] = tensor.numpy()
        if std is not None:
            results_std[begin : end] = std
    return results, results_std


def optimize(path : str, wos : WosVariable, wos_obj : WosVariable, objectives : mi.TensorXf,  coeff_str : str,
                 split : Split, bbox : list[list], resolution : list[int], spp : int, 
                 primal_spp : int,  seed : int, conf_per_iter : int, 
                 max_split_depth : int, num_iter : int, learning_rate : float,  post_process : callable, centered : bool, 
                 plot : bool = True, compute_std : bool = True, verbose : bool = False, 
                 max_range : list[float] = [0.1, 3], measure_time = False, vis_set = []):
    #set_matplotlib(9)
    num_conf = wos.input.shape.num_conf_d
    conf_per_iter = min(num_conf, conf_per_iter)
    
    # Compute the primals for getting the first loss values.
    primals, primals_std = compute_primals(wos, split, seed, bbox, resolution, primal_spp, centered, max_split_depth, compute_std, confs_iter = 16)
    losses = MSE_numpy(primals, objectives)

    coeff = wos.input.get_coefficient(coeff_str)
    coeff_obj = wos_obj.input.get_coefficient(coeff_str)

    
    plot_coeff(coeff, wos.input.shape, None, path, f"{coeff_str}-begin") 
    plot_coeff(coeff, wos.input.shape, max_range, path, f"{coeff_str}-begin-scaled") 

    
    # Create the optimizer
    opt = mi.ad.Adam(lr= learning_rate, params = wos.opt_params)
    wos.update(opt)

    loss_list = []
    #loss_reg_list = []
    # Record the objective optimization parameters
    if measure_time:
        primal_time = []
        grad_time = []
        vis_time = []
    path_npy = os.path.join(path, "npy")
    np.save(os.path.join(path_npy, "objectives.npy"), objectives)
    np.save(os.path.join(path_npy, "primal", "primal-0.npy"),  primals)
    if compute_std:
        np.save(os.path.join(path_npy, "primal", "std-0.npy"), primal_std)
    for key in opt.keys():
        np.save(os.path.join(path_npy, "tensor", f"{key}-0.npy"), opt[key].numpy().squeeze())
    if wos_obj is not None:
        np.save(os.path.join(path_npy, f"objective-tensor.npy"), wos_obj.input.get_coefficient(coeff_str).tensor.numpy().squeeze())
        
    
    initstate, initseq = tea(dr.arange(mi.UInt32, conf_per_iter), mi.UInt64(seed))
    sampler = mi.PCG32()
    sampler.seed(initstate, initseq)
    
    print("Optimization Started!")
    
    # Begin optimization
    for i in range(num_iter):
        if len(vis_set) > 0:
            if measure_time:
                dr.sync_thread()
                t0 = time.time()
            wos.change_seed(i+1)
            _, primal_vis, _ =  compute_primal(wos, split, bbox, resolution, primal_spp, centered, max_split_depth, 
                                            vis_set, compute_std=compute_std, verbose = verbose)
            np.save(os.path.join(path_npy, "primal", f"primalvis-{i}.npy"), primal_vis.numpy().squeeze())
            if measure_time:
                dr.sync_thread()
                t1 = time.time()
                t_vis = t1 - t0
                #print(f"Primal time: {t_primal}")
                vis_time.append(t_vis)
        
        seed_iter = np.random.randint(0, 2**15)
        wos.change_seed(seed_iter)  
        confs_iter = np.random.choice(range(num_conf), conf_per_iter, replace = False)
        confs_opaque = [dr.opaque(mi.UInt32, j, shape = (1)) for j in confs_iter.tolist()]
        obj_res_iter = objectives[confs_iter, :]
        if measure_time:
            dr.sync_thread()
            t0 = time.time()

        L, primal, primal_std = compute_primal(wos, split, bbox, resolution, primal_spp, centered, max_split_depth, 
                                                confs_opaque, compute_std=compute_std, verbose = verbose)
        dr.eval(primal)
        primal_np = primal.numpy()
        primals[confs_iter] = primal_np
        losses[confs_iter] = MSE_numpy(primal_np, obj_res_iter)
        if compute_std:
            primals_std[confs_iter] = primal_std

        if measure_time:
            dr.sync_thread()
            t1 = time.time()
            t_primal = t1 - t0
            #print(f"Primal time: {t_primal}")
            primal_time.append(t_primal)

        obj_opaque = mi.TensorXf(obj_res_iter.tolist())
        dr.make_opaque(obj_opaque)
        loss_grad = compute_loss_grad_vol(primal, obj_opaque)
        
        dL = compute_dL_vol(loss_grad, spp)
        points = create_bbox_points(bbox, resolution, spp, seed = seed, centered = centered)
        
        with dr.isolate_grad():
            _ = wos.solve_grad(points_in = points, split = split, dL = dL, max_depth_split = max_split_depth, 
                               conf_numbers=confs_opaque, verbose = verbose)

        
        coeff = wos.input.get_coefficient(coeff_str)
        grad_np = dr.grad(coeff.tensor).numpy()
        opt.step()
        post_process(opt)
        wos.update(opt)
        if measure_time:
            dr.sync_thread()
            t2 = time.time()
            t_grad = t2 - t1
            #print(f"Grad time: {t_grad}")
            grad_time.append(t_grad)
        
        np.save(os.path.join(path, "npy", "grad", f"grad-{i+1}.npy"), np.array(grad_np).squeeze())
        np.save(os.path.join(path, "npy", "primal", f"primal-{i+1}.npy"), np.array(primals).squeeze())
        if compute_std:
            np.save(os.path.join(path, "npy", "primal", f"std-{i+1}.npy"), np.array(primals_std).squeeze())
        
        for key in opt.keys():
            np.save(os.path.join(path, "npy", "tensor", f"{key}-{i+1}.npy"), opt[key].numpy().squeeze())
            
        loss_list.append(np.array(losses))   
        print(f"Iteration {i} is finished. Loss = {np.array(losses).sum()}")
    np.save(os.path.join(path, "npy", "losses.npy"), np.array(loss_list))
    if measure_time:
        np.save(os.path.join(path, "npy", "primal_time.npy"), np.array(primal_time))
        np.save(os.path.join(path, "npy", "grad_time.npy"), np.array(grad_time))
        np.save(os.path.join(path, "npy", "vis_time.npy"), np.array(vis_time))
    
    print("Optimization Ended!")
    plot_summary(loss_list,  path, log=False)
    plot_summary(loss_list, path, log=True)
    plot_coeff(coeff, wos.input.shape, max_range, path, f"{coeff_str}-end-scaled") 
    plot_coeff(coeff, wos.input.shape, None, path, f"{coeff_str}-end") 


    return wos
    