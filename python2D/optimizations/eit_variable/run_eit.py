import argparse
import numpy as np
import os
from python2D.optimizations.eit_variable.textures_eit import *
from python2D.optimizations.eit_variable.optimize_eit import *
from PDE2D.utils import *
from PDE2D.BoundaryShape import *
from PDE2D.Solver import *
from PDE2D import PATH, GreenSampling, Split



root_directory = os.path.join(PATH, "output2D", "optimizations", "variable-eit")
def main():
    parser = argparse.ArgumentParser(description='''Optimization Sphere''')
    parser.add_argument("--restensor", default = 24, type = int)
    parser.add_argument('--spe', default = 17, type=int)
    parser.add_argument('--dirichletspe', default = 23, type=int)
    parser.add_argument('--primalspe', default = 20, type=int)
    parser.add_argument('--objspe', default = 20, type=int)
    parser.add_argument('--seedobj', default = 16, type=int)
    parser.add_argument('--seed', default = 42, type=int)
    parser.add_argument('--confiter', default = 6, type=int)
    parser.add_argument('--iternum', default = 16, type=int)
    parser.add_argument("--lr", default = "0.1", type = float)
    parser.add_argument("--epsilon", default = "1e-3", type = float)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--normalizedgrad", action="store_true")
    parser.add_argument("--split", default = "normal", type = str)
    parser.add_argument("--noaccel", action="store_true")
    parser.add_argument("--splitdepth", default=254, type=int)
    parser.add_argument("--computevariance", action = "store_true")
    parser.add_argument("--numdirichlet", default = 1, type = int)
    parser.add_argument("--dirichletoffset", default = 0, type = float)
    parser.add_argument("--dirichletradius", default = 0.001,  type = float)
    parser.add_argument("--regL1", default = "0.0001", type = float)
    parser.add_argument("--regTV", default = "0.0001", type = float)
    parser.add_argument("--injectionset", default = "skip1-skip3-skip5-skip7", type = str)
    parser.add_argument("--visset", default = "none", type = str)
    parser.add_argument("--scaletexture", default = 1.0, type=float)
    parser.add_argument("--condthreshold", default = 1.2, type = float)
    parser.add_argument("--gradthreshold", default = 10, type = float)
    parser.add_argument("--mergedistance", default = 0.30, type = float)
    parser.add_argument("--conf", default = "1", type = str)
    parser.add_argument("--verbose", action = "store_true")
    parser.add_argument("--centeredsingle", action = "store_true")
    parser.add_argument("--kill", action = "store_true")
    parser.add_argument("--killstep", type = int, default = 150)
    parser.add_argument("--killrate", type = float, default = 0.99)
    parser.add_argument("--measuretime", action = "store_true")
    args = parser.parse_args()

    radius = 0.22
    bbox = [[-1.1 * radius,-1.1 * radius],[1.1 * radius, 1.1 * radius]]
    compute_variance = args.computevariance
    max_dirichlet = args.numdirichlet
    centered_dirichlet = args.centeredsingle
    dirichlet_offset = args.dirichletoffset
    dirichlet_radius = args.dirichletradius
    use_accel = not args.noaccel
    split_depth = args.splitdepth
    e_shell = args.epsilon * radius
    delete_injection = True
    normalized_grad = args.normalizedgrad
    plot = args.plot
    seed_obj = args.seedobj
    seed = args.seed
    is_delta = True
    spe_obj = 2 ** args.objspe
    spe = 2 ** args.spe
    spe_primal = 2 ** args.primalspe
    spe_dirichlet = 2 ** args.dirichletspe
    num_electrodes = 16
    conf_per_iter = args.confiter
    num_iter = args.iternum
    learning_rate = args.lr
    λ_L1 = args.regL1
    λ_TV = args.regTV
    bg_conductance = 1
    cond_threshold = args.condthreshold * bg_conductance
    grad_threshold = args.gradthreshold
    merge_distance = args.mergedistance
    res_tensor = args.restensor
    resolution_tensor = [res_tensor, res_tensor]
    

    if args.split == "none":
        split = Split.Naive
    elif args.split == "agressive":
        split = Split.Agressive
    elif args.split == "normal":
        split = Split.Normal
    else:
        raise Exception("No such split is defined.")
    
    
    conf_name = f"conf{args.conf}"
    normalized_name = "normalized" if normalized_grad else "unnormalized"
    if max_dirichlet == 0:
        dirichlet_name = "cn"
    else:
        dirichlet_name = "singlecentered" if centered_dirichlet else f"maxdirichlet{max_dirichlet}-merge{args.mergedistance}-spe{args.dirichletspe}"
    res_name = f"res{res_tensor}"
    spe_name = f"spe{args.primalspe}_{args.spe}"
    seed_name = f"seed{seed}"
    #e_name = f"epsilon{args.epsilon}"
    scale_name = f"scale{args.scaletexture}"
    reg_name = f"L1_{λ_L1}-TV_{λ_TV}"

    out_boundary = CircleWithElectrodes(radius = radius, num_electrodes=num_electrodes, is_delta = is_delta, 
                                        injection_set = args.injectionset, centered = True)
    # The primal configurations that are going to be evaluated at each iteration for visualization.
    vis_set = []
    if args.visset != "none":
        vis_set = out_boundary.get_injection_confs(args.injectionset, args.visset, num_electrodes=num_electrodes)
    
    num_conf = out_boundary.num_confs
    
    image_obj = objectives[conf_name]

    image_obj *= args.scaletexture
    image_obj += bg_conductance

    image = np.zeros(resolution_tensor)
    image[int(res_tensor * 3 / 8) : int(res_tensor * 5 / 8), 
                int(res_tensor * 3 / 8) : int(res_tensor * 5 / 8)] = 0.1
    image += bg_conductance

    kill_step = args.killstep if args.kill else dr.inf
    kill_rate = args.killrate


    grad_points = out_boundary.create_boundary_points(distance = 0, res = 512, spp = 1, discrete_points=True)[0]
    α_obj = TextureCoefficient("diffusion", bbox, image_obj, grad_zero_points=grad_points, out_val = bg_conductance)

    grad_points = out_boundary.create_boundary_points(distance = 0, res = 512, spp = 1, discrete_points=True)[0]
    α = TextureCoefficient("diffusion", bbox, image, grad_zero_points=grad_points, out_val = bg_conductance)

    opt_variable_name = "diffusion.texture.tensor"
    
    shape_obj = BoundaryWithDirichlets(out_boundary=out_boundary, dirichlet_boundaries=[], epsilon=e_shell, 
                                       dirichlet_values = [])
    data_holder_obj = DataHolder(shape_obj, α = α_obj)
    if centered_dirichlet:
        obj_dir_point = np.array([[0,0]])
    else:
        obj_dir_point = data_holder_obj.compute_high_conductance_points(max_num_points=1, cond_threshold=cond_threshold, 
                                                                    grad_threshold=grad_threshold, merge_distance=merge_distance)
    data_holder_obj.shape.update_in_boundaries_circle(origins = obj_dir_point, radius = dirichlet_radius * radius, 
                                                      dirichlet_values = [dirichlet_offset])
    wos_obj = WostVariable(data_holder_obj, green_sampling=GreenSampling.Polynomial, use_accelaration = use_accel)

    
    opt_params = [opt_variable_name]
    if max_dirichlet == 0:
        shape = BoundaryWithDirichlets(out_boundary=out_boundary, dirichlet_boundaries=[], 
                                        epsilon = e_shell, dirichlet_values = [])
    else:
        shape = BoundaryWithDirichlets(out_boundary=out_boundary, dirichlet_boundaries=[CircleShape(origin = Point2f(0, 0), radius = dirichlet_radius * radius)], 
                                   epsilon = e_shell, dirichlet_values = [dirichlet_offset])
    data_holder = DataHolder(shape, α = α)
    wos = WostVariable(data_holder, green_sampling=GreenSampling.Polynomial, use_accelaration = use_accel, opt_params = opt_params)

    shape_dummy = BoundaryWithDirichlets(out_boundary=out_boundary, dirichlet_boundaries=[CircleShape(radius = dirichlet_radius * radius)], 
                                         epsilon = e_shell, dirichlet_values = (np.ones([1, num_conf]) * dirichlet_offset).tolist())
    data_holder_dummy = DataHolder(shape_dummy, α = α)
    wos_dummy = WostVariable(data_holder_dummy, green_sampling=GreenSampling.Polynomial, use_accelaration = use_accel, opt_params = opt_params)
    
    def postprocess(opt, min_val, max_val):
        opt[opt_variable_name] = dr.clip(opt[opt_variable_name], min_val, max_val)
        
    post_process = lambda opt : postprocess(opt, bg_conductance, 1.1 * np.max(image_obj)) 

    folder0 = f"{conf_name}-{scale_name}"
    folder1 = f"{args.injectionset}"
    folder2 = dirichlet_name
    folder3 = f"{res_name}-{spe_name}-{seed_name}-{normalized_name}"
    folder4 = reg_name
    if args.kill:
        folder4 += f"-kill{kill_step}_{kill_rate}"
    path_obj = os.path.join(root_directory, "objectives", folder0, folder1)
    path = os.path.join(root_directory, folder0, folder1, folder2, folder3, folder4)
    print(path)
    create_path(path_obj)
    create_path(path)

    image_obj_ = plot_coeff(wos_obj.input.α, wos.input.shape, bbox, path, "objective", resolution = [256, 256])
    max_range = [bg_conductance, 1.1 * np.max(image_obj_)]
    
    
    obj_results = []
    for s in range(seed_obj):
        file = f"{s}.npy"
        file_el = f"elnums.npy"
        filepath = os.path.join(path_obj, file)
        filepath_el = os.path.join(path_obj, file_el)
        if not os.path.isfile(filepath):
            print(f"Generating objective results for seed {s}.")
            tensor, std, electrode_nums = compute_primals(wos_obj, split, spe_obj, 0, s,  delete_injection, 
                                                          split_depth, compute_variance, confs_iter = num_electrodes, 
                                                          num_electrodes=num_electrodes, conf_numbers = [dr.opaque(UInt32, i, ) for i in range(num_conf)])
            np.save(filepath, tensor)
            np.save(filepath_el, electrode_nums)
            if compute_variance:
                filepath_std = os.path.join(path_obj, f"{s}_std.npy")
                np.save(filepath_std, std)
        
        obj_iter = np.load(filepath, allow_pickle = True)
        obj_results.append(obj_iter)

    obj_results = np.mean(np.array(obj_results), axis = 0)
    electrode_nums = np.load(filepath_el, allow_pickle=True)


    print("Objective Results are loaded.")
    wos.input.shape.out_boundary.voltages = np.array(obj_results)
    obj_results_std = np.zeros_like(obj_results)
    wos.input.shape.out_boundary.voltages_std = obj_results_std

    
    #obj_results, obj_results_std, electrode_nums = compute_primals(wos_obj, split, spe_obj, seed_obj,  delete_injection, split_depth, compute_variance)
    #wos.input.shape.out_boundary.voltages = np.array(obj_results)
    #wos.input.shape.out_boundary.voltages_std = np.array(obj_results_std)
    if plot:
        iter_plot(wos_obj, bbox, path, "objective", obj_results, obj_results_std, electrode_nums, compute_std = False)

    optimize_eit(path = path, wos = wos, wos_obj = wos_obj, wos_dummy  = wos_dummy, split = split, 
                spe = spe, primal_spe = spe_primal, dirichlet_spe = spe_dirichlet, seed = seed, conf_per_iter = conf_per_iter, 
                max_split_depth = split_depth, num_iter = num_iter, learning_rate = learning_rate, λ_L1 = λ_L1, 
                λ_TV = λ_TV, post_process = post_process,  cond_threshold = cond_threshold, grad_threshold = grad_threshold, 
                max_dirichlet = max_dirichlet, dirichlet_radius = dirichlet_radius * radius, dirichlet_offset = dirichlet_offset, 
                merge_distance = merge_distance, normalize_grad = normalized_grad, plot = plot, bbox_plot = bbox,  
                delete_injection = delete_injection, compute_std = compute_variance, verbose = args.verbose, max_range = max_range, 
                centered_dirichlet = centered_dirichlet, kill_step = kill_step, kill_rate = kill_rate, measure_time=args.measuretime, 
                vis_confs=vis_set)
    
if __name__ == "__main__":
    main()