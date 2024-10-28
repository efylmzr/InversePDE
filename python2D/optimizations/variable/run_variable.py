import argparse
import mitsuba as mi 
mi.set_variant("cuda_ad_rgb")
import numpy as np
import os
from python2D.optimizations.variable.textures_variable import *
from python2D.optimizations.variable.optimize_variable import *
from python2D.optimizations.sketch import *
from PDE2D.utils import *
from PDE2D.BoundaryShape import *
from PDE2D.Solver import *
from PDE2D import PATH, GreenSampling, Split



root_directory = os.path.join(PATH, "output2D", "optimizations", "variable")
def main():
    parser = argparse.ArgumentParser(description='''Optimization Sphere''')
    parser.add_argument('--spp', default = 12, type=int)
    parser.add_argument('--primalspp', default = 14, type=int)
    parser.add_argument('--objspp', default = 14, type=int)
    parser.add_argument('--seedobj', default = 16, type=int)
    parser.add_argument('--seed', default = 243, type=int)
    parser.add_argument('--confiter', default = 6, type=int)
    parser.add_argument('--iternum', default = 512, type=int)
    parser.add_argument("--lr", default = "0.1", type = float)
    parser.add_argument("--epsilon", default = "1e-2", type = float)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--split", default = "normal", type = str)
    parser.add_argument("--noaccel", action="store_true")
    parser.add_argument("--splitdepth", default=250, type=int)
    parser.add_argument("--computevariance", action = "store_true")
    parser.add_argument("--regularization", default = "none", type = str)
    parser.add_argument("--regL", default = "0.01", type = float)
    parser.add_argument("--scaletexture", default = 1.0, type=float)
    parser.add_argument("--biastexture", default = 0.0, type=float)
    parser.add_argument("--conf", default = "1", type = str)
    parser.add_argument("--verbose", action = "store_true")
    parser.add_argument("--stepnum", default = 1, type = int)
    parser.add_argument("--res", default = 32, type = float)
    parser.add_argument("--averagepixel", action = "store_true")
    parser.add_argument("--coeff", default = "source", type = str)
    parser.add_argument("--dirichlet", action = "store_true")
    parser.add_argument("--restensor", default = 16, type = int)
    parser.add_argument("--screening", default = 0, type = float)
    parser.add_argument("--measuretime", action = "store_true")
    parser.add_argument("--confboundary", type = int, default = -1)
    parser.add_argument("--zeroboundary", action = "store_true")
    parser.add_argument("--visconf", default = -1, type = int)

    args = parser.parse_args()
    step_num = args.stepnum
    centered = not args.averagepixel
    bbox = [[-1,-1],[1, 1]]
    compute_variance = args.computevariance
    use_accel = not args.noaccel
    split_depth = args.splitdepth
    e_shell = args.epsilon
    plot = args.plot
    seed_obj = args.seedobj
    seed = args.seed
    res_primal = int(args.res)
    resolution_primal = [res_primal, res_primal]
    spp_obj = 2 ** args.objspp
    spp = 2 ** args.spp
    primal_spp = 2 ** args.primalspp
    conf_per_iter = args.confiter
    num_iter = args.iternum
    learning_rate = args.lr
    λ = args.regL
    bias = args.biastexture
    scale = args.scaletexture
    coeff_name = args.coeff
    only_dirichlet = args.dirichlet
    res_tensor = args.restensor
    resolution_tensor = [res_tensor, res_tensor]
    screening = args.screening

    if coeff_name == "diffusion" and bias == 0:
        raise Exception("Please give a positive bias value.")
    
    
    if args.split == "none":
        split = Split.Naive
    elif args.split == "agressive":
        split = Split.Agressive
    elif args.split == "normal":
        split = Split.Normal
    else:
        raise Exception("No such split is defined.")
    

    if args.regularization == "none":
        regularization = RegularizationType.none
    elif args.regularization == "L2":
        regularization = RegularizationType.L2
    elif args.regularization == "tensorL2":
        regularization = RegularizationType.tensoL2
    elif args.regularization == "L1":
        regularization = RegularizationType.L1
    elif args.regularization == "tensorL1":
        regularization = RegularizationType.tensorL1
    elif args.regularization == "TV":
        regularization = RegularizationType.TV
    elif args.regularization == "gradL1":
        regularization = RegularizationType.gradL1
    elif args.regularization == "gradL2":
        regularization = RegularizationType.gradL2
    elif args.regularization == "screeningL1":
        regularization = RegularizationType.screeningL1
    elif args.regularization == "screeningL2":
        regularization = RegularizationType.screeningL2
    else:
        raise Exception("No such regularization is defined.")
    
    conf_name = f"conf{args.conf}"
    centered_name = "centered" if centered else "avg"
    res_name = f"res{args.res}"
    screening_name = f"screen{screening}"
    spp_name = f"spp{args.primalspp}_{args.spp}"
    restensor_name = f"restensor{res_tensor}"
    seed_name = f"seed{seed}"
    e_name = f"epsilon{args.epsilon}"
    reg_name = "none" if args.regularization == "none" else f"{args.regularization}-{λ}"
    scale_name = f"scale{scale}-bias{bias}"
    boundary_name = "dirichlet" if only_dirichlet else "mixed"

    if args.visconf < 0:
        vis_confs = []
    else: 
        vis_confs = [dr.opaque(UInt32, args.visconf, shape =(1))]


    dirichlet, neumann = load_boundary_data(only_dirichlet | (coeff_name == "diffusion"), zero = (coeff_name == "source"))
    #dirichlet = [ConstantCoefficient("dirichlet", 0), ConstantCoefficient("dirichlet", 0.1)]
    #neumann = [ConstantCoefficient("neumann", 0)]
    if args.confboundary == -1:
        if coeff_name == "source":
            boundary_conf = 1
        elif coeff_name == "screening":
            boundary_conf = 2
        elif coeff_name == "diffusion":
            boundary_conf = 3
    else:
        boundary_conf = args.confboundary
    
    boundary = load_bunny(dirichlet = dirichlet, neumann = neumann, all_dirichlet = only_dirichlet, epsilon=e_shell, conf = boundary_conf)
    
    obj_name = f"{coeff_name}-{conf_name}"
    image_obj = objectives[obj_name]

    image_obj *= scale
    image_obj += bias

    
    image_begin = np.zeros(resolution_tensor)
    #image_begin[int(res_tensor * 3 / 8) : int(res_tensor * 5 / 8), 
    #            int(res_tensor * 3 / 8) : int(res_tensor * 5 / 8)] = 0.1
    image_begin *= scale
    image_begin += bias

    if coeff_name == "diffusion":
        if args.zeroboundary:
            grad_points = boundary.create_boundary_points(resolution = 64, spp = 2)
        elif not only_dirichlet:
            grad_points = boundary.create_neumann_points(resolution = 64, spp = 2)
        else:
            grad_points = None
        α_obj = TextureCoefficient("diffusion", bbox, image_obj, grad_zero_points=grad_points, out_val = bias)
        α = TextureCoefficient("diffusion", bbox, image_begin, grad_zero_points=grad_points, out_val = bias)
        σ = ConstantCoefficient("screening", screening)
        data_holder_obj = DataHolder(boundary, α = α_obj, σ = σ)
        data_holder = DataHolder(boundary, α = α, σ = σ)
    elif coeff_name == "screening":
        σ_obj = TextureCoefficient("screening", bbox, image_obj, out_val = bias)
        σ = TextureCoefficient("screening", bbox, image_begin, out_val = bias)
        data_holder_obj = DataHolder(boundary, σ = σ_obj)
        data_holder = DataHolder(boundary, σ = σ)
    elif coeff_name == "source":
        f_obj = TextureCoefficient("source", bbox, image_obj, out_val = bias)
        f = TextureCoefficient("source", bbox, image_begin, out_val = bias)
        σ = ConstantCoefficient("screening", screening)
        data_holder_obj = DataHolder(boundary, f = f_obj, σ = σ)
        data_holder = DataHolder(boundary, f = f, σ = σ)


    opt_variable_name = f"{coeff_name}.texture.tensor"
    wos_obj = WostVariable(data_holder_obj, green_sampling=GreenSampling.Polynomial, use_accelaration = use_accel)

    opt_params = [opt_variable_name]
    wos = WostVariable(data_holder, green_sampling=GreenSampling.Polynomial, use_accelaration = use_accel, opt_params = opt_params)

    
    def postprocess(opt, min_val, max_val):
        opt[opt_variable_name] = dr.clip(opt[opt_variable_name], min_val, max_val)
        
    post_process = lambda opt : postprocess(opt, bias, 1.2 * np.max(image_obj)) 

    folder0 = f"{conf_name}-{coeff_name}"
    if args.confboundary != -1:
        folder0 += f"b{boundary_conf}"
    folder1 = f"{boundary_name}-{res_name}-{scale_name}-{centered_name}"
    folder1 += f"-{screening_name}" if coeff_name!="screening" else ""
    folder2 = f"{spp_name}-{seed_name}-{e_name}"
    folder3 = f"{restensor_name}-{reg_name}"
    folder3 += f"-stepnum{step_num}" if step_num > 1 else ""
    path_obj = os.path.join(root_directory, "objectives", folder0, folder1)
    path = os.path.join(root_directory, folder0, folder1, folder2, folder3)
    print(path)
    create_path(path_obj)
    create_path(path)

    create_path(os.path.join(path, "npy", "primal"))
    create_path(os.path.join(path, "npy", "grad"))
    create_path(os.path.join(path, "npy", "tensor"))

    coeff_obj = wos_obj.input.get_coefficient(coeff_name)
    image_obj_ = plot_coeff(coeff_obj, wos.input.shape, bbox, path_obj, "objective", resolution = [256, 256], out_val = bias)
    image_obj_ = plot_coeff(coeff_obj, wos.input.shape, bbox, path, "objective", resolution = [256, 256], out_val = bias)

    obj_results = []
    for s in range(seed_obj):
        file = f"{s}.npy"
        filepath = os.path.join(path_obj, file)
        if not os.path.isfile(filepath):
            print(f"Generating objective results for seed {s}.")
            tensor, std = compute_primals(wos_obj, Split.Agressive, s, bbox, resolution_primal, spp_obj, centered, split_depth, compute_variance, confs_iter = 16)
            np.save(filepath, tensor)
            if compute_variance:
                filepath_std = os.path.join(path_obj, f"{s}_std.npy")
                np.save(filepath_std, std)
        obj_iter = np.load(filepath, allow_pickle = True)
        obj_results.append(obj_iter)

    obj_results = np.mean(np.array(obj_results), axis = 0)
    max_range = [bias, 1.1 * np.max(image_obj_)]

    print("Objective Results are loaded.")
    if plot:
        iter_plot(wos_obj, bbox, path, "objective", compute_std = compute_variance, out_val = bias, opt_param = coeff_name)

    wos = optimize_variable(path, wos, wos_obj, obj_results, coeff_name, split, bbox, resolution_primal, spp, primal_spp, seed, conf_per_iter, split_depth,
                            num_iter, learning_rate, regularization, λ, post_process, centered, plot, compute_variance, args.verbose, max_range, out_val = bias,
                            measure_time=args.measuretime, vis_confs=vis_confs)
            
    
    
if __name__ == "__main__":
    main()