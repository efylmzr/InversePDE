import argparse
import numpy as np
import mitsuba as mi
import drjit as dr
mi.set_variant("cuda_ad_rgb")
import os
from PDE3D.BoundaryShape import *
from PDE3D.Coefficient import *
from PDE3D.utils import *
from PDE3D.Solver import *
from PDE3D import PATH
from python3D.optimization.textures import *
from python3D.optimization.sketch import *
from python3D.optimization.optimize import *



root_directory = os.path.join(PATH, "output3D", "optimizations")
def main():
    parser = argparse.ArgumentParser(description='''Optimization Sphere''')
    parser.add_argument('--spp', default = 11, type=int)
    parser.add_argument('--primalspp', default = 13, type=int)
    parser.add_argument('--objspp', default = 13, type=int)
    parser.add_argument('--seedobj', default = 64, type=int)
    parser.add_argument('--seed', default = 243, type=int)
    parser.add_argument('--confiter', default = 6, type=int)
    parser.add_argument('--iternum', default = 512, type=int)
    parser.add_argument("--lr", default = "0.1", type = float)
    parser.add_argument("--epsilon", default = "1e-2", type = float)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--split", default = "normal", type = str)
    parser.add_argument("--splitdepth", default=250, type=int)
    parser.add_argument("--computevariance", action = "store_true")
    parser.add_argument("--scaletexture", default = 1.0, type=float)
    parser.add_argument("--biastexture", default = 0.0, type=float)
    parser.add_argument("--conf", default = "1", type = int)
    parser.add_argument("--verbose", action = "store_true")
    parser.add_argument("--stepnum", default = 1, type = int)
    parser.add_argument("--averagepixel", action = "store_true")
    parser.add_argument("--coeff", default = "source", type = str)
    parser.add_argument("--screening", default = 0, type = float)
    parser.add_argument("--measuretime", action = "store_true")
    parser.add_argument("--constantboundary", action = "store_true")
    parser.add_argument('--resprimal', default = 16, type=int)
    parser.add_argument('--restensor', default = 16, type=int)
    parser.add_argument('--visconf', default = -1, type = int)
    args = parser.parse_args()
    
    res_primal = [args.resprimal,args.resprimal,args.resprimal]
    res_tensor = [args.restensor,args.restensor,args.restensor]

   
    step_num = args.stepnum
    centered = not args.averagepixel
    bbox = [[-1,-1],[1, 1]]
    compute_variance = args.computevariance
    split_depth = args.splitdepth
    e_shell = args.epsilon
    plot = args.plot
    seed_obj = args.seedobj
    seed = args.seed
    spp_obj = 2 ** args.objspp
    spp = 2 ** args.spp
    primal_spp = 2 ** args.primalspp
    conf_per_iter = args.confiter
    num_iter = args.iternum
    learning_rate = args.lr
    bias = args.biastexture
    scale = args.scaletexture
    coeff_name = args.coeff
    screening = args.screening
    constant_boundary = args.constantboundary
    vis_conf = []
    if args.visconf >= 0:
        vis_conf = [dr.opaque(mi.UInt32, args.visconf, shape = (1))]

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
    

    conf_name = f"conf{args.conf}"
    centered_name = "centered" if centered else "avg"
    res_name = f"res{res_primal[0]}"
    screening_name = f"screen{screening}"
    spp_name = f"spp{args.primalspp}_{args.spp}"
    restensor_name = f"restensor{res_tensor[0]}"
    seed_name = f"seed{seed}"
    e_name = f"epsilon{args.epsilon}"
    scale_name = f"scale{scale}-bias{bias}"

    dirichlet = load_boundary_data(constant_boundary)
    
    name = "motorbike-engine"
    folder_name = os.path.join(PATH, "scenes", name)
    xml_name = os.path.join(folder_name, "scene.xml")
    sdf_data = np.load(os.path.join(folder_name, "sdf.npy"))
    boundary = SDF(sdf_data, dirichlet = dirichlet, scale = 12)
    
    obj_name = f"{coeff_name}-{conf_name}"

    bbox = boundary.bbox
    bbox_pad = (bbox.max - bbox.min) / 10
    bbox_coeff = mi.ScalarBoundingBox3f(bbox.min - bbox_pad, bbox.max + bbox_pad)
    image_obj = textures[int(args.conf - 1)]() * scale + bias

    image_begin = np.zeros(res_tensor)
    #image_begin[int(res_tensor[0] * 3 / 8) : int(res_tensor[0] * 5 / 8), 
    #            int(res_tensor[1] * 3 / 8) : int(res_tensor[1] * 5 / 8),
    #            int(res_tensor[2] * 3 / 8) : int(res_tensor[2] * 5 / 8)] = 0.1
    image_begin *= scale
    image_begin += bias

    if coeff_name == "diffusion":
        α_obj = TextureCoefficient("diffusion", bbox_coeff, image_obj)
        α = TextureCoefficient("diffusion", bbox_coeff, image_begin)
        σ = ConstantCoefficient("screening", screening)
        data_holder_obj = DataHolder(boundary, α = α_obj, σ = σ)
        data_holder = DataHolder(boundary, α = α, σ = σ)
    elif coeff_name == "screening":
        σ_obj = TextureCoefficient("screening", bbox_coeff, image_obj)
        σ = TextureCoefficient("screening", bbox_coeff, image_begin)
        data_holder_obj = DataHolder(boundary, σ = σ_obj)
        data_holder = DataHolder(boundary, σ = σ)
    elif coeff_name == "source":
        f_obj = TextureCoefficient("source", bbox_coeff, image_obj)
        f = TextureCoefficient("source", bbox_coeff, image_begin)
        σ = ConstantCoefficient("screening", screening)
        data_holder_obj = DataHolder(boundary, f = f_obj, σ = σ)
        data_holder = DataHolder(boundary, f = f, σ = σ)

    opt_variable_name = f"{coeff_name}.texture.tensor"
    wos_obj = WosVariable(data_holder_obj)

    opt_params = [opt_variable_name]
    wos = WosVariable(data_holder, opt_params = opt_params)

    input_range = get_range(boundary, bbox_coeff, image_obj)

    def postprocess(opt, min_val, max_val):
        opt[opt_variable_name] = dr.clip(opt[opt_variable_name], min_val, max_val)
        
    post_process = lambda opt : postprocess(opt, input_range[0] * 0.8, 1.2 * (input_range[1] - input_range[0]) + input_range[0]) 

    def create_path(path):
        if not os.path.exists(path):
            os.makedirs(path)

    folder0 = f"{conf_name}-{coeff_name}"
    folder0 += "-constDirichlet" if constant_boundary else "" 
    folder1 = f"{res_name}-{scale_name}-{centered_name}"
    folder1 += f"-{screening_name}" if coeff_name!="screening" else ""
    folder2 = f"{spp_name}-{seed_name}-{e_name}"
    folder3 = f"{restensor_name}"
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
    plot_coeff(coeff_obj, wos.input.shape, input_range, path_obj, "objective")

    obj_results = []
    for s in range(seed_obj):
        file = f"{s}.npy"
        filepath = os.path.join(path_obj, file)
        if not os.path.isfile(filepath):
            print(f"Generating objective results for seed {s}.")
            tensor, std = compute_primals(wos_obj, Split.Agressive, s, bbox, res_primal, spp_obj, centered, split_depth, compute_variance, confs_iter = 16)
            np.save(filepath, tensor)
            if compute_variance:
                filepath_std = os.path.join(path_obj, f"{s}_std.npy")
                np.save(filepath_std, std)
        obj_iter = np.load(filepath, allow_pickle = True)
        obj_results.append(obj_iter)

    obj_results = np.mean(np.array(obj_results), axis = 0)

    print("Objective Results are loaded.")

    wos = optimize(path, wos, wos_obj, obj_results, coeff_name, split, bbox, res_primal, spp, primal_spp, seed, conf_per_iter, split_depth,
                   num_iter, learning_rate, post_process, centered, plot, compute_variance, args.verbose, input_range,
                   measure_time=args.measuretime, vis_set = vis_conf)
    

    #if step_num == 1:
    #    wos = optimize_variable(path, wos, wos_obj, obj_results, coeff_name, split, bbox, resolution_primal, spp, primal_spp, seed, conf_per_iter, split_depth,
    #                            num_iter, learning_rate, post_process, centered, plot, compute_variance, args.verbose, max_range, out_val = bias,
    #                            measure_time=args.measuretime)
    #else:
    #    for i in range(step_num):
    #        path_iter = os.path.join(path, f"step{i}")
    #        create_path(path_iter)
    #        optimize_variable(path_iter, wos, wos_obj, obj_results, coeff_name, split, bbox, resolution_primal, spp, primal_spp, seed, conf_per_iter, split_depth,
    #                          num_iter, learning_rate, post_process, centered, plot, compute_variance, args.verbose, max_range, out_val = bias)
            #wos.input.upsample2(coeff_name)
            #wos.get_opt_params(wos.opt_params, opt_params)
            #print("The input tensor is upsampled, another optimization scheme starts.")
            
    
    
if __name__ == "__main__":
    main()