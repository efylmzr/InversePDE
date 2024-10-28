import argparse
import mitsuba as mi
import drjit as dr
mi.set_variant("cuda_ad_rgb")
import numpy as np
import sys
from PDE2D.Solver import WostConstant, DataHolder
from python2D.optimizations.eit_discrete.configurations import data
from python2D.optimizations.eit_discrete.optimize import *
from PDE2D.BoundaryShape import *
import os
sys.path.append("../../../")

from PDE2D import PATH

root_directory = f"{PATH}/output2D/optimizations/discrete-eit"

def main():
    parser = argparse.ArgumentParser(description='''Optimization Sphere''')
    parser.add_argument('--spe', default = 18, type=int)
    parser.add_argument('--objspe', default = 20, type=int) 
    parser.add_argument('--objseed', default = 32, type=int)
    parser.add_argument('--seed', default = 543, type=int)
    parser.add_argument("--distder", default = "5e-2", type = float)
    parser.add_argument('--resnormalder', default = 128, type=int)
    parser.add_argument('--sppnormalder', default = 18, type=int)
    parser.add_argument('--confiter', default = 4, type=int)
    parser.add_argument('--iternum', default = 256, type=int)
    parser.add_argument("--lr", default = "0.025", type = float)
    parser.add_argument("--epsilon", default = "1e-5", type = float)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--injectionset", default = "skip2", type = str)
    parser.add_argument("--visset", default = "none", type = str)
    parser.add_argument("--conf", default = 1, type = int)
    parser.add_argument("--measuretime", action = "store_true")
    parser.add_argument("--sdf", action = "store_true")
    parser.add_argument("--sdfres", type = int, default = 64)
    args = parser.parse_args()

    is_sdf = args.sdf
    e_shell = args.epsilon
    delete_injection = True
    #normalized_grad = args.normalizedgrad
    plot = args.plot

    seed_obj = args.objseed
    seed = args.seed
    dist_normalder = args.distder
    spp_normalder = 2 ** args.sppnormalder
    res_normalder = args.resnormalder

    spe_obj = 2 ** args.objspe
    spe = 2 ** args.spe

    num_iter = args.iternum
    learning_rate = args.lr
    conf = args.conf  

    if is_sdf:
        sdf_begin = data["begin-sdf"].repeat(args.sdfres / 16, axis = 0).repeat(args.sdfres / 16, axis = 1)
        sdf_obj = data[f"obj-sdf{conf}"]
        in_boundary = SDFGrid(tensor_np= sdf_begin, box_length=2.1, box_center=[0,0], 
                              epsilon = args.epsilon, redistance = True, name = "inboundary")
        in_boundary_obj = SDFGrid(tensor_np= sdf_obj, box_length=2.1, box_center=[0,0], 
                              epsilon = args.epsilon, redistance = True, name = "inboundaryobj")
        opt_params = ["inboundary.dirichlet.tensor"]
        
        def postprocess(opt, radius, box_length, box_center, bbox, resolution):
            shape1 = CircleShape(radius = radius)
            shape1 = shape1.generate_sdf_grid(resolution =resolution, box_length = box_length, box_center = box_center, redistance = False)
            tensor = opt["inboundary.dirichlet.tensor"].numpy().squeeze()
            shape2 = SDFGrid(tensor_np=tensor, box_length= box_length , box_center = box_center, redistance = False)
            new_tensor = get_intersection_tensor(shape1, shape2, tensor.shape, bbox).squeeze()
            opt["inboundary.dirichlet.tensor"] = mi.TensorXf(new_tensor[...,np.newaxis])
    
        post_process = lambda opt : postprocess(opt, 0.97, in_boundary.box_length, in_boundary.box_center, 
                                                            in_boundary.bbox, in_boundary.resolution) 


    else:
        origin = data[f"origin-conf{conf}"]
        radius = data[f"radius-conf{conf}"]
        origin_obj = data[f"obj-origin-conf{conf}"]
        radius_obj = data[f"obj-radius-conf{conf}"]
        in_boundary_obj = CircleShape(origin_obj, radius_obj, name = "inboundaryobj")
        in_boundary = CircleShape(origin, radius, name = "inboundary")
        opt_params = ["inboundary.dirichlet.origin", "inboundary.dirichlet.radius"]
        def postprocess(opt, min_radius, max_region):
            opt["inboundary.dirichlet.radius"] = dr.select(opt["inboundary.dirichlet.radius"] < min_radius, min_radius, opt["inboundary.dirichlet.radius"])
            origin_dir = dr.normalize(opt["inboundary.dirichlet.origin"])
            diff = dr.norm(opt["inboundary.dirichlet.origin"]) + opt["inboundary.dirichlet.radius"] - max_region
            opt["inboundary.dirichlet.origin"] = dr.select(diff >= 0, (max_region - opt["inboundary.dirichlet.radius"]) * origin_dir, opt["inboundary.dirichlet.origin"])
        post_process = lambda opt : postprocess(opt, 0.03 * out_radius, (1 - 0.0001) * out_radius) 

    out_radius = 1
    bbox_plot = [[-1.05 * out_radius, -1.05 * out_radius],[1.05 * out_radius, 1.05 * out_radius]]
    
    out_boundary = CircleWithElectrodes(radius = out_radius, injection_set = args.injectionset, is_delta = True)
    shape_obj = BoundaryWithDirichlets(out_boundary, [in_boundary_obj], dirichlet_values = [[0]], epsilon=e_shell)
    shape = BoundaryWithDirichlets(out_boundary, [in_boundary], dirichlet_values = [[0]], epsilon=e_shell)
    data_holder_obj = DataHolder(shape_obj)
    data_holder = DataHolder(shape)
    
    wos_obj = WostConstant(data_holder_obj, seed = args.objseed)
    wos = WostConstant(data_holder, opt_params = opt_params)

    num_conf = out_boundary.num_confs
    # The primal configurations that are going to be evaluated at each iteration for visualization.
    vis_set = []
    if args.visset != "none":
        vis_set = out_boundary.get_injection_confs(args.injectionset, args.visset, num_electrodes=16)


    folder_name0 = "sdf" if is_sdf else "circle"
    folder_name1 = f"config{conf}-{args.injectionset}"
    folder_name2 = f"distder{dist_normalder}-confperiter{args.confiter}"
    folder_name3 = f"spe{args.spe}-normalspp{args.sppnormalder}"
    path = os.path.join(root_directory, folder_name0, folder_name1, folder_name2, folder_name3)
    path_obj = os.path.join(root_directory, "objectives", folder_name0, folder_name1)

    def create_path(path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)

    create_path(path)
    create_path(path_obj)

    obj_results = []
    for s in range(seed_obj):
        file = f"{s}.npy"
        file_el = f"elnums.npy"
        filepath = os.path.join(path_obj, file)
        filepath_el = os.path.join(path_obj, file_el)
        if not os.path.isfile(filepath):
            print(f"Generating objective results for seed {s}.")
            tensor, electrode_nums = compute_primals(wos_obj, spe_obj, s,  delete_injection, confs_iter=16, 
                                                     num_electrodes=16, conf_numbers = [dr.opaque(UInt32, i, ) for i in range(num_conf)])
            np.save(filepath, tensor)
            np.save(filepath_el, electrode_nums)
        
        obj_iter = np.load(filepath, allow_pickle = True)
        obj_results.append(obj_iter)

    obj_results = np.mean(np.array(obj_results), axis = 0)
    electrode_nums = np.load(filepath_el, allow_pickle=True)

    print("Objective Results are loaded.")
    wos.input.shape.out_boundary.voltages = np.array(obj_results)


    optimize_eit(path, wos, wos_obj, spe, spe, seed, args.confiter,
                res_normalder, spp_normalder, dist_normalder,
                num_iter, learning_rate, post_process,
                plot, bbox_plot = bbox_plot,  
                delete_injection = delete_injection,
                measure_time = args.measuretime, 
                vis_confs = vis_set, is_sdf = is_sdf)


if __name__ == "__main__":
    main()