import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
import drjit as dr
import matplotlib.pyplot as plt
import sys
from PDE2D.Coefficient import *
from PDE2D.utils import *
from PDE2D.BoundaryShape import *
from PDE2D.Solver import *
from PDE2D import GreenSampling, Split,  PATH
import argparse
import os

root_directory = os.path.join(PATH, "output2D", "finite_differences", "discrete-circle", "fd-vals")

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

parser = argparse.ArgumentParser(description='''FD-computation sphere''')
parser.add_argument('--spe', default = 22, type=int)
parser.add_argument('--fdstep', default = 5e-3, type = float)
parser.add_argument("--objx", default = -0.4, type = float)
parser.add_argument("--objy", default = -0.3, type = float)
parser.add_argument("--objr", default = 0.3, type = float)
parser.add_argument("--objseed", default = 222, type = int)
parser.add_argument("--x", default = 0.2, type = float)
parser.add_argument("--y", default = 0.2, type = float)
parser.add_argument("--radius", default = 0.2, type = float)
parser.add_argument("--seed", default = 0, type = int)
parser.add_argument("--iter", default = 512, type = int)
parser.add_argument("--injection", default = "skip3", type = str)

args = parser.parse_args()

origin_obj = [args.objx, args.objy]
radius_obj = args.objr

origin = [args.x, args.y]
radius = args.radius
fd_step = args.fdstep

in_shape_obj = CircleShape(origin_obj, radius_obj)
in_shape = CircleShape(origin, radius)
out_boundary = CircleWithElectrodes(injection_set = args.injection, is_delta = True)

shape_obj = BoundaryWithDirichlets(out_boundary, [in_shape_obj], dirichlet_values = [[0]])
data_holder_obj = DataHolder(shape_obj)
wos_obj = WostConstant(data_holder_obj, seed = args.objseed)

num_conf = out_boundary.num_confs
#print(num_conf)
conf_numbers = [dr.opaque(UInt32, i, shape = (1)) for i in range(num_conf)]
points, active_conf, electrode_nums = out_boundary.create_electrode_points(2 ** args.spe, conf_numbers=conf_numbers)


L_obj, _ = wos_obj.solve(points, active_conf, all_inside = True, conf_numbers=conf_numbers)
elvals_obj = create_electrode_result(L_obj, 2 ** args.spe, electrode_nums, apply_normalization=False)

filename = f"{args.injection}-ref-{args.objx}-{args.objy}-{args.objr}-current-{args.x}-{args.y}-{radius}-fd{fd_step}"

path = os.path.join(root_directory, filename)
create_path(path)

seed = args.seed
for iter in range(args.iter):
    print(f"Iteration {iter} finished.")
    seed_iter = iter + args.seed
    for name in ["x", "y", "r"]:
        in_boundary1, in_boundary2 = in_shape.move_circle_fd(fd_step, name)
        shape1 = BoundaryWithDirichlets(out_boundary, [in_boundary1], dirichlet_values = [[0]])
        shape2 = BoundaryWithDirichlets(out_boundary, [in_boundary2], dirichlet_values = [[0]])
        data_holder1 = DataHolder(shape1)
        data_holder2 = DataHolder(shape2)
        wos1 = WostConstant(data_holder1, seed = seed_iter)
        wos2 = WostConstant(data_holder2, seed = seed_iter)

        L1, _ = wos1.solve(points, active_conf, all_inside = True, conf_numbers=conf_numbers)
        elvals1 = create_electrode_result(L1, 2 ** args.spe, electrode_nums, apply_normalization=False)
        L2, _ = wos2.solve(points, active_conf, all_inside = True, conf_numbers=conf_numbers)
        elvals2 = create_electrode_result(L2, 2 ** args.spe, electrode_nums, apply_normalization=False)
        val1 = MSE(elvals1, elvals_obj).numpy()
        val2 = MSE(elvals2, elvals_obj).numpy()
        #val1 = MSE(elvals1).numpy()
        #val2 = MSE(elvals2).numpy()
        np.save(os.path.join(path, f"{name}-{args.objseed}-{seed_iter}"), ((val1 - val2) / fd_step).sum(axis = 1))