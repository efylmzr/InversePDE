import mitsuba as mi 
mi.set_variant("cuda_ad_rgb")
import drjit as dr
import matplotlib.pyplot as plt
import sys
from PDE2D.Coefficient import *
from PDE2D.utils import *
from PDE2D.BoundaryShape import *
from PDE2D.Solver import *
from PDE2D import PATH
import argparse
import os

root_directory = os.path.join(PATH, "output2D", "finite_differences", "discrete-sdf")

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

parser = argparse.ArgumentParser(description='''Forward mode grad computation (translation)''')
parser.add_argument('--spe', default = 23, type=int)
parser.add_argument('--seed', default = 0, type=int)
parser.add_argument('--iter', default = 512, type = int)
parser.add_argument("--upsample", default = 1, type = int)
parser.add_argument("--fdstep", default = 5e-3, type = float)
args = parser.parse_args()

spe = 2 ** args.spe
fd_step = args.fdstep
seed = args.seed

sdf_array   = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1,-1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1,-1, 1,-1,-1,-1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1,-1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

sdf_array = sdf_array.repeat(args.upsample, axis = 0).repeat(args.upsample, axis = 1)
box_length = 2.1
box_center = [0,0]
in_boundary = SDFGrid(tensor_np= sdf_array, box_length=box_length, box_center=box_center, epsilon = 1e-5, redistance = True)

out_boundary = CircleWithElectrodes(injection_confs = [[0,10]], is_delta = True)
shape = BoundaryWithDirichlets(out_boundary, [in_boundary], dirichlet_values = [[0]])
data_holder = DataHolder(shape)
wos = WostConstant(data_holder)


num_conf = out_boundary.num_confs
conf_numbers = [dr.opaque(UInt32, i, shape = (1)) for i in range(num_conf)]


filename = f"fd{fd_step}"
path = os.path.join(root_directory, "fd", filename)
create_path(path)

points, active_conf, electrode_nums = out_boundary.create_electrode_points(spe, conf_numbers=conf_numbers)

grads_x = []
grads_y = []

for i in range(args.iter):
    seed_iter = i + args.seed

    wos.change_seed(seed_iter)
    wos.input.shape.in_boundaries[0].translation_x = dr.opaque(mi.Float, fd_step, shape = (1))
    
    L_x, _ = wos.solve(points, active_conf, conf_numbers=conf_numbers, all_inside = True)
    
    wos.input.shape.in_boundaries[0].translation_x = dr.opaque(mi.Float, -fd_step, shape = (1))
    L_x_, _ = wos.solve(points, active_conf, conf_numbers=conf_numbers, all_inside = True)
    
    grad_Lx = (L_x - L_x_) / (2 * fd_step)
    grad_x = create_electrode_result(grad_Lx, spe, electrode_nums, apply_normalization = True)

    
    wos.input.shape.in_boundaries[0].translation_x = dr.opaque(mi.Float, 0, shape = (1))
    wos.input.shape.in_boundaries[0].translation_y = dr.opaque(mi.Float, fd_step, shape = (1))
    
    L_y, _ = wos.solve(points, active_conf, conf_numbers=conf_numbers, all_inside = True)
    
    wos.input.shape.in_boundaries[0].translation_y = dr.opaque(mi.Float, -fd_step, shape = (1))
    L_y_, _ = wos.solve(points, active_conf, conf_numbers=conf_numbers, all_inside = True)


    grad_Ly = (L_y - L_y_) / (2 * fd_step)
    grad_y = create_electrode_result(grad_Ly, spe, electrode_nums, apply_normalization = True)

    grad_x_np = grad_x.numpy()
    grad_y_np = grad_y.numpy()

    grads_x.append(grad_x_np)
    grads_y.append(grad_y_np)

    np.save(os.path.join(path, f"x-{seed_iter}.npy"), grad_x_np) 
    np.save(os.path.join(path, f"y-{seed_iter}.npy"), grad_y_np) 

    print(f"Iteration {i} is finished!")

grad_x = np.sum(np.array(grads_x), axis = 0) / args.iter
grad_y = np.sum(np.array(grads_y), axis = 0) / args.iter


fig, ax = plt.subplots(layout='constrained', figsize = (12,5))
plot_primals(ax, grad_x[0], np.zeros_like(grad_x[0]), electrode_nums, 16, name1 = "grad_x", name2 = "-")
fig.savefig(f"{path}/grad_x.pdf", bbox_inches = "tight", dpi = 300)
plt.close(fig)

fig, ax = plt.subplots(layout='constrained', figsize = (12,5))
plot_primals(ax, grad_y[0], np.zeros_like(grad_y[0]), electrode_nums, 16, name1 = "grad_y", name2 = "-")
fig.savefig(f"{path}/grad_y.pdf", bbox_inches = "tight", dpi = 300)
plt.close(fig)
