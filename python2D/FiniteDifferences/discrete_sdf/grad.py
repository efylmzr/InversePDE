import mitsuba as mi 
mi.set_variant("cuda_ad_rgb_double")
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
parser.add_argument('--spe', default = 25, type=int)
parser.add_argument("--seednormal", default = 0, type = int)
parser.add_argument("--iternormal", default = 256, type = int)
parser.add_argument("--resnormal", default = 1024, type = int)
parser.add_argument("--distance", default = 0.01, type = float)
parser.add_argument("--upsample", default = 1, type = int)
parser.add_argument("--epsilon", default = 5e-6, type = float)
args = parser.parse_args()

distance = args.distance
spe = 2 ** args.spe

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
in_boundary = SDFGrid(tensor_np= sdf_array, box_length=box_length, box_center=box_center, epsilon = args.epsilon, redistance = True)

out_boundary = CircleWithElectrodes(injection_confs = [[0,10]], is_delta = True, epsilon=args.epsilon)
shape = BoundaryWithDirichlets(out_boundary, [in_boundary], dirichlet_values = [[0]], epsilon = args.epsilon)
data_holder = DataHolder(shape)
opt_params = ["inboundary.dirichlet.translation_x", "inboundary.dirichlet.translation_y"]
wos = WostConstant(data_holder, opt_params = opt_params)


normalder_file = os.path.join(root_directory, "normalder")
normal_der = []

for i in range(args.seednormal, args.seednormal + args.iternormal):
    normal_der.append(np.load(os.path.join(normalder_file, f"res{args.resnormal}-d{distance}-{i}.npy")))
normal_der = np.array(normal_der).sum(axis = 0) / args.iternormal

num_conf = out_boundary.num_confs
normal_der = mi.TensorXf(normal_der)

print(normal_der.shape)
wos.input.shape.in_boundaries[0].set_normal_derivative(normal_der)

conf_numbers = [dr.opaque(UInt32, i, shape = (1)) for i in range(num_conf)]

path = os.path.join(root_directory, "prb")
create_path(path)


points, active_conf, electrode_nums = out_boundary.create_electrode_points(spe, conf_numbers=conf_numbers)
grad_x = mi.Float(0)
grad_y = mi.Float(0)

dr.disable_grad(wos.input.shape.in_boundaries[0].translation_y)
dr.enable_grad(wos.input.shape.in_boundaries[0].translation_x)
dr.forward(wos.input.shape.in_boundaries[0].translation_x)

dL_x, _ = wos.solve(points, active_conf, conf_numbers=conf_numbers, all_inside = True, 
                    normal_derivative_dist=distance, mode = dr.ADMode.Forward)

grad_x += create_electrode_result(dL_x, spe, electrode_nums, apply_normalization = True)

dr.disable_grad(wos.input.shape.in_boundaries[0].translation_x)
dr.enable_grad(wos.input.shape.in_boundaries[0].translation_y)
dr.forward(wos.input.shape.in_boundaries[0].translation_y)

dL_y, _ = wos.solve(points, active_conf, conf_numbers=conf_numbers, 
                    all_inside = True, normal_derivative_dist=distance, mode = dr.ADMode.Forward)

grad_y += create_electrode_result(dL_y, spe, electrode_nums, apply_normalization = True)

np.save(os.path.join(path, f"gradx-d{distance}.npy"), grad_x.numpy()) 
np.save(os.path.join(path, f"grady-d{distance}.npy"), grad_y.numpy()) 

fig, ax = plt.subplots(layout='constrained', figsize = (12,5))
plot_primals(ax, grad_x[0], np.zeros_like(grad_x[0]), electrode_nums, 16, name1 = "grad_x", name2 = "-")
fig.savefig(f"{path}/grad_x-d{distance}.pdf", bbox_inches = "tight", dpi = 300)
plt.close(fig)

fig, ax = plt.subplots(layout='constrained', figsize = (12,5))
plot_primals(ax, grad_y[0], np.zeros_like(grad_y[0]), electrode_nums, 16, name1 = "grad_y", name2 = "-")
fig.savefig(f"{path}/grad_y-d{distance}.pdf", bbox_inches = "tight", dpi = 300)
plt.close(fig)
