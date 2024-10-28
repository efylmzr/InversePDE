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

root_directory = os.path.join(PATH, "output2D", "finite_differences", "discrete-circle")

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

parser = argparse.ArgumentParser(description='''FD-computation sphere''')
parser.add_argument('--spe', default = 20, type=int)
parser.add_argument("--objx", default = -0.4, type = float)
parser.add_argument("--objy", default = -0.3, type = float)
parser.add_argument("--objr", default = 0.3, type = float)
parser.add_argument("--objseed", default = 222, type = int)
parser.add_argument("--x", default = 0.2, type = float)
parser.add_argument("--y", default = 0.2, type = float)
parser.add_argument("--radius", default = 0.2, type = float)
parser.add_argument("--seednormal", default = 0, type = int)
parser.add_argument("--iternormal", default = 32, type = int)
parser.add_argument("--resnormal", default = 256, type = int)
parser.add_argument("--injection", default = "skip3", type = str)
parser.add_argument("--distance", default = 0.01, type = float)
args = parser.parse_args()


origin_obj = [args.objx, args.objy]
radius_obj = args.objr
origin = [args.x, args.y]
radius = args.radius
distance = args.distance
spe = 2 ** args.spe

in_boundary_obj = CircleShape(origin_obj, radius_obj, name = "inboundaryobj")
in_boundary = CircleShape(origin, radius, name = "inboundary")
out_boundary = CircleWithElectrodes(injection_set = args.injection, is_delta = True)
shape_obj = BoundaryWithDirichlets(out_boundary, [in_boundary_obj], dirichlet_values = [[0]])
shape = BoundaryWithDirichlets(out_boundary, [in_boundary], dirichlet_values = [[0]])
data_holder_obj = DataHolder(shape_obj)
data_holder = DataHolder(shape)
opt_params = ["inboundary.dirichlet.origin", "inboundary.dirichlet.radius"]
wos_obj = WostConstant(data_holder_obj, seed = args.objseed)
wos = WostConstant(data_holder, opt_params = opt_params)


normalder_file = os.path.join(root_directory, "normalder", f"{args.injection}-{args.x}-{args.y}-{radius}")
normal_der = []

for i in range(args.seednormal, args.seednormal + args.iternormal):
    normal_der.append(np.load(os.path.join(normalder_file, f"res{args.resnormal}-d{distance}-{i}.npy")))
normal_der = np.array(normal_der).sum(axis = 0) / args.iternormal

num_conf = out_boundary.num_confs
normal_ders = dr.zeros(ArrayXf, shape = (num_conf, args.resnormal))

for i in range(num_conf):
    normal_ders[i] = Float(normal_der[i])

wos.input.shape.in_boundaries[0].set_normal_derivative(normal_ders)

conf_numbers = [dr.opaque(UInt32, i, shape = (1)) for i in range(num_conf)]

filename = f"{args.injection}-ref-{args.objx}-{args.objy}-{args.objr}-current-{args.x}-{args.y}-{radius}"
path = os.path.join(root_directory, "prb", filename)
create_path(path)

points, active_conf, electrode_nums = out_boundary.create_electrode_points(spe, conf_numbers=conf_numbers)
L_obj, _ = wos_obj.solve(points, active_conf, conf_numbers=conf_numbers, all_inside = True)
el_tensor_obj = create_electrode_result(L_obj, spe, electrode_nums, apply_normalization=False)

L, _ = wos.solve(points, active_conf, conf_numbers=conf_numbers, all_inside = True)
el_tensor = create_electrode_result(L, spe, electrode_nums, apply_normalization=False)
# dr.eval(el_tensor)

loss_grad = compute_loss_grad(result = el_tensor, result_ref = el_tensor_obj)
#loss_grad = compute_loss_grad(result = el_tensor)
dL = compute_dL(L = L, loss_grad=loss_grad, spe = spe, electrode_nums=electrode_nums, apply_normalization=False)

opt = Adam(lr = 0.1, params = wos.opt_params)
wos.update(opt)


L, _ = wos.solve(points, active_conf, L_in=L, conf_numbers=conf_numbers, 
                 dL = dL, all_inside = True, normal_derivative_dist=distance, mode = dr.ADMode.Backward)

r_grad = dr.grad(wos.input.shape.in_boundaries[0].radius).numpy().squeeze()
o_grad = dr.grad(wos.input.shape.in_boundaries[0].origin).numpy().squeeze()

np.save(os.path.join(path, f"resnormal{args.resnormal}-d{args.distance}-{args.objseed}-r"), r_grad)
np.save(os.path.join(path, f"resnormal{args.resnormal}-d{args.distance}-{args.objseed}-x"), o_grad[0])
np.save(os.path.join(path, f"resnormal{args.resnormal}-d{args.distance}-{args.objseed}-y"), o_grad[1])

