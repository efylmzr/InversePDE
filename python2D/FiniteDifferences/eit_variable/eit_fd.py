# %%
import mitsuba as mi 
mi.set_variant("cuda_ad_rgb")
import drjit as dr
import matplotlib.pyplot as plt
import sys
from PDE2D.Coefficient import *
from PDE2D.utils import *
from PDE2D.BoundaryShape import *
from PDE2D.Solver import *
from PDE2D import GreenSampling, Split, PATH
from mitsuba import TensorXf, Texture2f
import argparse

parser = argparse.ArgumentParser(description='-')
parser.add_argument('--spe', default = 12, type=int)
parser.add_argument('--restensor', default = 16, type = int)
parser.add_argument('--fdstep', default = 1e-2, type = float)
args = parser.parse_args()

split = Split.Normal
green = GreenSampling.Polynomial
newton_steps = 6
weight_window = [0.3, 1.5]
fd_step = args.fdstep
use_accel = True
max_split_depth = 100
normalization = False
max_step_num = 100
spe = 2 ** args.spe
res_tensor = args.restensor

def step_texture(texture : Coefficient, i, j, fd_step):
    tex1 = texture.copy()
    tex2 = texture.copy()
    index = i * tex1.tensor.shape[0] + j
    dr.scatter_add(tex1.tensor.array, +fd_step, index)
    dr.scatter_add(tex2.tensor.array, -fd_step, index)
    dr.make_opaque(tex1.tensor)
    dr.make_opaque(tex2.tensor)
    tex1.update_texture()
    tex2.update_texture()
    return tex1, tex2

conf_number = 0
e_shell = 1e-4
parameters = {}
num_electrodes = 16
electrode_length = 0.1
conf_numbers = [UInt32(i) for i in range(16)]

radius = 1
out_boundary = CircleWithElectrodes(radius = radius, num_electrodes=num_electrodes, is_delta = True, 
                                          electrode_length=electrode_length, injection_set= "skip3", centered = True)

bbox = [[-1.1 * radius,-1.1 * radius],[1.1 * radius, 1.1 * radius]]
shape = BoundaryWithDirichlets(out_boundary, [], epsilon = e_shell)

out_val = 1
image = (np.arange(res_tensor, dtype = np.float32)) / res_tensor
image = np.tile(image, (res_tensor, 1))
image *= 2
image += out_val

grad_zero_points = out_boundary.create_boundary_points(distance = 0, res = 1024, spp = 2)[0]
#grad_zero_points = None
α = TextureCoefficient("diffusion", bbox, image, grad_zero_points=grad_zero_points, out_val = out_val)
default_majorant = 100
data_holder = DataHolder(shape, α = α, default_majorant=default_majorant)
name=  "diffusion.texture.tensor"
opt_params = [name]
wos = WostVariable(data_holder, green_sampling=green, use_accelaration = use_accel, opt_params = opt_params)

# %%
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=[9, 5])
resolution = [256, 256]
α.visualize(ax1, bbox, resolution = resolution)
wos.input.eff_screening_tex.visualize(ax2, bbox, resolution);
shape.sketch(ax1, bbox, resolution = resolution);
shape.sketch(ax2, bbox, resolution = resolution);
out_boundary.sketch_electrode_input(ax1, bbox, resolution)
out_boundary.sketch_electrode_input(ax2, bbox, resolution)
ax1.set_title("Diffusion")
ax2.set_title("Effective Screening")

# %%
opt = Adam(lr = 0.1, params = wos.opt_params)
wos.update(opt)
points, active_conf, electrode_nums = out_boundary.create_electrode_points(spe, conf_numbers=conf_numbers)

# %%
L, _ = wos.solve(points, active_conf, split = split, max_depth_split = max_split_depth, conf_numbers= conf_numbers, max_length=max_step_num, all_inside = True)
el_tensor = create_electrode_result(L, spe, electrode_nums, apply_normalization=normalization)


# %%

L, _ = wos.solve(points, active_conf, split = split, max_depth_split = max_split_depth, conf_numbers=conf_numbers, max_length=max_step_num, all_inside = True, verbose = False)
el_tensor = create_electrode_result(L, spe, electrode_nums, apply_normalization=normalization)
# dr.eval(el_tensor)
loss_grad = compute_loss_grad(result = el_tensor)
dL = compute_dL(L = L, loss_grad=loss_grad, spe = spe, apply_normalization=normalization)


# %%
L_grad, p = wos.solve_grad(points_in = points, active_conf_in = active_conf, split = split, dL = dL, max_depth_split = max_split_depth, 
                           conf_numbers=conf_numbers, max_length = max_step_num, all_inside = True, verbose = True)
grad_prb = dr.grad(α.tensor).numpy()

# %%
fig, ax = plt.subplots(1,1, figsize = (5,5))
plot_image(grad_prb, ax)

# %%
fd_grad = np.zeros_like(image)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        α1, α2 = step_texture(α, i, j, fd_step)
        data_holder1 = DataHolder(shape = shape, α = α1, α_split = α, default_majorant= default_majorant)
        data_holder2 = DataHolder(shape = shape, α = α2, α_split = α, default_majorant=default_majorant)
        wos1 = WostVariable(data_holder1, use_accelaration = use_accel, green_sampling = green)
        wos2 = WostVariable(data_holder2, use_accelaration = use_accel, green_sampling = green)
        L1, _ = wos1.solve(points, active_conf, split = split, max_depth_split = max_split_depth, conf_numbers=conf_numbers, 
                           max_length=max_step_num, all_inside = True, fd_forward=True)
        L2, _ = wos2.solve(points, active_conf, split = split, max_depth_split = max_split_depth, conf_numbers=conf_numbers, 
                           max_length=max_step_num, all_inside = True, fd_forward=True)
        el_tensor1 = create_electrode_result(L1, spe, electrode_nums, apply_normalization=normalization)
        el_tensor2 = create_electrode_result(L2, spe, electrode_nums, apply_normalization=normalization)
        val1 = dr.sum(MSE(el_tensor1))[0]
        val2 = dr.sum(MSE(el_tensor2))[0]
        fd_grad[i,j] = (val1 - val2) / (2 * fd_step)
        print(f"{i}, {j}")

# %%
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=  (14,5))
maxval = max(np.max(fd_grad), np.max(grad_prb))
minval = min(np.min(fd_grad), np.min(grad_prb))
max_range = max(maxval, -minval)
plot_image(fd_grad, ax1, input_range=(-max_range, max_range), cmap = "coolwarm")
plot_image(grad_prb, ax2, input_range=(-max_range, max_range), cmap = 'coolwarm')
plot_image(np.abs(fd_grad.squeeze()-grad_prb.squeeze()), ax3, cmap = 'coolwarm')
ax1.set_title("FD")
ax2.set_title("PRB")
ax3.set_title("Difference")


import os
def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
path = os.path.join(PATH, "output2D", "finite_differences", "eit", "diffusion")
create_path(path)

fig.savefig(os.path.join(path, f"diffusion{res_tensor}-fd{fd_step}.pdf"), bbox_inches='tight', pad_inches=0.04, dpi=200)
record = {}
record["prb"] = grad_prb
record["fd"] = fd_grad
np.save(os.path.join(path, f"diffusion{res_tensor}-fd{fd_step}.npy"), record)


