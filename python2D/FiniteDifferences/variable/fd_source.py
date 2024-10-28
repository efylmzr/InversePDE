import mitsuba as mi 
mi.set_variant("cuda_ad_rgb")
import matplotlib.patches as patches
import drjit as dr
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PDE2D.Coefficient import *
from PDE2D.utils import *
from PDE2D.BoundaryShape import *
from PDE2D.Solver import *
from PDE2D.Solver.constant.wos_constant import Particle
from mitsuba import Float, Point2f
from PDE2D import GreenSampling, Split, PATH
import argparse

parser = argparse.ArgumentParser(description='''Optimization Sphere''')
parser.add_argument('--spp', default = 8, type=int)
parser.add_argument('--resprimal', default = 5, type = int)
parser.add_argument('--restensor', default = 16, type = int)
parser.add_argument('--fdstep', default = 1e-2, type = float)
args = parser.parse_args()

green = GreenSampling.Polynomial
conf_numbers = [UInt32(0), UInt32(1)]
conf_vis = 0
epsilon = 1e-4
use_accel = True
bbox = [[-1, -1], [1, 1]]
resolution_image = [2 ** args.resprimal, 2 ** args.resprimal]
spp_image = 2 ** args.spp

split = Split.Normal
fd_step = args.fdstep
res_tensor = args.restensor


def boundary(points, parameters):
    angle = dr.atan2(points[0], points[1])
    return parameters["scale"] * dr.sin(angle * parameters["freq"]) + parameters["bias"]
parameters1_d = {}
parameters1_d["freq"] = 1
parameters1_d["bias"] = 6
parameters1_d["scale"] = 12
parameters2_d = {}
parameters2_d["freq"] = 8
parameters2_d["bias"] = 4
parameters2_d["scale"] = 8

dirichlet1 = FunctionCoefficient("dirichlet", parameters1_d, boundary)
dirichlet2 = FunctionCoefficient("dirichlet", parameters2_d, boundary)

shape = load_bunny(scale = 1, dirichlet = [dirichlet1, dirichlet2], neumann = [ConstantCoefficient("neumann", 10)], epsilon = epsilon)

out_val = 1
image = (np.arange(res_tensor, dtype = np.float32)) / res_tensor
image = np.tile(image, (res_tensor, 1))
image *= 2
image += out_val

α = ConstantCoefficient("diffusion", 1)
σ = ConstantCoefficient("screening", 0)
f = TextureCoefficient("source", bbox = bbox, tensor_np = image, out_val=out_val, grad_zero_points=None)
data_holder = DataHolder(shape = shape, α = α, σ = σ, f=f)


wos = WostVariable(data_holder, green_sampling=green, use_accelaration=use_accel, opt_params= ["source.texture.tensor"])
opt = Adam(lr = 0.1, params = wos.opt_params)
wos.update(opt)

points = create_image_points(bbox, resolution_image, spp_image)

L, p = wos.solve(points_in = points, conf_numbers = conf_numbers, split = split)
image_0, tensor = create_image_from_result(L, resolution_image)
fig, (ax1) = plt.subplots(1, 1, figsize=[5, 5])
plot_image(image_0[conf_vis], ax1)
ax1.set_title("Primal Result")

loss_grad = compute_loss_grad_image(result = tensor)
dL = compute_dL_image(loss_grad=loss_grad, spp = spp_image)

fig, ax = plt.subplots(1,1, figsize = (5,5))
plot_image(loss_grad[0].numpy(), ax)

dL_image,_ = create_image_from_result(dL, resolution_image)
fig, ax = plt.subplots(1,1, figsize = (5,5))
plot_image(dL_image[0] * spp_image, ax)

L_grad, p = wos.solve_grad(points_in = points, split = split, dL = dL, conf_numbers=conf_numbers, verbose = True)
grad_prb = dr.grad(f.tensor).numpy()

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

fd_grad = np.zeros_like(image)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        f1, f2 = step_texture(f, i, j, fd_step)
        data_holder1 = DataHolder(shape = shape, α = α, α_split = α, σ = σ, σ_split = σ, f = f1)
        data_holder2 = DataHolder(shape = shape, α = α, α_split = α, σ = σ, σ_split = σ, f = f2)
        wos1 = WostVariable(data_holder1, use_accelaration = use_accel, green_sampling = green)
        wos2 = WostVariable(data_holder2, use_accelaration = use_accel, green_sampling = green)
        L1, _ = wos1.solve(points, split = split, conf_numbers=conf_numbers, fd_forward=True, verbose = False)
        L2, _ = wos2.solve(points, split = split, conf_numbers=conf_numbers, fd_forward=True, verbose = False)
        image1, tensor1 = create_image_from_result(L1, resolution_image)
        image2, tensor2 = create_image_from_result(L2, resolution_image)
        val1 = np.sum(MSE_numpy(image1))
        val2 = np.sum(MSE_numpy(image2))
        print(val1, val2)
        fd_grad[i,j] = (val1 - val2) / (2 * fd_step)
        print(f"{i},{j}")

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=  (14,5))
maxval = max(np.max(fd_grad), np.max(grad_prb))
minval = min(np.min(fd_grad), np.min(grad_prb))
max_range = max(maxval, -minval)
plot_image(fd_grad, ax1, input_range=(-max_range, max_range), cmap = "coolwarm")
plot_image(grad_prb, ax2, input_range=(-max_range, max_range), cmap = 'coolwarm')
plot_image(np.abs(fd_grad.squeeze()-grad_prb.squeeze()), ax3, cmap = 'coolwarm')
ax1.set_title("FD")
ax2.set_title("PRB")

import os
def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
path = os.path.join(PATH, "output2D", "finite_differences", "variable", "source")
create_path(path)

fig.savefig(os.path.join(path, f"source{res_tensor}-fd{fd_step}.pdf"), bbox_inches='tight', pad_inches=0.04, dpi=200)
record = {}
record["prb"] = grad_prb
record["fd"] = fd_grad
np.save(os.path.join(path, f"source{res_tensor}-fd{fd_step}.npy"), record)