import numpy
import mitsuba as mi 
import drjit as dr 
mi.set_variant("cuda_ad_rgb_double")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PDE3D.Coefficient import *
from PDE3D.utils import *
from PDE3D.BoundaryShape import *
from PDE3D.Solver import *
import argparse
from PDE3D.utils import *
import os
from python3D.optimization.textures import *

parser = argparse.ArgumentParser(description='''Optimization Sphere''')
parser.add_argument('--spp', default = 6, type=int)
parser.add_argument('--resprimal', default = 4, type = int)
parser.add_argument('--restensor', default = 5, type = int)
parser.add_argument('--fdstep', default = 1e-3, type = float)
parser.add_argument('--param', default = "source", type = str)
args = parser.parse_args()


name = "motorbike-engine"
epsilon = 1e-2
spp = 2**args.spp
param = args.param
res_tex = 2**args.restensor
resolution_tex = [res_tex, res_tex, res_tex]
res_primal = 2**args.resprimal
res_primal = [res_primal, res_primal, res_primal]
fd_step = args.fdstep

import os
def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
path = os.path.join(PATH, "output3D", "finite_differences", "variable", f"{args.param}-resprimal{args.resprimal}")
create_path(path)

split = Split.Normal


def dirichlet(points, params):
    return dr.sin(points[0] * params["x"]) + dr.cos(points[1] * points[2] * params["yz"])
params1 = {}
params1["x"] = 0.2
params1["yz"] = 0.4

params2 = {}
params2["x"] = 4
params2["yz"] = 0.4

boundary_cond1 = FunctionCoefficient("dirichlet", params1, dirichlet)
boundary_cond2 = FunctionCoefficient("dirichlet", params2, dirichlet)
conf_numbers = [mi.UInt32(0), mi.UInt32(1)]

folder_name = os.path.join(PATH, "scenes", name)
xml_name = os.path.join(folder_name, "scene.xml")
sdf_data = np.load(os.path.join(folder_name, "sdf.npy"))

sdf = SDF(sdf_data, dirichlet = [boundary_cond1, boundary_cond2], epsilon=epsilon, scale = 12)

bbox = sdf.bbox
bbox_pad = (bbox.max - bbox.min) / 10
bbox_coeff = mi.ScalarBoundingBox3f(bbox.min - bbox_pad, bbox.max + bbox_pad)



source = textures[1]() * 10
screening = textures[3]() * 20 
diffusion = textures[5]() * 10 + 1

f = TextureCoefficient("source", bbox_coeff, source)
σ = TextureCoefficient("screening", bbox_coeff, screening)
α = TextureCoefficient("diffusion", bbox_coeff, diffusion)


points_bbox = create_bbox_points(bbox_coeff, resolution_tex, spp = 1, centered = True)


source_vals = f.get_value(points_bbox) 
vol_source, _ = create_volume_from_result(source_vals, resolution_tex)

screening_vals = σ.get_value(points_bbox) 
vol_screening, _ = create_volume_from_result(screening_vals, resolution_tex)

diffusion_vals = α.get_value(points_bbox)
vol_diffusion,_ = create_volume_from_result(diffusion_vals, resolution_tex)

f = TextureCoefficient("source", bbox_coeff, vol_source[0])
σ = TextureCoefficient("screening", bbox_coeff, vol_screening[0])
α = TextureCoefficient("diffusion", bbox_coeff, vol_diffusion[0])


data_holder = DataHolder(shape = sdf, α = α, σ = σ, f=f)
print(data_holder.σ_bar)
wos = WosVariable(data_holder, opt_params= [f"{param}.texture.tensor"])
opt = mi.ad.Adam(lr = 0.1, params = wos.opt_params)
wos.update(opt)

points = create_bbox_points(bbox, res_primal, spp, centered = True)

L, p = wos.solve(points_in = points, conf_numbers = conf_numbers, split = split)
image_0, tensor = create_volume_from_result(L, res_primal)

loss_grad = compute_loss_grad_vol(result = tensor)
dL = compute_dL_vol(loss_grad=loss_grad, spp = spp)

L_grad, p = wos.solve_grad(points_in = points, split = split, dL = dL, conf_numbers=conf_numbers, verbose = True)
if param == "source":
    grad_prb = dr.grad(f.tensor).numpy()
elif param == "screening":
    grad_prb = dr.grad(σ.tensor).numpy()
elif param == "diffusion":
    grad_prb = dr.grad(α.tensor).numpy()

np.save(os.path.join(path, f"{param}{args.restensor}-prb.npy"), grad_prb.squeeze())

def step_texture(texture : Coefficient, i, j, k, fd_step):
    tex1 = texture.copy()
    tex2 = texture.copy()
    index = i * tex1.tensor.shape[2] * tex1.tensor.shape[1] + j * tex1.tensor.shape[2] + k
    dr.scatter_add(tex1.tensor.array, +fd_step, index)
    dr.scatter_add(tex2.tensor.array, -fd_step, index)
    dr.make_opaque(tex1.tensor)
    dr.make_opaque(tex2.tensor)
    tex1.update_texture()
    tex2.update_texture()
    return tex1, tex2

def get_data_holder(i, j, k):
    if param == "source":
        f1, f2 = step_texture(f, i, j, k, fd_step)
        data_holder1 = DataHolder(shape = sdf, α = α, α_split = α, f = f1, σ_split = σ, σ = σ)
        data_holder2 = DataHolder(shape = sdf, α = α, α_split = α, f = f2, σ_split = σ, σ = σ)
    elif param == "screening":
        σ1, σ2 = step_texture(σ, i, j, k, fd_step)
        data_holder1 = DataHolder(shape = sdf, α = α, α_split = α, f = f, σ_split = σ, σ = σ1)
        data_holder2 = DataHolder(shape = sdf, α = α, α_split = α, f = f, σ_split = σ, σ = σ2)
    elif param == "diffusion":
        α1, α2 = step_texture(α, i, j, k, fd_step)
        data_holder1 = DataHolder(shape = sdf, α = α1, α_split = α, f = f, σ_split = σ, σ = σ)
        data_holder2 = DataHolder(shape = sdf, α = α2, α_split = α, f = f, σ_split = σ, σ = σ)
    return data_holder1, data_holder2

grad_fd = np.zeros(resolution_tex)
for i in range(resolution_tex[0]):
    for j in range(resolution_tex[1]):
        for k in range(resolution_tex[2]):
            data_holder1, data_holder2 = get_data_holder(i,j,k)
            wos1 = WosVariable(data_holder1)
            wos2 = WosVariable(data_holder2)
            points = create_bbox_points(bbox, res_primal, spp, centered = True)
            L1, _ = wos1.solve(points, split = split, conf_numbers=conf_numbers, fd_forward=True, verbose = False)
            L2, _ = wos2.solve(points, split = split, conf_numbers=conf_numbers, fd_forward=True, verbose = False)
            image1, tensor1 = create_volume_from_result(L1, res_primal)
            image2, tensor2 = create_volume_from_result(L2, res_primal)
            val1 = np.sum(MSE_numpy(image1))
            val2 = np.sum(MSE_numpy(image2))
            print(val1, val2)
            grad_fd[i,j,k] = (val1 - val2) / (2 * fd_step)
            print(f"{i},{j},{k}")

np.save(os.path.join(path, f"{param}{args.restensor}-fd{fd_step}.npy"), grad_fd)


prb_tex = TextureCoefficient("prb", bbox_coeff, np.squeeze(grad_prb), interpolation = "nearest")
fd_tex = TextureCoefficient("prb", bbox_coeff, np.squeeze(grad_fd), interpolation = "nearest")
diff_tex = TextureCoefficient("prb", bbox_coeff, (np.squeeze(grad_prb) - np.squeeze(grad_fd)), interpolation = "nearest")
cmap = "coolwarm"

cam_res = [512, 512]
res_slice = [512, 512]
spp = 64
downsample = 1
cam_origin = mi.ScalarPoint3f([7,7,10])
scale_cam = 1/5
cam_target = mi.ScalarPoint3f([0.0,0.0,0.0])
cam_up = mi.ScalarPoint3f([0,1,0])
slice = Slice(offset =0, scale = 7, axis = "z")

prb3D, prb_norm = sdf.visualize(colormap = cmap, cam_origin= cam_origin, spp = spp, image_res = cam_res, 
                            scale_cam=scale_cam, cam_up = cam_up, slice = slice, cam_target = cam_target, coeff= prb_tex, sym_colorbar=True)
fd3D, fd_norm = sdf.visualize(colormap = cmap, cam_origin= cam_origin, spp = spp, image_res = cam_res, 
                            scale_cam=scale_cam, cam_up = cam_up, slice = slice, cam_target = cam_target, coeff= fd_tex, sym_colorbar=True)
diff3D, diff_norm = sdf.visualize(colormap = cmap, cam_origin= cam_origin, spp = spp, image_res = cam_res, 
                            scale_cam=scale_cam, cam_up = cam_up, slice = slice, cam_target = cam_target, coeff= diff_tex, sym_colorbar=True)

fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize = (15,5))
plot_image_3D(prb3D, ax1, norm = prb_norm, cmap = cmap)
plot_image_3D(fd3D, ax2, norm = fd_norm, cmap = cmap)
plot_image_3D(diff3D, ax3, norm = diff_norm, cmap = cmap)
ax1.set_title("PRB")
ax2.set_title("FD")
ax3.set_title("Difference")
fig.savefig(os.path.join(path, f"{param}{args.restensor}-fd{fd_step}-spp{args.spp}.pdf"), bbox_inches='tight', pad_inches=0.04, dpi=200)