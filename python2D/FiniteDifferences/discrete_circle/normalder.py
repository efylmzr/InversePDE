import numpy
import drjit as dr
import mitsuba as mi 
mi.set_variant("cuda_ad_rgb")
import matplotlib.pyplot as plt
import sys
from PDE2D.Coefficient import *
from PDE2D.utils import *
from PDE2D.BoundaryShape import *
from PDE2D.Solver import *
from PDE2D import GreenSampling, Split, PATH
import argparse
import os

root_directory = os.path.join(PATH, "output2D", "finite_differences", "discrete-circle", "normalder")

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

parser = argparse.ArgumentParser(description='''FD-computation sphere''')
parser.add_argument("--res", default = 256, type = int)
parser.add_argument('--spp', default = 20, type=int)
parser.add_argument("--x", default = 0.2, type = float)
parser.add_argument("--y", default = 0.2, type = float)
parser.add_argument("--radius", default = 0.2, type = float)
parser.add_argument("--seed", default = 0, type = int)
parser.add_argument("--iter", default = 32, type = int)
parser.add_argument("--injection", default = "skip3", type = str)
parser.add_argument("--distance", default = 0.01, type = float)


args = parser.parse_args()
origin = [args.x, args.y]
radius = args.radius
distance = args.distance
spp = 2**args.spp

in_boundary = CircleShape(origin, radius)
out_boundary = CircleWithElectrodes(injection_set = args.injection, is_delta = True)
shape = BoundaryWithDirichlets(out_boundary, [in_boundary], dirichlet_values = [[0]])
data_holder = DataHolder(shape)
wos = WostConstant(data_holder)

num_conf = out_boundary.num_confs
conf_numbers = [dr.opaque(UInt32, i, shape = (1)) for i in range(num_conf)]

filename = f"{args.injection}-{args.x}-{args.y}-{radius}"
path = os.path.join(root_directory, filename)
create_path(path)

for iter in range(args.iter):
    print(f"Iteration {iter} finished.")
    seed_iter = iter + args.seed
    wos.change_seed(seed_iter)
    #dr.set_log_level(3)
    result, _ = wos.create_normal_derivative(args.res, spp, distance = distance, conf_numbers=conf_numbers)
    #dr.set_log_level(0)
    np.save(os.path.join(path, f"res{args.res}-d{distance}-{seed_iter}"), result.numpy())
    