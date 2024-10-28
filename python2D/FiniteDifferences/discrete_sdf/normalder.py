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

root_directory = os.path.join(PATH, "output2D", "finite_differences", "discrete-sdf", "normalder")

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

parser = argparse.ArgumentParser(description='''Normal derivative computation SDF.''')
parser.add_argument("--res", default = 1024, type = int)
parser.add_argument('--spp', default = 18, type=int)
parser.add_argument("--seed", default = 0, type = int)
parser.add_argument("--iter", default = 256, type = int)
parser.add_argument("--distance", default = 0.01, type = float)
parser.add_argument("--upsample", default = 1, type = int)
parser.add_argument("--epsilon", default = 5e-6, type = float)


args = parser.parse_args()
distance = args.distance
spp = 2**args.spp


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
wos = WostConstant(data_holder)

num_conf = out_boundary.num_confs
conf_numbers = [dr.opaque(UInt32, i, shape = (1)) for i in range(num_conf)]

path = root_directory
create_path(path)

for iter in range(args.iter):
    seed_iter = iter + args.seed
    wos.change_seed(seed_iter)
    result, _ = wos.create_normal_derivative(args.res, spp, distance = distance, conf_numbers=conf_numbers)
    np.save(os.path.join(path, f"res{args.res}-d{distance}-{seed_iter}"), result.numpy())
    print(f"Iteration {iter} finished.")
    