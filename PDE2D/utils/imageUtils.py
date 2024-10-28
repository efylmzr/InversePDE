import numpy as np
import drjit as dr
import mitsuba as mi
from mitsuba import PCG32, Float, Point2f, TensorXf
from PDE2D import ArrayXu, ArrayXf
import sys

def create_image_points(bbox : list, resolution : list[int], spp : int, seed : int = 64, centered = False) -> Point2f:
    # Generate the first points
    
    x, y = dr.meshgrid(dr.arange(Float, resolution[1]), 
                       dr.arange(Float, resolution[0]), indexing='xy')
    x = dr.repeat(x, spp)
    y = dr.repeat(y, spp)
    if not centered:
        npoints = resolution[0] * resolution[1] * spp
        np.random.seed(seed)
        init_state = np.random.randint(sys.maxsize, size = npoints)
        init_seq = np.random.randint(sys.maxsize, size = npoints)
        sampler = PCG32(npoints, initstate = init_state, initseq = init_seq)
        film_points =  Point2f(x,y) + Point2f(sampler.next_float32(), sampler.next_float32())
    else:
        film_points =  Point2f(x,y) + Point2f(0.5, 0.5)
    # The bounding box is defined as (bottom-left,up-right)
    points = (Point2f(bbox[0][0], bbox[1][1]) +  
              film_points / Point2f(resolution[1], resolution[0]) *  
              (Point2f(bbox[1][0], bbox[0][1]) - Point2f(bbox[0][0], bbox[1][1])))
    return points


def create_image_from_result(result, resolution = [256, 256], compute_std = False):
    if isinstance(result, Float):
        num_conf = 1
    else:
        if result.ndim == 1:
            num_conf = 1
        else:
            num_conf = result.shape[0]
    # Splat to film
    spp = int(dr.width(result) / (resolution[0] * resolution[1]))
    #active_lanes = dr.select(result != 0, 1, 0)
    #active_sum = dr.block_sum(active_lanes, spp)
    result_sum = dr.block_sum(result, spp) / spp
    #image_res = TensorXf(dr.select(active_sum > 0, result_sum / active_sum, 0))
    image_res = TensorXf(result_sum)

    shape = [num_conf, resolution[0], resolution[1]]
    tensor = dr.reshape(TensorXf, value = image_res, shape = shape)

    if not compute_std:
        return tensor.numpy(), tensor

    else:
        variance = TensorXf(dr.block_sum(dr.square(result), spp) / spp)
        variance = dr.reshape(TensorXf, value = variance, shape = shape) - dr.square(tensor)
        variance /= spp
    return tensor.numpy(), tensor, np.abs(variance.numpy()), variance

def create_circle_points(origin : list = [0,0], radius : float = 1.0, resolution = 1024,
                         spp = 256, seed : int = 14, centered = False, discrete_points = False, shift : float = 0):
    if not discrete_points:
        npoints = spp * resolution
        np.random.seed(seed)
        init_state = np.random.randint(sys.maxsize, size = npoints)
        init_seq = np.random.randint(sys.maxsize, size = npoints)
        sampler = PCG32(npoints, initstate = init_state, initseq = init_seq)
        film_points = dr.arange(Float, resolution)
        film_points = dr.repeat(film_points, spp) + sampler.next_float32()
        film_points -= 1/2 if centered else 0
        angles = film_points / resolution * 2 * dr.pi + shift
        points = Point2f(origin) + radius * Point2f(dr.sin(angles), dr.cos(angles))
    else:
        film_points = dr.arange(Float, resolution)
        film_points = dr.repeat(film_points, spp)
        film_points += 1/2 if centered else 0
        angles = film_points / resolution * 2 * dr.pi + shift
        points = Point2f(origin) + radius * Point2f(dr.sin(angles), dr.cos(angles))
    return points

def create_circle_from_result(result, resolution = 1024):
    # Splat to film
    spp = int(dr.width(result) / resolution)
    res_image = TensorXf(dr.block_sum(result, spp)) / spp
    return res_image.numpy(), res_image

def create_electrode_result(L, spe, electrode_nums : ArrayXu, apply_normalization = True, compute_std = False):
        #unnormalized =  dr.block_sum(L, spe) / spe
        unnormalized = dr.block_sum(L, spe) / spe
        num_active_electrodes = dr.width(electrode_nums)
        
        if apply_normalization:
            bias = dr.block_sum(unnormalized, dr.width(unnormalized)) / num_active_electrodes
            result = unnormalized - dr.select(unnormalized != 0, bias, 0)
        else:
            result = unnormalized
        
        if not compute_std:
            return result
        
        variance = dr.block_sum(dr.square(L), spe) / spe - dr.square(unnormalized)
        variance /= spe

        return result, dr.sqrt(variance)
    
'''
def block_sum(L : Float, spp : int) -> Float: #spe needs to be power of 2
    iternum = int(dr.log2(spp))
    sum = ArrayXf(L)
    for i in range(iternum):
        sum = dr.block_sum(sum, 2)
    return sum

def block_sum_(L : Float, spp : int) -> Float: # Kahan-compensated blocksum.
    num_bins = dr.width(L)//spp
    index = dr.arange(UInt32, num_bins)
    index = dr.repeat(index, spp)
    target1 = dr.zeros(Float, num_bins)
    target2 = dr.zeros(Float, num_bins)
    dr.scatter_add_kahan(target1, target2, L, index)
    return target1 + target2

'''

