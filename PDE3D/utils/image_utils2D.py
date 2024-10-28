import numpy as np
import mitsuba as mi
import drjit as dr
import sys

def create_image_points(bbox : list, resolution : list[int], spp : int, seed : int = 64, centered = False) -> mi.Point2f:
    # Generate the first points
    
    x, y = dr.meshgrid(dr.arange(mi.Float, resolution[1]), 
                       dr.arange(mi.Float, resolution[0]), indexing='xy')
    x = dr.repeat(x, spp)
    y = dr.repeat(y, spp)
    if not centered:
        npoints = resolution[0] * resolution[1] * spp
        np.random.seed(seed)
        init_state = np.random.randint(sys.maxsize, size = npoints)
        init_seq = np.random.randint(sys.maxsize, size = npoints)
        sampler = mi.PCG32(npoints, initstate = init_state, initseq = init_seq)
        film_points =  mi.Point2f(x,y) + mi.Point2f(sampler.next_float32(), sampler.next_float32())
    else:
        film_points =  mi.Point2f(x,y) + mi.Point2f(0.5, 0.5)
    # The bounding box is defined as (bottom-left,up-right)
    points = (mi.Point2f(bbox[0][0], bbox[1][1]) +  
              film_points / mi.Point2f(resolution[1], resolution[0]) *  
              (mi.Point2f(bbox[1][0], bbox[0][1]) - mi.Point2f(bbox[0][0], bbox[1][1])))
    return points


def create_image_from_result(result, resolution = [256, 256], compute_std = False):
    if isinstance(result, mi.Float):
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
    image_res = mi.TensorXf(result_sum)

    shape = [num_conf, resolution[0], resolution[1]]
    tensor = dr.reshape(mi.TensorXf, value = image_res, shape = shape)

    if not compute_std:
        return tensor.numpy(), tensor

    else:
        variance = mi.TensorXf(dr.block_sum(dr.square(result), spp) / spp)
        variance = dr.reshape(mi.TensorXf, value = variance, shape = shape) - dr.square(tensor)
        variance /= spp
    return tensor.numpy(), tensor, np.abs(variance.numpy()), variance

def create_circle_points(origin : list = [0,0], radius : float = 1.0, resolution = 1024,
                         spp = 256, seed : int = 14, centered = False, discrete_points = False, shift : float = 0):
    if not discrete_points:
        npoints = spp * resolution
        np.random.seed(seed)
        init_state = np.random.randint(sys.maxsize, size = npoints)
        init_seq = np.random.randint(sys.maxsize, size = npoints)
        sampler = mi.PCG32(npoints, initstate = init_state, initseq = init_seq)
        film_points = dr.arange(mi.Float, resolution)
        film_points = dr.repeat(film_points, spp) + sampler.next_float32()
        film_points -= 1/2 if centered else 0
        angles = film_points / resolution * 2 * dr.pi + shift
        points = mi.Array2f(origin) + radius * mi.Array2f(dr.sin(angles), dr.cos(angles))
    else:
        film_points = dr.arange(mi.Float, resolution)
        film_points = dr.repeat(film_points, spp)
        film_points += 1/2 if centered else 0
        angles = film_points / resolution * 2 * dr.pi + shift
        points = mi.Point2f(origin) + radius * mi.Point2f(dr.sin(angles), dr.cos(angles))
    return points


