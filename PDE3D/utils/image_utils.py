import mitsuba as mi
import drjit as dr
import numpy as np
import sys


def create_bbox_points(bbox : mi.BoundingBox3f, resolution : list[int], spp : int, seed : int = 64, centered = False) -> mi.Point3f:
    # Generate the first points
    
    x, y, z = dr.meshgrid(dr.arange(mi.Float, resolution[0]), 
                          dr.arange(mi.Float, resolution[1]),
                          dr.arange(mi.Float, resolution[2]), indexing='ij')
    x = dr.repeat(x, spp)
    y = dr.repeat(y, spp)
    z = dr.repeat(z, spp)
    if not centered:
        npoints = resolution[0] * resolution[1] * resolution[2] * spp
        np.random.seed(seed)
        init_state = np.random.randint(sys.maxsize, size = npoints)
        init_seq = np.random.randint(sys.maxsize, size = npoints)
        sampler = mi.PCG32(npoints, initstate = init_state, initseq = init_seq)
        film_points =  mi.Point3f(x,y,z) + mi.Point3f(sampler.next_float32(), sampler.next_float32(), sampler.next_float32())
    else:
        film_points =  mi.Point3f(x,y,z) + mi.Point3f(0.5, 0.5, 0.5)
    
    points = bbox.min + (bbox.max - bbox.min) * film_points / mi.Point3f(resolution)
    return points


def create_volume_from_result(result, resolution = [16, 16, 16], compute_std = False):
    if isinstance(result, mi.Float):
        num_conf = 1
    else:
        if result.ndim == 1:
            num_conf = 1
        else:
            num_conf = result.shape[0]
    # Splat to film
    spp = int(dr.width(result) / (resolution[0] * resolution[1] * resolution[2]))
    #active_lanes = dr.select(result != 0, 1, 0)
    #active_sum = dr.block_sum(active_lanes, spp)
    result_sum = dr.block_sum(result, spp) / spp
    #image_res = TensorXf(dr.select(active_sum > 0, result_sum / active_sum, 0))
    image_res = mi.TensorXf(result_sum)

    shape = [num_conf, resolution[0], resolution[1], resolution[2]]
    tensor = dr.reshape(mi.TensorXf, value = image_res, shape = shape)

    if not compute_std:
        return tensor.numpy(), tensor

    else:
        variance = mi.TensorXf(dr.block_sum(dr.square(result), spp) / spp)
        variance = dr.reshape(mi.TensorXf, value = variance, shape = shape) - dr.square(tensor)
        variance /= spp
    return tensor.numpy(), tensor, np.abs(variance.numpy()), variance


def create_slice_from_result(result, resolution = [256, 256], compute_std = False):
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



