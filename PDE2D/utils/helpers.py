import numpy as np
import drjit as dr
import mitsuba as mi
from math import sqrt
from mitsuba import Point2f, Float, UInt32
from collections.abc import Sequence

#def deviate_points(points, max = 0.1, repeat= 5, seed = 37):
#    sampler = mi.load_dict({'type': 'stratified'})
#    sampler.seed(seed, dr.width(points) * repeat)
#    return dr.repeat(points, repeat) + max * (sampler.next_2d() - 1/2)

def get_position_bbox(points : Float, bbox):
    "Get the new positions of the points normalized for the bbox."
    x = (points[0] - bbox[0][0]) / (bbox[1][0] - bbox[0][0])
    y = 1.0 - (points[1] - bbox[0][1]) / (bbox[1][1] - bbox[0][1])
    return x, y

def to_world_direction(direction : Point2f, normal : Point2f):
    return direction[1] * normal + direction[0] * Point2f(normal[1], -normal[0]) 

def to_normal_direction(direction : Point2f, normal: Point2f):
    normal_comp = dr.dot(direction, normal)
    other_comp = dr.dot(Point2f(normal[1], normal[0]), direction)
    return Point2f(other_comp, normal_comp)

@dr.syntax
def correct_angle(angle : Float):
    if angle<0:
        angle += 2 * dr.pi
    elif angle >= 2 * dr.pi:
        angle -= 2 * dr.pi
    return angle

"""
def upsample(tensor = TensorXf, upsample = [2,2]):
    n1, n2 = tensor.shape
    array = tensor.array
    array_new = dr.zeros(Float, n1 * n2 * upsample[0] * upsample[1])
    rows = dr.arange(UInt, n1 * upsample[0])
    cols = dr.arange(UInt, n2 * upsample[1])
    indices = dr.meshgrid(rows, cols)
    indices_low = Array2i(indices[0] // upsample[0], indices[1] // upsample[1])
    ind = indices[0] * n2 * upsample[1] + indices[1]
    ind_low = indices_low[0] * n2 + indices_low[1]
    low_vals = dr.gather(Float, array, ind_low)
    dr.scatter(array_new, low_vals, ind)
    tensor_new = TensorXf(array_new)
    tensor_new = dr.reshape(TensorXf, tensor_new, shape = [n1 * upsample[0], n2 * upsample[1]])
    return tensor_new
"""

def tea(v0: UInt32, v1: UInt32, rounds=4):
        sum = UInt32(0)
        for i in range(rounds):
            sum += 0x9e3779b9
            v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + sum) ^ ((v1>>5) + 0xc8013ea4)
            v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + sum) ^ ((v0>>5) + 0x7e95761e)
        return v0, v1

@dr.syntax
def k_means(points : Point2f, initial_means : Point2f, num_iter = 2):
    # Set initial vals.
    means = Point2f(initial_means)
    npoints = dr.width(points)
    nmeans = dr.width(means)
    groups = dr.zeros(UInt32, npoints)
    dist = dr.full(Float, dr.inf, npoints)
    # Start K-means.
    for i in range(num_iter):
        # Assign to the groups.
        for j in range(nmeans):
            mean = dr.gather(Point2f, means, j)
            dist_iter = dr.squared_norm(points - mean)
            if dist_iter < dist:
                groups = UInt32(j)
                dist = dist_iter
        # Recompute the means.
        next_means = dr.zeros(Point2f, nmeans)
        counter_sum = dr.zeros(Float, nmeans)
        dr.scatter_add(next_means, points, groups)
        dr.scatter_add(counter_sum, Float(1), groups)
        means = next_means / counter_sum
    return means, groups


def upsample(t, shape=None, scale_factor=None, align_corners=False):
    '''
    upsample(source, shape=None, scale_factor=None, align_corners=False)
    Up-sample the input tensor or texture according to the provided shape.

    Alternatively to specifying the target shape, a scale factor can be
    provided.

    The behavior of this function depends on the type of ``source``:

    1. When ``source`` is a Dr.Jit tensor, nearest neighbor up-sampling will use
    hence the target ``shape`` values must be multiples of the source shape
    values. When `scale_factor` is used, its values must be integers.

    2. When ``source`` is a Dr.Jit texture type, the up-sampling will be
    performed according to the filter mode set on the input texture. Target
    ``shape`` values are not required to be multiples of the source shape
    values. When `scale_factor` is used, its values must be integers.

    Args:
        source (object): A Dr.Jit tensor or texture type.

        shape (list): The target shape (optional)

        scale_factor (list): The scale factor to apply to the current shape
        (optional)

        align_corners (bool): Defines whether or not the corner pixels of the
        input and output should be aligned. This allows the values at the
        corners to be preserved. This flag is only relevant when ``source`` is
        a Dr.Jit texture type performing linear interpolation. The default is
        `False`.

    Returns:
        object: the up-sampled tensor or texture object. The type of the output
        will be the same as the type of the source.
    '''
    if  not getattr(t, 'IsTexture', False) and not dr.is_tensor_v(t):
        raise TypeError("upsample(): unsupported input type, expected Dr.Jit "
                        "tensor or texture type!")

    if shape is not None and scale_factor is not None:
        raise TypeError("upsample(): shape and scale_factor arguments cannot "
                        "be defined at the same time!")

    if shape is not None:
        if not isinstance(shape, Sequence):
            raise TypeError("upsample(): unsupported shape type, expected a list!")

        if len(shape) > len(t.shape):
            raise TypeError("upsample(): invalid shape size!")

        shape = list(shape) + list(t.shape[len(shape):])

        scale_factor = []
        for i, s in enumerate(shape):
            if type(s) is not int:
                raise TypeError("upsample(): target shape must contain integer values!")

            if s < t.shape[i]:
                raise TypeError("upsample(): target shape values must be larger "
                                "or equal to input shape! (%i vs %i)" % (s, t.shape[i]))

            if dr.is_tensor_v(t):
                factor = s / float(t.shape[i])
                if factor != int(factor):
                    raise TypeError("upsample(): target shape must be multiples of "
                                    "the input shape! (%i vs %i)" % (s, t.shape[i]))
    else:
        if not isinstance(scale_factor, Sequence):
            raise TypeError("upsample(): unsupported scale_factor type, expected a list!")

        if len(scale_factor) > len(t.shape):
            raise TypeError("upsample(): invalid scale_factor size!")

        scale_factor = list(scale_factor)
        for i in range(len(t.shape) - len(scale_factor)):
            scale_factor.append(1)

        shape = []
        for i, factor in enumerate(scale_factor):
            if type(factor) is not int:
                raise TypeError("upsample(): scale_factor must contain integer values!")

            if factor < 1:
                raise TypeError("upsample(): scale_factor values must be greater "
                                "than 0!")

            shape.append(factor * t.shape[i])

    if getattr(t, 'IsTexture', False):
        value_type = type(t.value())
        dim = len(t.shape) - 1

        if t.shape[dim] != shape[dim]:
            raise TypeError("upsample(): channel counts doesn't match input texture!")

        # Create the query coordinates
        coords = list(dr.meshgrid(*[
                dr.linspace(value_type, 0.0, 1.0, shape[i], endpoint=align_corners)
                for i in range(dim)
            ],
            indexing='ij'
        ))

        # Offset coordinates by half a voxel to hit the center of the new voxels
        if align_corners:
            for i in range(dim):
                coords[i] *= (1 - 1 / t.shape[i])
                coords[i] += 0.5 / t.shape[i]
        else:
            for i in range(dim):
                coords[i] += 0.5 / shape[i]

        # Reverse coordinates order according to dr.Texture convention
        coords.reverse()

        # Evaluate the texture at all voxel coordinates with interpolation
        values = t.eval(coords)

        # Concatenate output values to a flatten buffer
        channels = len(values)
        width = dr.width(values[0])
        index = dr.arange(dr.uint32_array_t(value_type), width)
        data = dr.zeros(value_type, width * channels)
        for c in range(channels):
            dr.scatter(data, values[c], channels * index + c)

        # Create the up-sampled texture
        texture = type(t)(shape[:-1], channels,
                          use_accel=t.use_accel(),
                          filter_mode=t.filter_mode(),
                          wrap_mode=t.wrap_mode())
        texture.set_value(data)

        return texture
    else:
        dim = len(shape)
        size = dr.prod(shape[:dim])
        base = dr.arange(dr.uint32_array_t(type(t.array)), size)

        index = 0
        stride = 1
        for i in reversed(range(dim)):
            ratio = shape[i] // t.shape[i]
            index += (base // ratio % t.shape[i]) * stride
            base //= shape[i]
            stride *= t.shape[i]

        return type(t)(dr.gather(type(t.array), t.array, index), tuple(shape))

