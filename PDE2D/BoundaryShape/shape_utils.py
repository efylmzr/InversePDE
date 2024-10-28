from .bezierquadratic import *
from PDE2D.Coefficient import *

def load_bunny(scale = 1, dirichlet  = None, neumann = None, all_dirichlet = False, epsilon = 1e-5, conf : int = 1):
    points = np.array([[ 36.0,  -28.6],
                    [ 49.9,  -25.2],
                    [ 66.6,  -38.7],
                    [ 67.2,  -47.3],
                    [ 71.2,  -52.8],
                    [ 65.1,  -55.7],
                    [ 61.7,  -56.0],
                    [ 40.7,  -57.0],
                    [ 14.2,  -56.8],
                    [ 12.0,  -54.3],
                    [ 14.3,  -50.4],
                    [ 13.6,  -44.4],
                    [ 12.9,  -41.0],
                    [ 11.0,  -40.0],
                    [  9.0,  -38.9],
                    [  8.2,  -29.0],
                    [ 18.3,  -20.9],
                    [ 25.5,  -9.2],
                    [ 32.7,  -5.2],
                    [ 33.5, -13.0],
                    [ 29.9, -20.5],
                    [ 31.1, -27.6]]) / 38 + np.array([-1 ,0.9])

    normals = np.array([[ -3.3,  9.3],
                    [  0.4,  10.0],
                    [  7.5, -0.0],
                    [  1.5,  4.6],
                    [  5.8, -1.4],
                    [ -1.6, -6.1],
                    [  1.8, -6.2],
                    [  0.0, -6.2],
                    [ -0.8, -5.0],
                    [ -4.1,  3.9],
                    [ -7.3,  0.5],
                    [ -7.3, -0.8],
                    [ -2.0,  -1.0],
                    [ -0.4,  -3.0],
                    [ -3.1, -4.5],
                    [ -7.3,  5.0],
                    [ -3.2,  8.1],
                    [ -8.3,  1.9],
                    [  0.2,  1.0],
                    [  7.9, -3.8],
                    [  7.9, -2.8],
                    [  4.7,  5.1]])
    
    if conf == 1:
        dirichlet_map = np.array([False, 
                                False, 
                                True, 
                                True, 
                                False, 
                                True, 
                                True, 
                                False, 
                                True, 
                                True, 
                                True, 
                                True, 
                                True, 
                                True, 
                                True, 
                                False, 
                                True, 
                                False, 
                                False,
                                True,
                                True,
                                True])

    elif conf == 2:
        dirichlet_map = np.array([True, 
                                True, 
                                False, 
                                True, 
                                True, 
                                True, 
                                False, 
                                False, 
                                True, 
                                True, 
                                True, 
                                True, 
                                True, 
                                True, 
                                False, 
                                False, 
                                True, 
                                False, 
                                False,
                                False,
                                True,
                                True])
    elif conf == 3:
        dirichlet_map = np.array([True, 
                                True, 
                                True, 
                                True, 
                                False, 
                                True, 
                                True, 
                                True, 
                                False, 
                                True, 
                                True, 
                                True, 
                                True, 
                                True, 
                                False, 
                                True, 
                                True, 
                                False, 
                                False,
                                True,
                                True,
                                True]) 
    elif conf == 4:
        dirichlet_map = np.array([False, 
                                False, 
                                True, 
                                False, 
                                False, 
                                True, 
                                False, 
                                False, 
                                True, 
                                True, 
                                False, 
                                True, 
                                True, 
                                True, 
                                False, 
                                False, 
                                True, 
                                False, 
                                False,
                                False,
                                True,
                                True])
    
    elif conf == 5:
        dirichlet_map = np.array([False, 
                                False, 
                                True, 
                                False, 
                                False, 
                                True, 
                                False, 
                                False, 
                                True, 
                                True, 
                                True, 
                                True, 
                                True, 
                                True, 
                                False, 
                                True, 
                                True, 
                                True, 
                                False,
                                False,
                                True,
                                True])

    
    
    if all_dirichlet:
        dirichlet_map = np.ones_like(dirichlet_map, dtype=np.bool_)
    points = Point2f(points.T)
    normals = dr.normalize(Point2f(normals.T))
    return QuadraticBezierShape(points.numpy(), normals.numpy(), dirichlet = dirichlet, 
                                neumann = neumann, epsilon = epsilon,
                                dirichlet_map = dirichlet_map, n_segment = 20, newton_steps = 5)

def load_boundary_data(only_dirichlet = False, constant = False, zero = False):
    dirichlet_coeffs = []
    neumann_coeffs = []

    if zero:
        return [ConstantCoefficient("coeff", 0)], [ConstantCoefficient("coeff", 0)]


    if only_dirichlet:
        constant_values = [0, 2, -2]
        for c in constant_values:
            dirichlet_coeffs.append(ConstantCoefficient("coeff", c))
    else:
        constant_values = [0, 2, 20, -2, -20]
        for c1 in constant_values:
            for c2 in constant_values:
                dirichlet_coeffs.append(ConstantCoefficient("coeff", c1))
                neumann_coeffs.append(ConstantCoefficient("coeff", c2))

    if constant:
        return dirichlet_coeffs, neumann_coeffs


    def ramp(points, parameters):
        direction = dr.normalize(parameters["direction"])
        z = dr.dot(points, direction)
        return z * parameters["ramp"] + parameters["bias"]
    
    def freq(points, parameters):
        direction = dr.normalize(parameters["direction"])
        z = dr.dot(points, direction)
        return parameters["power"] * dr.cos(2 * dr.pi * parameters["freq"] * z) + parameters["bias"]

    directions = [[0., 1.], [1, 0], [1., 1]]
    ramp_values = [1, 3, 10]
    for direction in directions:
        for ramp_v in ramp_values:
            for bias in [-ramp_v, 0, ramp_v]:
                p_ramp = {}  
                dir = Point2f(direction)
                dr.make_opaque(dir)
                p_ramp["direction"] = dir
                p_ramp["ramp"] = dr.opaque(Float, ramp_v, shape = (1))
                p_ramp["bias"] = dr.opaque(Float, bias, shape = (1))
                dirichlet_coeffs.append(FunctionCoefficient("coeff", dict(p_ramp), ramp))
                if not only_dirichlet:
                    neumann_coeffs.append(FunctionCoefficient("coeff", dict(p_ramp), ramp))


    freqs = [2, 4, 8]
    powers = [1, 10]
    for direction in directions:
        for f in freqs:
            for power in powers:
                for bias in [-power, 0, power]:
                    p_freq = {}  
                    dir = Point2f(direction)
                    dr.make_opaque(dir)
                    p_freq["direction"] = dir
                    p_freq["power"] = dr.opaque(Float, power, shape = (1))
                    p_freq["freq"] = dr.opaque(Float, f, shape = (1))
                    p_freq["bias"] = dr.opaque(Float, bias, shape = (1))
                    dirichlet_coeffs.append(FunctionCoefficient("coeff", dict(p_freq), freq))
                    if not only_dirichlet:
                        neumann_coeffs.append(FunctionCoefficient("coeff", dict(p_freq), freq))

    if len(neumann_coeffs) == 0:
        neumann_coeffs.append(ConstantCoefficient("coeff", 0))

    return dirichlet_coeffs, neumann_coeffs
