from .coefficient import *
from .elliptic_disk_map import *
from PDE2D.utils.helpers import get_position_bbox
from mitsuba import TensorXf

class DiskTextureCoefficient(Coefficient):
    def __init__(self, name: str, tensor_np: np.array,
                 origin : list = [0, 0], radius : float = 1, constant_thickness = 0.1,
                 out_val = 1, interpolation: str = "cubic"):
        self.is_zero = False
        self.interpolation = interpolation
        self.tensor = TensorXf(tensor_np.squeeze()[..., np.newaxis])
        self.name = name
        self.type = "disktexture"
        self.origin = mi.Point2f(origin)
        self.radius = mi.Float(radius)
        self.constant_thickness = constant_thickness
        self.inner_radius = self.radius - self.constant_thickness
        self.bbox = [[origin[0] - self.inner_radius, origin[1] - self.inner_radius], 
                     [origin[0] + self.inner_radius, origin[1] + self.inner_radius]]
        self.out_val = mi.Float(out_val)
        dr.make_opaque(self.out_val)
        dr.make_opaque(self.tensor)
        self.update_texture()
        
    def create_tensor(self, tensor : TensorXf, expand = 2):
        # Creating the texture!
        nx = tensor.shape[0] + 2 * expand
        ny = tensor.shape[1] + 2 * expand
        nz = tensor.shape[2]
        new_tensor = TensorXf(dr.repeat(self.out_val, nx * ny * nz))
        new_tensor = dr.reshape(TensorXf, new_tensor, shape = (nx, ny, nz))
        # Get middle indices to scatter
        i = dr.arange(mi.UInt32, expand, nx - expand)
        j = dr.arange(mi.UInt32, expand, ny - expand)
        ii, jj = dr.meshgrid(i, j)
        indices = (jj - expand) * tensor.shape[1] + (ii - expand)
        indices2 = jj * ny + ii

        scatter_vals = dr.gather(mi.Float, tensor.array, mi.UInt32(indices))
        dr.scatter(new_tensor.array, scatter_vals, mi.UInt32(indices2))
        return new_tensor

    def update_texture(self):
        self.tensor2 = self.create_tensor(self.tensor)
        self.texture = mi.Texture2f(self.tensor2, use_accel=False, migrate=False)
        
    def get_value(self, points : mi.Point2f):
        r = dr.norm(points - self.origin)
        inside = r < self.inner_radius
        points_square = disk_to_square(points, origin = self.origin, radius = self.inner_radius)
        x, y = get_position_bbox(points_square, self.bbox)
        if (self.interpolation == "cubic"):
            res = self.texture.eval_cubic(mi.Point2f(x, y))[0]
        elif (self.interpolation == "linear"):
            res =  self.texture.eval(mi.Point2f(x, y))[0]
        else:
            raise Exception(
                f"There is no interpolation called \"{self.interpolation}\"")
        return dr.select(inside, res, self.out_val)
        #return res
    def get_grad_laplacian(self, points: mi.Point2f, use_tensor_only = False):
        dilate_x = self.bbox[1][0] - self.bbox[0][0]
        dilate_y = self.bbox[1][1] - self.bbox[0][1]
        
        points_square = disk_to_square(points, origin = self.origin, radius = self.inner_radius)
        x, y = get_position_bbox(points_square, self.bbox)
        eval_result = self.texture.eval_cubic_hessian(mi.Point2f(x, y))
        grad_square = eval_result[1][0] / mi.Point2f(dilate_x, -dilate_y)
        hessian_square = eval_result[2][0]
        hessian_x = hessian_square[0, 0] / (dilate_x ** 2) 
        hessian_y = hessian_square[1, 1] / (dilate_y ** 2)
        hessian_xy = hessian_square[0, 1] / (-dilate_x * dilate_y)
        jak, jak2 = inverse_jakobian(points, origin = self.origin, radius = self.inner_radius)
        
        grad_x = grad_square[0] * jak[0, 0] + grad_square[1] * jak[1, 0]
        grad_y = grad_square[0] * jak[0, 1] + grad_square[1] * jak[1, 1]
        
        laplacian_u = (hessian_x * dr.sqr(jak[0,0]) + grad_square[0] * jak2[0,0] +
                       hessian_y * dr.sqr(jak[1,0]) + grad_square[1] * jak2[1,0] +
                        2 * hessian_xy * jak[0, 0] * jak[1, 0])
        laplacian_v = (hessian_x * dr.sqr(jak[0, 1]) + grad_square[0] * jak2[0, 1] +
                       hessian_y * dr.sqr(jak[1, 1]) + grad_square[1] * jak2[1, 1] +
                       2 * hessian_xy * jak[0, 1] * jak[1, 1])
        
        grad = mi.Point2f(grad_x, grad_y)
        laplacian = laplacian_u + laplacian_v
        r = dr.norm(points - self.origin)
        return dr.select(r<self.inner_radius, grad, 0), dr.select(r<self.inner_radius, laplacian, 0)
        #return grad, laplacian
    
    def get_opt_params(self, param_dict: dict, opt_params: list):
        for key in opt_params:
            vals = key.split(".")
            name = vals[0]
            type = vals[1]
            param = vals[2]
            if name == self.name and type == self.type:
                if param == "tensor":
                    param_dict[key] = self.tensor
                elif param == "outval":
                    param_dict[key] = self.outval
                else:
                    raise Exception(
                        f"DiskTexture ({self.name}) does not have a parameter called \"{param}\"")

    def update(self, optimizer):
        name_outval = f"{self.name}.{self.type}.outval"
        if name_outval in optimizer.keys():
            self.out_val = optimizer[name_outval]
        
        name_tensor = f"{self.name}.{self.type}.tensor"
        if name_tensor in optimizer.keys():
                self.tensor = optimizer[name_tensor]
        
        self.update_texture()

    def zero_grad(self):
        if dr.grad_enabled(self.tensor):
            dr.set_grad(self.tensor, 0.0)
        if dr.grad_enabled(self.out_val):
            dr.set_grad(self.out_val, 0.0)
            
    def upsample(self, scale_factor=[2, 2]):
        self.tensor = dr.upsample(self.tensor, scale_factor=scale_factor)
        self.update_texture()


    def copy(self):
        return DiskTextureCoefficient(name = self.name, tensor_np = self.tensor.numpy(),
                                      origin  = self.origin, radius = self.radius, constant_thickness = self.constant_thickness,
                                      out_val = self.out_val, interpolation = self.interpolation)