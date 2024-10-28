from .coefficient import *
import mitsuba as mi
import numpy as np

class TextureCoefficient(Coefficient):
    def __init__(self, name: str, bbox: mi.BoundingBox3f, tensor_np: np.array,
                 interpolation: str = "cubic", wrapping: str = "clamp"):
        #tensor_np = np.swapaxes(tensor_np, 0,2)
        self.is_zero = False
        self.interpolation = interpolation
        self.tensor = mi.TensorXf(tensor_np[..., np.newaxis])
        self.name = name
        self.type = "texture"
        self.wrapping = wrapping
        self.bbox = bbox
        dr.make_opaque(self.tensor)
        self.grad_zero_mask = None
        self.texture = None
        self.update_texture()

    def create_texture(self, tensor : mi.TensorXf):
        # Creating the texture!
        wrap_mode = None
        if self.wrapping == "clamp":
            wrap_mode = dr.WrapMode.Clamp
        elif self.wrapping == "mirror":
            wrap_mode = dr.WrapMode.Mirror
        elif self.wrapping == "repeat":
            wrap_mode = dr.WrapMode.Repeat
        else:
            raise Exception("Such wrapping is not defined.")
            
        filter_type = dr.FilterMode.Nearest if self.interpolation=="nearest" else dr.FilterMode.Linear
        return mi.Texture3f(tensor, wrap_mode=wrap_mode, use_accel=False, migrate=False, filter_mode=filter_type)

    def update_texture(self):
        if self.texture is None:
            self.texture = self.create_texture(self.tensor)
        else:
            self.texture.set_tensor(self.tensor)

    def get_value(self, points: mi.Point3f):
        p_ =  (points - self.bbox.min) / (self.bbox.max - self.bbox.min)
        p = mi.Point3f(p_[2], p_[1], p_[0])
        if (self.interpolation == "cubic"):
            return self.texture.eval_cubic(p)[0]
        elif (self.interpolation == "linear" or self.interpolation == "nearest"):
            return self.texture.eval(p)[0]
        else:
            raise Exception(
                f"There is no interpolation called \"{self.interpolation}\"")

    def get_grad_laplacian(self, points):
        return self.get_grad_laplacian_(points, self.texture)
    
    def get_grad_laplacian_(self, points: mi.Point3f, texture : mi.Texture3f):
        if not self.interpolation == "cubic":
            raise Exception("Laplacian is only defined for cubic interpolation.")
        dilate = self.bbox.max - self.bbox.min
        p_ = (points - self.bbox.min) / dilate
        p = mi.Point3f(p_[2], p_[1], p_[0])
        eval_result = texture.eval_cubic_hessian(p)
        grad = eval_result[1][0]
        hessian_image = eval_result[2][0]
        laplacian = hessian_image[0, 0] / dr.square(dilate[2]) + hessian_image[1, 1] / dr.square(dilate[1]) + hessian_image[2,2] / dr.square(dilate[0])
        return mi.Point3f(grad[2], grad[1], grad[0]) / dilate, mi.Float(laplacian)
    
    def get_opt_params(self, param_dict: dict, opt_params: list):
        for key in opt_params:
            vals = key.split(".")
            name = vals[0]
            type = vals[1]
            param = vals[2]
            if name == self.name and type == self.type:
                if param == "tensor":
                    param_dict[key] = self.tensor
                else:
                    raise Exception(
                        f"TextureCoefficient ({self.name}) does not have a parameter called \"{param}\"")
                
    def copy(self):
        new = TextureCoefficient(self.name, self.bbox, self.tensor.numpy()[...,0],
                                 self.interpolation, self.wrapping)
        new.zero_grad()
        return new

    def update(self, optimizer):
        name_tensor = f"{self.name}.{self.type}.tensor"
        if name_tensor in optimizer.keys():
            self.tensor = optimizer[name_tensor]
        self.update_texture()

    def zero_grad(self):
        if dr.grad_enabled(self.tensor):
            dr.set_grad(self.tensor, 0.0)
