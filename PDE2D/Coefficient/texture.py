from .coefficient import *
#from PDE2D import mi.TensorXf, mi.Float, mi.Point2f, mi.Texture2f, mi.UInt32, mi.TensorXb
from PDE2D.utils.helpers import get_position_bbox
from PDE2D.utils import upsample

class TextureCoefficient(Coefficient):
    def __init__(self, name: str, bbox: list, tensor_np: np.array,
                 interpolation: str = "cubic", wrapping: str = "clamp", 
                 grad_zero_points = None, out_val : mi.Float = None):
        self.is_zero = False
        self.interpolation = interpolation
        self.tensor = mi.TensorXf(tensor_np.squeeze()[..., np.newaxis])
        self.name = name
        self.type = "texture"
        self.wrapping = wrapping
        self.bbox = bbox
        dr.make_opaque(self.tensor)
        self.grad_zero_points = grad_zero_points
        self.out_val = out_val
        self.grad_zero_mask = None
        
        if self.grad_zero_points is not None:
            mask = self.compute_grad_zero_mask()
            self.grad_zero_mask = mi.TensorXb(mask, shape = (self.tensor.shape))
            dr.eval(self.grad_zero_mask)
            if self.out_val is None:
                raise Exception("If you want to force gradient to be zero in some locations, "
                                "please specify a forced texture value (out_val).")
            else:
                dr.make_opaque(self.out_val)
        
        self.texture = None
        self.update_texture()

    def compute_grad_zero_mask(self):
        tensor = dr.arange(mi.Float, dr.width(self.tensor.array))
        tensor = mi.TensorXf(tensor, shape = self.tensor.shape)
        dr.enable_grad(tensor)
        texture = self.create_texture(tensor)
        dr.backward(self.get_grad_laplacian_(self.grad_zero_points, texture)[0])
        mask = dr.abs(dr.grad(tensor).array) > 0
        dr.disable_grad(tensor)
        return mask
        

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
        return mi.Texture2f(tensor, wrap_mode=wrap_mode, use_accel=False, migrate=False, filter_mode=filter_type)

    def update_texture(self):
        tensor = mi.TensorXf(self.tensor)
        if self.grad_zero_mask is not None:
            tensor = dr.select(self.grad_zero_mask, mi.TensorXf(dr.detach(self.out_val)), tensor)
        if self.texture is None:
            self.texture = self.create_texture(tensor)
        else:
            self.texture.set_tensor(tensor)

    def get_value(self, points: mi.Point2f):
        x, y = get_position_bbox(points, self.bbox)
        if (self.interpolation == "cubic"):
            return self.texture.eval_cubic(mi.Point2f(x, y))[0]
        elif (self.interpolation == "linear" or self.interpolation == "nearest"):
            return self.texture.eval(mi.Point2f(x, y))[0]
        else:
            raise Exception(
                f"There is no interpolation called \"{self.interpolation}\"")

    def get_grad_laplacian(self, points):
        return self.get_grad_laplacian_(points, self.texture)
    
    def get_grad_laplacian_(self, points: mi.Point2f, texture : mi.Texture2f):
        if not self.interpolation == "cubic":
            raise Exception("Laplacian is only defined for cubic interpolation.")
        dilate_x = self.bbox[1][0] - self.bbox[0][0]
        dilate_y = self.bbox[1][1] - self.bbox[0][1]
        x, y = get_position_bbox(points, self.bbox)
        eval_result = texture.eval_cubic_hessian(mi.Point2f(x, y))
        grad = eval_result[1][0] / mi.Point2f(dilate_x, -dilate_y)
        hessian_image = eval_result[2][0]
        laplacian = hessian_image[0, 0] / \
            (dilate_x ** 2) + hessian_image[1, 1] / (dilate_y ** 2)
        return mi.Point2f(grad), mi.Float(laplacian)
    
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
                    param_dict[key] = self.out_val 
                else:
                    raise Exception(
                        f"TextureCoefficient ({self.name}) does not have a parameter called \"{param}\"")

    def update(self, optimizer):
        name_outval = f"{self.name}.{self.type}.outval"
        if name_outval in optimizer.keys():
            self.out_val = optimizer[name_outval]
        
        name_tensor = f"{self.name}.{self.type}.tensor"
        if name_tensor in optimizer.keys():
            #if self.grad_zero_mask is not None:
            self.tensor = optimizer[name_tensor]
        self.update_texture()


    def zero_grad(self):
        if dr.grad_enabled(self.tensor):
            dr.set_grad(self.tensor, 0.0)

    def copy(self):
        new = TextureCoefficient(self.name, self.bbox, self.tensor.numpy().squeeze(),
                                 self.interpolation, self.wrapping, self.grad_zero_points, self.out_val)
        new.zero_grad()
        return new

    def upsample(self, scale_factor=[2, 2]):
        self.tensor = dr.upsample(self.tensor, scale_factor=scale_factor)
        self.update_texture()

    def copy(self):
        return TextureCoefficient(name = self.name, bbox = self.bbox, tensor_np=self.tensor.numpy().squeeze(), 
                                  interpolation = self.interpolation, wrapping = self.wrapping, 
                                  grad_zero_points=self.grad_zero_points, out_val = self.out_val)
    
    def upsample2(self):
        self.tensor = upsample(self.tensor, scale_factor=[2,2])
        dr.eval(self.tensor)
        if self.grad_zero_points is not None:
            mask = self.compute_grad_zero_mask()
            self.grad_zero_mask = mi.TensorXb(mask, shape = (self.tensor.shape))
        self.update_texture()
        
