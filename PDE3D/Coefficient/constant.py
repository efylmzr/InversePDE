from .coefficient import *

class ConstantCoefficient(Coefficient):
    DRJIT_STRUCT = { 
        'value' : mi.Float
        }
    
    def __init__(self, name: str, value: float = 0):
        self.is_zero = (value == 0)
        self.value = mi.Float(value)
        self.name = name
        self.type = "constant"
        self.constant_thickness = dr.inf
        dr.make_opaque(self.value)

    def get_value(self, points : mi.Point3f):
        return self.value

    def get_grad_laplacian(self, points: mi.Point3f): # type: ignore
        return dr.zeros(mi.Point3f, dr.width(points)), dr.zeros(mi.Float, dr.width(points)) 

    def get_opt_params(self, param_dict: dict, opt_params: list):
        for key in opt_params:
            vals = key.split(".")
            name = vals[0]
            type = vals[1]
            param = vals[2]
            if name == self.name and type == self.type:
                if param == "value":
                    param_dict[key] = self.value
                else:
                    raise Exception(
                        f"ConstantCoefficient ({self.name}) does not have a parameter called \"{param}\"")

    def update(self, optimizer):
        for key in optimizer.keys():
            vals = key.split(".")
            name = vals[0]
            type = vals[1]
            param = vals[2]
            if (name == self.name) & (type == self.type) & (param == "value"):
                self.value = optimizer[key]

    def zero_grad(self):
        if dr.grad_enabled(self.value):
            dr.set_grad(self.value, 0.0)

    def copy(self):
        new = ConstantCoefficient(self.name, self.value)
        new.zero_grad()
        return new