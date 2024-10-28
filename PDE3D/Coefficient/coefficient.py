import mitsuba as mi
import numpy as np
import drjit as dr

class Coefficient(object):
    def __init__(self, name):
        self.name = name
        self.type = ""
        self.is_zero = False
        self.constant_thickness = 0
    
    def get_value(self, points : mi.Point3f) -> mi.Float:
        pass
    
    def get_grad_laplacian(self, points : mi.Point3f):
        pass
    
    def get_opt_params(self, param_dict : dict, opt_params : list["str"]):
        pass

    def update(self, optimizer):
        pass
    
    def zero_grad(self):
        pass
    
    def copy(self):
        pass