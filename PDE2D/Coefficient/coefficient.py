import numpy as np
import drjit as dr
import mitsuba as mi
from ..utils.sketch import plot_function

class Coefficient(object):
    def __init__(self, name):
        self.name = name
        self.type = ""
        self.is_zero = False
        self.constant_thickness = 0
    
    def get_value(self, points):
        pass
    def get_grad_laplacian(self, points):
        pass
    def get_opt_params(self, param_dict : dict, opt_params : list["str"]):
        pass
    def update(self, optimizer):
        pass
    def zero_grad(self):
        pass
    
    def copy(self):
        pass
    
    def visualize(self, ax, bbox, resolution = [1024, 1024], spp = 4, colorbar = True, input_range = [None, None], cmap = "viridis"):
        return plot_function(ax, self.get_value, bbox, resolution, spp, colorbar, input_range, cmap)
        
    def visualize_grad(self):
        pass

    def upsample2(self):
        pass
        

    

                
                

