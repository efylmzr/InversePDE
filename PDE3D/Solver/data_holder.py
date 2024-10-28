import mitsuba as mi
import drjit as dr
import numpy as np
from PDE3D.Coefficient import *
from PDE3D.BoundaryShape import *
from PDE3D.utils import *
from PDE3D.Sampling import *
from PDE3D.utils import *
from enum import IntEnum

class DataHolder(object):
    def __init__(self, shape: Shape = Shape(), bbox = [[-1, -1, -1], [1,1,1]],
                 max_z = 5,
                 α : Coefficient = ConstantCoefficient("diffusion", 1), 
                 σ : Coefficient = ConstantCoefficient("screening", 0), 
                 f : Coefficient = ConstantCoefficient("source", 0), 
                 α_split : Coefficient = None,
                 σ_split : Coefficient = None, 
                 opt_param_shape: list = [], opt_param_α: list = [], 
                 opt_param_σ: list = [], opt_param_f: list = [], 
                 majorant_safety_low: float = 1.2, 
                 majorant_safety_high : float = 1.2,
                 default_majorant : float = None, 
                 verbose = False):
        self.shape = shape
        self.bbox = mi.BoundingBox3f(bbox[0], bbox[1])
        self.verbose = verbose
        self.α = α
        self.σ = σ
        self.f = f
        # These are defined for fd computations.
        # When we deviate the coefficients, path splitting weights change 
        # We want fd forward computations to follow the same exact path.
        self.α_split = α_split if (α_split is not None) else α
        self.σ_split = σ_split if (σ_split is not None) else σ
        self.params_shape = opt_param_shape
        self.params_f = opt_param_f
        self.params_σ = opt_param_σ
        self.params_α = opt_param_α
        self.majorant_safety_high = majorant_safety_high
        self.majorant_safety_low = majorant_safety_low
        self.default_majorant = default_majorant
        self.max_z = mi.Float(max_z)

        if self.default_majorant is not None:
            self.σ_bar = self.default_majorant
        else:
            self.update_σbar()
            
    def σ_(self, σ, α, grad_α, laplacian_α):   # Equation 21 (2nd paper)
        return σ / α + 1/2 * (laplacian_α / α - dr.squared_norm(grad_α)/(2 * (α ** 2)))
    
    def update_σbar(self):
        if self.default_majorant is None:
            points = create_bbox_points(self.shape.scene.bbox(), [128, 128, 128], spp = 1, centered=True)
            mask = self.shape.inside_closed_surface_mask(points)
            σvals = self.σ_split.get_value(points)
            αvals = self.α_split.get_value(points)
            grad_α, laplacian_α = self.α_split.get_grad_laplacian(points)
            effective_σ = self.σ_(σvals, αvals, grad_α, laplacian_α)
            effective_σ = dr.select(mask, effective_σ, 0)
            majorant = dr.maximum(effective_σ * self.majorant_safety_high, -effective_σ * self.majorant_safety_low) 
            σ_bar =dr.max(majorant)
            σ_bar = dr.maximum(1e-3, σ_bar)
            self.σ_bar = dr.opaque(mi.Float, σ_bar[0], shape = (1))

    
    def get_coefficient(self, name : str = "diffusion"):
        if name == "diffusion":
            return self.α
        elif name == "screening":
            return self.σ
        elif name == "source":
            return self.f
        else:
            raise Exception("There is no such coefficient.")
    
    
    def get_opt_params(self, param_dict: dict, opt_params: list):
        #self.shape.get_opt_params_shape(param_dict, opt_params)
        self.α.get_opt_params(param_dict, opt_params)
        self.σ.get_opt_params(param_dict, opt_params)
        self.f.get_opt_params(param_dict, opt_params)

    def update(self, opt):
        #self.shape.update(opt)
        self.f.update(opt)
        self.σ.update(opt)
        self.α.update(opt)
        self.α_split = self.α
        self.σ_split = self.σ
        self.update_σbar()
                 
    def zero_grad(self):
        self.α.zero_grad()
        self.σ.zero_grad()
        self.f.zero_grad()
        self.shape.zero_grad()


    
    #def upsample2(self, coefficient = "diffusion"):
    #    coeff = self.get_coefficient(coefficient)
    #    coeff.upsample2()