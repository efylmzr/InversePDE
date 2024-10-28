
from .coefficient import *

class FunctionCoefficient(Coefficient):
    DRJIT_STRUCT = { 
        'parameters' : dict,
        'function_generator' : callable,
        }
    def __init__(self, name: str, parameters: dict, function_generator: callable,
                 grad_generator: callable = None, laplacian_generator: callable = None):
        self.is_zero = False
        self.parameters = parameters
        self.function_generator = function_generator
        self.grad_generator = grad_generator
        self.laplacian_generator = laplacian_generator
        self.name = name
        self.type = "CustomFunction"
        for key in parameters.keys():
            dr.make_opaque(parameters[key])
        self.update_function()

    def update_function(self):
        self.function = lambda points : self.function_generator(points, self.parameters)
        if ((self.grad_generator is not None) & (self.laplacian_generator is not None)):
            self.grad = lambda points : self.grad_generator(points, self.parameters)
            self.laplacian = lambda points : self.laplacian_generator(points, self.parameters)
        else:
            self.grad = None
            self.laplacian = None

    def get_value(self, points):
        return self.function(points)

    def get_grad_laplacian(self, points):
        if ((self.grad is not None) & (self.laplacian is not None)):
            return self.grad(points), self.laplacian(points)
        else:
            raise Exception(
                f"Laplacian or gradient is not defined for the function coefficient\"{self.name}\"!")

    def get_opt_params(self, param_dict: dict, opt_params: list):
        for i in opt_params:
            param_exists = False
            for j in self.parameters.keys():
                if i == j:
                    param_dict[f"{self.name}.{self.type}.{i}"] = self.parameters[i]
                    param_exists = True
            if not param_exists:
                raise Exception(
                    f"Function coefficient \"{self.name}\" of type \"{self.type}\" does not have parameter called \"{i}\".")

    def update(self, optimizer):
        param_exists = False
        for key in optimizer.keys():
            vals = key.split(".")
            name = vals[0]
            type = vals[1]
            param = vals[2]
            if (name == self.name) & (type == self.type):
                for p in self.parameters.keys():
                    if p == param:
                        param_exists = True
                        self.parameters[p] = optimizer[key]
        if param_exists:
            self.update_function()

    def zero_grad(self):
        for key in self.parameters.keys():
            if dr.grad_enabled(self.parameters[key]):
                dr.set_grad(self.parameters[key], 0.0)

    def copy(self):
        new = FunctionCoefficient(self.name, self.parameters, self.function_generator,
                                  self.grad_generator, self.laplacian_generator)
        return new
