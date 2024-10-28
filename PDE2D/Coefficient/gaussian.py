from .coefficient import *
from mitsuba import UInt

class GaussianMixtureCoefficient(Coefficient):
    def __init__(self, name, mean, std, power=1.0, corr=1, bias=0, num_lobes: int = 1):
        self.name = name
        self.is_zero = False
        self.type = "gaussian"
        self.power = mi.Float(power) if (dr.width(
            mi.Float(power)) == num_lobes) else dr.full(mi.Float, power, num_lobes)
        std0 = mi.Float(std[0]) if (dr.width(mi.Float(std[0]))
                                    == num_lobes) else dr.full(mi.Float, std[0], num_lobes)
        std1 = mi.Float(std[1]) if (dr.width(mi.Float(std[1]))
                                    == num_lobes) else dr.full(mi.Float, std[1], num_lobes)
        self.std = mi.Vector2f(std0, std1)
        self.corr = mi.Float(corr) if (
            dr.width(mi.Float(corr)) == num_lobes) else dr.full(mi.Float, corr, num_lobes)
        self.bias = mi.Float(bias)
        mean0 = mi.Float(mean[0]) if (dr.width(
            mi.Float(mean[0])) == num_lobes) else dr.full(mi.Float, mean[0], num_lobes)
        mean1 = mi.Float(mean[1]) if (dr.width(
            mi.Float(mean[1])) == num_lobes) else dr.full(mi.Float, mean[1], num_lobes)
        self.mean = mi.Vector2f(mean0, mean1)
        self.num_lobes = num_lobes

    def get_lobe_params(self, lobe_num: UInt):
        mean = dr.gather(mi.Vector2f, self.mean, lobe_num)
        std = dr.gather(mi.Vector2f, self.std, lobe_num)
        power = dr.gather(mi.Float, self.power, lobe_num)
        corr = dr.gather(mi.Float, self.corr, lobe_num)
        return mean, std, corr, power

    def get_value(self, points):
        value = mi.Float(0)
        for i in range(self.num_lobes):
            mean, std, corr, power = self.get_lobe_params(mi.Float(i))
            m = dr.rcp(1 - dr.sqr(corr))
            A = dr.rcp(2 * dr.pi * std[0] * std[1]) * dr.sqrt(m)
            XY = (points - mean) / std
            exponent = dr.sqr(XY[0]) + dr.sqr(XY[1]) - 2 * corr * XY[0] * XY[1]
            exponent *= (-m / 2)
            value += A * power * dr.exp(exponent)
        return value + self.bias

    def get_grad_laplacian(self, points):
        grad = mi.Vector2f(0)
        laplacian = mi.Float(0)
        for i in range(self.num_lobes):
            mean, std, corr, power = self.get_lobe_params(mi.Float(i))
            m = dr.rcp(1 - dr.sqr(corr))
            A = dr.rcp(2 * dr.pi * std[0] * std[1]) * dr.sqrt(m)
            XY = (points - mean) / std
            exponent = dr.sqr(XY[0]) + dr.sqr(XY[1]) - 2 * corr * XY[0] * XY[1]
            exponent *= (-m / 2)
            E = dr.exp(exponent)
            C_x = -m * (XY[0] - corr * XY[1]) / std[0]
            C_y = -m * (XY[1] - corr * XY[0]) / std[1]
            grad_ = A * E * power * mi.Vector2f(C_x, C_y)
            grad += grad_
            k = dr.rcp(dr.sqr(std[0])) + dr.rcp(dr.sqr(std[1]))
            laplacian += A * E * power * (dr.sqr(C_x) + dr.sqr(C_y) - m * k)
        return grad, laplacian

    def zero_grad(self):
        for param in [self.power, self.std, self.corr, self.bias, self.mean]:
            if dr.grad_enabled(param):
                dr.set_grad(param, 0.0)

    def get_opt_params(self, param_dict: dict, opt_params: list):
        for i in opt_params:
            if i == "mean":
                param_dict[f"{self.name}.{self.type}.mean"] = self.mean
            elif i == "std":
                param_dict[f"{self.name}.{self.type}.std"] = self.std
            elif i == "power":
                param_dict[f"{self.name}.{self.type}.power"] = self.power
            elif i == "correlation":
                param_dict[f"{self.name}.{self.type}.correlation"] = self.corr
            elif i == "bias":
                param_dict[f"{self.name}.{self.type}.bias"] = self.bias
            else:
                raise Exception(
                    f"Gaussian Coefficient ({self.name}) does not have a parameter called \"{i}\"")

    def update(self, optimizer):
        for key in optimizer.keys():
            vals = key.split(".")
            name = vals[0]
            type = vals[1]
            param = vals[2]
            if (name == self.name) & (type == self.type):
                if param == "mean":
                    self.mean = optimizer[key]
                elif param == "std":
                    self.std = optimizer[key]
                elif param == "power":
                    self.power = optimizer[key]
                elif param == "correlation":
                    self.corr = optimizer[key]
                elif param == "bias":
                    self.bias = optimizer[key]
                else:
                    raise Exception(
                        f"Gaussian Coefficient ({self.name}) does not have a parameter called \"{param}\"")
