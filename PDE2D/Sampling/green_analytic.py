import drjit as dr
import mitsuba as mi
from PDE2D import DIM
from PDE2D.Sampling.special import *
from .green import GreensFunction
 
z_threshold = mi.Float(0.05)

class GreensFunctionAnalytic(GreensFunction):
    def __init__(self, dim : DIM, grad : bool = False, newton_steps : int = 5) -> None:
        """
        The parameter ``newton_it`` specifies how many Newton iteration steps
        the implementation should perform in the ``.sample()`` method following
        initialization from a starting guess.
        """

        super().__init__(dim, grad, newton_steps)

    @dr.syntax # type: ignore
    def eval(self, r:mi.Float, radius:mi.Float, σ: mi.Float) -> mi.Float:
        z = radius * dr.sqrt(σ)
        y = r * dr.rcp(radius)
        yz = y * z
        rcpyz = dr.rcp(yz)
        rcpz = dr.rcp(z)
        val = mi.Float(0)

        if dr.hint(self.dim == DIM.Two, mode = 'scalar'):
            if dr.hint(self.is_grad, mode = 'scalar'):
                #raise Exception("Not Implemented.")
                if z < z_threshold:
                    val = 1 - dr.square(y)
                else:
                    val = yz * dGσ(y, z)
            else:
                if z < z_threshold:
                    val = dr.select(r ==0, 0, -r * dr.log(y))
                else:
                    val = r * Gσ(y, z) 
        else:
            #raise Exception("Not implemented.")
            if dr.hint(self.is_grad, mode = 'scalar'):
                if z < z_threshold:
                    val = 1 - y * dr.square(y)
                else:
                    val = yz * (dr.exp(-yz) * (1 + rcpyz) -
                                dr.exp(-z) * (1 + rcpz) *  ( (dr.cosh(yz) - dr.sinh(yz) * rcpyz) *  dr.rcp(dr.cosh(z) - dr.sinh(z) * rcpz) ))
                val = dr.select(y <= 0, 1, val)
                val = dr.select(y >= 1, 0, val)
        
            else:
                if z < z_threshold:
                    val = r * (1 - y)
                else:
                    val = radius * y * yz * (dr.exp(-yz) * dr.rcp(yz) - 
                                                dr.exp(-z) * dr.rcp(yz) * dr.sinh(yz) * dr.rcp(dr.sinh(z)))
                    val = dr.select(y == 0, 0, val)
                    val = dr.select(y == 1, 0, val)
                    
        val = dr.select((y>=0) & (y<=1), val, 0)
        return val
    

    @dr.syntax # type: ignore
    def eval_pdf(self, r: mi.Float, radius: mi.Float, σ : mi.Float) -> tuple[mi.Float, mi.Float, mi.Float]:
        norm = self.eval_norm(radius, σ)
        val = self.eval(r, radius, σ)
        pdf = val * dr.rcp(norm)
        cdf = mi.Float(0)
        y = r * dr.rcp(radius)
        z = radius * dr.sqrt(σ)
        coshz = dr.cosh(z)
        sinhz = dr.sinh(z)
        yz = y * z
        zyz = z - yz
        y2 = dr.square(y)

        if dr.hint(self.dim == DIM.Two, mode = 'scalar'):
            if dr.hint(self.is_grad, mode = 'scalar'):
            #    raise Exception("Not implemented")
                if z < z_threshold:
                    cdf = y * (1.5 - dr.square(y) * 0.5)
                else:
                    cdf = mi.Float(dr.nan) # Other case requires evaluation of very expensive and complex functions.
            else:
                if z < z_threshold:
                    cdf = dr.square(y) * (1 - 2 * dr.log(y))
                else:
                    cdf = Gσr_int(y,z) * dr.rcp(σ * norm)

        else:
            #raise Exception("Not implemented.")
            if dr.hint(self.is_grad, mode = 'scalar'):
                if z < z_threshold:
                    cdf = (4 * y - dr.square(y2)) / 3
                else:
                    cdf = ((-2* coshz + (2- yz * z) * dr.cosh(zyz) + 2 * z * dr.sinh(z) + (y-2) * z * dr.sinh(zyz)) / 
                            (2 - dr.square(z) - 2 * dr.cosh(z) + 2 * z * sinhz))
            else:
                if z < z_threshold:
                    cdf = dr.square(y) * (3 - 2 * y)
                else:
                    cdf = (yz * dr.cosh(zyz) - dr.sinh(z) + dr.sinh(zyz)) * dr.rcp(z - dr.sinh(z))
        
        if y <= 0:
            cdf = mi.Float(0)
        if y >= 1: 
            cdf = mi.Float(1)
        return pdf, cdf, norm
    
    @dr.syntax # type: ignore
    def eval_norm(self, radius : mi.Float, σ : mi.Float) -> mi.Float:
        norm = mi.Float(0)
        z = radius * dr.sqrt(σ)
        coshz = dr.cosh(z)
        sinhz = dr.sinh(z)  
        
        if dr.hint(self.dim == DIM.Two, mode = 'scalar'):
            if dr.hint(self.is_grad, mode = 'scalar'):
                raise Exception("Not Implemented")
                if z < z_threshold:
                    norm = 2 * radius / 3
                else:
                    norm = mi.Float(dr.nan) # Other case requires evaluation of very expensive and complex functions.
            else:
                if z < z_threshold:
                    norm = dr.square(radius) / 4 
                else:
                    norm = dr.rcp(σ) * (1.0 - dr.rcp(i0(z)))
                    
        else:
            #raise Exception("Not Implemented")
            if dr.hint(self.is_grad, mode = 'scalar'):
                if z < z_threshold:
                    norm = 3 * radius / 4
                else: 
                    norm = radius * (2 - dr.square(z) - 2 * coshz + 2 * z * sinhz) * dr.rcp(z * (z * coshz - sinhz))
            else:
                if z < z_threshold:
                    norm = dr.square(radius) / 6
                else:
                    norm = dr.rcp(σ) * (1 - z * dr.rcp(dr.sinh(z)))
        return norm

    @dr.syntax # type: ignore
    def sample(self, x: mi.Float, radius: mi.Float, σ: mi.Float) -> tuple[mi.Float, mi.Float]:
        # The expression to initialize the Newton iteration is numerically
        # unstable when 'z' is too small. Clamp to 1e-1 (for this part only)
        z = dr.sqrt(σ)
        z_init = dr.maximum(z, 1e-1)
        b = None

        if dr.hint(not self.is_grad, mode='scalar'):
            if dr.hint(self.dim == DIM.Two, mode='scalar'):
                # Based on 'Sample3Composed2' from the Mathematica notebook
                sqrt_x = dr.sqrt(x)
                b = 1 - dr.acosh(dr.fma(dr.cosh(z_init), 1 - sqrt_x, sqrt_x)) / z_init
            elif self.dim == DIM.Three:
                # Based on 'Sample2Composed1' from the Mathematica notebook
                b = (1 - dr.acosh(dr.fma(dr.cosh(z_init), 1 - x, x)) / z_init) ** (2 / 3)
            else:
                raise RuntimeError("Unsupported number of dimensions!")
        else:
            # No good sampling strategy yet
            b = (1 - dr.sqrt(1-x))

        # Bracketing interval
        a, c = mi.Float(0), mi.Float(1)

        # Iteration counter
        i = mi.UInt32(0)
        norm = mi.Float(0)
        while i < self.newton_steps:
            # Perform a Newton step
            deriv, cdf, norm = self.eval_pdf(b * radius, radius, σ)
            deriv *= radius 
            b = b - (cdf - x) / deriv

            # Newton-Bisection: potentially reject the Newton step
            bad_step = ~((b >= a) & (b <= c))
            b = dr.select(bad_step, (a + c) / 2, b)

            # Update bracketing interval
            is_neg = self.eval_pdf(b * radius, radius, σ)[1] - x < 0
            a = dr.select(is_neg, b, a)
            c = dr.select(is_neg, c, b)

            i += 1
        return b * radius, norm 
    
    @dr.syntax # type: ignore
    def eval_poisson_kernel(self, r : mi.Float, radius : mi.Float, σ : mi.Float):
        
        # There is no such relation for poisson kernel in gradient.
        # I did not look to the 3D case.
        assert (not self.is_grad) & (self.dim == DIM.Two)
        
        z = radius * dr.sqrt(σ)
        y = r/radius

        result = mi.Float(0)
        if z < z_threshold:
            result =  1 - dr.square(y) * (1 - 2 * dr.log(y)) * self.eval_norm(radius, σ) *  σ
        else: 
            result = 1- Gσr_int(r/radius, z)
        return result
        