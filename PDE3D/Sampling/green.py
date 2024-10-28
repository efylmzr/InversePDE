import drjit as dr
import mitsuba as mi
 
z_threshold = mi.Float(0.05)

class GreensFunction():
    def __init__(self, grad : bool = False, newton_steps : int = 5) -> None:
        """
        The parameter ``newton_it`` specifies how many Newton iteration steps
        the implementation should perform in the ``.sample()`` method following
        initialization from a starting guess.
        """
        self.newton_steps = newton_steps
        self.is_grad = grad
        #super().__init__(dim, grad, newton_steps)

    @dr.syntax # type: ignore
    def eval(self, r:mi.Float, radius:mi.Float, σ: mi.Float) -> mi.Float:
        z = radius * dr.sqrt(σ)
        y = r * dr.rcp(radius)
        
        val = mi.Float(0)
        
        #raise Exception("Not implemented.")
        if dr.hint(self.is_grad, mode = 'scalar'):
            if z < z_threshold:
                val = 1 - y * dr.square(y)
            else:
                rcpyz = dr.rcp(yz)
                rcpz = dr.rcp(z)
                yz = y * z
                val = yz * (dr.exp(-yz) * (1 + rcpyz) -
                            dr.exp(-z) * (1 + rcpz) *  ( (dr.cosh(yz) - dr.sinh(yz) * rcpyz) *  dr.rcp(dr.cosh(z) - dr.sinh(z) * rcpz) ))
            val = dr.select(y <= 0, 1, val)
            val = dr.select(y >= 1, 0, val)
        else:
            if z < z_threshold:
                val = r * (1 - y)
            else:
                yz = y * z
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
        

        if dr.hint(self.is_grad, mode = 'scalar'):
            if z < z_threshold:
                y2 = dr.square(y)
                cdf = (4 * y - dr.square(y2)) / 3
            else:
                yz = y * z
                zyz = z - yz
                coshz = dr.cosh(z)
                
                cdf = ((-2* coshz + (2- yz * z) * dr.cosh(zyz) + 2 * z * dr.sinh(z) + (y-2) * z * dr.sinh(zyz)) / 
                        (2 - dr.square(z) - 2 * dr.cosh(z) + 2 * z * sinhz))
        else:
            if z < z_threshold:
                cdf = dr.square(y) * (3 - 2 * y)
            else:
                yz = y * z
                zyz = z - yz
                sinhz = dr.sinh(z)
                cdf = (yz * dr.cosh(zyz) - sinhz + dr.sinh(zyz)) * dr.rcp(z - sinhz)
        
        if y <= 0:
            cdf = mi.Float(0)
        if y >= 1: 
            cdf = mi.Float(1)
        return pdf, cdf, norm
    
    @dr.syntax # type: ignore
    def eval_norm(self, radius : mi.Float, σ : mi.Float) -> mi.Float:
        norm = mi.Float(0)
        z = radius * dr.sqrt(σ)
        
        #raise Exception("Not Implemented")
        if dr.hint(self.is_grad, mode = 'scalar'):
            if z < z_threshold:
                norm = 3 * radius / 4
            else: 
                coshz = dr.cosh(z)
                sinhz = dr.sinh(z)  
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
        z = radius * dr.sqrt(σ)
        z_init = dr.maximum(z, 1e-1)
        b = None

        if dr.hint(not self.is_grad, mode='scalar'):
            # Based on 'Sample2Composed1' from the Mathematica notebook
            b = (1 - dr.acosh(dr.fma(dr.cosh(z_init), 1 - x, x)) / z_init) ** (2 / 3)
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
    def eval_poisson_kernel(self,  radius : mi.Float, σ : mi.Float):
        
        # There is no such relation for poisson kernel in gradient.
        # I did not look to the 3D case.
        
        z = radius * dr.sqrt(σ)

        result = mi.Float(0)
        if z < z_threshold:
            result =  mi.Float(1)
        else: 
            result = z / dr.sinh(z)
        return result
        