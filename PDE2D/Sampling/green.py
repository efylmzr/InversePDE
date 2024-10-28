import mitsuba as mi
from PDE2D import DIM
from PDE2D.Sampling.special import *
from mitsuba import Float
 
z_threshold = Float(0.05)

class GreensFunction:
    def __init__(self, dim : DIM, grad : bool = False, newton_steps : int = 5) -> None:
        """
        The parameter ``newton_it`` specifies how many Newton iteration steps
        the implementation should perform in the ``.sample()`` method following
        initialization from a starting guess.
        """
        self.dim = dim
        self.newton_steps = newton_steps
        self.is_grad = grad
    
    def initialize(self, z : Float) -> None:
        pass

    def eval(self, r:Float, radius:Float, σ: Float) -> Float:
        return Float(0)
    
    def eval_pdf(self, r: Float, radius: Float, σ : Float) -> tuple[Float, Float, Float]:
        return Float(0), Float(0), Float(0)
    
    def eval_norm(self, radius : Float, σ : Float) -> Float:
        return Float(0)

    def sample(self, x: Float, radius: Float, σ: Float) -> tuple[Float, Float]:
        return Float(0), Float(0)

    def eval_poisson_kernel(self, r : Float, radius : Float, σ : Float) -> Float:
        return Float(0)
    
    def eval_pdf_only(self, r : Float, radius : Float, σ : Float) -> Float:
        norm = self.eval_norm(radius, σ)
        val = self.eval(r, radius, σ)
        pdf = val * dr.rcp(norm)
        return pdf