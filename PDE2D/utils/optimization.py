import drjit as dr
import mitsuba as mi
from PDE2D import ArrayXf
from mitsuba import TensorXf
import numpy as np

def MSE(img , img_ref = 0):
    val = img
    val_ref = img_ref
    if isinstance(val, TensorXf):
        val = val.array
    if isinstance(val_ref, TensorXf):
        val_ref = val_ref.array
    return dr.block_sum(dr.square(val - val_ref), dr.width(val)) / dr.width(val)


def MSE_image(img , img_ref = 0):
    return dr.sum(dr.square(img - img_ref).array) / (img.shape[1] * img.shape[2])

def MSE_numpy(val :np.array , val_ref : np.array = 0):
    return np.sum(np.square(val - val_ref), axis = tuple(range(1, val.ndim))) / (np.size(val) / val.shape[0])


def compute_loss_grad(result, result_ref=0):
    return (2 * (result - result_ref)) / dr.width(result)

def compute_dL(L, loss_grad, spe, electrode_nums = None, apply_normalization = False):
    if not apply_normalization:
        # The commented lines show that there is no difference between 
        # applying normalization to the primal computation.
        #normalization = dr.sum(L) / dr.width(L)
        #L = L - normalization
        #dr.enable_grad(L)
        #result = dr.block_sum(L, spe) / spe
        #dr.set_grad(result, adjoint_result)
        #dr.backward(result)
        dL = dr.repeat(loss_grad, spe) / spe
    else:
        num_active_electrodes = dr.width(electrode_nums)
        unnormalized =  dr.block_sum(L, spe) / spe
        #unnormalized =  self.block_sum(L, spe) / spe
        dr.enable_grad(unnormalized)
        bias = dr.block_sum(unnormalized, dr.width(unnormalized)) / num_active_electrodes
        result = unnormalized - dr.select(unnormalized != 0, bias, 0)
        #result = unnormalized - dr.sum(unnormalized) / dr.width(unnormalized)
        dr.enable_grad(result)
        dr.set_grad(result, loss_grad)
        dr.enqueue(dr.ADMode.Backward, result)
        dr.traverse(dr.ADMode.Backward)
        grad = dr.grad(unnormalized)
        dL = dr.repeat(grad, spe) / spe    
    return dL

def compute_loss_grad_image(result, result_ref = 0):
    return (2 * (result - result_ref)) / (result.shape[1] * result.shape[2])


def compute_dL_image(loss_grad, spp):
    size = loss_grad.shape[1] * loss_grad.shape[2] * spp
    dL = dr.zeros(ArrayXf, shape = (loss_grad.shape[0], size))
    for i in range(loss_grad.shape[0]):
        dL[i] = dr.repeat(loss_grad[i].array, spp) / spp
    return dL
