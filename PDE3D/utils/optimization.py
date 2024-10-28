import drjit as dr
import mitsuba as mi
import numpy as np
from PDE3D import ArrayXf

def MSE(img , img_ref = 0):
    val = img
    val_ref = img_ref
    if isinstance(val, mi.TensorXf):
        val = val.array
    if isinstance(val_ref, mi.TensorXf):
        val_ref = val_ref.array
    return dr.block_sum(dr.square(val - val_ref), dr.width(val)) / dr.width(val)


def MSE_numpy(val :np.array , val_ref : np.array = 0):
    return np.sum(np.square(val - val_ref), axis = tuple(range(1, val.ndim))) / (np.size(val) / val.shape[0])

def MSE_vol(img , img_ref = 0):
    return dr.sum(dr.square(img - img_ref).array) / (img.shape[1] * img.shape[2] * img.shape[3])

def MSE_slice(img , img_ref = 0):
    return dr.sum(dr.square(img - img_ref).array) / (img.shape[1] * img.shape[2])

def compute_loss_grad_slice(result, result_ref = 0):
    return (2 * (result - result_ref)) / (result.shape[1] * result.shape[2])

def compute_dL_slice(loss_grad, spp):
    size = loss_grad.shape[1] * loss_grad.shape[2] * spp
    dL = dr.zeros(ArrayXf, shape = (loss_grad.shape[0], size))
    for i in range(loss_grad.shape[0]):
        dL[i] = dr.repeat(loss_grad[i].array, spp) / spp
    return dL

def compute_loss_grad_vol(result, result_ref = 0):
    return (2 * (result - result_ref)) / (result.shape[1] * result.shape[2] * result.shape[3])

def compute_dL_vol(loss_grad, spp):
    size = loss_grad.shape[1] * loss_grad.shape[2] * loss_grad.shape[3] * spp
    dL = dr.zeros(ArrayXf, shape = (loss_grad.shape[0], size))
    for i in range(loss_grad.shape[0]):
        dL[i] = dr.repeat(loss_grad[i].array, spp) / spp
    return dL