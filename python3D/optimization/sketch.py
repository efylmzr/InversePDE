from PDE3D.Coefficient import *
from PDE3D.BoundaryShape import *
from PDE3D.utils import *
import os

def plot_coeff(coeff : Coefficient, shape : Shape, 
               input_range, path : str, tag : str, 
               slice_ax = "z", cam_res = [512, 512], spp=128, cam_origin =  mi.ScalarPoint3f([7,7,10]),  
               cam_target = mi.ScalarPoint3f([0.0,0.0,0.0]), cam_up = mi.ScalarPoint3f([0,1,0]), scale_cam = 1/5,
               cmap = "viridis", start_slice = -3.5, end_slice =3.0, num_slices = 9):
    slices = []
    offsets = np.linspace(start_slice, end_slice, num_slices)
    for offset in offsets:
        slices.append(Slice(offset = offset, scale = 9, axis = slice_ax,))
    fig, ax = plt.subplots(num_slices//3, 3, figsize = (12, (num_slices//3) * 4))
    for i in range(num_slices):
        r = i // 3
        c = i % 3
        coeff3D, coeff_norm = shape.visualize(colormap = cmap, cam_origin= cam_origin, spp = spp, image_res = cam_res, 
                                            scale_cam=scale_cam, cam_up = cam_up, slice = slices[i], cam_target = cam_target, coeff=coeff,
                                            input_range = input_range)
        plot_image_3D(coeff3D, ax[r,c], norm = coeff_norm, cmap = cmap)
    
    plt.savefig(os.path.join(path, f"{tag}.pdf"), dpi = 250)

def plot_summary(loss_list, path, log = False, save_npy = True):
    losses = np.array(loss_list)
    if save_npy:
        np.save(os.path.join(path, "losses.npy"),losses)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 5))
    loss_all = losses.sum(axis = 1).squeeze()
    iters = np.arange(0, len(loss_all))
    ax1.plot(iters, loss_all)
    ax1.grid()
    ax1.set_title("Sum of All Setups")
    for i in range(losses.shape[1]):
        ax2.plot(iters, losses[:,i], label = f"Setup {i}")
    ax2.grid()
    #ax2.legend()
    if log:
        ax1.set_yscale("log")
        ax2.set_yscale("log")
    ax2.set_title("Across different setups")
    fig.suptitle("Evolution of the loss function")
    path_loss_pdf = os.path.join(path, "loss")
    if log:
        fig.savefig(f"{path_loss_pdf}-log-setups.pdf", bbox_inches='tight', pad_inches=0.04, dpi=300)
    else:
        fig.savefig(f"{path_loss_pdf}-lin-setups.pdf", bbox_inches='tight', pad_inches=0.04, dpi=300)
    plt.close(fig)

