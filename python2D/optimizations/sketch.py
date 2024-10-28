import numpy as np
import os
import drjit as dr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PDE2D.utils.sketch import *
from PDE2D import PATH
import matplotlib.animation as animation
dpi = 200
PER_ROW = 4
SAVEDATA = True
PNG = False
PDF = True

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def iter_plot(wos, bbox, path, name, el_vals = None, el_std = None, electrode_nums = None, std = True,
              opt_param = "diffusion", resolution = [256, 256], wos_obj = None, compute_std = False, 
              max_range = [None, None], out_val : float = 1.0):
    path_reconstruction = os.path.join(path, "reconstruction")
    path_primal = os.path.join(path, "primal")
    path_grad = os.path.join(path, "grad")
    path_distance = os.path.join(path, "stepdistance")
    path_screening = os.path.join(path, "eff-screening")

    create_path(path_reconstruction)
    create_path(path_primal)
    create_path(path_grad)
    create_path(path_distance)
    create_path(path_screening)

    if SAVEDATA:
        create_path(os.path.join(path_reconstruction, "npy"))
        create_path(os.path.join(path_primal, "npy"))
        create_path(os.path.join(path_grad, "npy"))
        create_path(os.path.join(path_distance, "npy"))
        create_path(os.path.join(path_screening, "npy"))

    coeff = wos.input.get_coefficient(opt_param)
    coeff_obj = None
    if wos_obj is not None:
        coeff_obj = wos_obj.input.get_coefficient(opt_param)
    # Plot the coefficients
    plot_coeff(coeff, wos.input.shape, bbox, path_reconstruction, f"{opt_param}-{name}-scaled", resolution, coeff_obj, max_range = max_range, out_val=out_val)
    plot_coeff(coeff, wos.input.shape, bbox, path_reconstruction, f"{opt_param}-{name}", resolution, coeff_obj, out_val = out_val)
    
    if SAVEDATA:
        np.save(os.path.join(path_reconstruction, "npy", f"tensor-{name}.npy"), np.array(coeff.tensor))
        if wos_obj is not None:
            np.save(os.path.join(path_reconstruction, "npy", f"tensor-obj.npy"), np.array(coeff_obj.tensor))
    
    
    # Plot the gradient
    grad = dr.grad(coeff.tensor).numpy()
    fig1, ax1 = plt.subplots(1,1,figsize = (5,5))
    range_grad = max(np.max(grad), -np.min(grad))
    plot_image(grad, ax1, input_range = [-range_grad, range_grad], cmap = "coolwarm")

    fig2, ax2 = plt.subplots(1,1,figsize = (5,5))
    coeff_tensor = coeff.tensor.numpy().squeeze()
    plot_image(coeff_tensor, ax2)
    if PDF:
        fig1.savefig(os.path.join(path_grad, f"grad-{name}.pdf"), bbox_inches='tight', pad_inches=0.04, dpi=200)
        fig2.savefig(os.path.join(path_grad, f"tensor-{name}.pdf"), bbox_inches='tight', pad_inches=0.04, dpi=200)
    if PNG:
        fig1.savefig(os.path.join(path_grad, f"grad-{name}.png"), bbox_inches='tight', pad_inches=0.04, dpi=dpi)
        fig1.savefig(os.path.join(path_grad, f"tensor-{name}.png"), bbox_inches='tight', pad_inches=0.04, dpi=dpi)
    plt.close(fig1)
    plt.close(fig2)
    if SAVEDATA:
        np.save(os.path.join(path_grad, "npy", f"grad-{name}.npy"), grad)
        np.save(os.path.join(path_grad, "npy", f"tensor-{name}.npy"), coeff_tensor)

    # Plot effective screening
        fig, ax = plt.subplots(1,1, figsize = (5,5))
        wos.input.eff_screening_tex.visualize(ax, bbox, resolution)
        if PDF:
            fig.savefig(os.path.join(path_screening, f"{name}.pdf"), bbox_inches='tight', pad_inches=0.04, dpi=200)
        if PNG:
            fig.savefig(os.path.join(path_screening, f"{name}.pdf"), bbox_inches='tight', pad_inches=0.04, dpi=200)
        plt.close(fig)

    if wos.use_accel:
        plot_coeff(wos.input.r_best_tex, wos.input.shape, bbox, path_distance, f"distance-{name}", resolution)
        plot_coeff(wos.input.σ_best_tex, wos.input.shape, bbox, path_distance, f"sigma-{name}", resolution)
        if SAVEDATA:
            np.save(os.path.join(path_distance, "npy", f"distance-{name}.npy"), wos.input.r_best_tex.tensor.numpy())
            np.save(os.path.join(path_distance, "npy", f"sigma-{name}.npy"), wos.input.σ_best_tex.tensor.numpy())
    
    if el_vals is not None:
        s = wos.input.shape.out_boundary
        plot_iter_result(el_vals, el_std, s.voltages, s.voltages_std, s.injections, 
                        electrode_nums, s.num_electrodes, path_primal, f"{name}", std = compute_std)


def plot_coeff(coeff, shape, bbox, path, name, resolution = [1024,1024], coeff_obj = None, max_range = [None, None], out_val : float = None):
    points = create_image_points(bbox, resolution, spp=1, centered = True)
    vals = coeff.get_value(points)
    if out_val is not None:
        mask = shape.inside_closed_surface_mask(points)
        vals = dr.select(mask, vals, out_val)
    image, _ = create_image_from_result(vals, resolution)
    if coeff_obj is None:
        fig, ax = plt.subplots(1,1,figsize = (5,5))
        plot_image(image[0], ax, input_range = max_range)
        shape.sketch(ax, bbox, resolution, sketch_center = True)
    else:
        vals_obj = coeff_obj.get_value(points)
        if out_val is not None:
            mask = shape.inside_closed_surface_mask(points)
            vals_obj = dr.select(mask, vals_obj, out_val)
        image_obj, _ = create_image_from_result(vals_obj, resolution)
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10,5))
        plot_image(image_obj[0], ax1, input_range = max_range)
        plot_image(image[0], ax2, input_range = max_range)
        shape.sketch(ax1, bbox, resolution, sketch_in_boundaries = False)
        shape.sketch(ax2, bbox, resolution, sketch_center = True)
        ax1.set_title("Objective")
        ax2.set_title("Iteration")
    if PDF:
        fig.savefig(os.path.join(path, f"reconstruction-{name}.pdf"), bbox_inches='tight', pad_inches=0.04, dpi=200)
    if PNG:
        fig.savefig(os.path.join(path, f"reconstruction{name}.png"), bbox_inches='tight', pad_inches=0.04, dpi=dpi)
    plt.close(fig)
    return image


def plot_iter_result(signals, signals_std, objectives, objectives_std, injection_confs, electrode_nums, num_electrodes, 
                     path, name, selected_confs = None, std = True, name1 = "Iteration", name2 = "Objective"):
    num_confs = len(signals)
    selected_mask = np.zeros(num_confs, dtype = bool)
    if selected_confs is not None:
        selected_mask[selected_confs] = True
    
    if SAVEDATA:
        np.save(os.path.join(path, "npy", f"signal-{name}.npy"), np.array(signals))
        if std:
            np.save(os.path.join(path, "npy", f"signal_{name}-std.npy"), np.array(signals_std))
    per_row = PER_ROW
    num_rows = int(num_confs / per_row) + 1
    fig = plt.figure(figsize= (per_row * 8, num_rows * 4))
    
    g = gridspec.GridSpec(num_rows, per_row, figure = fig, wspace = 0.1, hspace=0.3)
    for i, (injection, signal, signal_std, objective, objective_std, el_nums) in enumerate(zip(injection_confs, signals, signals_std, objectives, 
                                                                    objectives_std, electrode_nums)):
        row = int(i / per_row)
        col = i % per_row
        ax = fig.add_subplot(g[row, col])
        
        #ax.set_axis_off()
        plot_primals(ax, signal, objective, el_nums, num_electrodes, std1=signal_std, std2=objective_std, fontsize = 5, label = False, name1 = name1, name2 = name2)
        
        if (selected_mask[i] == False):
            ax.set_title(f"Conf. {i}, E{injection[0]}-E{injection[1]}")
        else:
            ax.set_title(f"Conf. {i}, E{injection[0]}-E{injection[1]}", color = "red")
    
    path_iter_pdf = os.path.join(path, f"primal-{name}")
    path_iter_png = os.path.join(path, f"primal-{name}")
    if PDF:
        fig.savefig(f"{path_iter_pdf}.pdf", bbox_inches='tight', pad_inches=0.04, dpi=300)
    if PNG:
        fig.savefig(f"{path_iter_png}.png", bbox_inches='tight', pad_inches=0.04, dpi=dpi)
    plt.close(fig)

    if std:
        fig = plt.figure(figsize= (per_row * 8, num_rows * 4))

        g = gridspec.GridSpec(num_rows, per_row, figure = fig, wspace = 0.1, hspace=0.3)
        for i, (injection, signal, signal_std, objective, objective_std, el_nums) in enumerate(zip(injection_confs, signals, signals_std, objectives, 
                                                                        objectives_std, electrode_nums)):
            row = int(i / per_row)
            col = i % per_row
            ax = fig.add_subplot(g[row, col])
            
            #ax.set_axis_off()
            plot_diff_primal(ax, signal, objective, el_nums, num_electrodes, std1=signal_std, std2=objective_std, fontsize = 5)

            if (selected_mask[i] == False):
                ax.set_title(f"Conf. {i}, E{injection[0]}-E{injection[1]}")
            else:
                ax.set_title(f"Conf. {i}, E{injection[0]}-E{injection[1]}", color = "red")

        path_iter_pdf = os.path.join(path, f"primal-{name}-diff")
        path_iter_png = os.path.join(path, f"primal-{name}-diff")
        if PDF:
            fig.savefig(f"{path_iter_pdf}.pdf", bbox_inches='tight', pad_inches=0.04, dpi=300)
        if PNG:
            fig.savefig(f"{path_iter_png}.png", bbox_inches='tight', pad_inches=0.04, dpi=dpi)
        plt.close(fig)



def plot_summary(loss_list, loss_reg_list, path, log = False, save_npy = SAVEDATA):
    losses = np.array(loss_list)
    losses_reg = np.array(loss_reg_list)
    if save_npy:
        np.save(os.path.join(path, "losses.npy"),losses)
        np.save(os.path.join(path, "losses_reg.npy"),losses_reg)

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
    path_loss_png = os.path.join(path, "loss")
    if log:
        if PDF:
            fig.savefig(f"{path_loss_pdf}-log-setups.pdf", bbox_inches='tight', pad_inches=0.04, dpi=300)
        if PNG:
            fig.savefig(f"{path_loss_png}-log-setups.png", bbox_inches='tight', pad_inches=0.04, dpi=dpi)
    else:
        if PDF:
            fig.savefig(f"{path_loss_pdf}-lin-setups.pdf", bbox_inches='tight', pad_inches=0.04, dpi=300)
        if PNG:
            fig.savefig(f"{path_loss_png}-lin-setups.png", bbox_inches='tight', pad_inches=0.04, dpi=dpi)
    plt.close(fig)
    
    if losses_reg.sum() > 0:
        # Now only plot the regularization loss.
        fig, ax = plt.subplots(1,1, figsize = (5,5))
        ax.plot(iters, losses_reg)
        ax.grid()
        ax.set_title("Regularization Loss")
        if log:
            ax.set_yscale("log")
            if PDF:
                fig.savefig(f"{path_loss_pdf}-log-reg.pdf", bbox_inches='tight', pad_inches=0.04, dpi=300)
            if PNG:
                fig.savefig(f"{path_loss_png}-log-reg.png", bbox_inches='tight', pad_inches=0.04, dpi=dpi)
        else:
            if PDF:
                fig.savefig(f"{path_loss_pdf}-lin-reg.pdf", bbox_inches='tight', pad_inches=0.04, dpi=300)
            if PNG:
                fig.savefig(f"{path_loss_png}-lin-reg.png", bbox_inches='tight', pad_inches=0.04, dpi=dpi)
        plt.close(fig)

        # Now only plot the regularization loss.
        fig, ax = plt.subplots(1,1, figsize = (5,5))
        ax.plot(iters, losses_reg + loss_all)
        ax.grid()
        ax.set_title("Loss")
        if log:
            ax.set_yscale("log")
            if PDF:
                fig.savefig(f"{path_loss_pdf}-log-all.pdf", bbox_inches='tight', pad_inches=0.04, dpi=300)
            if PNG:
                fig.savefig(f"{path_loss_png}-log-all.png", bbox_inches='tight', pad_inches=0.04, dpi=dpi)
        else:
            if PDF:
                fig.savefig(f"{path_loss_pdf}-lin-all.pdf", bbox_inches='tight', pad_inches=0.04, dpi=300)
            if PNG:
                fig.savefig(f"{path_loss_png}-lin-all.png", bbox_inches='tight', pad_inches=0.04, dpi=dpi)
        plt.close(fig)

def create_animation(record, path, iternum, bbox, wos, max_range = None, wos_obj = None, resolution = [512, 512], 
                     opt_param = "diffusion.texture.tensor", fileset = None, out_val = None):

    name = "reconstruction" if max_range is None else "reconstruction-scaled"
    if (wos_obj is None) and (fileset is None):
        fig , (ax1, ax2) = plt.subplots(1,2, figsize = (10, 5))
    else:
        fig = plt.figure(figsize=(8, 6))
        g = gridspec.GridSpec(16, 20, figure = fig, wspace = 0.5, hspace=0.1)
        ax0 = fig.add_subplot(g[0:9,0:9])
        ax1 = fig.add_subplot(g[0:9,10:19])
        ax2 = fig.add_subplot(g[10:,:19])

    coeff_name = opt_param.split(".")[0]

    coeff = wos.input.get_coefficient(coeff_name)
    
    if fileset is not None:
        image_file = os.path.join(PATH ,f"eit-data/target_photos/fantom_{fileset}.jpg")
        image = plt.imread(image_file)
        ax0.imshow(image)
        ax0.set_axis_off()
        ax0.set_title("Objective")

    elif wos_obj is not None:
        coeff_obj = wos_obj.input.get_coefficient(coeff_name)

        points = create_image_points(bbox, resolution, spp=1, centered = True)
        vals = coeff_obj.get_value(points)
        if out_val is not None:
            mask = wos_obj.input.shape.inside_closed_surface_mask(points)
            vals = dr.select(mask, vals, out_val)
        obj_image, _ = create_image_from_result(vals, resolution)
        if max_range is None:
            plot_image(obj_image[0], ax0)
        else:
            plot_image(obj_image[0], ax0, input_range = max_range)
        ax0.set_title("Objective")
        wos_obj.input.shape.sketch(ax0, bbox, resolution, sketch_center = True)

    if max_range is None:
        maxval  = -np.inf
        minval = np.inf 
        for i in range(iternum):
            tensor = TensorXf(record[f"{opt_param}-{i}"]).numpy()
            maxval = max(maxval, np.max(tensor))
            minval = min(minval, np.min(tensor))
        max_range = [minval, maxval]
    coeff.tensor = TensorXf(record[f"{opt_param}-0"])
    coeff.update_texture()

    dirichlet_str = "dirichletpoints-0"
    dirichlet_points = None
    if dirichlet_str in record:
        dirichlet_points_ = record[f"dirichletpoints-0"]
        if dirichlet_points_.shape[0] > 0:
            dirichlet_points = Point2f(dirichlet_points_.T)
            dirichlet_points = point2sketch(dirichlet_points, bbox, resolution).numpy()
    
    im, s, line = start_animation(ax1, ax2, record, bbox, wos, dirichlet_points, coeff, resolution, max_range, out_val = out_val)
    update = lambda iteration : update_animation(wos, iteration, record, coeff, resolution, im, s, line, bbox, opt_param, out_val = out_val)

    ani = animation.FuncAnimation(fig=fig, func=update, frames=iternum+1, interval=30)
    writervideo = animation.FFMpegWriter(fps=25) 
    ani.save(filename=f"{path}/{name}.gif", writer="pillow")
    ani.save(f"{path}/{name}.mp4", writer=writervideo)

def start_animation(ax1, ax2, record, bbox, wos, in_boundary_points, coeff, resolution, max_range, out_val : float = None):
    points = create_image_points(bbox, resolution, spp = 4, centered = False)
    vals = coeff.get_value(points)
    if out_val is not None:
        mask = wos.input.shape.inside_closed_surface_mask(points)
        vals = dr.select(mask, vals, out_val)
    image, _ = create_image_from_result(vals, resolution)
    im = plot_image(image[0], ax1, input_range = max_range)
    wos.input.shape.sketch(ax1, bbox, resolution, sketch_center = True, sketch_in_boundaries = False)
    s = None
    if in_boundary_points is not None:
        s = ax1.scatter(in_boundary_points[0], in_boundary_points[1], s = 5, color = "red")
    
    ax1.set_title("Reconstruction")

    loss = record["loss"].sum(axis = 1).squeeze()
    loss_reg = record["loss-reg"]
    iters = np.arange(0, len(loss))
    ax2.plot(iters, loss + loss_reg, color = "grey", ls = "-.")
    line = ax2.plot(iters[0], loss[0], color = "red")[0]
    ax2.yaxis.tick_right()
    ax2.set_yscale("log")
    ax2.set_yscale("log")
    ax2.grid()
    ax2.set_ylabel("Loss")
    return im, s, line
    
def update_animation(wos, iteration, record, coeff, resolution, im, s, line, bbox, opt_param, out_val : float = None):
    coeff.tensor = TensorXf(record[f"{opt_param}-{iteration}"])
    coeff.update_texture()
    points = create_image_points(bbox, resolution, spp = 4, centered = False)
    vals = coeff.get_value(points)
    if out_val is not None:
        mask = wos.input.shape.inside_closed_surface_mask(points)
        vals = dr.select(mask, vals, out_val)
    image, _ = create_image_from_result(vals, resolution)
    im.set_data(image[0])
    if s is not None:
        dirichlet_points = record[f"dirichletpoints-{iteration}"]
        if dirichlet_points.shape[0] > 0:
            dirichlet_points = Point2f(dirichlet_points.T)
            dirichlet_points = point2sketch(dirichlet_points, bbox, resolution).numpy()
            s.set_offsets(dirichlet_points.T)

    loss = record["loss"].sum(axis = 1)[:iteration].squeeze()
    loss_reg = record["loss-reg"][:iteration]
    iters = np.arange(0, iteration)
    line.set_xdata(iters)
    line.set_ydata(loss + loss_reg)

