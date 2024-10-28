import numpy as np
from PDE2D import PATH
from PDE2D.utils.imageUtils import create_image_points, create_image_from_result
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import matplotlib.animation as animation
import os
'''
def create_animation(record, path, iternum, bbox, wos, max_range = None, wos_obj = None, resolution = [512, 512], 
                     opt_param = "diffusion.texture.tensor", fileset = None, out_val = None, 
                     optimize_electrode = True, primal_range = ):

    name = "reconstruction" if max_range is None else "reconstruction-scaled"
    if (wos_obj is None) and (fileset is None):
        fig , ((ax1, ax2), (ax3, ax4)) = plt.subplots(1,2, figsize = (10, 10))
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

'''