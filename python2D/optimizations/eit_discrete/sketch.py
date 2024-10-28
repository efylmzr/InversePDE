import matplotlib.pyplot as plt 
from matplotlib.patches import Polygon
from PDE2D.Coefficient import *
from PDE2D.utils import *
from PDE2D.BoundaryShape import *
from PDE2D.Solver import *
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.animation as animation

def create_animation_shape(record, path, iternum, bbox, wos, wos_obj = None, resolution = [1024, 1024], type = "sdf", 
                           plot_center = False):
    
    fig = plt.figure(figsize= (10, 5))
    g = gridspec.GridSpec(16, 32, figure = fig, wspace = 0.0, hspace=0.0)
    ax1 = fig.add_subplot(g[:16, :16])
    ax2 = fig.add_subplot(g[:, 16:31])
    
    if type == "sphere":
        image_obj, ax_image, line = start_animation_sphere(ax1, ax2, record, bbox, wos, wos_obj, resolution, plot_center)
        update = lambda iteration : update_sphere(iteration, record, resolution, line, ax_image, image_obj, bbox)
    elif type == "sdf":
        image_obj, ax_image, line = start_animation_sdf(ax1, ax2, record, bbox, wos, wos_obj, resolution, plot_center,)
        update = lambda iteration : update_sdf(iteration, record, resolution, line, ax_image, image_obj, bbox)
        
    fig.subplots_adjust(left=0.01, bottom=0.05, right=0.97, top=0.95, wspace=0.01, hspace=0.05)

    ani = animation.FuncAnimation(fig=fig, func=update, frames=iternum+1, interval=30)
    writervideo = animation.FFMpegWriter(fps=25) 
    if wos_obj is not None:
        ani.save(filename=f"{path}/reconstruction.gif", writer="pillow")
        ani.save(f"{path}/reconstruction.mp4", writer=writervideo)
    else:
        ani.save(filename=f"{path}/reconstruction_wo_obj.gif", writer="pillow")
        ani.save(f"{path}/reconstruction_wo_obj.mp4", writer=writervideo)
    
def start_animation_sphere(ax1, ax2, record, bbox, wos, wos_obj = None, resolution = [1024, 1024], 
                           plot_center =  False):
    ax1.set_axis_off()
    image_obj = None
    
    wos.input.shape.sketch_image(ax1, bbox, resolution)
    
    if wos_obj is not None:
        image_obj = wos_obj.input.shape.in_boundaries[0].sketch_image(ax1, bbox, resolution, channel = 0)
    
    radius_iter = record["inboundary.dirichlet.radius-0"]
    origin_iter = record["inboundary.dirichlet.origin-0"]
    circle = CircleShape(radius = radius_iter, origin = origin_iter) 
    
    if image_obj is not None:
        image_obj_ = np.array(image_obj)
    else:
        image_obj_ = None

    image_iter = circle.sketch_image(ax1, bbox, resolution, image = image_obj_, channel = 2)
    image_ax = ax1.imshow(image_iter)
    if plot_center:
        origin = point2sketch(wos.shape_holder.out_boundary.origin, bbox, resolution).numpy().squeeze()
        ax1.scatter(origin[0], origin[1], color = "white", s = 0.5)
    loss = record["loss"].sum(axis = 1)
    iters = np.arange(0, len(loss))
    ax2.plot(iters, loss, color = "grey", ls = "-.")
    line = ax2.plot(iters[0], loss[0], color = "red")[0]
    ax2.set_yscale("log")
    ax2.set_yscale("log")
    ax2.grid()
    #ax2.spines[['right', 'top']].set_visible(False)

    
    ax1.set_title("Reconstruction")
    ax2.set_title("Evolution of the Loss")
    
    return image_obj, image_ax, line
    
def update_sphere(iteration, record, resolution, line, image_ax, image_obj, bbox):
    radius_iter = record[f"inboundary.dirichlet.radius-{iteration}"]
    origin_iter = record[f"inboundary.dirichlet.origin-{iteration}"]
    circle = CircleShape(radius = radius_iter, origin = origin_iter) 
    fig, (ax_dummy) = plt.subplots(1, 1, figsize=[5, 5])
    if image_obj is not None:
        image_obj_ = np.array(image_obj)
    else:
        image_obj_ = None
    image_i = circle.sketch_image(ax_dummy, bbox, resolution, image = image_obj_, channel = 2)
    plt.close(fig)
    loss = record["loss"].sum(axis = 1)[:iteration]
    iters = np.arange(0, iteration)
    line.set_xdata(iters)
    line.set_ydata(loss)
    image_ax.set_data(image_i)


def start_animation_sdf(ax1, ax2, record, bbox, wos, wos_obj, resolution = [1024, 1024], plot_center = False,
                        center_bg = 1, deviate_bg = 0.02):
    ax1.set_axis_off()
    image_obj = None
    
    wos.input.shape.sketch_image(ax1, bbox, resolution)
    
    if wos_obj is not None:
        image_obj = wos_obj.input.shape.in_boundaries[0].sketch_image(ax1, bbox, resolution, channel = 0)
    
    image_iter = record["inboundary.dirichlet.tensor-0"].squeeze()
    box_length = bbox[1][0] - bbox[0][0]
    box_center = [(bbox[0][0] + bbox[1][0])/2, (bbox[0][1] + bbox[1][1])/2]
    sdf = SDFGrid(image_iter, box_length=box_length, box_center = box_center)
    if image_obj is not None:
        image_obj_ = np.array(image_obj)
    else:
        image_obj_ = None
    image_iter = sdf.sketch_image(ax1, bbox, resolution, image = image_obj_, channel = 2)
    image_ax = ax1.imshow(image_iter * 0.8)
    if plot_center:
        origin = point2sketch(wos.shape_holder.out_boundary.origin, bbox, resolution).numpy().squeeze()
        ax1.scatter(origin[0], origin[1], color = "white", s = 0.3)

    loss = record["loss"].sum(axis = 1)
    iters = np.arange(0, len(loss))
    ax2.plot(iters, loss, color = "grey", ls = "-.")
    line = ax2.plot(iters[0], loss[0], color = "red")[0]
    ax2.set_yscale("log")
    ax2.set_yscale("log")
    ax2.grid()

    
    ax1.set_title("Reconstruction")
    ax2.set_title("Evolution of the Loss")
    return image_obj, image_ax, line
    
def update_sdf(iteration, record, resolution, line, image_ax, image_obj, bbox):
    image_iter = record[f"inboundary.dirichlet.tensor-{iteration}"].squeeze()
    box_length = bbox[1][0] - bbox[0][0]
    box_center = [(bbox[0][0] + bbox[1][0])/2, (bbox[0][1] + bbox[1][1])/2 ]
    sdf = SDFGrid(image_iter, box_length=box_length, box_center = box_center)
    fig, (ax_dummy) = plt.subplots(1, 1, figsize=[5, 5])
    if image_obj is not None:
        image_obj_ = np.array(image_obj)
    else:
        image_obj_ = None
    image_iter = sdf.sketch_image(ax_dummy, bbox, resolution, image = image_obj_, channel = 2)
    plt.close(fig)
    loss = record["loss"].sum(axis = 1)[:iteration]
    iters = np.arange(iteration)
    line.set_xdata(iters)
    line.set_ydata(loss)
    
    image_ax.set_data(image_iter * 0.8)

def plot_summary(loss_list, loss_reg_list, path, log = False, save_npy = True):
    losses = np.array(loss_list)
    losses_reg = np.array(loss_reg_list)
    if save_npy:
        np.save(os.path.join(path, "losses.npy"),losses)
        np.save(os.path.join(path, "losses_reg.npy"),losses_reg)