import numpy as np
import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt
from ..Coefficient import TextureCoefficient
from PDE2D.utils.sketch import *
from PDE2D.Solver import *
from .boundary_shape import *
import skfmm
from .sdf_grid import *
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from PDE2D.utils.common import *

def disable_ticks(ax):
    """Disable ticks around plot (useful for displaying images)"""
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])


def disable_border(ax):
    """Disable border around plot"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def get_image_shape(image_name, image_center, image_length_x, sdf_res, sdf_center, sdf_length, square = False, high_res = 2048):
    image = plt.imread(image_name)
    image = image[...,:3].sum(axis = 2)/3
    ratio = image.shape[0] / image.shape[1]
    image_length_y = image_length_x if square else image_length_x * ratio
    image[image > 0.5] = 1
    image[image<0.5] = -1
    image_bbox = [[image_center[0] - image_length_x/2, image_center[1] - image_length_y/2],
                  [image_center[0] + image_length_x/2, image_center[1] + image_length_y/2]]
    bbox = [[sdf_center[0] - sdf_length/2, sdf_center[1] - sdf_length/2],
            [sdf_center[0] + sdf_length/2, sdf_center[1] + sdf_length/2]]
    tex = TextureCoefficient("a", image_bbox, image)
    res_high = [high_res, high_res]
    upsample = int(high_res / sdf_res)
    points = create_image_points(bbox = bbox, resolution = res_high, 
                                              spp = 1, centered = True)
    vals = tex.get_value(points)
    image, _= create_image_from_result(vals, resolution = res_high)
    image = skfmm.distance(image, dx = sdf_length/high_res)
    image = image[int(upsample/2-1)::upsample, int(upsample/2-1)::upsample]
    return SDFGrid(image, box_length = sdf_length, box_center = sdf_center, redistance = False)


def get_intersection_tensor(shape1 : Shape, shape2 : Shape, resolution, bbox):
    points = create_image_points(bbox, resolution, spp = 1, centered = True)
    vals1 = shape1.get_closest_dist(points)
    vals2 = shape2.get_closest_dist(points)
    new = dr.maximum(vals1, vals2)
    image, _ = create_image_from_result(new, resolution)
    return image

def visualize1(bbox, resolution, sdf1, sdf2, bpoints = None, name1 = "Old", name2 = "Optimized", res_angle = 32, max = None):
    d1, c1, gx1, gy1, ng1 = sdf1.vis_images(bbox = bbox, resolution = resolution)
    d2, c2, gx2, gy2, ng2 = sdf2.vis_images(bbox = bbox, resolution = resolution)
    fig, ax = plt.subplots(4, 3, figsize = (13, 16))
    ax[0][0].set_title(name1)
    ax[0][1].set_title(name2)
    ax[0][2].set_title("Difference")
    
    
    # distance
    ax[0][0].set_ylabel("Distance")
    d_range = get_common_range(d1, d2)
    plot_image(d1, ax[0][0], input_range = d_range)
    plot_image(d2, ax[0][1], input_range = d_range) 
    plot_image(np.abs(d1 - d2), ax[0][2], input_range = [None, max])
    

    # grad_norm
    ax[1][0].set_ylabel("|Grad|")
    ng_range = get_common_range(ng1, ng2)
    plot_image(ng1, ax[1][0], input_range = ng_range)
    plot_image(ng2, ax[1][1], input_range = ng_range) 
    plot_image(ng1 - ng2, ax[1][2])
    
    # modified grad_norm
    ax[2][0].set_ylabel("||grad|-1|")
    ng1_modified = ng1.copy()
    ng2_modified = ng2.copy()
    #ng1_modified[d1 < -0.1] = np.nan
    #ng2_modified[d2 < -0.1] = np.nan
    ng1_modified = np.abs(ng1_modified - 1)
    ng2_modified = np.abs(ng2_modified - 1) 
    ng_modified_range = get_common_range(ng1_modified, ng2_modified)
    plot_image(ng1_modified, ax[2][0], input_range = ng_modified_range)
    plot_image(ng2_modified, ax[2][1], input_range = ng_modified_range) 
    plot_image(ng1_modified - ng2_modified, ax[2][2])
    

    # grad direction
    ax[3][2].set_axis_off()
    ax[3,0].set_ylabel("Direction Grad")
    
    gx1_tex = TextureCoefficient("gx1", bbox, gx1, "nearest")
    gx2_tex = TextureCoefficient("gx2", bbox, gx2, "nearest")
    gy1_tex = TextureCoefficient("gy1", bbox, gy1, "nearest")
    gy2_tex = TextureCoefficient("gy2", bbox, gy2, "nearest")
    
    plot_image(c1, ax[3][0])
    plot_image(c2, ax[3][1])
    
    xlength = bbox[1][0] - bbox[0][0]
    ylength = bbox[1][1] - bbox[0][1]
    scale = mi.Point2f(xlength/res_angle, ylength/res_angle)
    
    points = create_image_points(bbox, resolution = [res_angle, res_angle], spp = 1, centered = True)
    
    dx1 = dr.normalize(mi.Point2f(gx1_tex.get_value(points), gy1_tex.get_value(points))) * scale * 0.48
    dx2 = dr.normalize(mi.Point2f(gx2_tex.get_value(points), gy2_tex.get_value(points))) * scale * 0.48
    
    points_sketch = point2sketch(points, bbox, resolution = resolution).numpy().T
    dir1_sketch = dir2sketch(dx1, bbox, resolution = resolution).numpy().T
    dir2_sketch = dir2sketch(dx2, bbox, resolution = resolution).numpy().T


    
    for point, dir1, dir2 in zip(points_sketch, dir1_sketch, dir2_sketch):
        ax[3][0].arrow(point[0], point[1], dir1[0], dir1[1])
        ax[3][1].arrow(point[0], point[1], dir2[0], dir2[1])
    
    if bpoints is not None:
        bpoints_sketch = point2sketch(bpoints, bbox, resolution).numpy()
        for aa in ax:
            for a in aa:
                a.scatter(bpoints_sketch[0] - 0.5, bpoints_sketch[1] - 0.5, s = 0.4, color  ="white")


def plot_shape_sdf(out_boundary, in_boundary, ax, bbox, resolution, linewidth = 3, e_size = 5):
    image = in_boundary.sketch_image(ax, bbox = bbox, resolution = resolution, channel = 2)
    black_region = image.sum(axis = 2) < 0.1
    image[black_region]  = 1
    ax.imshow(image)
    ax.axis("on")
    out_boundary.sketch(ax, bbox = bbox, resolution = resolution, lw = linewidth, e_size = e_size)

def visualize2(wos, bbox, curvature_distance = 0.0, 
               resolution = [1024, 1024], num_points = 10, step = 0.03, spp = 18,
               distances = None, color_points = "green", colors = None, lw = 1, e_size = 2, image_width = 3.36):

    fig = plt.figure(figsize= (image_width, image_width))
    pad1 = 3
    pad2 = 3
    s = 3
    imsize=  20
    g = gridspec.GridSpec(2 * imsize + pad2 + s, 2 * imsize + pad1 + s, figure = fig, wspace = 0, hspace=0)
    ax = fig.add_subplot(g[:,:])
    disable_ticks(ax)
    plt.setp(ax.spines.values(), color="white")
    ax1 = fig.add_subplot(g[pad1 : pad1 + imsize, pad2 : pad2 + imsize])
    ax2 = fig.add_subplot(g[pad1 : pad1 + imsize, pad2 + imsize + s :  pad2 + 2 * imsize + s])
    ax3 = fig.add_subplot(g[pad1 + s + imsize : pad1 + 2 * imsize + s, pad2 : pad2 + imsize])
    ax4 = fig.add_subplot(g[pad1 + imsize + s : pad1 + 2 * imsize + s, pad2 + imsize + s :  pad2 + 2 * imsize + s])
    #fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize = (8,8))
    plot_shape_sdf(wos.input.shape.out_boundary, wos.input.shape.in_boundaries[0], ax1, bbox, resolution, linewidth=lw, e_size=e_size)
    points_polyline = wos.input.shape.in_boundaries[0].get_boundary_polyline(step = step)
    sketch_points = wos.input.shape.in_boundaries[0].sketch_boundary_polyline(ax1, bbox, resolution, esize = e_size)
    sketch_points_plot = sketch_points[::num_points]
    disable_border(ax1)
    disable_ticks(ax1)

    indices = np.arange(dr.width(wos.input.shape.in_boundaries[0].polyline))
    normal = wos.input.shape.in_boundaries[0].get_normal(wos.input.shape.in_boundaries[0].polyline)
    points_curvature = wos.input.shape.in_boundaries[0].polyline  + curvature_distance * normal
    curvature = wos.input.shape.in_boundaries[0].compute_curvature(points_curvature).numpy()
    ax2.plot(indices, curvature, label = f"Curvature")

    spp = 2 ** spp
    distances = [1e-2, 5e-2, 1e-1] if distances is None else distances
    colors = ["cyan", "green", "blue"] if colors is None else colors
    points_normalders = []
    range_all = [np.nan, -np.nan]
    for distance, color in zip(distances, colors):
        print(f"distance = {distance}")
        normal = wos.input.shape.in_boundaries[0].get_normal(wos.input.shape.in_boundaries[0].polyline)
        points_normal_der = wos.input.shape.in_boundaries[0].polyline  + distance * normal
        points_normalders.append(points_normal_der)
        points_polyline = dr.repeat(points_normal_der, spp)
        normals = dr.repeat(normal, spp)
        L, _ = wos.solve(conf_numbers = [mi.UInt32(0)], points_in = points_polyline, derivative_dir = normals)
        result = dr.block_sum(L, spp) / spp
        result = result.numpy()
        
        result_corrected = result * (1 - distance * curvature)

        range_ = get_common_range(result_corrected, result)
        range_all = get_common_range(np.array(range_all), range_)
        
        ax3.plot(indices, result[0], label = f"t = {distance}", color = color, linewidth = 1)
        ax4.plot(indices, result_corrected[0], label = f"t = {distance}", color = color, linewidth = 1)
        
    range_all[0] -= 1
    range_all[1] += 1
    ax3.set_ylim(range_all)
    ax4.set_ylim(range_all)

    #ax2.legend()
    ax3.legend(framealpha = 1)
    ax4.legend(framealpha = 1)

    ax2.axhline(0, lw = 0.5, color = "black")
    ax3.axhline(0, lw = 0.5, color = "black")
    ax4.axhline(0, lw = 0.5, color = "black")
    
    ax2.axvline(0, lw = 0.5, color = "red")
    ax3.axvline(0, lw = 0.5, color = "red")
    ax4.axvline(0, lw = 0.5, color = "red")
    ax2.set_xticks([])
    ax3.set_xticks([])
    ax4.set_xticks([])
    
    ax2.set_xlim([-1, sketch_points.shape[0]])
    ax3.set_xlim([-1, sketch_points.shape[0]])
    ax4.set_xlim([-1, sketch_points.shape[0]])

    for i in range(num_points, sketch_points.shape[0], num_points):
        ax2.axvline(i, lw = 0.5, color = color_points)
        ax3.axvline(i, lw = 0.5, color = color_points)
        ax4.axvline(i, lw = 0.5, color = color_points)
    pad = 4
    ax1.set_title("(a) Shape", pad = pad, fontsize = DEFAULT_FONTSIZE_SMALL)
    ax3.set_title("(c) Normal derivatives", pad = pad, fontsize = DEFAULT_FONTSIZE_SMALL)
    ax2.set_title(f"(b) Curvatures", pad = pad, fontsize = DEFAULT_FONTSIZE_SMALL)
    ax4.set_title("(d) Corrected normal deriv.", pad = pad, fontsize = DEFAULT_FONTSIZE_SMALL)

    ax1.scatter(sketch_points_plot[0,0], sketch_points_plot[0,1], color = "red", zorder = 1, s = e_size)
    ax1.scatter(sketch_points_plot[1:,0], sketch_points_plot[1:,1], color = color_points, zorder = 1, s = e_size)

    #fig.tight_layout()
    
def visualize_grad(sdf, bbox, resolution, bpoints = None, res_angle = 32, range_d =None, range = 1e-2, norm =True, cmap = "coolwarm", col_width_image = 3.36):
    xlength = bbox[1][0] - bbox[0][0]
    ylength = bbox[1][1] - bbox[0][1]
    d, c, gx, gy, ng = sdf.vis_images(bbox = bbox, resolution = resolution)
    imsize = 250
    num_images = 3
    cbar_offset = 15
    pre_cbar_offset = 10
    post_cbar_offset = 60
    total_cbar_offset = cbar_offset + pre_cbar_offset + post_cbar_offset
    scale = col_width_image / (num_images * imsize + 2 * total_cbar_offset)
    cbar_b = 1
    fig = plt.figure(figsize = ((num_images * imsize + 2 * total_cbar_offset) * scale, imsize * scale))
    g = gridspec.GridSpec(imsize, imsize * num_images + total_cbar_offset * 2, wspace = 0.0, hspace=0.0)
    ax = fig.add_subplot(g[:,:])
    disable_ticks(ax)
    plt.setp(ax.spines.values(), color="white")
    ax1 = fig.add_subplot(g[:, :imsize])
    ax2 = fig.add_subplot(g[:, imsize + total_cbar_offset :2 * imsize + total_cbar_offset])
    ax3 = fig.add_subplot(g[:, 2 * imsize + 2 * total_cbar_offset :3 * imsize + 2 * total_cbar_offset])

    ax1_cbar = fig.add_subplot(g[cbar_b: -cbar_b, imsize + pre_cbar_offset : imsize + pre_cbar_offset + cbar_offset])
    ax2_cbar = fig.add_subplot(g[cbar_b: -cbar_b, 2 * imsize + total_cbar_offset + pre_cbar_offset : 2*imsize+ total_cbar_offset+ cbar_offset + pre_cbar_offset])
    
    if range_d is not None:
        im1 = plot_image(d, ax1, input_range = [-range_d, range_d], colorbar= False, cmap = cmap)
    else:
        im1 = plot_image(d, ax1, colorbar = False, cmap = cmap)
    if norm:
        im2 = plot_image(ng, ax2, input_range = [1-range, 1+range], cmap = cmap, colorbar = False)
    else:    
        im2 = plot_image(np.abs(ng-1), ax2, input_range = [0, range], colorbar = False)

    cbar1 = plt.colorbar(im1, cax = ax1_cbar)
    tick_locator = ticker.MaxNLocator(nbins=3)
    cbar1.locator = tick_locator
    cbar1.formatter.set_powerlimits((0, 0))
    cbar1.ax.yaxis.set_offset_position('left') 
    cbar1.update_ticks()
    cbar2 = plt.colorbar(im2, cax = ax2_cbar)
    tick_locator = ticker.MaxNLocator(nbins=3)
    cbar2.locator = tick_locator
    cbar2.formatter.set_powerlimits((0, 0))
    cbar2.ax.yaxis.set_offset_position('right') 
    cbar2.update_ticks()
    #if bpoints is not None:
    #    mask = (bpoints[0] > bbox[0][0]) &  (bpoints[0] < bbox[1][0]) & (bpoints[1] > bbox[0][1]) & (bpoints[1] < bbox[1][1])
    #    mask = mask.numpy().squeeze()
    #    if len(mask) > 0:
    #        bpoints_np = bpoints.numpy().squeeze()
    #        bpoints_ = mi.Point2f(bpoints_np[mask])
    #        bpoints_sketch = point2sketch(bpoints_, bbox, resolution).numpy()
    #        ax2.scatter(bpoints_sketch[:,0], bpoints_sketch[:,1], s = ps, color = "green")
    #        ax1.scatter(bpoints_sketch[:,0], bpoints_sketch[:,1], s = ps, color = "white")

    x = (np.arange(resolution[0]) + 0.5) / 1024 * d.shape[0]
    y = (np.arange(resolution[1]) + 0.5) / 1024 * d.shape[0]
    X, Y = np.meshgrid(x, y)
    # Creating contour plot
    ax1.contour(X, Y, d, colors = ["white"], lw = 5, levels = np.array([0]))
    ax2.contour(X, Y, d, colors = ["white"], lw = 5, levels = np.array([0]))

    gx_tex = TextureCoefficient("gx", bbox, gx, "nearest")
    gy_tex = TextureCoefficient("gy", bbox, gy, "nearest")
    plot_image(-2 * c+1, ax3, colorbar = False, cmap = "coolwarm", input_range = [-2, 2])
    scale = mi.Point2f(xlength/res_angle, ylength/res_angle)
    points = create_image_points(bbox, resolution = [res_angle, res_angle], spp = 1, centered = True)
    dx = dr.normalize(mi.Point2f(gx_tex.get_value(points), gy_tex.get_value(points))) * scale * 0.48
    
    points_sketch = point2sketch(points, bbox, resolution = resolution).numpy().T
    dir_sketch = dir2sketch(dx, bbox, resolution = resolution).numpy().T

    for point, dir, in zip(points_sketch, dir_sketch):
        ax3.arrow(point[0], point[1], dir[0], dir[1])
    pad = 4
    ax1.set_title("$d(x)$", fontsize = DEFAULT_FONTSIZE_SMALL, pad = pad)
    if norm:
        ax2.set_title("$||\\nabla d(x)||$", fontsize = DEFAULT_FONTSIZE_SMALL, pad = pad)
    else:
        ax2.set_title("$||\\nabla d(x)|| - 1|$", fontsize = DEFAULT_FONTSIZE_SMALL, pad = pad)
    ax3.set_title("Direction", fontsize = DEFAULT_FONTSIZE_SMALL, pad = pad)
    #fig.tight_layout()

        