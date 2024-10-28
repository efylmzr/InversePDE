import numpy as np
import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .imageUtils import *
from mitsuba import Point2f, Float


def set_matplotlib(fontsize = 12):
    import matplotlib
    # Override any style changes by VSCode
    matplotlib.style.use('default')
    matplotlib.rcParams['font.size'] = fontsize
    #matplotlib.rcParams['font.weight'] = "bold"
    #matplotlib.rcParams['text.usetex'] = True
    #matplotlib.rcParams['text.latex.preamble'] = r"""\usepackage{libertine}
    #\usepackage{amsmath}"""
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

def get_common_range(image1_, image2_, equal = False):
    image1 = np.copy(image1_)
    image2 = np.copy(image2_)
    image1[np.isnan(image1)] = 0
    image2[np.isnan(image2)] = 0
    max1 = image1.max()
    min1 = image1.min()
    max2 = image2.max()
    min2 = image2.min()
    maxval = max(max1, max2)
    minval = min(min1, min2)
    if equal:
        val = max(maxval, -minval)
        return [-val, val]
    else:
        return [minval, maxval]

def plot_image(image, ax, colorbar = True, input_range = [None,None], cmap = "viridis", axis = True, axis_ticks = False):
    im = ax.imshow(image, vmin = input_range[0], 
                   vmax=input_range[1], cmap = cmap)
    
    if(colorbar):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.formatter.set_useMathText(True)
        cbar.formatter.set_useOffset(True)
        cbar.ax.locator_params(nbins=5)
        cbar.ax.yaxis.set_offset_position("right")
        plt.setp(cbar.ax.yaxis.get_offset_text(), weight='semibold')
    #else:
    #    cax.axis("off")
    
    ax.set_xticks([])
    ax.set_yticks([])
    if not axis:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    return im
    
    

def plot_function(ax, function, bbox, resolution = [1024, 1024], spp = 4, colorbar=True, input_range = [None,None], cmap = "viridis"):
    points = create_image_points(bbox, resolution, spp)
    fnc_vals = function(points)
    image, tensor = create_image_from_result(fnc_vals, resolution)
    plot_image(image[0], ax, colorbar, input_range, cmap)
    return image
    
def sketch_diff(image1, image2, equal_range = True, title = "Difference", max_range = None):
    image1[np.isnan(image1)] = 0
    image2[np.isnan(image2)] = 0
    diff = image1 -image2
    diff_flat = diff.flatten()
    diff_flat = diff[np.isfinite(diff)] 
    print("Mean:")
    print(np.mean(diff_flat))
    print("Variance:")
    print(np.var(diff_flat))
    
    range_ = max(diff.max(), -diff.min())
    if(max_range != None):
        diff[diff > max_range] = max_range
        diff[diff < -max_range] = -max_range
        range_ = min(max_range, range_)
    
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12.3,6))
    diff_im = None
    if(equal_range):
        diff_im = ax1.imshow(diff, vmin = -range_, vmax=range_ ,cmap = "coolwarm")
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(diff_im, cax=cax)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax1.set_title("Diff. Image")
    
    diff = diff.flatten()
    diff = diff[diff !=0]
    
    ax2.hist(diff, bins = 100, range = (-range_, range_))
    ax2.set_title("Histogram")
    
    fig.suptitle(title, fontsize=18,fontweight =5, color = "purple")
    plt.show()
    
def plot_primals(ax, data1, data2, entry_nums, num_entries, std1 = None, std2 = None, name1 = "Iteration", name2 = "Objective", fontsize = 7, label = True, 
                 scale = 1.2, grid = True):
    assert len(data1) == len(data2)
    electrodes = np.arange(num_entries)
    obj_vals = np.zeros(num_entries)
    obj_vals= data2
    iter_vals = np.zeros(num_entries)
    iter_vals = data1
    plot_std = (std1 is not None) and (std2 is not None)
    if plot_std:
        obj_vals_std = np.zeros(num_entries)
        obj_vals_std = std2
        iter_vals_std = np.zeros(num_entries)
        iter_vals_std = std1
    
    if not plot_std:
        el_names = [f"E{electrodes[i]}" for i in range(num_entries)]
        el_means = {
            name2 : obj_vals,
            name1 : iter_vals,
        }
        x = np.arange(num_entries)  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0

        for attribute, measurement in el_means.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute, align = "center")
            if label:
                ax.bar_label(rects, padding=3, fmt="%.2e", rotation = "vertical", 
                                label_type = "center", fontsize = fontsize)
            multiplier += 1
    else:
        el_names = [f"E{electrodes[i]}" for i in range(num_entries)]
        el_means = {
            name2 : [obj_vals, obj_vals_std],
            name1 : [iter_vals, iter_vals_std],
        }
        x = np.arange(num_entries)  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0

        for attribute, measurement in el_means.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement[0], width, label=attribute, align = "center", yerr = measurement[1])
            if label:
                ax.bar_label(rects, padding=3, fmt="%.2e", rotation = "vertical", 
                                label_type = "center", fontsize = fontsize)
            multiplier += 1
        
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Voltage V')
    ax.set_title('Primal Comparison')
    ax.set_xticks(x + width, el_names)
    ax.legend(loc='upper left')
    max_val = max(np.max(obj_vals), np.max(iter_vals))
    min_val = min(np.min(obj_vals), np.min(iter_vals))
    min_val = min(min_val, 0)
    max_val = max(max_val, 0)
    ax.set_ylim(scale * min_val, scale * max_val)
    if grid:
        ax.grid()
    else:
        ax.axhline(y = 0, linewidth = 1, color = "black")


def plot_diff_primal(ax, data1, data2, entry_nums, num_entries, std1 = None, std2 = None, fontsize = 7,
                 scale = 1.2, grid = True):
    electrodes = np.arange(num_entries)
    vals = np.zeros(num_entries)
    vals[entry_nums] = np.abs(data2 - data1)
    vals_std = np.zeros(num_entries)
    vals_std[entry_nums] = np.sqrt(std1 ** 2 + std2 ** 2)
    
    el_names = [f"E{electrodes[i]}" for i in range(num_entries)]
    x = np.arange(num_entries)  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0.5
    offset = width * multiplier
    rects = ax.bar(x + offset, vals, width, align = "center", yerr = vals_std)
    multiplier += 1
        
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Voltage V')
    ax.set_title('Standard Deviation Diff.')
    ax.set_xticks(x + width, el_names)
    max_val = np.max(vals + vals_std)
    min_val = min(np.min(vals - vals_std), 0)
    ax.set_ylim(scale * min_val, scale * max_val)
    if grid:
        ax.grid()
    else:
        ax.axhline(y = 0, linewidth = 1, color = "black")

def point2sketch(point : Point2f, bbox : list, resolution : list):
    xscale = bbox[1][0] - bbox[0][0]
    yscale = bbox[1][1] - bbox[0][1]
    pointx = (point[0]-bbox[0][0])/xscale*resolution[1]
    pointy = resolution[0] - (point[1]-bbox[0][1])/yscale*resolution[0]
    return Point2f(pointx, pointy )
    
def dir2sketch(vec : Point2f, bbox : list, resolution : list):
    xscale = bbox[1][0] - bbox[0][0]
    yscale = bbox[1][1] - bbox[0][1]
    vecx = vec[0] / xscale * resolution[1]
    vecy = vec[1] / yscale * resolution[0]
    return Point2f(vecx, -vecy)
    
def dist2sketch(dist : Float, bbox : list, resolution : list):
    xscale = bbox[1][0] - bbox[0][0]
    yscale = bbox[1][1] - bbox[0][1]
    x = resolution[1] / xscale
    y = resolution[0] / yscale
    return x * dist, y * dist, dr.sqrt(dr.sqr(x) + dr.sqr(y)) * dist

def disable_border(ax):
    """Disable border around plot"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


