import numpy as np
import drjit as dr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

def plot_image(image, ax, colorbar = True, input_range = [None,None], cmap = "viridis", axis = True):
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

def plot_image_3D(image, ax, norm = None, cmap = "viridis"):
    im = ax.imshow(image)
    if norm is not None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=cax, orientation='vertical')
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
    #ax.axes.get_xaxis().set_visible(False)
    #ax.axes.get_yaxis().set_visible(False)
    return im

def disable_border(ax):
    """Disable border around plot"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def disable_ticks(ax):
    """Disable ticks around plot (useful for displaying images)"""
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
