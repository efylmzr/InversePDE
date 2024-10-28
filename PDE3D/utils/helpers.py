import mitsuba as mi 
import matplotlib.cm as cm
import numpy as np

def tea(v0: mi.UInt32, v1: mi.UInt32, rounds=4):
        sum = mi.UInt32(0)
        for i in range(rounds):
            sum += 0x9e3779b9
            v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + sum) ^ ((v1>>5) + 0xc8013ea4)
            v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + sum) ^ ((v0>>5) + 0x7e95761e)
        return v0, v1

def get_rgb_from_colormap(value, colormap_name="viridis"):
    """
    Returns the RGB values from a given value and colormap name.
    
    Parameters:
    - value: A float between 0 and 1.
    - colormap_name: The name of the colormap (default is "viridis").
    
    Returns:
    - A tuple containing the RGB values.
    """
    # Ensure the value is within the range [0, 1]
    value = np.maximum(0, np.minimum(1, value))
    
    # Get the colormap
    colormap = cm.get_cmap(colormap_name)
    
    # Get the RGB values
    rgb = colormap(value)
    
    return rgb.T