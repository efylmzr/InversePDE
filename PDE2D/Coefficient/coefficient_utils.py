from matplotlib.patches import Polygon
import numpy as np

def build_grid_squares(points_per_edge, resolution, bbox, ax=None):
    xscale = bbox[1][0] - bbox[0][0]
    yscale = bbox[1][1] - bbox[0][1]
    binx = xscale / resolution[0]
    biny = yscale / resolution[1]
    stepx = binx / points_per_edge
    stepy = biny / points_per_edge
    points = np.zeros([0, 2])
    for i in range(resolution[0]):
        for j in range(resolution[1]):
            vec = np.zeros([4 * points_per_edge + 1, 2])
            vec[0:points_per_edge, 0] = np.arange(
                i * binx, (i+1) * binx, stepx)[0:points_per_edge]
            vec[0:points_per_edge, 1] = -j * biny
            vec[points_per_edge: 2 * points_per_edge, 0] = (i+1) * binx
            vec[points_per_edge: 2 * points_per_edge,
                1] = np.arange(-j * biny, -(j+1) * biny, -stepy)[0:points_per_edge]
            vec[2 * points_per_edge: 3 * points_per_edge,
                0] = np.arange((i+1) * binx, i * binx, -stepx)[0:points_per_edge]
            vec[2 * points_per_edge: 3 * points_per_edge, 1] = - (j + 1) * biny
            vec[3 * points_per_edge: 4 * points_per_edge, 0] = i * binx
            vec[3 * points_per_edge: 4 * points_per_edge,
                1] = np.arange(-(j+1) * biny, -j * biny, stepy)[0:points_per_edge]
            vec[-1] = vec[0]
            vec += np.array([bbox[0][0], bbox[1][1]])
            points = np.vstack([points, vec])
            if (ax is not None):
                color = 'red' if (i+j) % 2 == 1 else "blue"
                polygon = Polygon(vec, edgecolor=None, facecolor=color)
                ax.add_patch(polygon)
    return points.T

def sketch_grid_squares(points, points_per_edge, resolution,  ax):
    num_points = points.shape[0] // (4 * points_per_edge + 1)
    for i in range(num_points):
        p = points[i * (4 * points_per_edge + 1): (i+1)
                   * (4 * points_per_edge + 1)]
        color = 'red' if (
            ((i // resolution[0]) + (i % resolution[1])) % 2 == 0) else "blue"
        polygon = Polygon(p, edgecolor=None, facecolor=color)
        ax.add_patch(polygon)
    disable_ticks(ax)

def disable_ticks(ax):
    """Disable ticks around plot (useful for displaying images)"""
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])