import numpy as np
import scipy.stats
import libmaproom.contour as py_contour

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


def contour_layer(particle_layer, contour_param, percent_levels=None):
    """Calculate contour segments for each level

    If contour levels are not specified, a default collection of levels is used
    """
    if percent_levels is None:
        percent_levels = [0.1, 0.4, 0.8, 1]
    percent_levels.sort()

    x, y = particle_layer.calc_contour_points()
    xmin = x.min()
    ymin = y.min()
    xmax = x.max()
    ymax = y.max()
    # x = np.asarray(particle_layer.points.x, dtype=np.float64)# - xmin
    # y = np.asarray(particle_layer.points.y, dtype=np.float64)# - ymin
    print(x)
    print(y)
    xy = np.vstack([x, y])
    print("XY")
    print(xy)

    weights = particle_layer.calc_contour_weights(contour_param)
    if weights is None:
        total_weight = 1.0
    else:
        total_weight = weights.sum()
    progress_log.info("Calculating gaussian_kde...")
    kernel = scipy.stats.gaussian_kde(xy, weights=weights)

    binsize = 301
    xdelta = (xmax - xmin)# / 2.0
    ydelta = (ymax - ymin)# / 2.0
    # xdelta = 0.0
    # ydelta = 0.0
    x_flat = np.linspace(x.min() - xdelta, x.max() + xdelta, binsize)
    y_flat = np.linspace(y.min() - ydelta, y.max() + ydelta, binsize)
    xx,yy = np.meshgrid(x_flat,y_flat)
    grid_coords = np.append(xx.reshape(-1,1),yy.reshape(-1,1),axis=1)

    progress_log.info("Calculating values from kernel...")
    values = kernel(grid_coords.T) * total_weight
    values = values.reshape(binsize,binsize)

    max_density = values.max()
    particle_contours = [lev * max_density for lev in percent_levels]

    values = np.ascontiguousarray(values.T)
    progress_log.info("Calculating contour...")
    segs = py_contour.contour(values, x_flat, y_flat, particle_contours)

    return segs, ((xmin, ymin), (xmax, ymax))


def segments_to_line_layer_data(seg_list):
    """Calculate numpy array to be used as input to line layer

    py_contour returns list of tuples, each tuple represents a line segment. This
    converts to a numpy array and a list of segment connectivity.
    """
    count = len(seg_list)
    points = np.zeros((count * 2, 2), dtype=np.float64)
    lines = np.empty((count, 2), dtype=np.uint32)
    i = 0
    i2 = 0
    while (i < count):
        points[i2][0] = seg_list[i][0][0]
        points[i2][1] = seg_list[i][0][1]
        lines[i][0] = i2
        i2 += 1
        points[i2][0] = seg_list[i][1][0]
        points[i2][1] = seg_list[i][1][1]
        lines[i][1] = i2
        i2 += 1
        i += 1
    return points, lines       


def contour_layer_to_line_layer_data(particle_layer, contour_param, percent_levels=None):
    """Calculate segment lists suitable for use in line layers

    Because py_contour appears to return disjointed segments, no attempt is made
    to convert the contour lines into contiguous polylines. Instead, a list of points
    is returned such that a line segment list can be created connecting point 0 to point 1,
    point 2 to point 3, point 4 to point 5 (etc) and can be fed into a normal line layer
    using the set_disjointed_segment_data method.
    """
    segs, bbox = contour_layer(particle_layer, contour_param, percent_levels)
    levels = {}
    for level in segs.keys():
        # polys = py_contour.sort_segments(segs[level])  # SUPER SLOW!!!!!
        # print(polys)
        # print(level)
        points, line_segment_indexes = segments_to_line_layer_data(segs[level])
        points[:,0]# += bbox[0][0]
        points[:,1]# += bbox[0][1]
        # print(points)
        # print(line_segment_indexes)
        levels[level] = points, line_segment_indexes
    return levels
