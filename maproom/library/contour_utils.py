import numpy as np
import scipy.stats
import py_contour

import logging
log = logging.getLogger(__name__)


def contour_layer(particle_layer, contour_param, percent_levels=None):
    """Calculate contour segments for each level

    If contour levels are not specified, a default collection of levels is used
    """
    if percent_levels is None:
        percent_levels = [0.1, 0.4, 0.8, 1]
    percent_levels.sort()

    xmin = particle_layer.points.x.min()
    ymin = particle_layer.points.y.min()
    xmax = particle_layer.points.x.max()
    ymax = particle_layer.points.y.max()
    x = particle_layer.points.x - xmin
    y = particle_layer.points.y - ymin
    xy = np.vstack([x, y])

    weights = particle_layer.scalar_vars[contour_param]
    total_weight = weights.sum()
    kernel = scipy.stats.gaussian_kde(xy, weights=weights)

    binsize = 101
    x_flat = np.linspace(x.min(), x.max(), binsize)
    y_flat = np.linspace(y.min(), y.max(), binsize)
    xx,yy = np.meshgrid(x_flat,y_flat)
    grid_coords = np.append(xx.reshape(-1,1),yy.reshape(-1,1),axis=1)

    values = kernel(grid_coords.T) * total_weight
    values = values.reshape(binsize,binsize)

    max_density = values.max()
    particle_contours = [lev * max_density for lev in percent_levels]

    segs = py_contour.contour(values, x_flat, y_flat, particle_contours)
    
    return segs, ((xmin, ymin), (xmax, ymax))


def segments_to_polylines(seg_list):
    """Calculate polylines from list of point1 -> point2 tuples

    Simplistic code, assuming the next tuple matches either the first or second
    point the current tuple.
    """
    count = len(seg_list)
    workspace = np.zeros((count, 2), dtype=np.float32)
    polylines = []
    i = 0
    while (i < count):
        sx = seg_list[i][0][0]
        sy = seg_list[i][0][1]
        ex = seg_list[i][1][0]
        ey = seg_list[i][1][1]
        print("start", sx, sy, ex, ey, "at", i)
        i += 1
        if i < count:
            ax = seg_list[i][0][0]
            ay = seg_list[i][0][1]
            bx = seg_list[i][1][0]
            by = seg_list[i][1][1]
            print("first group", ax, ay, bx, by, "at", i)
            if ax == sx and ay == sy:
                # found continuation: e -> s -> b (since s == a)
                workspace[0][0] = ex
                workspace[0][1] = ey
                workspace[1][0] = sx
                workspace[1][1] = sy
                workspace[2][0] = bx
                workspace[2][1] = by
                ex = bx
                ey = by
                w = 2  # index into workspace of current endpoint
            elif ax == ex and ay == ey:
                # found continuation: s -> e -> b (since e == a)
                workspace[0][0] = sx
                workspace[0][1] = sy
                workspace[1][0] = ex
                workspace[1][1] = ey
                workspace[2][0] = bx
                workspace[2][1] = by
                ex = bx
                ey = by
                w = 2  # index into workspace of current endpoint
            elif bx == sx and by == sy:
                # found continuation: e -> s -> a (since s == b)
                workspace[0][0] = ex
                workspace[0][1] = ey
                workspace[1][0] = sx
                workspace[1][1] = sy
                workspace[2][0] = ax
                workspace[2][1] = ay
                ex = ax
                ey = ay
                w = 2  # index into workspace of current endpoint
            elif bx == ex and by == ey:
                # found continuation: s -> e -> b (since e == b)
                workspace[0][0] = sx
                workspace[0][1] = sy
                workspace[1][0] = ex
                workspace[1][1] = ey
                workspace[2][0] = ax
                workspace[2][1] = ay
                ex = ax
                ey = ay
                w = 2  # index into workspace of current endpoint
            else:
                w = 1  # flag: a-b segment isn't connected to s-e
        else:
            w = 1  # no more segments

        if w == 1:
            print("SINGLE SEGMENT at", i)
            # a-b segment isn't connected to the s-e segment, so s-e
            # is a two-point segment
            workspace[0][0] = sx
            workspace[0][1] = sy
            workspace[1][0] = ex
            workspace[1][1] = ey
        else:
            # continue to look for more segments
            i += 1
            while (i < count):
                ax = seg_list[i][0][0]
                ay = seg_list[i][0][1]
                bx = seg_list[i][1][0]
                by = seg_list[i][1][1]
                print("looking for", ex, ey, "at", i, ax, ay, bx, by)
                if ax == ex and ay == ey:
                    # found continuation: e -> b (since e == a)
                    workspace[w][0] = bx
                    workspace[w][1] = by
                    ex = bx
                    ey = by
                    w += 1
                elif bx == ex and by == ey:
                    # found continuation: e -> b (since e == b)
                    workspace[w][0] = ax
                    workspace[w][1] = ay
                    ex = ax
                    ey = ay
                    w += 1
                else:
                    # doesn't match either point, so we must be starting a
                    # new polyline.
                    print("DOESNT MATCH at ", i)
                    break
                i += 1

        # end of segment, store it and check the next one
        w += 1
        polyline = np.zeros((w, 2), dtype=np.float32)
        polyline[0:w,:] = workspace[0:w,:]
        polylines.append(polyline)
    return polylines       


def contour_layer_to_polylines(particle_layer, contour_param, percent_levels=None):
    """Calculate polylines for each contour level

    Instead of disjointed segments as returned by contour_layer, this returns a list
    of polylines for each contour level. Each polyline is a list of points connected
    sequentially. Each contour level may contain one or more polylines, as each
    polyline does not have any gaps.
    """
    segs, bbox = contour_layer(particle_layer, contour_param, percent_levels)
    levels = {}
    for level in segs.keys():
        print(level)
        polylines = segments_to_polylines(segs[level])
        # for i, p in enumerate(polylines):
        #     p[:,0] += bbox[0][0]
        #     p[:,1] += bbox[0][1]
        #     print("  ", level, i, p)
        levels[level] = polylines
        break
    return levels
