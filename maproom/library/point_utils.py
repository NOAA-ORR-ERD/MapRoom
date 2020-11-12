import random

import numpy as np

from maproom.renderer.gl.data_types import POINT_XY_VIEW_DTYPE
from .shapefile_utils import GeomInfo
from ..layers.shapefile import ShapefileLayer

from libmaproom.pytriangle import triangulate_simple


import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


def create_convex_hull(layers, manager):
    progress_log.info("Creating convex hull...")

    params = "-c"  # allows triangulation without explicit boundary

    count = 0
    for ly in layers:
        count += len(ly.points)
    depths = np.zeros(count, np.float32)

    # we need to use projected points for the triangulation
    projected_points = np.zeros([count,2], dtype=np.float64)
    index = 0
    for ly in layers:
        count = len(ly.points)
        projected_points[index:index+count,:] = ly.points.view(POINT_XY_VIEW_DTYPE).xy
        index += count

    hole_points_xy = np.zeros([0,2], dtype=np.float64)
    line_segment_indexes = np.zeros([0,2], dtype=np.uint32)

    (triangle_points_xy,
     triangle_points_z,
     triangle_line_segment_indexes,  # not needed
     triangles) = triangulate_simple(
        params,
        projected_points,
        depths,
        line_segment_indexes,
        hole_points_xy)

#    output_count = 20
#    output = np.zeros([count,2], dtype=np.float64)
#    points = random.choices(projected_points, k=output_count)
    print(triangle_line_segment_indexes[:,0])
    index_order = triangle_line_segment_indexes[:,0]
    print(index_order)
    points = triangle_points_xy[index_order]
    print(points)
    print(triangle_line_segment_indexes)
    #points = np.asarray(points, dtype=np.float64)
    item = ["Polygon", GeomInfo(0, len(points), "convex hull", 1, "blah")]
    layer = ShapefileLayer(manager=manager)
    layer.set_geometry(points, [item])

    return layer, ""
