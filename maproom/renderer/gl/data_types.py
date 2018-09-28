import numpy as np

import logging
log = logging.getLogger(__name__)


# data type used for point coordinates in the renderer
POINT_COORD_VIEW_DTYPE = np.float32


POINT_DTYPE = np.dtype([
    ("x", np.float64),
    ("y", np.float64),
    ("z", np.float32),
    ("color", np.uint32),
    ("state", np.uint32)
])
POINT_XY_VIEW_DTYPE = np.dtype([
    ("xy", "2f8"),
    ("z", np.float32),
    ("color", np.uint32),
    ("state", np.uint32)
])
LINE_SEGMENT_DTYPE = np.dtype([
    ("point1", np.uint32),  # Index of first point in line segment.
    ("point2", np.uint32),  # Index of second point in line segment.
    ("color", np.uint32),  # Color of this line segment.
    ("state", np.uint32)
])
LINE_SEGMENT_POINTS_VIEW_DTYPE = np.dtype([
    ("points", "2u4"),
    ("color", np.uint32),
    ("state", np.uint32)
])
TRIANGLE_DTYPE = np.dtype([
    ("point1", np.uint32),  # Index of first point in triangle.
    ("point2", np.uint32),  # Index of second point in triangle.
    ("point3", np.uint32),  # Index of third point in triangle.
    ("color", np.uint32),
    ("state", np.uint32)
])
TRIANGLE_POINTS_VIEW_DTYPE = np.dtype([
    ("point_indexes", "3u4"),
    ("color", np.uint32),
    ("state", np.uint32)
])

# point adjacency flag can be:
# bit 0: if 1, connect previous point to this point
# bit 1: last point
# bit 2: only checked on last point: connect to starting point
# any points after this in the ring are unconnected points
# bit 31: if 1, start of new ring and bits 0 - 30 are number of points
#
# the meaning of state depends on its position in the points array
# entry corresponding to starting point holds state for entire polygon (see maproom/layers/state.py)
# next entry is feature code, because if there's only one entry, it's a point
# next entry holds fill color for entire polygon, because it requires at least 3 points to have a color
RING_ADJACENCY_DTYPE = np.dtype([  # parallels the points array
    ("point_flag", np.int32),
    ("state", np.int32),
])

POLYGON_ADJACENCY_DTYPE = np.dtype([  # parallels the points array
    ("next", np.uint32),   # Index of next adjacent point in ring
    ("ring_index", np.uint32)  # Index of ring this point is in
])
POLYGON_DTYPE = np.dtype([
    ("start", np.uint32),  # Index of arbitrary point in this polygon.
    ("count", np.uint32),  # Number of points in this polygon.
    ("group", np.uint32),  # An outer polygon and all of its holes have the same opaque group id.
    ("color", np.uint32),  # Color of this polygon.
    ("state", np.uint32)   # standard maproom object states, plus polygon type
])
QUAD_VERTEX_DTYPE = np.dtype(
    [("x_lb", np.float32), ("y_lb", np.float32),
     ("x_lt", np.float32), ("y_lt", np.float32),
     ("x_rt", np.float32), ("y_rt", np.float32),
     ("x_rb", np.float32), ("y_rb", np.float32)]
)
QUAD_VERTEX_DUPLICATE_DTYPE = np.dtype(
    [("x_lb", np.float32), ("y_lb", np.float32),
     ("x_lt", np.float32), ("y_lt", np.float32),
     ("x_rt", np.float32), ("y_rt", np.float32),
     ("x_rb", np.float32), ("y_rb", np.float32),
     ("xprime_lb", np.float32), ("yprime_lb", np.float32),
     ("xprime_lt", np.float32), ("yprime_lt", np.float32),
     ("xprime_rt", np.float32), ("yprime_rt", np.float32),
     ("xprime_rb", np.float32), ("yprime_rb", np.float32)]
)
TEXTURE_COORDINATE_DTYPE = np.dtype(
    [("u_lb", np.float32), ("v_lb", np.float32),
     ("u_lt", np.float32), ("v_lt", np.float32),
     ("u_rt", np.float32), ("v_rt", np.float32),
     ("u_rb", np.float32), ("v_rb", np.float32)]
)
TEXTURE_COORDINATE_DUPLICATE_DTYPE = np.dtype(
    [("u_lb", np.float32), ("v_lb", np.float32),
     ("u_lt", np.float32), ("v_lt", np.float32),
     ("u_rt", np.float32), ("v_rt", np.float32),
     ("u_rb", np.float32), ("v_rb", np.float32),
     ("uprime_lb", np.float32), ("vprime_lb", np.float32),
     ("uprime_lt", np.float32), ("vprime_lt", np.float32),
     ("uprime_rt", np.float32), ("vprime_rt", np.float32),
     ("uprime_rb", np.float32), ("vprime_rb", np.float32)]
)


def make_points(count):
    return np.repeat(
        np.array([(np.nan, np.nan, np.nan, 0, 0)], dtype=POINT_DTYPE),
        count,
    ).view(np.recarray)


def make_points_from_xy(xy_values):
    xy_values = np.asarray(xy_values, dtype=np.float32)
    print(xy_values)
    points = make_points(len(xy_values))
    points.z = 0.
    points.x = xy_values[:,0]
    points.y = xy_values[:,1]
    return points


def make_polygons(count):
    return np.repeat(
        np.array([(0, 0, 0, 0, 0)], dtype=POLYGON_DTYPE),
        count,
    ).view(np.recarray)


def make_ring_adjacency_array(count):
    # default point_flag is to connect last point
    return np.repeat(
        np.array([(1, 0)], dtype=RING_ADJACENCY_DTYPE),
        count,
    ).view(np.recarray)


def make_point_adjacency_array(count):
    return np.repeat(
        np.array([(0, 0)], dtype=POLYGON_ADJACENCY_DTYPE),
        count,
    ).view(np.recarray)


def compute_projected_point_data(points, projection, hidden_points=None):
    view = points.view(POINT_XY_VIEW_DTYPE).xy.astype(np.float32)
    projected_point_data = np.zeros(
        (len(points), 2),
        dtype=np.float32
    )
    projected_point_data[:,0], projected_point_data[:,1] = projection(view[:,0], view[:,1])
    if hidden_points is not None:
        # OpenGL doesn't draw points that have a coordinate set to NaN
        projected_point_data[hidden_points] = np.nan
    return projected_point_data


def iter_geom(geom_list):
    for item in geom_list:
        log.debug(f"processing geometry list item {item}")
        geom_type = item[0]
        if geom_type == "MultiPolygon":
            # MultiPolygon is list of lists
            for subgeom in item[1:]:
                geom_type = subgeom[0]
                for subitem in subgeom[1:]:
                    yield geom_type, subitem
        else:
            for subitem in item[1:]:
                yield geom_type, subitem


def compute_rings(point_list, geom_list, feature_code_to_color):
    ring_adjacency = make_ring_adjacency_array(len(point_list))
    flattened_geom_list = []
    default_color = feature_code_to_color.get("default", 0x12345678)
    for geom_type, item in iter_geom(geom_list):
        log.debug(f"Adding geometry {item}")
        flattened_geom_list.append(item)
        ring_adjacency[item.start_index]['point_flag'] = -item.count
        ring_adjacency[item.start_index + item.count - 1]['point_flag'] = 2
        ring_adjacency[item.start_index]['state'] = 0
        if item.count > 0:
            ring_adjacency[item.start_index + 1]['state'] = item.feature_code
        if item.count > 1:
            ring_adjacency[item.start_index + 2]['state'] = feature_code_to_color.get(item.feature_code, default_color)
    return flattened_geom_list, ring_adjacency
