import numpy as np


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


def make_polygons(count):
    return np.repeat(
        np.array([(0, 0, 0, 0, 0)], dtype=POLYGON_DTYPE),
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


def compute_rings(f_ring_starts, f_ring_counts, f_ring_groups=None, f_ring_colors=0):
    n_rings = np.alen(f_ring_starts)
    rings = make_polygons(n_rings)
    rings.start[0:n_rings] = f_ring_starts
    rings.count[0:n_rings] = f_ring_counts
    rings.color[0:n_rings] = f_ring_colors
    if f_ring_groups is None:
        # if not otherwise specified, each polygon is in its own group
        rings.group = np.arange(n_rings)
    else:
        # grouping of rings allows for holes: the first polygon is
        # the outer boundary and subsequent rings in the group are
        # the holes
        rings.group = np.asarray(f_ring_groups, dtype=np.uint32)
    n_points = f_ring_starts[-1] + f_ring_counts[-1]
    point_adjacency_array = make_point_adjacency_array(n_points)

    total = f_ring_starts[0]
    for p in range(n_rings):
        c = rings.count[p]
        point_adjacency_array.ring_index[total: total + c] = p
        point_adjacency_array.next[total: total + c] = np.arange(total + 1, total + c + 1)
        point_adjacency_array.next[total + c - 1] = total
        total += c

    return rings, point_adjacency_array
