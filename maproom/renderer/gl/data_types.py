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
POINT_SIMPLE_DTYPE = np.dtype([
    ("x", np.float64),
    ("y", np.float64),
    ("color", np.uint32),
    ("state", np.uint32)
])
POINT_XY_VIEW_SIMPLE_DTYPE = np.dtype([
    ("xy", "2f8"),
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
    ("next", np.uint32),   # Index of next adjacent point in polygon.
    ("polygon", np.uint32)  # Index of polygon this point is in.
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
TEXTURE_COORDINATE_DTYPE = np.dtype(
    [("u_lb", np.float32), ("v_lb", np.float32),
     ("u_lt", np.float32), ("v_lt", np.float32),
     ("u_rt", np.float32), ("v_rt", np.float32),
     ("u_rb", np.float32), ("v_rb", np.float32)]
)

