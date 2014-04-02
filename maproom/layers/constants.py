"""
    point: x, y, z (depth), color, state
    state = selected | flagged | deleted | edited | added | land_polygon | water_polygon | other_polygon
"""
from ..renderer import color_to_int, data_types

STATE_NONE = 0
STATE_SELECTED = 1
STATE_FLAGGED = 2
STATE_DELETED = 4
STATE_EDITED = 8
STATE_ADDED = 16
STATE_LAND_POLYGON = 32
STATE_WATER_POLYGON = 64
STATE_OTHER_POLYGON = 128

DEFAULT_DEPTH = 1.0
DEFAULT_POINT_COLOR = color_to_int(0.5, 0.5, 0, 1.0)
DEFAULT_COLORS = [
    color_to_int(0, 0, 1.0, 1),
    color_to_int(0, 0.75, 0, 1),
    color_to_int(0.5, 0, 1.0, 1),
    color_to_int(1.0, 0.5, 0, 1),
    color_to_int(0.5, 0.5, 0, 1),
]
DEFAULT_LINE_SEGMENT_COLOR = color_to_int(0.5, 0, 0.5, 1.0)
