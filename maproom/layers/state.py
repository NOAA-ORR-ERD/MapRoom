"""
    point: x, y, z (depth), color, state
    state = selected | flagged | deleted | edited | added | land_polygon | water_polygon | other_polygon
"""

CLEAR = 0
SELECTED = 1
FLAGGED = 2
DELETED = 4
EDITED = 8
ADDED = 16
LAND_POLYGON = 32
WATER_POLYGON = 64
OTHER_POLYGON = 128

EXTRA_MASK = 0xffff0000
POLYGON_NUMBER_SHIFT = 256 * 256
