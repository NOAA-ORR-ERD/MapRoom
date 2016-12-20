import os
import os.path
import time
import sys
import numpy as np

from shapely.geometry import box, Polygon, MultiPolygon, LineString, MultiLineString
from shapely.ops import triangulate

# Enthought library imports.
from traits.api import Int, Unicode, Any, Str, Float, Enum, Property, List

from ..library import rect
from ..library.accumulator import flatten
from ..library.projection import Projection
from ..library.Boundary import Boundaries, PointsError
from ..renderer import color_floats_to_int, data_types
from ..library.accumulator import accumulator
from ..library.shapely_utils import shapely_to_polygon

from point import PointLayer
from polygon import PolygonLayer
from constants import *

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


class PolygonShapefileLayer(PolygonLayer):
    """Layer for shapely objects rendered as polygons.
    
    """
    type = Str("shapefile")
    
    geometry = List
    
    layer_info_panel = ["Layer name", "Shapefile Objects", "Polygon count"]

    def __str__(self):
        num = len(self.geometry)
        return "ShapefileLayer %s: %d objects" % (self.name, num)
    
    def get_info_panel_text(self, prop):
        if prop == "Shapefile Objects":
            return str(len(self.geometry))
        return PolygonLayer.get_info_panel_text(self, prop)

    def empty(self):
        """
        We shouldn't allow saving of a layer with no content, so we use this method
        to determine if we can save this layer.
        """
        return len(self.geometry) == 0

    def set_layer_style_defaults(self):
        self.style.use_next_default_color()
        self.style.line_width = 1

    def compute_bounding_rect(self, mark_type=STATE_NONE):
        bounds = rect.NONE_RECT

        if (len(self.geometry) > 0):
            for o in self.geometry:
                l, b, r, t = o.bounds
                bounds = rect.accumulate_rect(bounds, ((l, b), (r, t)))

        return bounds

    def set_geometry(self, geom):
        self.geometry = geom
        
        (self.load_error_string,
         f_polygon_points,
         f_polygon_starts,
         f_polygon_counts,
         f_polygon_identifiers,
         f_polygon_groups) = shapely_to_polygon(self.geometry)
        self.set_data(f_polygon_points, f_polygon_starts, f_polygon_counts,
                 f_polygon_identifiers, f_polygon_groups)
