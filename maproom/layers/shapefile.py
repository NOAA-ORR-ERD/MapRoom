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

from point import PointLayer
from constants import *

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


class ShapefileLayer(PointLayer):
    """Layer for shapely objects.
    
    """
    type = Str("shapefile")
    
    mouse_mode_toolbar = Str("BaseLayerToolBar")
    
    geometry = List

    triangles = Any

    visibility_items = ["points", "triangles", "labels"]
    
    layer_info_panel = ["Layer name", "Triangle count", "Show depth shading"]

    def __str__(self):
        try:
            triangles = len(self.triangles)
        except:
            triangles = 0
        return PointLayer.__str__(self) + ", %d triangles" % triangles
    
    def get_info_panel_text(self, prop):
        if prop == "Triangle count":
            if self.triangles is not None:
                return str(len(self.triangles))
            return "0"
        return PointLayer.get_info_panel_text(self, prop)

    def empty(self):
        """
        We shouldn't allow saving of a layer with no content, so we use this method
        to determine if we can save this layer.
        """
        no_points = (self.points is None or len(self.points) == 0)
        no_triangles = (self.triangles is None or len(self.triangles) == 0)

        return no_points and no_triangles
        
    def visibility_item_exists(self, label):
        """Return keys for visibility dict lookups that currently exist in this layer
        """
        if label in ["points", "labels"]:
            return self.points is not None
        if label == "triangles":
            return self.triangles is not None
        raise RuntimeError("Unknown label %s for %s" % (label, self.name))

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
        self.update_bounds()

    def set_data(self, f_points, f_triangles):
        n = np.alen(f_points)
        self.set_layer_style_defaults()
        self.points = self.make_points(n)
        if (n > 0):
            self.points.view(data_types.POINT_XY_VIEW_DTYPE).xy[
                0: n
            ] = f_points
            self.points.z[0: n] = 0.0
            self.points.color = self.style.line_color
            self.points.state = 0

            n = len(f_triangles)
            if n > 0:
                self.triangles = self.make_triangles(n)
                self.triangles.view(data_types.TRIANGLE_POINTS_VIEW_DTYPE).point_indexes = f_triangles

    def can_save_as(self):
        return True

    def triangles_to_json(self):
        return self.triangles.tolist()

    def triangles_from_json(self, json_data):
        self.triangles = np.array([tuple(i) for i in json_data['triangles']], data_types.TRIANGLE_DTYPE).view(np.recarray)


    def make_triangles(self, count):
        return np.repeat(
            np.array([(0, 0, 0, 0, 0)], dtype=data_types.TRIANGLE_DTYPE),
            count,
        ).view(np.recarray)

    def geometry_to_triangles(self):
        tri_points = accumulator(block_shape=(2,), dtype=np.float32)
        tri_indexes = accumulator(block_shape=(3,), dtype=np.uint32)
        point_index = 0
        for geom in self.geometry:
            triangles = triangulate(geom)
            print geom
            for tri in triangles:
                print "Triangle"
                print tri
                print tri.exterior.coords
                tri_points.extend(tri.exterior.coords[0:3])
                tri_indexes.append((point_index, point_index + 1, point_index + 2))
                point_index += 3

        self.set_data(np.asarray(tri_points), np.asarray(tri_indexes))

    def rebuild_renderer(self, renderer, in_place=False):
        """Update display canvas data with the data in this layer
        
        """
        self.geometry_to_triangles()
        projected_point_data = self.compute_projected_point_data()
        renderer.set_points(projected_point_data, self.points.z, self.points.color.copy().view(dtype=np.uint8))
        triangles = self.triangles.view(data_types.TRIANGLE_POINTS_VIEW_DTYPE).point_indexes
        tri_points_color = np.zeros(len(self.points), dtype=np.uint32)
        tri_points_color[:] = color_floats_to_int(.5, .5, .9, 1)
        renderer.set_triangles(triangles, tri_points_color)
    
    def render_projected(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        log.log(5, "Rendering line layer!!! pick=%s" % (picker))
        if picker.is_active:
            return

        renderer.draw_triangles(self.style.line_width, layer_visibility["triangles"])

        if layer_visibility["labels"]:
            renderer.draw_labels_at_points(self.points.z, s_r, p_r)
