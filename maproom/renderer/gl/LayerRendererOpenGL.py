import os
import os.path
import time
import sys
import numpy as np
import Point_and_line_set_renderer
import Triangle_set_renderer
import Polygon_set_renderer
import Label_set_renderer
import Image_set_renderer
from maproom.library.color import *
import maproom.library.rect as rect
from maproom.Editor import *


"""
    point: x, y, z (depth), color, state
    state = selected | flagged | deleted | edited | added | land_polygon | water_polygon | other_polygon
"""

# for the picker, we only have a total of 255 "layers" that it tracks;
# so we give each Layer 10 slots, 5 for the point-and-line renderer and
# 5 for the polygon renderer; within each renderer, these 5 are further
# divided to distinguish between points, lines, and fills; so the assumption
# here is that we won't have more than 25 actively "pickable" Layers, and
# each layer won't have more than 10 subcategories of pickable item types
# among its renderers
POINTS_AND_LINES_SUB_LAYER_PICKER_OFFSET = 0
POLYGONS_SUB_LAYER_PICKER_OFFSET = 5

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

MAX_LABEL_CHARATERS = 1000 * 5


class LayerRendererOpenGL():

    """
    OpenGL Renderer for MapRoom Layers
    """

    def __init__(self, renderer, layer):
        self.renderer = renderer
        self.layer = layer

        self.point_and_line_set_renderer = None
        self.triangle_set_renderer = None
        self.label_set_renderer = None
        self.polygon_set_renderer = None
        self.image_set_renderer = None

    def __repr__(self):
        return self.name

    def create_necessary_renderers(self):
        if (self.layer.triangle_points != None and self.triangle_set_renderer == None):
            self.rebuild_triangle_set_renderer()

        if (self.layer.points != None and self.point_and_line_set_renderer == None):
            if (self.layer.line_segment_indexes == None):
                self.layer.line_segment_indexes = self.layer.make_line_segment_indexes(0)

            self.rebuild_point_and_line_set_renderer(create=True)

        if self.layer.polygons != None and self.polygon_set_renderer == None:
            self.polygon_set_renderer = Polygon_set_renderer.Polygon_set_renderer(
                self.renderer.opengl_renderer,
                self.layer.points.view(self.layer.POINT_XY_VIEW_DTYPE).xy[: len(self.layer.points)].copy(),
                self.layer.polygon_adjacency_array,
                self.layer.polygons,
                self.renderer.projection,
                self.renderer.projection_is_identity)

        if self.layer.images and not self.image_set_renderer:
            self.image_set_renderer = Image_set_renderer.Image_set_renderer(
                self.renderer.opengl_renderer,
                self.layer.images,
                self.layer.image_sizes,
                self.layer.image_world_rects,
                self.renderer.projection,
                self.renderer.projection_is_identity)

        self.set_up_labels()

    def set_up_labels(self):
        if (self.layer.points != None and self.label_set_renderer == None):
            self.label_set_renderer = Label_set_renderer.Label_set_renderer(self.renderer.opengl_renderer, MAX_LABEL_CHARATERS)

    def rebuild_triangle_set_renderer(self):
        if self.triangle_set_renderer:
            self.triangle_set_renderer.destroy()

        self.triangle_set_renderer = Triangle_set_renderer.Triangle_set_renderer(
            self.renderer.opengl_renderer,
            self.layer.triangle_points.view(self.layer.POINT_XY_VIEW_DTYPE).xy,
            self.layer.triangle_points.color.copy().view(dtype=np.uint8),
            self.layer.triangles.view(self.layer.TRIANGLE_POINTS_VIEW_DTYPE).point_indexes,
            self.renderer.projection,
            self.renderer.projection_is_identity)

    def rebuild_point_and_line_set_renderer(self, create=False):
        if self.point_and_line_set_renderer:
            create = True
            self.point_and_line_set_renderer.destroy()

        t0 = time.clock()
        if create:
            self.point_and_line_set_renderer = Point_and_line_set_renderer.Point_and_line_set_renderer(
                self.renderer.opengl_renderer,
                self.layer.points.view(self.layer.POINT_XY_VIEW_DTYPE).xy,
                self.layer.points.color.copy().view(dtype=np.uint8),
                self.layer.line_segment_indexes.view(self.layer.LINE_SEGMENT_POINTS_VIEW_DTYPE)["points"],
                self.layer.line_segment_indexes.color,
                self.renderer.projection,
                self.renderer.projection_is_identity)

        t = time.clock() - t0  # t is wall seconds elapsed (floating point)
        print "rebuilt point and line set renderer in {0} seconds".format(t)

    def reproject(self, projection, projection_is_identity):
        if (self.point_and_line_set_renderer != None):
            self.point_and_line_set_renderer.reproject(self.layer.points.view(self.layer.POINT_XY_VIEW_DTYPE).xy,
                                                       projection,
                                                       projection_is_identity)
        if (self.polygon_set_renderer != None):
            self.polygon_set_renderer.reproject(projection, projection_is_identity)
        """
        if ( self.label_set_renderer != None ):
            self.label_set_renderer.reproject( self.layer.points.view( self.POINT_XY_VIEW_DTYPE ).xy,
                                               projection,
                                               projection_is_identity )
        """
        if (self.image_set_renderer != None):
            self.image_set_renderer.reproject(projection, projection_is_identity)

    def render(self, render_window, layer_visibility, layer_index_base, pick_mode=False):
        if (not layer_visibility["layer"]):
            return

        s_r = render_window.get_screen_rect()
        p_r = render_window.get_projected_rect_from_screen_rect(s_r)
        w_r = render_window.get_world_rect_from_projected_rect(p_r)

        # the images
        if (self.image_set_renderer != None and layer_visibility["images"]):
            self.image_set_renderer.render(-1, pick_mode)

        # the polygons
        if (self.polygon_set_renderer != None and layer_visibility["polygons"]):
            self.polygon_set_renderer.render(layer_index_base + POLYGONS_SUB_LAYER_PICKER_OFFSET,
                                             pick_mode,
                                             self.layer.polygons.color,
                                             color_to_int(0, 0, 0, 1.0),
                                             1)  # , self.get_selected_polygon_indexes()

        # the triangle points and triangle line segments
        if (self.triangle_set_renderer != None and layer_visibility["triangles"]):
            self.triangle_set_renderer .render(pick_mode,
                                               self.layer.point_size + 10,
                                               self.layer.triangle_line_width)

        # the points and line segments
        if (self.point_and_line_set_renderer != None):
            self.point_and_line_set_renderer.render(layer_index_base + POINTS_AND_LINES_SUB_LAYER_PICKER_OFFSET,
                                                    pick_mode,
                                                    self.layer.point_size,
                                                    self.layer.line_width,
                                                    layer_visibility["points"],
                                                    layer_visibility["lines"],
                                                    self.layer.get_selected_point_indexes(),
                                                    self.layer.get_selected_point_indexes(STATE_FLAGGED),
                                                    self.layer.get_selected_line_segment_indexes(),
                                                    self.layer.get_selected_line_segment_indexes(STATE_FLAGGED))

            # the labels
            if (self.label_set_renderer != None and layer_visibility["labels"] and self.point_and_line_set_renderer.vbo_point_xys != None):
                self.label_set_renderer.render(-1, pick_mode, s_r,
                                               MAX_LABEL_CHARATERS, self.layer.points.z,
                                               self.point_and_line_set_renderer.vbo_point_xys.data,
                                               p_r, render_window.projected_units_per_pixel)

        # render selections after everything else
        if (self.point_and_line_set_renderer != None and not pick_mode):
            if layer_visibility["lines"]:
                self.point_and_line_set_renderer.render_selected_line_segments(self.layer.line_width, self.layer.get_selected_line_segment_indexes())

            if layer_visibility["points"]:
                self.point_and_line_set_renderer.render_selected_points(self.layer.point_size,
                                                                        self.layer.get_selected_point_indexes())

    def __del__(self):
        if (self.point_and_line_set_renderer != None):
            self.point_and_line_set_renderer.destroy()
            self.point_and_line_set_renderer = None
        if (self.polygon_set_renderer != None):
            self.polygon_set_renderer.destroy()
            self.polygon_set_renderer = None
        if (self.label_set_renderer != None):
            self.label_set_renderer.destroy()
            self.label_set_renderer = None
        if (self.image_set_renderer != None):
            self.image_set_renderer.destroy()
            self.image_set_renderer = None
