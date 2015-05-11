import os
import os.path
import time
import sys
import math

import numpy as np

import wx

# Enthought library imports.
from traits.api import on_trait_change, Unicode, Str, Any, Float, Bool, Int
from pyface.api import YES

from ..library import rect
from ..mouse_commands import MoveControlPointCommand
from ..renderer import color_to_int, int_to_color

from line import LineLayer
from constants import *

import logging
log = logging.getLogger(__name__)


class VectorObjectLayer(LineLayer):
    """Layer for a vector object
    
    Vector objects have control points (the points that can be moved by the
    user) and rasterized points, the points created by the control points
    in order to generate whatever shape is needed.  E.g.  a simple spline
    might have 4 control points, but only those endpoints on the drawn object
    actually go through the control points so the rasterized points are
    computed from those control points.
    
    The self.points array contains the control points
    """
    name = Unicode("Vector Object Layer")

    type = Str("vector_object")
    
    mouse_mode_toolbar = Str("AnnotationLayerToolBar")

    alpha = Float(1.0)
    
    rebuild_needed = Bool(False)
    
    def has_alpha(self):
        return True
    
    def set_layer_style_defaults(self):
        self.style.line_color = self.manager.default_style.line_color
    
    @on_trait_change('alpha')
    def mark_rebuild(self):
        self.rebuild_needed = True

    def set_visibility_when_selected(self, layer_visibility):
        layer_visibility['points'] = True

    def clear_visibility_when_deselected(self, layer_visibility):
        layer_visibility['points'] = False
    
    def rebuild_renderer(self, in_place=False):
        """Update renderer
        
        """
        projected_point_data = self.compute_projected_point_data()
        r, g, b, a = int_to_color(self.style.line_color)
        color = color_to_int(r, g, b, self.alpha)
#        self.rasterize(projected_point_data, self.points.z, self.points.color.copy().view(dtype=np.uint8))
        self.rasterize(projected_point_data, self.points.z, self.style.line_color, color)
        self.rebuild_needed = False

    def render_projected(self, w_r, p_r, s_r, layer_visibility, layer_index_base, picker):
        """Renders the outline of the vector object.
        
        If the vector object subclass is fillable, subclass from
        FillableVectorObject instead of this base class.
        """
        log.log(5, "Rendering vector object %s!!! visible=%s, pick=%s" % (self.name, layer_visibility["layer"], picker))
        if (not layer_visibility["layer"]):
            return
        if self.rebuild_needed:
            self.rebuild_renderer()
        self.renderer.outline_object(layer_index_base, picker, self.style)
        if layer_visibility["points"]:
            self.renderer.draw_points(layer_index_base, picker, self.point_size)


class LineVectorObject(VectorObjectLayer):
    """Line uses 3 control points in the self.points array.  The midpoint is an
    additional control point, which is constrained and not independent of the
    ends.  This is used as the control point when translating.
    
    """
    name = Unicode("Line")
    
    layer_info_panel = ["Layer name", "Line Color", "Transparency"]
    
    selection_info_panel = ["Point coordinates"]

    corners = np.asarray((0, 1, 2, 3), dtype=np.uint8)
    lines = np.asarray(((0, 1),), dtype=np.uint8)
    center_point_index = 2
    display_center_control_point = False
    
    # return the anchor point of the index point. E.g. anchors_of[0] = 1
    anchors_of = np.asarray((1, 0, 2), dtype=np.uint8)
    
    # anchor modification array: apply dx,dy values to each control point based
    # on the anchor point.  Used when moving/resizing
    anchor_dxdy = np.asarray((
        ((0,0), (1,1), (0.5,0.5)), # anchor point is 0 (drag point is 1)
        ((1,1), (0,0), (0.5,0.5)), # anchor point is 1, etc.
        ((1,1), (1,1), (1,1)), # center point acts as rigid move
        ), dtype=np.float32)

    def set_opposite_corners(self, p1, p2):
        p = np.concatenate((p1, p2), 0)  # flatten to 1D
        c = p[self.corners].reshape(-1,2)
        self.set_control_points_from_corners(c)
        self.set_data(self.cp, 0.0, self.lines)
    
    def set_control_points_from_corners(self, c):
        num_cp = self.center_point_index + 1
        cp = np.empty((num_cp,2), dtype=np.float32)
        cp[0:self.center_point_index] = c
        cp[self.center_point_index] = c.mean(0)
        self.cp = cp

    def find_anchor_of(self, point_index):
        self.anchor_point = self.anchors_of[point_index]

    def set_anchor_point(self, point_index, maintain_aspect=False):
        self.clear_all_selections()
        self.select_point(point_index)
        self.drag_point = point_index
        self.find_anchor_of(point_index)
        
    def dragging_selected_objects(self, world_dx, world_dy):
        cmd = MoveControlPointCommand(self, self.drag_point, self.anchor_point, world_dx, world_dy)
        return cmd
    
    def move_control_point(self, drag, anchor, dx, dy):
        """Moving the control point changes the size of the bounding rectangle.
        
        Assuming the drag point is one of the corners and the anchor is the
        opposite corner, the points are constrained as follows: the drag point
        moves by both dx & dy.  The anchor point doesn't move at all, and of
        the other points: one only uses dx and the other dy.
        """
        scale = self.anchor_dxdy[anchor]
        xoffset = scale.T[0] * dx
        yoffset = scale.T[1] * dy
        
        # FIXME: Why does specifying .x work for a range, but not for a single
        # element? Have to use the dict notation for a single element.
        self.points.x += xoffset
        self.points.y += yoffset
    
    def rasterize(self, projected_point_data, z, cp_color, line_color):
        n = np.alen(self.points)
        if not self.display_center_control_point:
            n -= 1
        colors = np.empty(n, dtype=np.uint32)
        colors.fill(cp_color)
        self.renderer.set_points(projected_point_data, z, colors, num_points=n)
        colors = np.empty(np.alen(self.line_segment_indexes), dtype=np.uint32)
        colors.fill(line_color)
        self.renderer.set_lines(projected_point_data, self.line_segment_indexes.view(data_types.LINE_SEGMENT_POINTS_VIEW_DTYPE)["points"], colors)


class FillableVectorObject(LineVectorObject):
    # Fillable objects should (in general) display their center control point
    display_center_control_point = True

    @on_trait_change('alpha')
    def mark_rebuild(self):
        r, g, b, a = int_to_color(self.style.fill_color)
        self.style.fill_color = color_to_int(r, g, b, self.alpha)
        self.rebuild_needed = True
    
    def set_layer_style_defaults(self):
        self.style.line_color = self.manager.default_style.line_color
        self.style.fill_color = self.manager.default_style.fill_color

    def render_projected(self, w_r, p_r, s_r, layer_visibility, layer_index_base, picker):
        log.log(5, "Rendering vector object %s!!! visible=%s, pick=%s" % (self.name, layer_visibility["layer"], picker))
        if (not layer_visibility["layer"]):
            return
        if self.rebuild_needed:
            self.rebuild_renderer()
        self.renderer.fill_object(layer_index_base, picker, self.style)
        self.renderer.outline_object(layer_index_base, picker, self.style)
        if layer_visibility["points"]:
            self.renderer.draw_points(layer_index_base, picker, self.point_size)
            

class RectangleVectorObject(FillableVectorObject):
    """Rectangle uses 4 control points in the self.points array, and nothing in
    the polygon points array.  All corner points can be used as control points.
    
    The center is an additional control point, which is constrained and not
    independent of the corners.
    
     3           2
      o---------o
      |         |
      |    o 4  |
      |         |
      o---------o
     0           1
    """
    name = Unicode("Rectangle")
    
    layer_info_panel = ["Layer name", "Line Color", "Fill Color", "Transparency"]
    
    selection_info_panel = ["Point coordinates"]

    corners = np.asarray((0, 1, 2, 1, 2, 3, 0, 3), dtype=np.uint8)
    lines = np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.uint8)
    center_point_index = 4
    
    # return the anchor point of the index point. E.g. anchors_of[0] = 2
    anchors_of = np.asarray((2, 3, 0, 1, 4), dtype=np.uint8)
    
    # anchor modification array: apply dx,dy values to each control point based
    # on the anchor point.  Used when moving/resizing
    anchor_dxdy = np.asarray((
        ((0,0), (1,0), (1,1), (0,1), (0.5,0.5)), # anchor point is 0 (drag point is 2)
        ((1,0), (0,0), (0,1), (1,1), (0.5,0.5)), # anchor point is 1 (drag is 3)
        ((1,1), (0,1), (0,0), (1,0), (0.5,0.5)), # anchor point is 2, etc.
        ((0,1), (1,1), (1,0), (0,0), (0.5,0.5)),
        ((1,1), (1,1), (1,1), (1,1), (1,1)), # center point acts as rigid move
        ), dtype=np.float32)


class EllipseVectorObject(RectangleVectorObject):
    """Rectangle uses 4 control points in the self.points array, and nothing in
    the polygon points array.  All corner points can be used as control points.
    
    """
    name = Unicode("Ellipse")
    
    def rasterize(self, projected_point_data, z, cp_color, line_color):
        colors = np.empty(np.alen(self.points), dtype=np.uint32)
        colors.fill(cp_color)
        self.renderer.set_points(projected_point_data, z, colors)
        p = projected_point_data
        
        # FIXME: this only supports axis aligned ellipses
        width = p[1][0] - p[0][0]
        height = p[2][1] - p[1][1]
        sx = width / 2
        sy = height / 2
        cx = p[4][0]
        cy = p[4][1]
         
        num_segments = 128
        xy = np.zeros((num_segments, 2), dtype=np.float32)
        
        dtheta = 2 * 3.1415926 / num_segments
        theta = 0.0
        x = sx # we start at angle = 0 
        y = 0
        i = 0
        while i < num_segments:
            xy[i] = (cx + sx*math.cos(theta), cy + sy*math.sin(theta))
            theta += dtheta
            i += 1
        
        # create line segment list from one point to the next
        i1 = np.arange(num_segments, dtype=np.uint32)
        i2 = np.arange(1, num_segments+1, dtype=np.uint32)
        i2[-1] = 0
        lsi = np.vstack((i1, i2)).T  # zip arrays to get line segment indexes
        
        # set_lines expects a color list for each point, not a single color
        colors = np.empty(num_segments, dtype=np.uint32)
        colors.fill(line_color)
        self.renderer.set_lines(xy, lsi, colors)
