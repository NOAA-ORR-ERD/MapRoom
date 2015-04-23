import os
import os.path
import time
import sys
import math

import numpy as np

import wx

# Enthought library imports.
from traits.api import Unicode, Str, Any, Float
from pyface.api import YES

from ..library import rect

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

    def rebuild_renderer(self, in_place=False):
        """Update renderer
        
        """
        projected_point_data = self.compute_projected_point_data()
        self.rasterize(projected_point_data, self.points.z, self.points.color.copy().view(dtype=np.uint8))

    def render_projected(self, w_r, p_r, s_r, layer_visibility, layer_index_base, picker):
        log.log(5, "Rendering vector object %s!!! visible=%s, pick=%s" % (self.name, layer_visibility["layer"], picker))
        if (not layer_visibility["layer"]):
            return

        self.draw_object(p_r, s_r, layer_index_base, picker)
    
    def draw_object(self, p_r, s_r, layer_index_base, picker):
        pass

class RectangleLayer(VectorObjectLayer):
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
    
    corners = np.asarray((0, 1, 2, 1, 2, 3, 0, 3), dtype=np.uint8)
    lines = np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.uint8)

    def set_opposite_corners(self, p1, p2):
        p = np.concatenate((p1, p2), 0)  # flatten to 1D
        c = p[self.corners].reshape(-1,2)
        self.set_control_points_from_corners(c)
        self.set_data(self.cp, 0.0, self.lines)
    
    def set_control_points_from_corners(self, c):
        cp = np.empty((5,2), dtype=np.float32)
        cp[0:4] = c
        cp[4] = c.mean(0)
        self.cp = cp

    def rasterize(self, projected_point_data, z, color):
        self.renderer.set_points(projected_point_data, z, color)
        self.renderer.set_lines(projected_point_data, self.line_segment_indexes.view(data_types.LINE_SEGMENT_POINTS_VIEW_DTYPE)["points"], self.line_segment_indexes.color)

    def draw_object(self, p_r, s_r, layer_index_base, picker):
        self.renderer.draw_object(layer_index_base, picker, 4)

class EllipseLayer(RectangleLayer):
    """Rectangle uses 4 control points in the self.points array, and nothing in
    the polygon points array.  All corner points can be used as control points.
    
    """
    name = Unicode("Ellipse")
    
    def rasterize(self, projected_point_data, z, color):
        self.renderer.set_points(projected_point_data, z, color)
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
        colors.fill(self.color)
        self.renderer.set_lines(xy, lsi, colors)

    def draw_object(self, p_r, s_r, layer_index_base, picker):
        self.renderer.draw_object(layer_index_base, picker, 4)
