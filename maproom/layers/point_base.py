"""
Layer type to be used as a base class for layers with points

"""
import os
import os.path
import time
import sys
import tempfile
import shutil
from StringIO import StringIO
import numpy as np
import wx
from pytriangle import triangulate_simple
from shapely.geometry import box, Polygon, MultiPolygon, LineString, MultiLineString

# Enthought library imports.
from traits.api import Int, Unicode, Any, Str, Float, Enum, Property

from ..library import rect
from ..library.accumulator import flatten
from ..library.Boundary import Boundaries, PointsError
from ..renderer import color_floats_to_int, data_types

from base import ProjectedLayer
from constants import *

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


class PointBaseLayer(ProjectedLayer):
    """
    Layer for just points
    
    """
    name = Unicode("Point Layer")
    
    type = Str("base_point")
    
    points = Any
    
    visibility_items = ["points"]
    
    layer_info_panel = ["Layer name", "Point count"]
    
    selection_info_panel = []

    def __str__(self):
        try:
            points = len(self.points)
        except TypeError:
            points = 0
        return "%s layer '%s': %d points" % (self.type, self.name, points)
    
    def get_info_panel_text(self, prop):
        if prop == "Point count":
            return str(len(self.points))
        return ProjectedLayer.get_info_panel_text(self, prop)

    def new(self):
        super(PointBaseLayer, self).new()
        self.new_points()

    def has_points(self):
        return True

    def new_points(self, num=0):
        #fixme: this should be done differently...
        self.set_layer_style_defaults()
        self.points = self.make_points(num)

    def empty(self):##fixme: make a property?
        """
        We shouldn't allow saving of a layer with no content, so we use this method
        to determine if we can save this layer.
        """
        no_points = (self.points is None or len(self.points) == 0)

        return no_points

    ##fixme: can we remove all the visibility stuff???    
    ## and if not -- this shouldn't have any references to labels
    def get_visibility_dict(self):
        ##fixme: why not call self.get_visibility_dict ?
        d = ProjectedLayer.get_visibility_dict(self)
        ## fixme: and why do I need to mess with label visibility here?
        d["labels"] = False
        return d
    
    def set_data(self, f_points):
        n = np.alen(f_points)
        self.set_layer_style_defaults()
        self.points = self.make_points(n)
        if (n > 0):
            self.points.view(data_types.POINT_XY_VIEW_DTYPE).xy[0:n] = f_points
            self.points.z = 0.0
            self.points.color = self.style.line_color
            self.points.state = 0

        self.update_bounds()
    
    def set_color(self, color):
        self.points.color = color
    
    def set_style(self, style):
        ProjectedLayer.set_style(self, style)
        if style.line_color is not None:
            self.set_color(style.line_color)

    def make_points(self, count):
        return np.repeat(
            np.array([(np.nan, np.nan, np.nan, 0, 0)], dtype=data_types.POINT_DTYPE),
            count,
        ).view(np.recarray)

    def compute_bounding_rect(self, mark_type=STATE_NONE):
        bounds = rect.NONE_RECT

        if (self.points is not None and len(self.points) > 0):
            if (mark_type == STATE_NONE):
                points = self.points
            else:
                points = self.points[self.get_selected_point_indexes(mark_type)]
            # fixme -- could be more eficient numpy-wise
            l = points.x.min()
            r = points.x.max()
            b = points.y.min()
            t = points.y.max()
            bounds = rect.accumulate_rect(bounds, ((l, b), (r, t)))

        return bounds

    def get_state(self, index):
        ##fixme -- is this needed -- should all points have a state?
        return self.points.state[index]

    def compute_selected_bounding_rect(self):
        bounds = self.compute_bounding_rect(STATE_SELECTED)
        return bounds

    def clear_all_selections(self, mark_type=STATE_SELECTED):
        self.clear_all_point_selections(mark_type)
        self.increment_change_count()

    def clear_all_point_selections(self, mark_type=STATE_SELECTED):
        if (self.points is not None):
            self.points.state = self.points.state & (0xFFFFFFFF ^ mark_type)
            self.increment_change_count()

    def clear_flagged(self, refresh=False):
        self.clear_all_selections(STATE_FLAGGED)
        if refresh:
            self.manager.dispatch_event('refresh_needed')
    
    def has_selection(self):
        return self.get_num_points_selected() > 0

    def has_flagged(self):
        return self.get_num_points_flagged() > 0

    def select_point(self, point_index, mark_type=STATE_SELECTED):
        self.points.state[point_index] = self.points.state[point_index] | mark_type
        self.increment_change_count()

    def deselect_point(self, point_index, mark_type=STATE_SELECTED):
        self.points.state[point_index] = self.points.state[point_index] & (0xFFFFFFFF ^ mark_type)
        self.increment_change_count()

    def is_point_selected(self, point_index, mark_type=STATE_SELECTED):
        return self.points is not None and (self.points.state[point_index] & mark_type) != 0

    def select_points(self, indexes, mark_type=STATE_SELECTED):
        self.points.state[indexes] |= mark_type
        self.increment_change_count()

    def deselect_points(self, indexes, mark_type=STATE_SELECTED):
        self.points.state[indexes] &= (0xFFFFFFFF ^ mark_type)
        self.increment_change_count()
    
    def select_flagged(self, refresh=False):
        indexes = self.get_selected_point_indexes(STATE_FLAGGED)
        self.deselect_points(indexes, STATE_FLAGGED)
        self.select_points(indexes, STATE_SELECTED)
        if refresh:
            self.manager.dispatch_event('refresh_needed')

    def select_points_in_rect(self, is_toggle_mode, is_add_mode, w_r, mark_type=STATE_SELECTED):
        if (not is_toggle_mode and not is_add_mode):
            self.clear_all_point_selections()
        indexes = np.where(np.logical_and(
            np.logical_and(self.points.x >= w_r[0][0], self.points.x <= w_r[1][0]),
            np.logical_and(self.points.y >= w_r[0][1], self.points.y <= w_r[1][1])))
        if (is_add_mode):
            self.points.state[indexes] |= mark_type
        if (is_toggle_mode):
            self.points.state[indexes] ^= mark_type
        self.increment_change_count()

    def get_selected_point_indexes(self, mark_type=STATE_SELECTED):
        if (self.points is None):
            return []
        return np.where((self.points.state & mark_type) != 0)[0]

    def get_selected_and_dependent_point_indexes(self, mark_type=STATE_SELECTED):
        """Get all points from selected objects.
        
        Subclasses should override to provide a list of points that are
        implicitly selected by an object being selected.
        """
        return self.get_selected_point_indexes(mark_type)

    def get_num_points_selected(self, mark_type=STATE_SELECTED):
        return len(self.get_selected_point_indexes(mark_type))

    def get_num_points_flagged(self):
        return len(self.get_selected_point_indexes(STATE_FLAGGED))
    
    def is_mergeable_with(self, other_layer):
        return hasattr(other_layer, "points")
    
    def find_merge_layer_class(self, other_layer):
        return type(self)

    def merge_from_source_layers(self, layer_a, layer_b):
        # for now we only handle merging of points and lines
        self.new()
        
        self.merged_points_index = len(layer_a.points)

        n = len(layer_a.points) + len(layer_b.points)
        self.points = self.make_points(n)
        self.points[
            0: len(layer_a.points)
        ] = layer_a.points.copy()
        self.points[
            len(layer_a.points): n
        ] = layer_b.points.copy()
        # self.points.state = 0
    
    def compute_projected_point_data(self):
        projection = self.manager.project.layer_canvas.projection
        view = self.points.view(data_types.POINT_XY_VIEW_DTYPE).xy.astype(np.float32)
        projected_point_data = np.zeros(
            (len(self.points), 2),
            dtype=np.float32
        )
        projected_point_data[:, 0], projected_point_data[:, 1] = projection(view[:,0], view[:,1])
        return projected_point_data
    
    def rebuild_renderer(self, in_place=False):
        """Update renderer
        
        """
        projected_point_data = self.compute_projected_point_data()
        self.renderer.set_points(projected_point_data, self.points.z, self.points.color.copy().view(dtype=np.uint8))

    def render_projected(self, w_r, p_r, s_r, layer_visibility, layer_index_base, picker):
        log.log(5, "Rendering line layer!!! visible=%s, pick=%s" % (layer_visibility["layer"], picker))
        if (not layer_visibility["layer"]):
            return

        if layer_visibility["points"]:
            self.renderer.draw_points(layer_index_base, picker, self.point_size,
                                      self.get_selected_point_indexes(),
                                      self.get_selected_point_indexes(STATE_FLAGGED))

        # the labels
        if layer_visibility["labels"]:
            self.renderer.draw_labels_at_points(self.points.z, s_r, p_r)
                
        # render selections after everything else
        if (not picker.is_active):
            if layer_visibility["points"]:
                self.renderer.draw_selected_points(
                    self.point_size,
                    self.get_selected_point_indexes())
