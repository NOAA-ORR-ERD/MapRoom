import os
import os.path
import time
import sys
import numpy as np

# Enthought library imports.
from traits.api import Int, Unicode, Any, Str, Float, Enum, Property

from ..library import rect
from ..library.Boundary import Boundaries, PointsError
from ..library.textparse import parse_int_string, int_list_to_string
from ..renderer import color_to_int, data_types
from ..command import UndoInfo
from ..mouse_commands import MovePointsCommand

from point_base import PointBaseLayer
from constants import *

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


class PointLayer(PointBaseLayer):
    """
    Layer for points with depth and labels..

    Points are selectable, and all that

    """
    name = Unicode("Point Layer")
    
    type = Str("point")
    
    mouse_mode_toolbar = Str("VectorLayerToolBar")
      
    merged_points_index = Int(0)
    
    default_depth = Float(1.0)
    
    depth_unit = Property(Str)
    
    _depth_unit = Enum("unknown", "meters", "feet", "fathoms")
    
    pickable = True # is this a layer that support picking?

    visibility_items = ["points", "labels"]
    
    layer_info_panel = ["Layer name", "Point count", "Flagged points", "Default depth", "Depth unit", "Color"]
    
    selection_info_panel = ["Selected points", "Point index", "Point depth", "Point coordinates"]

    # Trait setters/getters
    def _get_depth_unit(self):
        return self._depth_unit
    
    def _set_depth_unit(self, unit):
        unit = unit.lower()
        if unit in ['meter', 'meters', 'm']:
            unit = 'meters'
        elif unit in ['foot', 'feet', 'ft']:
            unit = 'feet'
        elif unit in ['fathom', 'fathoms', 'ftm']:
            unit = 'fathoms'
        else:
            log.warning("Depth unit '%s' in %s; set to 'unknown'" % (unit, self.file_path))
            unit = 'unknown'
        self._depth_unit = unit
    
    def get_info_panel_text(self, prop):
        if prop == "Selected points":
            return str(self.get_num_points_selected())
        return PointBaseLayer.get_info_panel_text(self, prop)

    def highlight_exception(self, e):
        if hasattr(e, "points") and e.points is not None:
            self.clear_all_selections(STATE_FLAGGED)
            for p in e.points:
                self.select_point(p, STATE_FLAGGED)
            self.manager.dispatch_event('refresh_needed')
    
    def clear_flagged(self, refresh=False):
        self.clear_all_selections(STATE_FLAGGED)
        if refresh:
            self.manager.dispatch_event('refresh_needed')
    
    def set_data(self, f_points, f_depths, f_line_segment_indexes):
        n = np.alen(f_points)
        self.determine_layer_color()
        self.points = self.make_points(n)
        if (n > 0):
            self.points.view(data_types.POINT_XY_VIEW_DTYPE).xy[0:n] = f_points
            self.points.z[0:n] = f_depths
            self.points.color = self.color
            self.points.state = 0

            n = np.alen(f_line_segment_indexes)
            self.line_segment_indexes = self.make_line_segment_indexes(n)
            self.line_segment_indexes.view(data_types.LINE_SEGMENT_POINTS_VIEW_DTYPE).points[
                0: n
            ] = f_line_segment_indexes
            self.line_segment_indexes.color = self.color
            self.line_segment_indexes.state = 0
        
        self.update_bounds()
    
    def can_save(self):
        return self.can_save_as() and bool(self.file_path)
    
    def serialize_json(self, index):
        json = Layer.serialize_json(self, index)
        update = {
            'has encoded data': True,
            'points': self.points.tolist(),
            'default_depth': self.default_depth,
            'depth_unit': self.depth_unit,
        }
        json.update(update)
        return json
    
    def unserialize_json_version1(self, json_data):
        Layer.unserialize_json_version1(self, json_data)
        # numpy can't restore an array of arrays; must be array of tuples
        self.points = np.array([tuple(i) for i in json_data['points']], data_types.POINT_DTYPE).view(np.recarray)
        self.default_depth = json_data['default_depth']
        self.depth_unit = json_data['depth_unit']
        self.update_bounds()
    
    def make_points(self, count):
        return np.repeat(
            np.array([(np.nan, np.nan, np.nan, 0, 0)], dtype=data_types.POINT_DTYPE),
            count,
        ).view(np.recarray)

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
    
    def dragging_selected_objects(self, world_dx, world_dy):
        indexes = self.get_selected_and_dependent_point_indexes()
        cmd = MovePointsCommand(self, indexes, world_dx, world_dy)
        return cmd
    
    def insert_point(self, world_point):
        if self.points is None:
            index = -1
        else:
            index = len(self.points)
        return self.insert_point_at_index(index, world_point, self.default_depth, self.color, STATE_SELECTED)

    def insert_point_at_index(self, point_index, world_point, z, color, state):
        t0 = time.clock()
        # insert it into the layer
        p = np.array([(world_point[0], world_point[1], z, color, state)],
                     dtype=data_types.POINT_DTYPE)
        undo = UndoInfo()
        if (self.points is None):
            self.new_points(1)
            self.points[0] = p
            point_index = 0
            undo.flags.refresh_needed = True
            undo.flags.select_layer = self
        else:
            self.points = np.insert(self.points, point_index, p).view(np.recarray)
        undo.index = point_index
        undo.data = np.copy(p)
        undo.flags.items_moved = True
        undo.flags.layer_contents_added = self

        # update point indexes in the line segements to account for the inserted point
        self.update_after_insert_point_at_index(point_index)

        return undo

    def update_after_insert_point_at_index(self, point_index):
        """Hook for subclasses to update dependent items when a point is inserted.
        """
        pass

    def delete_point(self, point_index):
        if (self.find_points_connected_to_point(point_index) != []):
            raise Exception()

        undo = UndoInfo()
        p = self.points[point_index]
        print "LABEL: deleting point: %s" % str(p)
        undo.index = point_index
        undo.data = np.copy(p)
        undo.flags.refresh_needed = True
        undo.flags.items_moved = True
        undo.flags.layer_contents_deleted = self
        self.points = np.delete(self.points, point_index, 0)

        # update point indexes in the line segements to account for the deleted point
        self.update_after_delete_point(point_index)

        return undo

    # def find_points_connected_to_point(self, point_index):
    #     ## fixme -- are an points connected to a pint without a lines layer? i.e. this belongs in LineLayer]
    #     ##          ormore tot the point , not needed
    #     return []
    
    def update_after_delete_point(self, point_index):
        """Hook for subclasses to update dependent items when a point is deleted.
        """
        pass
    
    def create_renderer(self, renderer):
        """Create the graphic renderer for this layer.
        
        There may be multiple views of this layer (e.g.  in different windows),
        so we can't just create the renderer as an attribute of this object.
        The storage parameter is attached to the view and independent of
        other views of this layer.
        
        """
        ## fixme: in theory, this is points only, so why the point_and_line_set_renderer ?
        ##        could probably be refactored to use just the point and labels renderer\
        ##        but we don't have a points with depth layer to test..
        if self.points is not None and renderer.point_and_line_set_renderer is None:
            if (self.line_segment_indexes is None):
                self.line_segment_indexes = self.make_line_segment_indexes(0)

            renderer.rebuild_point_and_line_set_renderer(self, create=True)

        renderer.set_up_labels(self)
    
    def rebuild_renderer(self, renderer, in_place=False):
        renderer.rebuild_point_and_line_set_renderer(self, in_place=in_place)

    def render_projected(self, renderer, w_r, p_r, s_r, layer_visibility, layer_index_base, pick_mode=False):
        log.log(5, "Rendering point layer!!! visible=%s, pick=%s" % (layer_visibility["layer"], pick_mode))
        if (not layer_visibility["layer"]):
            return

        # the points and line segments
        if (renderer.point_and_line_set_renderer is not None):
            renderer.point_and_line_set_renderer.render(layer_index_base + renderer.POINTS_AND_LINES_SUB_LAYER_PICKER_OFFSET,
                                                    pick_mode,
                                                    self.point_size,
                                                    self.line_width,
                                                    layer_visibility["points"],
                                                    layer_visibility["lines"],
                                                    layer_visibility["triangles"],
                                                    self.triangle_line_width,
                                                    self.get_selected_point_indexes(),
                                                    self.get_selected_point_indexes(STATE_FLAGGED),
                                                    self.get_selected_line_segment_indexes(),
                                                    self.get_selected_line_segment_indexes(STATE_FLAGGED))

            # the labels
            if (renderer.label_set_renderer is not None and layer_visibility["labels"] and renderer.point_and_line_set_renderer.vbo_point_xys is not None):
                renderer.label_set_renderer.render(-1, pick_mode, s_r,
                                               renderer.MAX_LABEL_CHARACTERS, self.points.z,
                                               renderer.point_and_line_set_renderer.vbo_point_xys.data,
                                               p_r, renderer.canvas.projected_units_per_pixel)

        # render selections after everything else
        if (renderer.point_and_line_set_renderer is not None and not pick_mode):
            if layer_visibility["lines"]:
                renderer.point_and_line_set_renderer.render_selected_line_segments(self.line_width, self.get_selected_line_segment_indexes())

            if layer_visibility["points"]:
                renderer.point_and_line_set_renderer.render_selected_points(self.point_size,
                                                                        self.get_selected_point_indexes())


