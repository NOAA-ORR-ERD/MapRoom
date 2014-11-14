import os
import os.path
import time
import sys
import numpy as np

# Enthought library imports.
from traits.api import Int, Unicode, Any, Str, Float, Enum, Property

from ..library import rect
from ..library.Boundary import Boundaries, PointsError
from ..renderer import color_to_int, data_types
from ..layer_undo import *
from ..command import UndoInfo
from ..mouse_commands import MovePointsCommand

from base import Layer, ProjectedLayer
from constants import *

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


class PointLayer(ProjectedLayer):
    """Layer for points/lines/polygons.
    
    """
    name = Unicode("Point Layer")
    
    type = Str("point")
    
    mouse_mode_toolbar = Str("VectorLayerToolBar")
    
    points = Any
    
    merged_points_index = Int(0)
    
    default_depth = Float(1.0)
    
    depth_unit = Property(Str)
    
    _depth_unit = Enum("unknown", "meters", "feet", "fathoms")
    
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


    def __str__(self):
        try:
            points = len(self.points)
        except:
            points = 0
        return "%s layer '%s': %d points" % (self.type, self.name, points)

    def new(self):
        Layer.new(self)
        self.new_points()
    
    def new_points(self, num=0):
        self.determine_layer_color()
        self.points = self.make_points(num)

    def empty(self):
        """
        We shouldn't allow saving of a layer with no content, so we use this method
        to determine if we can save this layer.
        """
        no_points = (self.points is None or len(self.points) == 0)

        return no_points
    
    def highlight_exception(self, e):
        if hasattr(e, "points") and e.points != None:
            self.clear_all_selections(STATE_FLAGGED)
            for p in e.points:
                self.select_point(p, STATE_FLAGGED)
            self.manager.dispatch_event('refresh_needed')
    
    def clear_flagged(self, refresh=False):
        self.clear_all_selections(STATE_FLAGGED)
        if refresh:
            self.manager.dispatch_event('refresh_needed')
    
    def get_visibility_items(self):
        """Return allowable keys for visibility dict lookups for this layer
        """
        return ["points", "lines", "labels"]
    
    def visibility_item_exists(self, label):
        """Return keys for visibility dict lookups that currently exist in this layer
        """
        if label in ["points", "labels"]:
            return self.points is not None
        raise RuntimeError("Unknown label %s for %s" % (label, self.name))

    def set_data(self, f_points, f_depths, f_line_segment_indexes):
        n = np.alen(f_points)
        self.determine_layer_color()
        self.points = self.make_points(n)
        if (n > 0):
            self.points.view(data_types.POINT_XY_VIEW_DTYPE).xy[
                0: n
            ] = f_points
            self.points.z[
                0: n
            ] = f_depths
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
    
    def update_bounds(self):
        self.bounds = self.compute_bounding_rect()

    def make_points(self, count):
        return np.repeat(
            np.array([(np.nan, np.nan, np.nan, 0, 0)], dtype=data_types.POINT_DTYPE),
            count,
        ).view(np.recarray)

    def determine_layer_color(self):
        if not self.color:
            self.color = DEFAULT_COLORS[
                Layer.next_default_color_index
            ]

            Layer.next_default_color_index = (
                Layer.next_default_color_index + 1
            ) % len(DEFAULT_COLORS)

    def compute_bounding_rect(self, mark_type=STATE_NONE):
        bounds = rect.NONE_RECT

        if (self.points != None and len(self.points) > 0):
            if (mark_type == STATE_NONE):
                points = self.points
            else:
                points = self.points[self.get_selected_point_indexes(mark_type)]
            l = points.x.min()
            r = points.x.max()
            b = points.y.min()
            t = points.y.max()
            bounds = rect.accumulate_rect(bounds, ((l, b), (r, t)))

        return bounds
    
    def compute_selected_bounding_rect(self):
        bounds = self.compute_bounding_rect(STATE_SELECTED)
        return bounds

    def clear_all_selections(self, mark_type=STATE_SELECTED):
        self.clear_all_point_selections(mark_type)
        self.increment_change_count()

    def clear_all_point_selections(self, mark_type=STATE_SELECTED):
        if (self.points != None):
            self.points.state = self.points.state & (0xFFFFFFFF ^ mark_type)
            self.increment_change_count()

    def has_points(self):
        return True

    def select_point(self, point_index, mark_type=STATE_SELECTED):
        self.points.state[point_index] = self.points.state[point_index] | mark_type
        self.increment_change_count()

    def deselect_point(self, point_index, mark_type=STATE_SELECTED):
        self.points.state[point_index] = self.points.state[point_index] & (0xFFFFFFFF ^ mark_type)
        self.increment_change_count()

    def is_point_selected(self, point_index, mark_type=STATE_SELECTED):
        return self.points != None and (self.points.state[point_index] & mark_type) != 0

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
        if (self.points == None):
            return []
        #
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
    
    def get_state(self, index):
        return self.points.state[index]

    def offset_selected_objects(self, world_d_x, world_d_y):
        self.offset_selected_points(world_d_x, world_d_y)

    def offset_selected_points(self, world_d_x, world_d_y):
        if (self.points != None):
            # offset our own copy of the points (which automatically updates our own line segments)
            s_p_i_s = self.get_selected_point_indexes()
            for point_index in s_p_i_s:
                self.offset_point(point_index, world_d_x, world_d_y, True)
            # self.offset_points( s_p_i_s, world_d_x, world_d_y, True )
            
            # Rebuilding the renderer by using the event layer_contents_changed
            # is super slow for large data sets.  This is a different event
            # that updates the point positions given the requirement that no
            # points have been added or removed
            self.manager.dispatch_event('layer_contents_changed_in_place', self)
            self.increment_change_count()

    def offset_point(self, point_index, world_d_x, world_d_y, add_undo_info=False):
        self.points.x[point_index] += world_d_x
        self.points.y[point_index] += world_d_y
        """
        # we don't set the undo information here because this function is called repeatedly as the mouse moves
        if ( add_undo_info ):
            params = ( world_d_x, world_d_y )
            self.manager.add_undo_operation_to_operation_batch( OP_MOVE_POINT, self, point_index, params )
        """
    
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
        if (self.points == None):
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

    def find_points_connected_to_point(self, point_index):
        return []
    
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
        if self.points != None and renderer.point_and_line_set_renderer == None:
            if (self.line_segment_indexes == None):
                self.line_segment_indexes = self.make_line_segment_indexes(0)

            renderer.rebuild_point_and_line_set_renderer(self, create=True)

        renderer.set_up_labels(self)

    def render_projected(self, renderer, w_r, p_r, s_r, layer_visibility, layer_index_base, pick_mode=False):
        log.log(5, "Rendering point layer!!! visible=%s, pick=%s" % (layer_visibility["layer"], pick_mode))
        if (not layer_visibility["layer"]):
            return

        # the points and line segments
        if (renderer.point_and_line_set_renderer != None):
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
            if (renderer.label_set_renderer != None and layer_visibility["labels"] and renderer.point_and_line_set_renderer.vbo_point_xys != None):
                renderer.label_set_renderer.render(-1, pick_mode, s_r,
                                               renderer.MAX_LABEL_CHARACTERS, self.points.z,
                                               renderer.point_and_line_set_renderer.vbo_point_xys.data,
                                               p_r, renderer.canvas.projected_units_per_pixel)

        # render selections after everything else
        if (renderer.point_and_line_set_renderer != None and not pick_mode):
            if layer_visibility["lines"]:
                renderer.point_and_line_set_renderer.render_selected_line_segments(self.line_width, self.get_selected_line_segment_indexes())

            if layer_visibility["points"]:
                renderer.point_and_line_set_renderer.render_selected_points(self.point_size,
                                                                        self.get_selected_point_indexes())


