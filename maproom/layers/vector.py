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
from ..library.scipy_ckdtree import cKDTree
from ..library.accumulator import flatten
from ..library.Projection import Projection
from ..library.Boundary import Boundaries, PointsError
from ..renderer import color_to_int, data_types
from ..layer_undo import *

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
    
    mouse_selection_mode = Str("VectorLayer")
    
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

    def insert_point(self, world_point):
        if self.points is None:
            index = -1
        else:
            index = len(self.points)
        return self.insert_point_at_index(index, world_point, self.default_depth, self.color, STATE_SELECTED, True)

    def insert_point_at_index(self, point_index, world_point, z, color, state, add_undo_info):
        t0 = time.clock()
        # insert it into the layer
        p = np.array([(world_point[0], world_point[1], z, color, state)],
                     dtype=data_types.POINT_DTYPE)
        if (self.points == None):
            self.new_points(1)
            self.points[0] = p
            point_index = 0
            self.manager.dispatch_event('refresh_needed')
            self.manager.project.layer_tree_control.select_layer(self)
        else:
            self.points = np.insert(self.points, point_index, p).view(np.recarray)
        t = time.clock() - t0  # t is wall seconds elapsed (floating point)
        # print "inserted new point in {0} seconds".format( t )

        # update point indexes in the line segements to account for the inserted point
        self.update_after_insert_point_at_index(point_index)

        if (add_undo_info):
            params = ((self.points.x[point_index], self.points.y[point_index]), self.points.z[point_index], self.points.color[point_index], self.points.state[point_index])
            self.manager.add_undo_operation_to_operation_batch(OP_ADD_POINT, self, point_index, params)

        # insert it into the point_and_line_set_renderer (by simply rebuilding it)
        # we don't update the point_and_line_set_renderer if not adding undo info, because that means we are undoing or redoing
        # and the point_and_line_set_renderer for all affected layers will get rebuilt at the end of the process
        if (add_undo_info):
            self.manager.dispatch_event('layer_contents_changed', self)

        """
        t0 = time.clock()
        # insert it into the label_set_renderer
        if ( self.label_set_renderer != None ):
            self.label_set_renderer.insert_point( len( self.points ) - 1,
                                                  str( self.default_depth ),
                                                  world_point,
                                                  self.manager.project.control.projection,
                                                  self.manager.project.control.projection_is_identity )
        t = time.clock() - t0 # t is wall seconds elapsed (floating point)
        print "inserted into label set renderer in {0} seconds".format( t )
        """

        return point_index

    def update_after_insert_point_at_index(self, point_index):
        """Hook for subclasses to update dependent items when a point is inserted.
        """
        pass

    def delete_point(self, point_index, add_undo_info):
        if (self.find_points_connected_to_point(point_index) != []):
            raise Exception()

        if (add_undo_info):
            params = ((self.points.x[point_index], self.points.y[point_index]), self.points.z[point_index], self.points.color[point_index], self.points.state[point_index])
            self.manager.add_undo_operation_to_operation_batch(OP_DELETE_POINT, self, point_index, params)
        self.points = np.delete(self.points, point_index, 0)

        # update point indexes in the line segements to account for the deleted point
        self.update_after_delete_point(point_index)

        self.manager.dispatch_event('layer_contents_changed', self)

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


class LineLayer(PointLayer):
    """Layer for points/lines/polygons.
    
    """
    name = Unicode("Ugrid Layer")
    
    type = Str("line")
    
    line_segment_indexes = Any

    def __str__(self):
        try:
            lines = len(self.line_segment_indexes)
        except:
            lines = 0
        return PointLayer.__str__(self) + ", %d lines" % lines
    
    def get_visibility_items(self):
        """Return allowable keys for visibility dict lookups for this layer
        """
        return ["points", "lines", "labels"]
    
    def visibility_item_exists(self, label):
        """Return keys for visibility dict lookups that currently exist in this layer
        """
        if label in ["points", "labels"]:
            return self.points is not None
        if label == "lines":
            return self.line_segment_indexes is not None
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

    def can_save_as(self):
        return True
    
    def serialize_json(self, index):
        json = PointLayer.serialize_json(self, index)
        update = {
            'lines': self.line_segment_indexes.tolist(),
        }
        json.update(update)
        return json
    
    def unserialize_json_version1(self, json_data):
        PointLayer.unserialize_json_version1(self, json_data)
        self.line_segment_indexes = np.array([tuple(i) for i in json_data['lines']], data_types.LINE_SEGMENT_DTYPE).view(np.recarray)
    
    def check_for_problems(self, window):
        # determine the boundaries in the parent layer
        boundaries = Boundaries(self, allow_branches=False, allow_self_crossing=False)
        boundaries.check_errors(True)

    def make_line_segment_indexes(self, count):
        return np.repeat(
            np.array([(0, 0, 0, 0)], dtype=data_types.LINE_SEGMENT_DTYPE),
            count,
        ).view(np.recarray)

    def clear_all_selections(self, mark_type=STATE_SELECTED):
        self.clear_all_point_selections(mark_type)
        self.clear_all_line_segment_selections(mark_type)
        self.increment_change_count()

    def clear_all_line_segment_selections(self, mark_type=STATE_SELECTED):
        if (self.line_segment_indexes != None):
            self.line_segment_indexes.state = self.line_segment_indexes.state & (0xFFFFFFFF ^ mark_type)
            self.increment_change_count()

    def select_line_segment(self, line_segment_index, mark_type=STATE_SELECTED):
        self.line_segment_indexes.state[line_segment_index] = self.line_segment_indexes.state[line_segment_index] | mark_type
        self.increment_change_count()

    def deselect_line_segment(self, line_segment_index, mark_type=STATE_SELECTED):
        self.line_segment_indexes.state[line_segment_index] = self.line_segment_indexes.state[line_segment_index] & (0xFFFFFFFF ^ mark_type)
        self.increment_change_count()

    def is_line_segment_selected(self, line_segment_index, mark_type=STATE_SELECTED):
        return self.line_segment_indexes != None and (self.line_segment_indexes.state[line_segment_index] & mark_type) != 0

    def select_line_segments_in_rect(self, is_toggle_mode, is_add_mode, w_r, mark_type=STATE_SELECTED):
        if (not is_toggle_mode and not is_add_mode):
            self.clear_all_line_segment_selections()
        point_indexes = np.where(np.logical_and(
            np.logical_and(self.points.x >= w_r[0][0], self.points.x <= w_r[1][0]),
            np.logical_and(self.points.y >= w_r[0][1], self.points.y <= w_r[1][1])))[0]
        indexes = np.where(np.logical_or(
            np.in1d(self.line_segment_indexes.point1, point_indexes),
            np.in1d(self.line_segment_indexes.point2, point_indexes)))
        if (is_add_mode):
            self.line_segment_indexes.state[indexes] |= mark_type
        if (is_toggle_mode):
            self.line_segment_indexes.state[indexes] ^= mark_type
        self.increment_change_count()

    def get_selected_and_dependent_point_indexes(self, mark_type=STATE_SELECTED):
        indexes = np.arange(0)
        if (self.points != None):
            indexes = np.append(indexes, self.get_selected_point_indexes(mark_type))
        if (self.line_segment_indexes != None):
            l_s_i_s = self.get_selected_line_segment_indexes(mark_type)
            indexes = np.append(indexes, self.line_segment_indexes[l_s_i_s].point1)
            indexes = np.append(indexes, self.line_segment_indexes[l_s_i_s].point2)
        #
        return np.unique(indexes)

    def get_num_points_selected(self, mark_type=STATE_SELECTED):
        return len(self.get_selected_and_dependent_point_indexes(mark_type))

    def get_selected_line_segment_indexes(self, mark_type=STATE_SELECTED):
        if (self.line_segment_indexes == None):
            return []
        #
        return np.where((self.line_segment_indexes.state & mark_type) != 0)[0]

    def get_all_line_point_indexes(self):
        indexes = np.arange(0)
        if (self.line_segment_indexes != None):
            indexes = np.append(indexes, self.line_segment_indexes.point1)
            indexes = np.append(indexes, self.line_segment_indexes.point2)
        #
        return np.unique(indexes)

    def find_points_on_shortest_path_from_point_to_selected_point(self, point_index):
        return self.follow_point_paths_to_selected_point([[point_index]])

    def follow_point_paths_to_selected_point(self, list_of_paths):
        while True:
            if (list_of_paths == []):
                return []

            new_paths = []
            for path in list_of_paths:
                # consider the last point in the path
                # find all other points connected to this point that are not already in path
                # if one such point is selected, we found the path
                # otherwise, add the point to the path to be followed further
                p = path[len(path) - 1]
                connections = self.find_points_connected_to_point(p)
                for q in connections:
                    if (not q in path):
                        extended = []
                        extended.extend(path)
                        extended.append(q)
                        if (self.is_point_selected(q)):
                            return extended
                        else:
                            new_paths.append(extended)

            list_of_paths = new_paths

    def find_points_connected_to_point(self, point_index):
        if (self.line_segment_indexes == None):
            return []

        result = []
        indexes = self.line_segment_indexes[self.line_segment_indexes.point1 == point_index]
        result.extend(indexes.point2)
        indexes = self.line_segment_indexes[self.line_segment_indexes.point2 == point_index]
        result.extend(indexes.point1)

        return list(set(result))

    def are_points_connected(self, point_index_1, point_index_2):
        return point_index_2 in self.find_points_connected_to_point(point_index_1)

    def find_lines_on_shortest_path_from_line_to_selected_line(self, line_segment_index):
        return self.follow_line_paths_to_selected_line([[line_segment_index]])

    def follow_line_paths_to_selected_line(self, list_of_paths):
        while True:
            if (list_of_paths == []):
                return []

            new_paths = []
            for path in list_of_paths:
                # consider the last line segment in the path
                # find all other line segments connected to this line segment that are not already in path
                # if one such line segment is selected, we found the path
                # otherwise, add the line segment to the path to be followed further
                i = path[len(path) - 1]
                connections = self.find_lines_connected_to_line(i)
                for j in connections:
                    if (not j in path):
                        extended = []
                        extended.extend(path)
                        extended.append(j)
                        if (self.is_line_segment_selected(j)):
                            return extended
                        else:
                            new_paths.append(extended)

            list_of_paths = new_paths

    def find_lines_connected_to_line(self, line_segment_index):
        if (self.line_segment_indexes == None):
            return []

        p1 = self.line_segment_indexes.point1[line_segment_index]
        p2 = self.line_segment_indexes.point2[line_segment_index]
        result = np.arange(0)
        result = np.append(result, np.where(self.line_segment_indexes.point1 == p1))
        result = np.append(result, np.where(self.line_segment_indexes.point1 == p2))
        result = np.append(result, np.where(self.line_segment_indexes.point2 == p1))
        result = np.append(result, np.where(self.line_segment_indexes.point2 == p2))

        s = set(result)
        s.remove(line_segment_index)

        return list(s)

    def offset_selected_points(self, world_d_x, world_d_y):
        if (self.points != None):
            # offset our own copy of the points (which automatically updates our own line segments)
            s_p_i_s = self.get_selected_and_dependent_point_indexes()
            for point_index in s_p_i_s:
                self.offset_point(point_index, world_d_x, world_d_y, True)
            # self.offset_points( s_p_i_s, world_d_x, world_d_y, True )
            self.manager.dispatch_event('layer_contents_changed', self)
            self.increment_change_count()

    def delete_all_selected_objects(self):
        point_indexes = self.get_selected_point_indexes()
        l_s_i_s = None
        if (self.get_selected_line_segment_indexes != None):
            l_s_i_s = self.get_selected_line_segment_indexes()
        if ((point_indexes != None and len(point_indexes)) > 0 or (l_s_i_s != None and len(l_s_i_s) > 0)):
            self.delete_points_and_lines(point_indexes, l_s_i_s, True)
        self.increment_change_count()

    def delete_points_and_lines(self, point_indexes, l_s_i_s, add_undo_info):
        line_segment_indexes_to_be_deleted = None
        if (self.line_segment_indexes != None):
            # (1) delete any lines whose points are going away
            line_segment_indexes_to_be_deleted = np.where(np.in1d(self.line_segment_indexes.point1, point_indexes))
            line_segment_indexes_to_be_deleted = np.append(line_segment_indexes_to_be_deleted, np.where(np.in1d(self.line_segment_indexes.point2, point_indexes)))
            line_segment_indexes_to_be_deleted = np.unique(line_segment_indexes_to_be_deleted)
            # (2) add in the line segments that are being deleted explicitly
            if (l_s_i_s != None):
                line_segment_indexes_to_be_deleted = np.unique(np.append(line_segment_indexes_to_be_deleted, l_s_i_s))

            if (add_undo_info):
                # add everything to the undo stack in an order such that if it was undone from last to first it would all work
                l = list(line_segment_indexes_to_be_deleted)
                l.reverse()
                for i in l:
                    params = (self.line_segment_indexes.point1[i], self.line_segment_indexes.point2[i], self.line_segment_indexes.color[i], self.line_segment_indexes.state[i])
                    self.manager.add_undo_operation_to_operation_batch(OP_DELETE_LINE, self, i, params)

            # adjust the point indexes of the remaining line segments
            offsets = np.zeros(np.alen(self.line_segment_indexes))
            for index in point_indexes:
                offsets += np.where(self.line_segment_indexes.point1 > index, 1, 0)
            self.line_segment_indexes.point1 -= offsets
            offsets[: np.alen(offsets)] = 0
            for index in point_indexes:
                offsets += np.where(self.line_segment_indexes.point2 > index, 1, 0)
            self.line_segment_indexes.point2 -= offsets

        if (add_undo_info):
            # add everything to the undo stack in an order such that if it was undone from last to first it would all work
            l = list(point_indexes)
            l.reverse()
            for i in l:
                params = ((self.points.x[i], self.points.y[i]), self.points.z[i], self.points.color[i], self.points.state[i])
                self.manager.add_undo_operation_to_operation_batch(OP_DELETE_POINT, self, i, params)

        # delete them from the layer
        self.points = np.delete(self.points, point_indexes, 0)
        if (line_segment_indexes_to_be_deleted != None):
            # then delete the line segments
            self.line_segment_indexes = np.delete(self.line_segment_indexes, line_segment_indexes_to_be_deleted, 0)

        # delete them from the point_and_line_set_renderer (by simply rebuilding it)

        """
        # delete them from the label_set_renderer
        if ( self.label_set_renderer != None ):
            self.label_set_renderer.delete_points( point_indexes )
            self.label_set_renderer.reproject( self.points.view( data_types.POINT_XY_VIEW_DTYPE ).xy,
                                               self.manager.project.control.projection,
                                               self.manager.project.control.projection_is_identity )
        """

        # when points are deleted from a layer the indexes of the points in the existing merge dialog box
        # become invalid; so force the user to re-find duplicates in order to create a valid list again
        self.manager.dispatch_event('layer_contents_deleted', self)

    def update_after_insert_point_at_index(self, point_index):
        # update point indexes in the line segements to account for the inserted point
        if (self.line_segment_indexes != None):
            offsets = np.zeros(np.alen(self.line_segment_indexes))
            offsets += np.where(self.line_segment_indexes.point1 >= point_index, 1, 0)
            self.line_segment_indexes.point1 += offsets
            offsets[: np.alen(offsets)] = 0
            offsets += np.where(self.line_segment_indexes.point2 >= point_index, 1, 0)
            self.line_segment_indexes.point2 += offsets
    
    def insert_point_in_line(self, world_point, line_segment_index):
        new_point_index = self.insert_point(world_point)
        point_index_1 = self.line_segment_indexes.point1[line_segment_index]
        point_index_2 = self.line_segment_indexes.point2[line_segment_index]
        color = self.line_segment_indexes.color[line_segment_index]
        state = self.line_segment_indexes.state[line_segment_index]
        depth = (self.points.z[point_index_1] + self.points.z[point_index_2])/2
        self.points.z[new_point_index] = depth
        self.delete_line_segment(line_segment_index, True)
        self.insert_line_segment_at_index(len(self.line_segment_indexes), point_index_1, new_point_index, color, state, True)
        self.insert_line_segment_at_index(len(self.line_segment_indexes), new_point_index, point_index_2, color, state, True)

        return new_point_index

    """
    def connect_points_to_point( self, point_indexes, point_index ):
        connected_points = self.find_points_connected_to_point( point_index )
        num_connections_made = 0
        for p_i in point_indexes:
            if ( p_i != point_index and not ( p_i in connected_points ) ):
                l_s = np.array( [ ( p_i, point_index, DEFAULT_LINE_SEGMENT_COLOR, STATE_NONE ) ],
                                dtype = data_types.LINE_SEGMENT_DTYPE ).view( np.recarray )
                self.line_segment_indexes = np.append( self.line_segment_indexes, l_s ).view( np.recarray )
                num_connections_made += 1
        if ( num_connections_made > 0 ):
            self.rebuild_point_and_line_set_renderer()
    """

    def insert_line_segment(self, point_index_1, point_index_2):
        return self.insert_line_segment_at_index(len(self.line_segment_indexes), point_index_1, point_index_2, self.color, STATE_NONE, True)

    def insert_line_segment_at_index(self, l_s_i, point_index_1, point_index_2, color, state, add_undo_info):
        l_s = np.array([(point_index_1, point_index_2, color, state)],
                       dtype=data_types.LINE_SEGMENT_DTYPE).view(np.recarray)
        self.line_segment_indexes = np.insert(self.line_segment_indexes, l_s_i, l_s).view(np.recarray)

        if (add_undo_info):
            params = (self.line_segment_indexes.point1[l_s_i], self.line_segment_indexes.point2[l_s_i], self.line_segment_indexes.color[l_s_i], self.line_segment_indexes.state[l_s_i])
            self.manager.add_undo_operation_to_operation_batch(OP_ADD_LINE, self, l_s_i, params)

        # we don't update the point_and_line_set_renderer if not adding undo info, because that means we are undoing or redoing
        # and the point_and_line_set_renderer for all affected layers will get rebuilt at the end of the process
        if (add_undo_info):
            self.manager.dispatch_event('layer_contents_changed', self)

        return l_s_i

    def update_after_delete_point(self, point_index):
        if (self.line_segment_indexes != None):
            offsets = np.zeros(np.alen(self.line_segment_indexes))
            offsets += np.where(self.line_segment_indexes.point1 > point_index, 1, 0)
            self.line_segment_indexes.point1 -= offsets
            offsets[: np.alen(offsets)] = 0
            offsets += np.where(self.line_segment_indexes.point2 > point_index, 1, 0)
            self.line_segment_indexes.point2 -= offsets

    def delete_line_segment(self, l_s_i, add_undo_info):
        if (add_undo_info):
            params = (self.line_segment_indexes.point1[l_s_i], self.line_segment_indexes.point2[l_s_i], self.line_segment_indexes.color[l_s_i], self.line_segment_indexes.state[l_s_i])
            self.manager.add_undo_operation_to_operation_batch(OP_DELETE_LINE, self, l_s_i, params)
        self.line_segment_indexes = np.delete(self.line_segment_indexes, l_s_i, 0)

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

        n = len(layer_a.line_segment_indexes) + len(layer_b.line_segment_indexes)
        self.line_segment_indexes = self.make_line_segment_indexes(n)
        self.line_segment_indexes[
            0: len(layer_a.line_segment_indexes)
        ] = layer_a.line_segment_indexes.copy()
        l_s_i_s = layer_b.line_segment_indexes.copy()
        # offset line segment point indexes to account for their new position in the merged array
        l_s_i_s.point1 += len(layer_a.points)
        l_s_i_s.point2 += len(layer_a.points)
        self.line_segment_indexes[
            len(layer_a.line_segment_indexes): n
        ] = l_s_i_s
        # self.line_segment_indexes.state = 0

    # returns a list of pairs of point indexes
    def find_duplicates(self, distance_tolerance_degrees, depth_tolerance_percentage=-1):
        if (self.points == None or len(self.points) < 2):
            return []

        points = self.points.view(
            data_types.POINT_XY_VIEW_DTYPE
        ).xy[:].copy()

        # cKDTree doesn't handle NaNs gracefully, but it does handle infinity
        # values. So replace all the NaNs with infinity.
        points = points.view(np.float64)
        points[np.isnan(points)] = np.inf

        tree = cKDTree(points)

        (_, indices_list) = tree.query(
            points,
            2,  # number of points to return per point given.
            distance_upper_bound=distance_tolerance_degrees
        )

        duplicates = set()

        for (n, indices) in enumerate(indices_list):
            # cKDTree uses the point count (the number of points in the input list) to indicate a missing neighbor, so
            # filter out those values from the results.
            indices = [
                index for index in sorted(indices)
                if index != len(self.points)
            ]

            if len(indices) < 2:
                continue

            # If this layer was merged from two layers, and if
            # all point indices in the current list are from the same source
            # layer, then skip this list of duplicates.
            if self.merged_points_index > 0:
                unique_sources = set()
                for index in indices:
                    if (index < self.merged_points_index):
                        unique_sources.add(0)
                    else:
                        unique_sources.add(1)

                if len(unique_sources) < 2:
                    continue

            # Filter out points not within the depth tolerance from one another.
            depth_0 = self.points.z[indices[0]]
            depth_1 = self.points.z[indices[1]]
            smaller_depth = min(abs(depth_0), abs(depth_1)) or 1.0

            depth_difference = abs((depth_0 - depth_1) / smaller_depth) * 100.0

            if (depth_tolerance_percentage > -1 and depth_difference > depth_tolerance_percentage):
                continue

            duplicates.add(tuple(indices))

            """
            if n % 100 == 0:
                scheduler.switch()
            """

        return list(duplicates)

    def merge_duplicates(self, indexes, points_in_lines):
        points_to_delete = set()

        for sublist in indexes:
            (point_0, point_1) = sublist
            point_0_in_line = point_0 in points_in_lines
            point_1_in_line = point_1 in points_in_lines

            # If each point in the pair is within a line, then skip it since
            # we don't know how to merge such points.
            if (point_0_in_line and point_1_in_line):
                continue

            # If only one of the points is within a line, then delete the
            # other point in the pair.
            if (point_0_in_line):
                points_to_delete.add(point_1)
            elif (point_1_in_line):
                points_to_delete.add(point_0)
            # Otherwise, arbitrarily delete one of the points
            else:
                points_to_delete.add(point_1)

        if (len(points_to_delete) > 0):
            self.delete_points_and_lines(list(points_to_delete), None, True)

        self.manager.dispatch_event('refresh_needed')
    
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
        log.log(5, "Rendering line layer!!! visible=%s, pick=%s" % (layer_visibility["layer"], pick_mode))
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
                                                    False,
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


class TriangleLayer(PointLayer):
    """Layer for triangles.
    
    """
    type = Str("triangle")
    
    triangles = Any

    def __str__(self):
        try:
            triangles = len(self.triangles)
        except:
            triangles = 0
        return PointLayer.__str__(self) + ", %d triangles" % triangles

    def empty(self):
        """
        We shouldn't allow saving of a layer with no content, so we use this method
        to determine if we can save this layer.
        """
        no_points = (self.points is None or len(self.points) == 0)
        no_triangles = (self.triangles is None or len(self.triangles) == 0)

        return no_points and no_triangles
    
    def get_visibility_items(self):
        """Return allowable keys for visibility dict lookups for this layer
        """
        return ["points", "triangles", "labels"]
    
    def visibility_item_exists(self, label):
        """Return keys for visibility dict lookups that currently exist in this layer
        """
        if label in ["points", "labels"]:
            return self.points is not None
        if label == "triangles":
            return self.triangles is not None
        raise RuntimeError("Unknown label %s for %s" % (label, self.name))

    def set_data(self, f_points, f_depths, f_triangles):
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

            n = len(f_triangles)
            if n > 0:
                self.triangles = self.make_triangles(n)
                self.triangles.view(data_types.TRIANGLE_POINTS_VIEW_DTYPE).point_indexes = f_triangles
        
        self.update_bounds()

    def can_save_as(self):
        return True

    def serialize_json(self, index):
        json = PointLayer.serialize_json(self, index)
        update = {
            'triangles': self.triangles.tolist(),
        }
        json.update(update)
        return json
    
    def unserialize_json_version1(self, json_data):
        PointLayer.unserialize_json_version1(self, json_data)
        self.triangles = np.array([tuple(i) for i in json_data['triangles']], data_types.TRIANGLE_DTYPE).view(np.recarray)

    def update_after_insert_point_at_index(self, point_index):
        # update point indexes in the triangles to account for the inserted point
        if (self.triangles != None):
            offsets = np.zeros(np.alen(self.triangles))
            offsets += np.where(self.triangles.point1 >= point_index, 1, 0)
            self.triangles.point1 += offsets
            offsets[: np.alen(offsets)] = 0
            offsets += np.where(self.triangles.point2 >= point_index, 1, 0)
            self.triangles.point2 += offsets
            offsets[: np.alen(offsets)] = 0
            offsets += np.where(self.triangles.point3 >= point_index, 1, 0)
            self.triangles.point3 += offsets

    def insert_triangle(self, point_index_1, point_index_2, point_index_3):
        return self.insert_triangle_at_index(len(self.triangles), (point_index_1, point_index_2, point_index_3, self.color, STATE_NONE), True)

    def insert_triangle_at_index(self, index, params, add_undo_info):
        entry = np.array([params],
                       dtype=data_types.TRIANGLE_DTYPE).view(np.recarray)
        self.triangles = np.insert(self.triangles, index, entry).view(np.recarray)

        if (add_undo_info):
            self.manager.add_undo_operation_to_operation_batch(OP_ADD_LINE, self, index, params)

            # we don't update the point_and_line_set_renderer if not adding
            # undo info, because that means we are undoing or redoing and the
            # point_and_line_set_renderer for all affected layers will get
            # rebuilt at the end of the process
            self.manager.dispatch_event('layer_contents_changed', self)

        return index

    def delete_triangle(self, index, add_undo_info):
        if (add_undo_info):
            t = self.triangles[index]
            params = (t.point1, t.point2, t.color, t.state)
            self.manager.add_undo_operation_to_operation_batch(OP_DELETE_TRIANGLE, self, index, params)
        self.triangles = np.delete(self.triangles, index, 0)

    def delete_all_selected_objects(self):
        point_indexes = self.get_selected_point_indexes()
        if point_indexes != None and len(point_indexes) > 0:
            self.delete_points_and_triangles(point_indexes, True)
        self.increment_change_count()

    def update_after_delete_point(self, point_index):
        # update point indexes in the triangles to account for the inserted point
        if (self.triangles != None):
            offsets = np.zeros(np.alen(self.triangles))
            offsets += np.where(self.triangles.point1 >= point_index, 1, 0)
            self.triangles.point1 -= offsets
            offsets[: np.alen(offsets)] = 0
            offsets += np.where(self.triangles.point2 >= point_index, 1, 0)
            self.triangles.point2 -= offsets
            offsets[: np.alen(offsets)] = 0
            offsets += np.where(self.triangles.point3 >= point_index, 1, 0)
            self.triangles.point3 -= offsets

    def delete_points_and_triangles(self, point_indexes, add_undo_info):
        triangle_indexes_to_be_deleted = None
        if (self.triangles != None):
            # (1) delete any triangles whose points are going away
            triangle_indexes_to_be_deleted = np.where(np.in1d(self.triangles.point1, point_indexes))
            triangle_indexes_to_be_deleted = np.append(triangle_indexes_to_be_deleted, np.where(np.in1d(self.triangles.point2, point_indexes)))
            triangle_indexes_to_be_deleted = np.append(triangle_indexes_to_be_deleted, np.where(np.in1d(self.triangles.point3, point_indexes)))
            triangle_indexes_to_be_deleted = np.unique(triangle_indexes_to_be_deleted)

            if (add_undo_info):
                # add everything to the undo stack in an order such that if it was undone from last to first it would all work
                l = list(triangle_indexes_to_be_deleted)
                l.reverse()
                for i in l:
                    params = (self.triangles.point1[i], self.triangles.point2[i], self.triangles.point3[i], self.triangles.color[i], self.triangles.state[i])
                    self.manager.add_undo_operation_to_operation_batch(OP_DELETE_TRIANGLE, self, i, params)

            # adjust the point indexes of the remaining triangles
            offsets = np.zeros(np.alen(self.triangles))
            for index in point_indexes:
                offsets += np.where(self.triangles.point1 > index, 1, 0)
            self.triangles.point1 -= offsets
            offsets[: np.alen(offsets)] = 0
            for index in point_indexes:
                offsets += np.where(self.triangles.point2 > index, 1, 0)
            self.triangles.point2 -= offsets
            offsets[: np.alen(offsets)] = 0
            for index in point_indexes:
                offsets += np.where(self.triangles.point3 > index, 1, 0)
            self.triangles.point3 -= offsets

        if (add_undo_info):
            # add everything to the undo stack in an order such that if it was undone from last to first it would all work
            l = list(point_indexes)
            l.reverse()
            for i in l:
                params = ((self.points.x[i], self.points.y[i]), self.points.z[i], self.points.color[i], self.points.state[i])
                self.manager.add_undo_operation_to_operation_batch(OP_DELETE_POINT, self, i, params)

        # delete them from the layer
        self.points = np.delete(self.points, point_indexes, 0)
        if (triangle_indexes_to_be_deleted != None):
            # then delete the line segments
            self.triangles = np.delete(self.triangles, triangle_indexes_to_be_deleted, 0)

        # delete them from the point_and_line_set_renderer (by simply rebuilding it)

        """
        # delete them from the label_set_renderer
        if ( self.label_set_renderer != None ):
            self.label_set_renderer.delete_points( point_indexes )
            self.label_set_renderer.reproject( self.points.view( data_types.POINT_XY_VIEW_DTYPE ).xy,
                                               self.manager.project.control.projection,
                                               self.manager.project.control.projection_is_identity )
        """

        # when points are deleted from a layer the indexes of the points in the existing merge dialog box
        # become invalid; so force the user to re-find duplicates in order to create a valid list again
        self.manager.dispatch_event('layer_contents_deleted', self)

    def make_triangles(self, count):
        return np.repeat(
            np.array([(0, 0, 0, 0, 0)], dtype=data_types.TRIANGLE_DTYPE),
            count,
        ).view(np.recarray)

    def get_triangulated_points(self, layer, q, a):
        # determine the boundaries in the parent layer
        boundaries = Boundaries(layer, allow_branches=True, allow_self_crossing=False)
        boundaries.check_errors(True)
        
        progress_log.info("Triangulating...")

        # calculate a hole point for each boundary
        hole_points_xy = np.empty(
            (len(boundaries), 2), np.float64,
        )

        for (boundary_index, boundary) in enumerate(boundaries):
            print boundary_index, boundary

        for (boundary_index, boundary) in enumerate(boundaries):
            if (len(boundary.point_indexes) < 3):
                continue

            # the "hole" point for the outer boundary (first in the list) should be outside of it
            if boundary_index == 0:
                hole_points_xy[boundary_index] = boundary.generate_outside_hole_point()
            else:
                hole_points_xy[boundary_index] = boundary.generate_inside_hole_point()

        params = "V"
        if (q is not None):
            params = params + "q" + str(q)
        if (a is not None):
            params = params + "a" + str(a)

        # we need to use projected points for the triangulation
        projected_points = layer.points.view(data_types.POINT_XY_VIEW_DTYPE).xy[: len(layer.points)].view(np.float64).copy()
        if (self.manager.project.control.projection_is_identity):
            projected_points[:, 0] = layer.points.x[:]
            projected_points[:, 1] = layer.points.y[:]
        else:
            projected_points[:, 0], projected_points[:, 1] = self.manager.project.control.projection(layer.points.x, layer.points.y)
            hole_points_xy[:, 0], hole_points_xy[:, 1] = self.manager.project.control.projection(hole_points_xy[:, 0], hole_points_xy[:, 1])
#        print "params: " + params
#        print "hole points:"
#        print hole_points_xy
        (triangle_points_xy,
         triangle_points_z,
         triangle_line_segment_indexes,  # not needed
         triangles) = triangulate_simple(
            params,
            projected_points,
            layer.points.z[: len(layer.points)].copy(),
            layer.line_segment_indexes.view(data_types.LINE_SEGMENT_POINTS_VIEW_DTYPE).points[: len(layer.line_segment_indexes)].view(np.uint32).copy(),
            hole_points_xy)
        return (triangle_points_xy,
         triangle_points_z,
         triangles)

    def unproject_triangle_points(self, points):
        if (not self.manager.project.control.projection_is_identity):
            # import code; code.interact( local = locals() )
            points.x, points.y = self.manager.project.control.projection(points.x, points.y, inverse=True)
    
    def triangulate_from_data(self, points, depths, triangles):
        self.set_data(points, depths, triangles)
        self.unproject_triangle_points(self.points)
        self.manager.dispatch_event('layer_contents_changed', self)
        self.manager.dispatch_event('layer_metadata_changed', self)

    def triangulate_from_layer(self, parent_layer, q, a):
        points, depths, triangles = self.get_triangulated_points(parent_layer, q, a)
        self.triangulate_from_data(points, depths, triangles)
    
    def color_interp(self, z, colormap, alpha):
        c0 = colormap[0]
        if z < c0[0]:
            return color_to_int(c0[1]/255., c0[2]/255., c0[3]/255., alpha)
        for c in colormap[1:]:
            if z >= c0[0] and z <= c[0]:
                perc = (z - c0[0]) / float(c[0] - c0[0])
                return color_to_int((c0[1] + (c[1] - c0[1]) * perc)/255.,
                                    (c0[2] + (c[2] - c0[2]) * perc)/255.,
                                    (c0[3] + (c[3] - c0[3]) * perc)/255.,
                                    alpha)
            c0 = c
        return color_to_int(c[1]/255., c[2]/255., c[3]/255., alpha)
    
    def get_triangle_point_colors(self, alpha=.9):
        colors = np.zeros(len(self.points), dtype=np.uint32)
        if self.points != None:
            
            # Lots of points in the colormap doesn't help because the shading
            # is only applied linearly based on the depth of the endpoints.
            # So if there are colors at depths 10, 25, 50, and 100, but the
            # triangle has points of depth 0 and 100, the naive GL shader
            # skips the colors at 25 and 50.  If it were written as a GLSL
            # shader, maybe a complicated palette could be implemented.  But I
            # know nothing of GLSL at this moment.
            colormap = (
                (-10, 0xf0, 0xeb, 0xc3),
                (-0.01, 0xf0, 0xeb, 0xc3),
                (0, 0xd6, 0xea, 0xeb),
#                (10, 0x9b, 0xd3, 0xe0),
#                (20, 0x54, 0xc0, 0xdc),
#                (30, 0x00, 0xa0, 0xcc),
#                (40, 0x00, 0x6a, 0xa4),
#                (50, 0x1f, 0x48, 0x8a),
                (100, 0x00, 0x04, 0x69),
                )
                
            for i in range(len(colors)):
                d = self.points.z[i]
                colors[i] = self.color_interp(d, colormap, alpha)
        return colors

    def create_renderer(self, renderer):
        """Create the graphic renderer for this layer.
        
        There may be multiple views of this layer (e.g.  in different windows),
        so we can't just create the renderer as an attribute of this object.
        The storage parameter is attached to the view and independent of
        other views of this layer.
        
        """
        if self.points != None and renderer.point_and_line_set_renderer == None:

            renderer.rebuild_point_and_line_set_renderer(self, create=True)

        renderer.set_up_labels(self)

    def render_projected(self, renderer, w_r, p_r, s_r, layer_visibility, layer_index_base, pick_mode=False):
        log.log(5, "Rendering triangle layer!!! visible=%s, pick=%s" % (layer_visibility["layer"], pick_mode))
        if (not layer_visibility["layer"]):
            return

        # the points and line segments
        if (renderer.point_and_line_set_renderer != None):
            renderer.point_and_line_set_renderer.render(layer_index_base + renderer.POINTS_AND_LINES_SUB_LAYER_PICKER_OFFSET,
                                                    pick_mode,
                                                    self.point_size,
                                                    self.line_width,
                                                    layer_visibility["points"],
                                                    False,
                                                    layer_visibility["triangles"],
                                                    self.triangle_line_width,
                                                    self.get_selected_point_indexes(),
                                                    self.get_selected_point_indexes(STATE_FLAGGED),
                                                    [], [])

            # the labels
            if (renderer.label_set_renderer != None and layer_visibility["labels"] and renderer.point_and_line_set_renderer.vbo_point_xys != None):
                renderer.label_set_renderer.render(-1, pick_mode, s_r,
                                               renderer.MAX_LABEL_CHARACTERS, self.points.z,
                                               renderer.point_and_line_set_renderer.vbo_point_xys.data,
                                               p_r, renderer.canvas.projected_units_per_pixel)

        # render selections after everything else
        if (renderer.point_and_line_set_renderer != None and not pick_mode):
            if layer_visibility["points"]:
                renderer.point_and_line_set_renderer.render_selected_points(self.point_size,
                                                                        self.get_selected_point_indexes())


class PolygonLayer(PointLayer):
    """Layer for polygons.
    
    """
    type = Str("polygon")
    
    mouse_selection_mode = Str("PolygonLayer")
    
    polygons = Any
    
    polygon_adjacency_array = Any  # parallels the points array
    
    polygon_identifiers = Any

    def __str__(self):
        try:
            points = len(self.points)
        except:
            points = 0
        try:
            polygons = len(self.polygons)
        except:
            polygons = 0
        return "PolygonLayer %s: %d points, %d polygons" % (self.name, points, polygons)

    def empty(self):
        """
        We shouldn't allow saving of a layer with no content, so we use this method
        to determine if we can save this layer.
        """
        no_points = (self.points is None or len(self.points) == 0)
        no_polygons = (self.polygons is None or len(self.polygons) == 0)

        return no_points and no_polygons
    
    def get_visibility_items(self):
        """Return allowable keys for visibility dict lookups for this layer
        """
        return ["points", "polygons"]
    
    def visibility_item_exists(self, label):
        """Return keys for visibility dict lookups that currently exist in this layer
        """
        if label in ["points"]:
            return self.points is not None
        if label == "polygons":
            return self.polygons is not None
        raise RuntimeError("Unknown label %s for %s" % (label, self.name))
    
    def set_data(self, f_polygon_points, f_polygon_starts, f_polygon_counts,
                 f_polygon_identifiers):
        self.determine_layer_color()
        n_points = np.alen(f_polygon_points)
        self.points = self.make_points(n_points)
        if (n_points > 0):
            n_polygons = np.alen(f_polygon_starts)
            self.points.view(data_types.POINT_XY_VIEW_DTYPE).xy[
                0: n_points
            ] = f_polygon_points
            self.polygons = self.make_polygons(n_polygons)
            self.polygons.start[
                0: n_polygons
            ] = f_polygon_starts
            self.polygons.count[
                0: n_polygons
            ] = f_polygon_counts
            # TODO: for now we assume each polygon is its own group
            self.polygons.group = np.arange(n_polygons)
            self.polygon_adjacency_array = self.make_polygon_adjacency_array(n_points)
            
            # set up feature code to color map
            green = color_to_int(0.25, 0.5, 0, 0.75)
            blue = color_to_int(0.0, 0.0, 0.5, 0.75)
            gray = color_to_int(0.5, 0.5, 0.5, 0.75)
            color_array = np.array((0, green, blue, gray), dtype=np.uint32)
            
            total = 0
            for p in xrange(n_polygons):
                c = self.polygons.count[p]
                self.polygon_adjacency_array.polygon[total: total + c] = p
                self.polygon_adjacency_array.next[total: total + c] = np.arange(total + 1, total + c + 1)
                self.polygon_adjacency_array.next[total + c - 1] = total
                total += c
                self.polygons.color[p] = color_array[np.clip(f_polygon_identifiers[p]['feature_code'], 1, 3)]

            self.polygon_identifiers = list(f_polygon_identifiers)
            self.points.state = 0
        self.update_bounds()


    def can_save_as(self):
        return True

    def serialize_json(self, index):
        json = PointLayer.serialize_json(self, index)
        update = {
            'polygons': self.polygons.tolist(),
            'adjacency': self.polygon_adjacency_array.tolist(),
            'identifiers': self.polygon_identifiers,
        }
        json.update(update)
        return json
    
    def unserialize_json_version1(self, json_data):
        PointLayer.unserialize_json_version1(self, json_data)
        self.polygons = np.array([tuple(i) for i in json_data['polygons']], data_types.POLYGON_DTYPE).view(np.recarray)
        self.polygon_adjacency_array = np.array([tuple(i) for i in json_data['adjacency']], data_types.POLYGON_ADJACENCY_DTYPE).view(np.recarray)
        self.polygon_identifiers = json_data['identifiers']

    def check_for_problems(self, window):
        problems = []
        # record log messages from the shapely package
        templog = logging.getLogger("shapely.geos")
        buf = StringIO()
        handler = logging.StreamHandler(buf)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        templog.addHandler(handler)
        for n in range(np.alen(self.polygons)):
            poly = self.get_shapely_polygon(n)
            if not poly.is_valid:
                problems.append(poly)
                #print "\n".join(str(a) for a in list(poly.exterior.coords))
                try:
                    templog.warning("in polygon #%d (%d points in polygon)" % (n, len(poly.exterior.coords)))
                except:
                    templog.warning("in polygon #%d\n" % (n,))
        templog.removeHandler(handler)
        handler.flush()
        buf.flush()
        errors = buf.getvalue()
        raise PointsError(errors)
    
    def make_polygons(self, count):
        return np.repeat(
            np.array([(0, 0, 0, 0, 0)], dtype=data_types.POLYGON_DTYPE),
            count,
        ).view(np.recarray)

    def make_polygon_adjacency_array(self, count):
        return np.repeat(
            np.array([(0, 0)], dtype=data_types.POLYGON_ADJACENCY_DTYPE),
            count,
        ).view(np.recarray)

    def clear_all_polygon_selections(self, mark_type=STATE_SELECTED):
        if (self.polygons != None):
            self.polygons.state = self.polygons.state & (0xFFFFFFFF ^ mark_type)
            self.increment_change_count()

    def select_polygon(self, polygon_index, mark_type=STATE_SELECTED):
        self.polygons.state[polygon_index] = self.polygons.state[polygon_index] | mark_type
        self.increment_change_count()

    def deselect_polygon(self, polygon_index, mark_type=STATE_SELECTED):
        self.polygons.state[polygon_index] = self.polygons.state[polygon_index] & (0xFFFFFFFF ^ mark_type)
        self.increment_change_count()

    def is_polygon_selected(self, polygon_index, mark_type=STATE_SELECTED):
        return self.polygons != None and (self.polygons.state[polygon_index] & mark_type) != 0

    def get_selected_polygon_indexes(self, mark_type=STATE_SELECTED):
        if (self.polygons == None):
            return []
        #
        return np.where((self.polygons.state & mark_type) != 0)[0]

    def offset_selected_objects(self, world_d_x, world_d_y):
        self.offset_selected_points(world_d_x, world_d_y)
        self.offset_selected_polygons(world_d_x, world_d_y)

    def offset_selected_polygons(self, world_d_x, world_d_y):
        self.increment_change_count()
    
    def insert_line_segment(self, point_index_1, point_index_2):
        raise RuntimeError("Not implemented yet for polygon layer!")
    
    def can_crop(self):
        return True
    
    def get_polygon(self, index):
        start = self.polygons.start[index]
        count = self.polygons.count[index]
        boundary = self.points
        points = np.c_[boundary.x[start:start + count], boundary.y[start:start + count]]
        points = np.require(points, np.float64, ["C", "OWNDATA"])
        return points, self.polygon_identifiers[index]
    
    def iter_polygons(self):
        for n in range(np.alen(self.polygons)):
            poly = self.get_polygon(n)
            yield poly
    
    def get_shapely_polygon(self, index, debug=False):
        points, ident = self.get_polygon(index)
        if np.alen(points) > 2:
            poly = Polygon(points)
        else:
            poly = LineString(points)
        if debug:
            print "points tuples:", points
            print "numpy:", points.__array_interface__, points.shape, id(points), points.flags
            print "shapely polygon:", poly.bounds
        return poly
    
    def crop_rectangle(self, w_r, add_undo_info=True):
        print "Cropping to %s" % str(w_r)
        
        crop_rect = box(w_r[0][0], w_r[1][1], w_r[1][0], w_r[0][1])
        
        class AccumulatePolygons(object):
            """Helper class to store results from stepping through the clipped
            polygons
            """
            def __init__(self):
                self.p_points = None
                self.p_starts = []
                self.p_counts = []
                self.p_identifiers = []
                self.total_points = 0
            
            def add_polygon(self, cropped_poly, ident):
                points = np.require(cropped_poly.exterior.coords.xy, np.float64, ["C", "OWNDATA"])
                num_points = points.shape[1]
                if self.p_points is None:
                    self.p_points = np.zeros((num_points, 2), dtype=np.float64)
                    # Need an array that owns its own data, otherwise the
                    # subsequent resize can be messed up
                    self.p_points = np.require(points.T, requirements=["C", "OWNDATA"])
                else:
                    self.p_points.resize((self.total_points + num_points, 2))
                    self.p_points[self.total_points:,:] = points.T
                self.p_starts.append(self.total_points)
                self.p_counts.append(num_points)
                self.p_identifiers.append(ident)
                self.total_points += num_points
        
        new_polys = AccumulatePolygons()
        for n in range(np.alen(self.polygons)):
            poly = self.get_shapely_polygon(n)
            try:
                cropped_poly = crop_rect.intersection(poly)
            except Exception, e:
                print "Shapely intersection exception", e
                print poly
                print poly.is_valid
                raise
                
            if not cropped_poly.is_empty:
                if cropped_poly.geom_type == "MultiPolygon":
                    for i, p in enumerate(cropped_poly):
                        ident = dict({
                            'name': '%s (cropped part #%d)' % (self.polygon_identifiers[n]['name'], i + 1),
                            'feature_code': self.polygon_identifiers[n]['feature_code'],
                            })
                        new_polys.add_polygon(p, ident)
                    continue
                elif not hasattr(cropped_poly, 'exterior'):
                    print "Temporarily skipping %s" % cropped_poly.geom_type
                    continue
                new_polys.add_polygon(cropped_poly, self.polygon_identifiers[n])
                
        old_state = self.get_restore_state()
        self.set_data(new_polys.p_points,
                      np.asarray(new_polys.p_starts, dtype=np.uint32),
                      np.asarray(new_polys.p_counts, dtype=np.uint32),
                      new_polys.p_identifiers)
        
        if add_undo_info:
            new_state = self.get_restore_state()
            self.manager.add_undo_operation_to_operation_batch(OP_CLIP, self, -1, (old_state, new_state))

        # NOTE: Renderers need to be updated after setting new data, but this
        # can't be done here because 1) we don't know the layer control that
        # contains the renderer data, and 2) it may affect multiple views of
        # this layer.  Need to rely on the caller to rebuild renderers.
    
    def get_restore_state(self):
        return self.points.copy(), self.polygons.copy(), self.polygon_adjacency_array.copy(), list(self.polygon_identifiers)
    
    def set_state(self, params):
        self.points, self.polygons, self.polygon_adjacency_array, self.polygon_identifiers = params
        self.update_bounds()
        self.manager.renderer_rebuild_event = True

    def create_renderer(self, renderer):
        """Create the graphic renderer for this layer.
        
        There may be multiple views of this layer (e.g.  in different windows),
        so we can't just create the renderer as an attribute of this object.
        The storage parameter is attached to the view and independent of
        other views of this layer.
        
        """
        if self.polygons != None and renderer.polygon_set_renderer == None:
            renderer.rebuild_polygon_set_renderer(self)

    def render_projected(self, renderer, w_r, p_r, s_r, layer_visibility, layer_index_base, pick_mode=False):
        log.log(5, "Rendering polygon layer!!! visible=%s, pick=%s" % (layer_visibility["layer"], pick_mode))
        if (not layer_visibility["layer"]):
            return

        # the polygons
        if (renderer.polygon_set_renderer != None and layer_visibility["polygons"]):
            renderer.polygon_set_renderer.render(layer_index_base + renderer.POLYGONS_SUB_LAYER_PICKER_OFFSET,
                                             pick_mode,
                                             self.polygons.color,
                                             color_to_int(0, 0, 0, 1.0),
                                             1)  # , self.get_selected_polygon_indexes()
