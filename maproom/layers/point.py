import time
import numpy as np

# Enthought library imports.
from traits.api import Enum
from traits.api import Float
from traits.api import Int
from traits.api import Property
from traits.api import Str
from traits.api import Unicode

from ..renderer import data_types
from ..command import UndoInfo
from ..mouse_commands import MovePointsCommand

from point_base import PointBaseLayer
import state

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

    pickable = True  # is this a layer that support picking?

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
            self.clear_all_selections(state.FLAGGED)
            for p in e.points:
                self.select_point(p, state.FLAGGED)
            self.manager.dispatch_event('refresh_needed')

    def set_data(self, f_points, f_depths, f_line_segment_indexes, style=None):
        n = np.alen(f_points)
        if style is None:
            self.set_layer_style_defaults()
        else:
            self.style = style
        self.points = self.make_points(n)
        if (n > 0):
            self.points.view(data_types.POINT_XY_VIEW_DTYPE).xy[0:n] = f_points
            self.points.z[0:n] = f_depths
            self.points.color = self.style.line_color
            self.points.state = 0

            n = np.alen(f_line_segment_indexes)
            self.line_segment_indexes = self.make_line_segment_indexes(n)
            self.line_segment_indexes.view(data_types.LINE_SEGMENT_POINTS_VIEW_DTYPE).points[
                0: n
            ] = f_line_segment_indexes
            self.line_segment_indexes.color = self.style.line_color
            self.line_segment_indexes.state = 0

        self.update_bounds()

    def can_save(self):
        return self.can_save_as() and bool(self.file_path)

    # JSON Serialization

    def default_depth_to_json(self):
        return self.default_depth

    def default_depth_from_json(self, json_data):
        self.default_depth = json_data['default_depth']

    def depth_unit_to_json(self):
        return self.depth_unit

    def depth_unit_from_json(self, json_data):
        self.depth_unit = json_data['depth_unit']

    def dragging_selected_objects(self, world_dx, world_dy, snapped_layer, snapped_cp, about_center=False):
        indexes = self.get_selected_and_dependent_point_indexes()
        cmd = MovePointsCommand(self, indexes, world_dx, world_dy)
        return cmd

    def insert_point(self, world_point):
        if self.points is None:
            index = -1
        else:
            index = len(self.points)
        return self.insert_point_at_index(index, world_point, self.default_depth, self.style.line_color, state.SELECTED)

    def insert_point_at_index(self, point_index, world_point, z, color, state):
        t0 = time.clock()
        # insert it into the layer
        p = np.array([(world_point[0], world_point[1], z, color, state)],
                     dtype=data_types.POINT_DTYPE)
        undo = UndoInfo()
        lf = undo.flags.add_layer_flags(self)
        if (self.points is None):
            self.new_points(1)
            self.points[0] = p
            point_index = 0
            undo.flags.refresh_needed = True
            lf.select_layer = True
        else:
            self.points = np.insert(self.points, point_index, p).view(np.recarray)
        undo.index = point_index
        undo.data = np.copy(p)
        lf.layer_items_moved = True
        lf.layer_contents_added = True

        # update point indexes in the line segements to account for the inserted point
        self.update_after_insert_point_at_index(point_index)

        return undo

    def update_after_insert_point_at_index(self, point_index):
        """Hook for subclasses to update dependent items when a point is inserted.
        """

    def delete_point(self, point_index):
        if (self.find_points_connected_to_point(point_index) != []):
            raise Exception()

        undo = UndoInfo()
        p = self.points[point_index]
        print "LABEL: deleting point: %s" % str(p)
        undo.index = point_index
        undo.data = np.copy(p)
        undo.flags.refresh_needed = True
        lf = undo.flags.add_layer_flags(self)
        lf.layer_items_moved = True
        lf.layer_contents_deleted = True
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
