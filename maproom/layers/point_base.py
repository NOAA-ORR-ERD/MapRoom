"""
Layer type to be used as a base class for layers with points

"""
import numpy as np

from ..library import rect
from ..library.depth_utils import convert_units
from ..renderer import data_types

from .base import ProjectedLayer
from . import state

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


class PointBaseLayer(ProjectedLayer):
    """
    Layer for just points

    """
    name = "Point"

    type = "base_point"

    layer_info_panel = ["Point count"]

    selection_info_panel = []

    def __init__(self, manager):
        super().__init__(manager)

        self.points = None
        self.hidden_points = None  # numpy array listing indexes of points to hide
        self.point_size = 4.0
        self.selected_point_size = 15.0
        self.selected_line_width = 10.0

    def __str__(self):
        return ProjectedLayer.__str__(self) + ", %d points" % (self.num_points)

    @property
    def num_points(self):
        try:
            return len(self.points)
        except TypeError:
            return 0

    def test_contents_equal(self, other):
        return self.num_points == other.num_points and ProjectedLayer.test_contents_equal(self, other)

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
        # fixme: this should be done differently...
        self.points = data_types.make_points(num)

    def empty(self):  # fixme: make a property?
        """
        We shouldn't allow saving of a layer with no content, so we use this method
        to determine if we can save this layer.
        """
        no_points = (self.points is None or len(self.points) == 0)

        return no_points

    # fixme: can we remove all the visibility stuff???
    # and if not -- this shouldn't have any references to labels
    def get_visibility_dict(self, project):
        # fixme: why not call self.get_visibility_dict ?
        d = ProjectedLayer.get_visibility_dict(self, project)
        # fixme: and why do I need to mess with label visibility here?
        d["labels"] = False
        return d

    def set_data(self, f_points, style=None):
        n = len(f_points)
        if style is not None:
            self.style = style
        self.points = data_types.make_points(n)
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
        if self.style.line_color is not None and self.points is not None:
            self.set_color(self.style.line_color)

    def make_points(self, count):
        return np.repeat(
            np.array([(np.nan, np.nan, np.nan, 0, 0)], dtype=data_types.POINT_DTYPE),
            count,
        ).view(np.recarray)

    def copy_points(self):
        return self.points.copy().view(np.recarray)

    def copy_bounds(self):
        return rect.copy(self.bounds)

    def get_undo_info(self):
        """ Return a copy of any data needed to restore the state of the layer

        It must be a copy, not a referece, so that it can be stored unchanged
        even if the layer has further changes by commands in the future.
        """
        return (self.copy_points(), self.copy_bounds())

    def restore_undo_info(self, info):
        """ Restore the state of the layer given the data previously generated
        by get_undo_info
        """
        self.points = info[0]
        self.bounds = info[1]

    # JSON Serialization

    def points_to_json(self):
        if self.points is not None:
            return self.points.tolist()

    def points_from_json(self, json_data):
        jd = json_data['points']
        if jd is not None:
            self.points = np.array([tuple(i) for i in jd], data_types.POINT_DTYPE).view(np.recarray)
        else:
            self.points = jd

    def point_size_to_json(self):
        return self.point_size

    def point_size_from_json(self, json_data):
        self.point_size = json_data.get('point_size', 4.0)

    def compute_bounding_rect(self, mark_type=state.CLEAR):
        if (self.points is not None and len(self.points) > 0):
            if (mark_type == state.CLEAR):
                points = self.points
            else:
                points = self.points[self.get_selected_point_indexes(mark_type)]
            return self.compute_bounding_rect_from_points(points)
        return rect.NONE_RECT

    def compute_bounding_rect_from_points(self, points):
        # fixme -- could be more eficient numpy-wise
        l = points.x.min()
        r = points.x.max()
        b = points.y.min()
        t = points.y.max()
        return ((l, b), (r, t))

    def get_state(self, index):
        # fixme -- is this needed -- should all points have a state?
        return self.points.state[index]

    def compute_selected_bounding_rect(self):
        bounds = self.compute_bounding_rect(state.SELECTED)
        return bounds

    def clear_all_selections(self, mark_type=state.SELECTED):
        self.clear_all_point_selections(mark_type)
        self.increment_change_count()

    def clear_all_point_selections(self, mark_type=state.SELECTED):
        if (self.points is not None):
            self.points.state = self.points.state & (0xFFFFFFFF ^ mark_type)
            self.increment_change_count()

    def clear_flagged(self, refresh=False):
        self.clear_all_selections(state.FLAGGED)
        if refresh:
            self.manager.project.refresh(None)

    def has_selection(self):
        return self.get_num_points_selected() > 0

    def has_flagged(self):
        return self.get_num_points_flagged() > 0

    def select_point(self, point_index, mark_type=state.SELECTED):
        self.points.state[point_index] = self.points.state[point_index] | mark_type
        self.increment_change_count()

    def select_all_points(self, mark_type=state.SELECTED):
        if (self.points is not None):
            self.points.state = self.points.state | mark_type
            self.increment_change_count()

    def select_nearest_point(self, world_point):
        c = self.copy_points()
        c.x -= world_point[0]
        c.y -= world_point[1]
        diff = np.empty([len(c)], dtype=np.float64)
        diff[:] = np.abs(c.x + c.y)
        s = np.argsort(diff, 0)
        index = s[0]
        self.select_point(index)

    def deselect_point(self, point_index, mark_type=state.SELECTED):
        self.points.state[point_index] = self.points.state[point_index] & (0xFFFFFFFF ^ mark_type)
        self.increment_change_count()

    def is_point_selected(self, point_index, mark_type=state.SELECTED):
        return self.points is not None and (self.points.state[point_index] & mark_type) != 0

    def select_points(self, indexes, mark_type=state.SELECTED):
        self.points.state[indexes] |= mark_type
        self.increment_change_count()

    def deselect_points(self, indexes, mark_type=state.SELECTED):
        self.points.state[indexes] &= (0xFFFFFFFF ^ mark_type)
        self.increment_change_count()

    def select_flagged(self, refresh=False):
        indexes = self.get_flagged_point_indexes()
        self.deselect_points(indexes, state.FLAGGED)
        self.select_points(indexes, state.SELECTED)
        if refresh:
            self.manager.project.refresh(True)

    def get_flagged_point_indexes(self):
        return self.get_selected_point_indexes(state.FLAGGED)

    def select_points_in_rect(self, is_toggle_mode, is_add_mode, w_r, mark_type=state.SELECTED):
        if (not is_toggle_mode and not is_add_mode):
            self.clear_all_point_selections()
            is_add_mode = True
        indexes = np.where(np.logical_and(
            np.logical_and(self.points.x >= w_r[0][0], self.points.x <= w_r[1][0]),
            np.logical_and(self.points.y >= w_r[0][1], self.points.y <= w_r[1][1])))
        if (is_add_mode):
            self.points.state[indexes] |= mark_type
        if (is_toggle_mode):
            self.points.state[indexes] ^= mark_type
        self.increment_change_count()

    def get_selected_point_indexes(self, mark_type=state.SELECTED):
        if (self.points is None):
            return []
        return np.where((self.points.state & mark_type) != 0)[0]

    def get_selected_and_dependent_point_indexes(self, mark_type=state.SELECTED):
        """Get all points from selected objects.

        Subclasses should override to provide a list of points that are
        implicitly selected by an object being selected.
        """
        return self.get_selected_point_indexes(mark_type)

    def get_num_points_selected(self, mark_type=state.SELECTED):
        return len(self.get_selected_point_indexes(mark_type))

    def get_num_points_flagged(self):
        return len(self.get_selected_point_indexes(state.FLAGGED))

    def is_mergeable_with(self, other_layer):
        return hasattr(other_layer, "points")

    def find_merge_layer_class(self, other_layer):
        return type(self)

    def merge_from_source_layers(self, layer_a, layer_b, depth_unit=""):
        # for now we only handle merging of points and lines
        self.new()

        self.merged_points_index = len(layer_a.points)

        n = len(layer_a.points) + len(layer_b.points)
        self.points = data_types.make_points(n)
        self.points[0: self.merged_points_index] = layer_a.points.copy()
        if depth_unit and layer_a.depth_unit != depth_unit:
            convert_units(self.points[0: self.merged_points_index].z, layer_a.depth_unit, depth_unit)

        self.points[self.merged_points_index: n] = layer_b.points.copy()
        if depth_unit and layer_b.depth_unit != depth_unit:
            convert_units(self.points[self.merged_points_index: n].z, layer_b.depth_unit, depth_unit)

        if depth_unit:
            self.depth_unit = depth_unit
        # self.points.state = 0

    def normalize_longitude(self):
        l = self.points.x.min()
        r = self.points.x.max()
        if l > 0 and r > 0:
            self.points.x -= 360.0

    def swap_lat_lon(self):
        self.points.x, self.points.y = self.points.y, self.points.x.copy()

    def compute_projected_point_data(self, projection):
        return data_types.compute_projected_point_data(self.points, projection, self.hidden_points)

    def update_affected_points(self,
        indexes_affected: "list of indexes that have been moved"
        ):
        """Hook to update additional layer attributes when the points
        specified by the index have been changed.
        """
        pass

    #### contour

    @property
    def can_contour(self):
        return True

    def calc_contour_points(self):
        return self.points.x, self.points.y

    #### rendering

    def rebuild_renderer(self, renderer, in_place=False):
        """Update renderer

        """
        projection = renderer.canvas.projection
        projected_point_data = self.compute_projected_point_data(projection)
        renderer.set_points(projected_point_data, self.points.z, self.points.color.copy().view(dtype=np.uint8))

    def render_projected(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        log.log(5, "Rendering line layer!!! pick=%s" % (picker))

        if layer_visibility["points"]:
            renderer.draw_points(self, picker, self.point_size,
                                 self.get_selected_point_indexes(),
                                 self.get_selected_point_indexes(state.FLAGGED),
                                 style=self.style)

        # the labels
        if layer_visibility["labels"]:
            renderer.draw_labels_at_points(self.points.z, s_r, p_r)

        # render selections after everything else
        if (not picker.is_active):
            if layer_visibility["points"]:
                renderer.draw_selected_points(self.point_size, self.get_selected_point_indexes())
