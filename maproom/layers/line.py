import numpy as np

# Enthought library imports.
from traits.api import Any
from traits.api import Int
from traits.api import Str
from traits.api import Unicode

from ..library.scipy_ckdtree import cKDTree
from ..library.Boundary import Boundaries
from ..library.shapely_utils import shapely_to_polygon
from ..renderer import data_types
from ..command import UndoInfo
from ..mouse_commands import DeleteLinesCommand, MergePointsCommand

from .point import PointLayer
from . import state

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


class LineLayer(PointLayer):
    """Layer for points/lines/polygons.

    """
    name = "Ugrid"

    type = "line"

    line_segment_indexes = Any

    point_identifiers = Any

    pickable = True  # this is a layer that supports picking

    use_color_cycling = True

    visibility_items = ["points", "lines", "labels"]

    layer_info_panel = ["Point count", "Line segment count", "Show depth", "Flagged points", "Default depth", "Depth unit", "Color"]

    def __str__(self):
        return PointLayer.__str__(self) + ", %d lines" % self.num_lines

    @property
    def num_lines(self):
        try:
            return len(self.line_segment_indexes)
        except TypeError:
            return 0

    def test_contents_equal(self, other):
        return self.num_lines == other.num_lines and PointLayer.test_contents_equal(self, other)

    def new(self):
        super(LineLayer, self).new()
        self.line_segment_indexes = self.make_line_segment_indexes(0)

    def get_info_panel_text(self, prop):
        if prop == "Line segment count":
            if self.line_segment_indexes is not None:
                return str(len(self.line_segment_indexes))
            return "0"
        return PointLayer.get_info_panel_text(self, prop)

    def visibility_item_exists(self, label):
        """Return keys for visibility dict lookups that currently exist in this layer
        """
        # fixme == does this need to be hard-coded?
        if label in ["points", "labels"]:
            return self.points is not None
        elif label == "lines":
            return self.line_segment_indexes is not None
        else:
            raise RuntimeError("Unknown label %s for %s" % (label, self.name))

    def set_data(self, f_points, f_depths, f_line_segment_indexes, update_bounds=True, style=None):
        n = np.alen(f_points)
        if style is not None:
            self.style = style
        self.points = self.make_points(n)
        if (n > 0):
            self.points.view(data_types.POINT_XY_VIEW_DTYPE).xy[
                0: n
            ] = f_points
            self.points.z[
                0: n
            ] = f_depths
            self.points.color = self.style.line_color
            self.points.state = 0

        n = np.alen(f_line_segment_indexes)
        self.line_segment_indexes = self.make_line_segment_indexes(n)
        if (n > 0):
            self.line_segment_indexes.view(data_types.LINE_SEGMENT_POINTS_VIEW_DTYPE).points[
                0: n
            ] = f_line_segment_indexes
            self.line_segment_indexes.color = self.style.line_color
            self.line_segment_indexes.state = 0

        if update_bounds:
            self.update_bounds()

    def set_simple_data(self, points):
        count = np.alen(points)
        lines = np.empty((count, 2), dtype=np.uint32)
        lines[:,0] = np.arange(0, count, dtype=np.uint32)
        lines[:,1] = np.arange(1, count + 1, dtype=np.uint32)
        lines[count - 1, 1] = 0
        self.set_data(points, 0.0, lines)

    def set_data_from_geometry(self, geom, style=None):
        error, points, starts, counts, identifiers, groups = shapely_to_polygon([geom])
        count = np.alen(points)
        lines = np.empty((count, 2), dtype=np.uint32)
        self.point_identifiers = []
        # there could be multiple rings if the geometry has holes or is a
        # MultiPolygon, so each subset needs to be matched to its identifier
        for s, c, ident in zip(starts, counts, identifiers):
            # generate list connecting each point to the next
            lines[s:s + c, 0] = np.arange(s, s + c, dtype=np.uint32)
            lines[s:s + c, 1] = np.arange(s + 1, s + c + 1, dtype=np.uint32)
            # but replace the last point with the first to close the loop
            lines[s + c - 1, 1] = s
            self.point_identifiers.append((s, s + c, ident))

        self.set_data(points, 0.0, lines, style=style)
        for i, (s, c, ident) in enumerate(zip(starts, counts, identifiers)):
            self.line_segment_indexes.state[s:s + c] = state.POLYGON_NUMBER_SHIFT * i

    def set_data_from_boundary_points(self, points, style=None):
        print(f"data_from_boundary: {points.shape}")
        count = np.alen(points)
        lines = np.empty((count, 2), dtype=np.uint32)
        lines[0:count, 0] = np.arange(0, count, dtype=np.uint32)
        lines[0:count, 1] = np.arange(1, count + 1, dtype=np.uint32)
        lines[count - 1, 1] = 0
        self.set_data(points, 0.0, lines, style=style)

    def get_point_identifier(self, point_num):
        for s, e, ident in self.point_identifiers:
            if point_num >= s and point_num < e:
                return s, e, ident
        raise IndexError("Point number %d not found!" % point_num)

    def set_color(self, color):
        self.style.line_color = color
        self.points.color = color
        self.line_segment_indexes.color = color

    def can_save_as(self):
        return True

    def lines_to_json(self):
        if self.line_segment_indexes is not None:
            return self.line_segment_indexes.tolist()

    def lines_from_json(self, json_data):
        jd = json_data['lines']
        if jd is not None:
            self.line_segment_indexes = np.array([tuple(i) for i in jd], data_types.LINE_SEGMENT_DTYPE).view(np.recarray)
        else:
            self.line_segment_indexes = jd

    def check_for_problems(self, window):
        # determine the boundaries in the parent layer
        boundaries = Boundaries(self, allow_branches=False, allow_self_crossing=False)
        boundaries.check_errors(True)

    def has_boundaries(self):
        return True

    def get_all_boundaries(self):
        b = Boundaries(self, True, True)
        return b.boundaries

    def get_points_lines(self):
        points = self.points.view(data_types.POINT_XY_VIEW_DTYPE).xy
        lines = self.line_segment_indexes.view(data_types.LINE_SEGMENT_POINTS_VIEW_DTYPE).points
        return points, lines

    def select_outer_boundary(self):
        # determine the boundaries in the parent layer
        boundaries = Boundaries(self, allow_branches=True, allow_self_crossing=True)
        if len(boundaries) > 0:
            self.select_points(boundaries[0].point_indexes)
        else:
            return None
        return boundaries[0]

    def make_line_segment_indexes(self, count):
        return np.repeat(
            np.array([(0, 0, 0, 0)], dtype=data_types.LINE_SEGMENT_DTYPE),
            count,
        ).view(np.recarray)

    def clear_all_selections(self, mark_type=state.SELECTED):
        self.clear_all_point_selections(mark_type)
        self.clear_all_line_segment_selections(mark_type)
        self.increment_change_count()

    def clear_all_line_segment_selections(self, mark_type=state.SELECTED):
        if (self.line_segment_indexes is not None):
            self.line_segment_indexes.state = self.line_segment_indexes.state & (0xFFFFFFFF ^ mark_type)
            self.increment_change_count()

    def select_line_segment(self, line_segment_index, mark_type=state.SELECTED):
        self.line_segment_indexes.state[line_segment_index] = self.line_segment_indexes.state[line_segment_index] | mark_type
        self.increment_change_count()

    def deselect_line_segment(self, line_segment_index, mark_type=state.SELECTED):
        self.line_segment_indexes.state[line_segment_index] = self.line_segment_indexes.state[line_segment_index] & (0xFFFFFFFF ^ mark_type)
        self.increment_change_count()

    def is_line_segment_selected(self, line_segment_index, mark_type=state.SELECTED):
        return self.line_segment_indexes is not None and (self.line_segment_indexes.state[line_segment_index] & mark_type) != 0

    def select_line_segments_in_rect(self, is_toggle_mode, is_add_mode, w_r, mark_type=state.SELECTED):
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

    def get_selected_and_dependent_point_indexes(self, mark_type=state.SELECTED):
        indexes = np.arange(0)
        if (self.points is not None):
            indexes = np.append(indexes, self.get_selected_point_indexes(mark_type))
        if (self.line_segment_indexes is not None):
            l_s_i_s = self.get_selected_line_segment_indexes(mark_type)
            indexes = np.append(indexes, self.line_segment_indexes[l_s_i_s].point1)
            indexes = np.append(indexes, self.line_segment_indexes[l_s_i_s].point2)
        #
        return np.unique(indexes)

    def get_num_points_selected(self, mark_type=state.SELECTED):
        return len(self.get_selected_and_dependent_point_indexes(mark_type))

    def get_selected_line_segment_indexes(self, mark_type=state.SELECTED):
        if (self.line_segment_indexes is None):
            return []
        #
        return np.where((self.line_segment_indexes.state & mark_type) != 0)[0]

    def get_all_line_point_indexes(self):
        indexes = np.arange(0)
        if (self.line_segment_indexes is not None):
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
                    if (q not in path):
                        extended = []
                        extended.extend(path)
                        extended.append(q)
                        if (self.is_point_selected(q)):
                            return extended
                        else:
                            new_paths.append(extended)

            list_of_paths = new_paths

    def find_points_connected_to_point(self, point_index):
        if (self.line_segment_indexes is None):
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
                    if (j not in path):
                        extended = []
                        extended.extend(path)
                        extended.append(j)
                        if (self.is_line_segment_selected(j)):
                            return extended
                        else:
                            new_paths.append(extended)

            list_of_paths = new_paths

    def find_lines_connected_to_line(self, line_segment_index):
        if (self.line_segment_indexes is None):
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

    def delete_all_selected_objects(self):
        point_indexes = self.get_selected_point_indexes()
        l_s_i_s = None
        if (self.get_selected_line_segment_indexes is not None):
            l_s_i_s = self.get_selected_line_segment_indexes()
        if ((point_indexes is not None and len(point_indexes)) > 0 or (l_s_i_s is not None and len(l_s_i_s) > 0)):
            cmd = DeleteLinesCommand(self, point_indexes, l_s_i_s)
            return cmd

    def get_lines_connected_to_points(self, point_indexes):
        if point_indexes is None:
            return []
        attached = np.where(np.in1d(self.line_segment_indexes.point1, point_indexes))
        attached = np.append(attached, np.where(np.in1d(self.line_segment_indexes.point2, point_indexes)))
        attached = np.unique(attached)
        return attached.astype(np.uint32)

    def remove_points_and_lines(self, point_indexes, line_segment_indexes_to_be_deleted):
        # adjust the point indexes of the remaining line segments
        offsets = np.zeros(np.alen(self.line_segment_indexes)).astype(np.uint32)
        for index in point_indexes:
            offsets += np.where(self.line_segment_indexes.point1 > index, 1, 0).astype(np.uint32)
        self.line_segment_indexes.point1 -= offsets
        offsets[: np.alen(offsets)] = 0
        for index in point_indexes:
            offsets += np.where(self.line_segment_indexes.point2 > index, 1, 0).astype(np.uint32)
        self.line_segment_indexes.point2 -= offsets

        # delete them from the layer
        self.points = np.delete(self.points, point_indexes, 0)
        if (line_segment_indexes_to_be_deleted is not None):
            # then delete the line segments
            self.line_segment_indexes = np.delete(self.line_segment_indexes, line_segment_indexes_to_be_deleted, 0)

    def update_after_insert_point_at_index(self, point_index):
        # update point indexes in the line segements to account for the inserted point
        if (self.line_segment_indexes is not None):
            offsets = np.zeros(np.alen(self.line_segment_indexes), dtype=np.uint32)
            offsets += np.where(self.line_segment_indexes.point1 >= point_index, 1, 0).astype(np.uint32)
            self.line_segment_indexes.point1 += offsets
            offsets[: np.alen(offsets)] = 0
            offsets += np.where(self.line_segment_indexes.point2 >= point_index, 1, 0).astype(np.uint32)
            self.line_segment_indexes.point2 += offsets

    def insert_point_in_line(self, world_point, line_segment_index):
        new_point_index = self.insert_point(world_point)
        point_index_1 = self.line_segment_indexes.point1[line_segment_index]
        point_index_2 = self.line_segment_indexes.point2[line_segment_index]
        color = self.line_segment_indexes.color[line_segment_index]
        state = self.line_segment_indexes.state[line_segment_index]
        depth = (self.points.z[point_index_1] + self.points.z[point_index_2]) / 2
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
                l_s = np.array( [ ( p_i, point_index, DEFAULT_LINE_SEGMENT_COLOR, state.CLEAR ) ],
                                dtype = data_types.LINE_SEGMENT_DTYPE ).view( np.recarray )
                self.line_segment_indexes = np.append( self.line_segment_indexes, l_s ).view( np.recarray )
                num_connections_made += 1
        if ( num_connections_made > 0 ):
            self.rebuild_point_and_line_set_renderer()
    """

    def insert_line_segment(self, point_index_1, point_index_2):
        return self.insert_line_segment_at_index(len(self.line_segment_indexes), point_index_1, point_index_2, self.style.line_color, state.CLEAR)

    def insert_line_segment_at_index(self, l_s_i, point_index_1, point_index_2, color, state):
        l_s = np.array([(point_index_1, point_index_2, color, state)],
                       dtype=data_types.LINE_SEGMENT_DTYPE).view(np.recarray)
        self.line_segment_indexes = np.insert(self.line_segment_indexes, l_s_i, l_s).view(np.recarray)

        undo = UndoInfo()
        undo.index = l_s_i
        undo.data = np.copy(l_s)
        lf = undo.flags.add_layer_flags(self)
        lf.layer_contents_added = True

        return undo

    def update_after_delete_point(self, point_index):
        if (self.line_segment_indexes is not None):
            offsets = np.zeros(np.alen(self.line_segment_indexes)).astype(np.uint32)
            offsets += np.where(self.line_segment_indexes.point1 > point_index, 1, 0).astype(np.uint32)
            self.line_segment_indexes.point1 -= offsets
            offsets[: np.alen(offsets)] = 0
            offsets += np.where(self.line_segment_indexes.point2 > point_index, 1, 0).astype(np.uint32)
            self.line_segment_indexes.point2 -= offsets

    def delete_line_segment(self, l_s_i):
        undo = UndoInfo()
        p = self.line_segment_indexes[l_s_i]
        undo.index = l_s_i
        undo.data = np.copy(p)
        undo.flags.refresh_needed = True
        lf = undo.flags.add_layer_flags(self)
        lf.layer_contents_added = True

        self.line_segment_indexes = np.delete(self.line_segment_indexes, l_s_i, 0)
        return undo

    def merge_from_source_layers(self, layer_a, layer_b, depth_unit=""):
        PointLayer.merge_from_source_layers(self, layer_a, layer_b, depth_unit)

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
        if (self.points is None or len(self.points) < 2):
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
            return MergePointsCommand(self, list(points_to_delete))

    def rebuild_renderer(self, renderer, in_place=False):
        """Update display canvas data with the data in this layer

        """
        projected_point_data = self.compute_projected_point_data()
        renderer.set_points(projected_point_data, self.points.z, self.points.color.copy().view(dtype=np.uint8))
        renderer.set_lines(projected_point_data, self.line_segment_indexes.view(data_types.LINE_SEGMENT_POINTS_VIEW_DTYPE)["points"], self.line_segment_indexes.color)

    def render_projected(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        """Actually draw the screen using the current display canvas renderer

        """
        log.log(5, "Rendering line layer!!! visible=%s, pick=%s" % (layer_visibility["layer"], picker))
        if (not layer_visibility["layer"]):
            return

        # the points and line segments
        if layer_visibility["lines"]:
            renderer.draw_lines(self, picker, self.style,
                                self.get_selected_line_segment_indexes(),
                                self.get_selected_line_segment_indexes(state.FLAGGED))

        if layer_visibility["points"]:
            renderer.draw_points(self, picker, self.point_size,
                                 self.get_selected_point_indexes(),
                                 self.get_selected_point_indexes(state.FLAGGED))

        # the labels
        if layer_visibility["labels"]:
            renderer.draw_labels_at_points(self.points.z, s_r, p_r)

        # render selections after everything else
        if (not picker.is_active):
            if layer_visibility["lines"]:
                renderer.draw_selected_lines(self.style, self.get_selected_line_segment_indexes())

            if layer_visibility["points"]:
                renderer.draw_selected_points(self.point_size, self.get_selected_point_indexes())


class LineEditLayer(LineLayer):
    """Layer for points/lines/rings.

    """
    name = "Line Edit"

    type = "line_edit"

    parent_layer = Any

    object_type = Int

    object_index = Int

    layer_info_panel = ["Point count", "Line segment count", "Show depth", "Flagged points", "Default depth", "Depth unit", "Color"]

    transient_edit_layer = True

    def get_new_points_after_move(self, indexes):
        new_points = {}
        for i in indexes:
            # find the ring that contains the selected point
            for s, e, line_layer_ident in self.point_identifiers:
                if i >= s and i < e:
                    sub_index = line_layer_ident['sub_index']
                    ring_index = line_layer_ident['ring_index']
                    geom, geom_ident = self.parent_layer.get_geometry_from_object_index(self.object_index, sub_index, ring_index)
                    geom_index = geom_ident['geom_index']
                    # use a dict to coalesce changes in multiple points on the
                    # same sub/ring to create a single change
                    new_points[(geom_index, sub_index, ring_index)] = (geom_ident, self.points.view(data_types.POINT_XY_VIEW_DTYPE).xy[s:e])
        return list(new_points.values())

    def get_new_points_after_insert(self, pt1, pt2, new_index):
        new_points = {}
        s, e, ident = self.get_point_identifier(pt1)
        s2, e2, ident2 = self.get_point_identifier(pt2)
        if ident != ident2:
            log.error("Two points somehow aren't on the same ring")
        sub_index = ident['sub_index']
        ring_index = ident['ring_index']
        geom, geom_ident = self.parent_layer.get_geometry_from_object_index(self.object_index, sub_index, ring_index)
        # geom_index = geom_ident['geom_index']
        new_points = self.make_points(e - s + 1)  # start to pt1; add new index, pt1 + 1 to e
        insertion_point = pt1 - s + 1
        new_points[0:insertion_point] = self.points[s:s + insertion_point]
        new_points[insertion_point] = self.points[new_index]
        new_points[insertion_point + 1:e - s + 1] = self.points[pt1 + 1:e]
        geom_points = [(geom_ident, new_points.view(data_types.POINT_XY_VIEW_DTYPE).xy)]
        return geom_points

    def rebuild_from_parent_layer(self):
        geom, ident = self.parent_layer.get_geometry_from_object_index(self.object_index, 0, 0)
        style_save = self.style.get_copy()
        self.set_data_from_geometry(geom, style=style_save)
        self.style = style_save

    def update_transient_layer(self, command):
        log.debug("Updating transient layer %s with %s" % (self.name, command))
        if command and hasattr(command, 'transient_geometry_update'):
            command.transient_geometry_update(self)
        return self.parent_layer
