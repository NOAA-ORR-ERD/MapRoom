import numpy as np

# Enthought library imports.
from traits.api import Any
from traits.api import Int
from traits.api import Str
from traits.api import Unicode

from ..library.scipy_ckdtree import cKDTree
from ..library.Boundary import Boundaries
from ..library.shapely_utils import shapely_to_polygon
from ..renderer import color_floats_to_int, data_types
from ..command import UndoInfo
from ..mouse_commands import DeleteLinesCommand, MergePointsCommand

from . import LineLayer, Folder, state

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


class PolygonBoundaryLayer(LineLayer):
    """Layer for points/lines/polygons.

    """
    name = "Polygon Boundary"

    type = "polygon_boundary"

    parent_point_index = Int

    use_color_cycling = False

    layer_info_panel = ["Point count", "Line segment count", "Flagged points", "Color"]

    selection_info_panel = ["Selected points", "Point index", "Point latitude", "Point longitude"]

    draw_on_top_when_selected = True

    @property
    def area(self):
        # numpy version of shoelace formula, from
        # https://stackoverflow.com/questions/24467972
        if len(self.line_segment_indexes) < 3:
            return 0.0
        start = self.line_segment_indexes.point1
        end = self.line_segment_indexes.point2
        x = self.points.x
        y = self.points.y
        return 0.5 * np.dot(x[start] + x[end], y[start] - y[end])

    @property
    def is_clockwise(self):
        return self.area > 0.0

    @property
    def pretty_name(self):
        if self.grouped:
            prefix = self.grouped_indicator_prefix
        else:
            prefix = ""
        return prefix + self.name + (" cw" if self.is_clockwise else " ccw") + " " + str(self.area)

    def verify_winding(self, positive_area=True):
        area = self.area
        cw = self.is_clockwise
        log.debug(f"area={area} cw={cw}, should be cw={positive_area}")
        if (positive_area and not cw) or (not positive_area and cw):
            self.reverse_line_direction()
            log.debug(f"reversed: area={area} cw={self.is_clockwise}")

    def layer_selected_hook(self):
        parent = self.manager.get_layer_parent(self)
        c = self.manager.project.layer_canvas
        c.rebuild_renderer_for_layer(parent)

    def calc_ring_fill_color(self):
        return color_floats_to_int(0.25, 0.5, 0, 0.10)

    def set_data_from_parent_points(self, parent_points, index, count):
        self.parent_point_index = index
        self.parent_point_map = np.arange(index, index + count, dtype=np.uint32)
        self.points = parent_points[self.parent_point_map]  # Copy!
        log.debug(f"polygon point index={index} count={count}")
        self.points.color = self.style.line_color
        self.points.state = 0
        lsi = self.make_line_segment_indexes(count)
        lsi.point1[0:count] = np.arange(0, count, dtype=np.uint32)
        lsi.point2[0:count] = np.arange(1, count + 1, dtype=np.uint32)
        lsi.point2[count - 1] = 0
        lsi.color = self.style.line_color
        lsi.state = 0
        self.line_segment_indexes = lsi
        self.update_bounds()

    def update_affected_points(self, indexes):
        indexes = np.asarray(indexes, dtype=np.uint32)
        print(f"points changed: {indexes}")
        print(f"points changed in parent: {self.parent_point_map[indexes]}")
        parent = self.manager.get_layer_parent(self)
        changed_points = self.points[indexes]
        parent.update_child_points(self.parent_point_map[indexes], changed_points)
        return parent

    def render_projected(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        if not self.manager.project.layer_tree_control.is_edit_layer(self):
            log.debug(f"not edit layer, skipping verdat editing for {self}")
            return
        LineLayer.render_projected(self, renderer, w_r, p_r, s_r, layer_visibility, picker)


class PolygonParentLayer(Folder, LineLayer):
    """Parent folder for group of polygons. Direct children will be
    PolygonBoundaryLayer objects (with grandchildren will be HoleLayers) or
    PointLayer objects.

    """
    name = "Polygon"

    type = "shapefile"

    point_list = Any

    visibility_items = ["points", "lines", "labels"]

    layer_info_panel = ["Point count", "Line segment count", "Flagged points", "Color"]

    selection_info_panel = ["Selected points", "Point index", "Point latitude", "Point longitude"]

    def _style_default(self):
        style = self.manager.get_default_style_for(self)
        style.use_next_default_color()
        log.debug("_style_default for %s: %s" % (self.type, str(style)))
        return style

    @property
    def is_renderable(self):
        return True

    def has_groupable_objects(self):
        return True

    def get_info_panel_text(self, prop):
        if prop == "Point count":
            total = 0
            for child in self.get_child_layers():
                try:
                    total += len(self.points)
                except TypeError:
                    pass
            return str(total)
        elif prop == "Line segment count":
            total = 0
            for child in self.get_child_layers():
                try:
                    total += len(self.line_segment_indexes)
                except TypeError:
                    pass
            return str(total)
        return str(None)

    def set_parent_points(self, parent_points):
        self.points = parent_points
        log.debug(f"parent_points={parent_points[0:10]}")
        self.rings = None

    def create_rings(self):
        n_rings = 0
        ring_starts = []
        ring_counts = []
        ring_groups = []
        ring_color = []
        current_group_number = 0
        point_start_index = 0
        for child in self.get_child_layers():
            n_rings += 1
            ring_starts.append(child.parent_point_index)
            ring_counts.append(len(child.points))
            ring_color.append(child.calc_ring_fill_color())
            if child.is_clockwise:
                current_group_number += 1
            ring_groups.append(current_group_number)
        self.rings, self.point_adjacency_array = data_types.compute_rings(ring_starts, ring_counts, ring_groups)
        for c in ring_color:
            self.rings.color = c
        log.debug(f"ring list: {self.rings} {type(self.rings)}")
        log.debug(f"points: {point_start_index}, from rings: {self.rings[-1][0] + self.rings[-1][1]}")

    def update_child_points(self, indexes, values):
        self.points[indexes] = values
        # projection = self.manager.project.layer_canvas.projection
        # projected_point_data = data_types.compute_projected_point_data(self.points[indexes], projection)

    def check_for_problems(self, window):
        pass

    def rebuild_renderer(self, renderer, in_place=False):
        print("REBUILDING POLYGON2")
        if self.rings is None:
            self.create_rings()
        projection = self.manager.project.layer_canvas.projection
        projected_point_data = data_types.compute_projected_point_data(self.points, projection)
        renderer.set_points(projected_point_data, self.points.z, self.points.color.copy().view(dtype=np.uint8))
        renderer.set_polygons(self.rings, self.point_adjacency_array)

    def can_render_for_picker(self, renderer):
        return renderer.canvas.project.layer_tree_control.is_edit_layer(self)

    def render_projected(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        log.log(5, "Rendering polygon folder layer!!! pick=%s" % (picker))
        if picker.is_active and not self.can_render_for_picker(renderer):
            return
        # the rings
        if layer_visibility["polygons"]:
            renderer.draw_polygons(self, picker,
                                   self.rings.color,
                                   color_floats_to_int(0, 0, 0, 1.0),
                                   1)
