import numpy as np

# Enthought library imports.
from traits.api import Any
from traits.api import Int
from traits.api import Str
from traits.api import Bool

from ..library.scipy_ckdtree import cKDTree
from ..library.Boundary import Boundaries
from ..library.shapely_utils import shapely_to_polygon
from ..renderer import color_floats_to_int, data_types
from ..command import UndoInfo
from ..mouse_commands import DeleteLinesCommand, MergePointsCommand

from . import PointLayer, LineLayer, Folder, state

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


feature_code_to_color = {
    1: color_floats_to_int(0.25, 0.5, 0, 0.10),  # green
    2: color_floats_to_int(0.0, 0.0, 1.0, 0.10),  # blue
    3: color_floats_to_int(0.5, 0.5, 0.5, 0.10),  # gray
    4: color_floats_to_int(0.9, 0.9, 0.9, 0.15),  # mapbounds
    5: color_floats_to_int(0.0, 0.2, 0.5, 0.15),  # spillable
    "default": color_floats_to_int(0.8, 0.8, 0.8, 0.10),  # light gray
}


class RingEditLayer(LineLayer):
    """Layer for editing rings

    """
    name = "Ring Edit"

    type = "ring_edit"

    parent_point_index = Int

    ring_fill_color = Int

    ring_index = Int

    feature_code = Int

    feature_name = Str

    layer_info_panel = ["Point count", "Line segment count", "Flagged points", "Color"]

    selection_info_panel = ["Selected points", "Point index", "Point latitude", "Point longitude"]

    draw_on_top_when_selected = True

    parent_layer = Any

    transient_edit_layer = True

    def _style_default(self):
        style = self.manager.get_default_style_for(self)
        style.line_color = style.default_highlight_color
        log.debug("_style_default for %s: %s" % (self.type, str(style)))
        return style

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

    def set_data_from_parent_points(self, parent_points, index, count, feature_code, feature_name):
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
        self.feature_code = feature_code
        self.feature_name = feature_name
        self.ring_fill_color = color_array.get(feature_code, color_array[1])
        self.update_bounds()

    def set_data_from_geometry(self, points):
        self.set_simple_data(points)

    def update_transient_layer(self, command):
        return None

    ##### User interface

    def calc_context_menu_actions(self, object_type, object_index, world_point):
        """Return actions that are appropriate when the right mouse button
        context menu is displayed over a particular object within the layer.
        """
        from ..actions import SaveRingEditAction

        return [SaveRingEditAction]


class PolygonParentLayer(PointLayer):
    """Parent folder for group of polygons. Direct children will be
    PolygonBoundaryLayer objects (with grandchildren will be HoleLayers) or
    PointLayer objects.

    """
    name = "Polygon"

    type = "shapefile"

    rebuild_needed = Bool(False)

    point_list = Any

    geometry_list = Any

    ring_adjacency = Any

    rings = Any

    visibility_items = ["points", "lines", "labels"]

    layer_info_panel = ["Point count", "Polygon count", "Flagged points", "Color"]

    selection_info_panel = ["Selected points", "Point index", "Point latitude", "Point longitude"]

    def _style_default(self):
        style = self.manager.get_default_style_for(self)
        style.use_next_default_color()
        log.debug("_style_default for %s: %s" % (self.type, str(style)))
        return style

    def get_info_panel_text(self, prop):
        if prop == "Point count":
            if self.points is not None:
                return str(len(self.points) - 1)  # zeroth point is a NaN
            return "0"
        if prop == "Polygon count":
            if self.ring_adjacency is not None:
                polygon_count = len(np.where(self.ring_adjacency['point_flag'] < 0)[0])
                return str(polygon_count)
            return "0"
        return LineLayer.get_info_panel_text(self, prop)

    def get_undo_info(self):
        return (self.copy_points(), self.ring_adjacency.copy())

    def restore_undo_info(self, info):
        self.points = info[0]
        self.update_bounds()
        self.ring_adjacency = info[1]
        self.create_rings()

    def get_ring_start_end(self, ring_index):
        r = self.rings[ring_index]
        start = r['start']
        return start, start + r['count']

    def get_geometry_from_object_index(self, object_index, sub_index, ring_index):
        start, end = self.get_ring_start_end(object_index)
        geom = self.points.view(data_types.POINT_XY_VIEW_DTYPE).xy[start:end]
        return geom, None

    def create_rings(self):
        print("creating rings from", self.ring_adjacency)
        polygon_starts = np.where(self.ring_adjacency['point_flag'] < 0)[0]
        print("polygon_starts", polygon_starts)
        polygon_counts = -self.ring_adjacency[polygon_starts]['point_flag']
        print("polygon counts", polygon_counts)
        polys = data_types.make_polygons(len(polygon_counts))
        paa = data_types.make_point_adjacency_array(len(self.points))
        group_index = 0
        for ring_index, (start, count) in enumerate(zip(polygon_starts, polygon_counts)):
            end = start + count
            print("poly:", start, end)
            paa[start:end]['next'] = np.arange(start+1, end+1, dtype=np.uint32)
            paa[end-1]['next'] = start
            paa[start:end]['ring_index'] = ring_index
            polys[ring_index]['start'] = start
            polys[ring_index]['count'] = count
            is_hole = count > 1 and self.ring_adjacency[start + 1]['state'] < 0
            if not is_hole:
                group_index += 1
            polys[ring_index]['group'] = group_index
            if count > 2:
                color = self.ring_adjacency[start + 2]['state']
            else:
                color = 0x12345678
            polys[ring_index]['color'] = color
        print(paa)
        print(polys)
        self.rings = polys
        self.point_adjacency_array = paa

    def set_geometry(self, point_list, geom_list):
        self.set_data(point_list)
        print("points", self.points)
        self.geometry_list, self.ring_adjacency = data_types.compute_rings(point_list, geom_list, feature_code_to_color)
        print("adjacency", self.ring_adjacency)

    def commit_editing_layer(self, layer):
        log.debug(f"commiting layer {layer}, ring_index={layer.ring_index if layer is not None else -1}")
        if layer is None:
            return
        log.warning("committing polygon edits, but only handling outer boundary at the moment")
        boundary = layer.select_outer_boundary()
        if boundary is not None:
            self.replace_ring_with_resizing(layer, boundary)
        else:
            log.error("no boundary found; not committing layer")
            self.manager.project.window.error("Incomplete boundary; not updating polygon")

    def replace_ring_with_resizing(self, layer, boundary):
        # print(f"before: points={self.points} rings={self.ring_adjacency}")
        insert_index, old_after_index = self.get_ring_start_end(layer.ring_index)
        old_num_points = len(self.points)

        num_before = insert_index
        num_replace = old_after_index - insert_index
        num_insert = len(boundary.points)
        num_after = old_num_points - old_after_index

        new_after_index = insert_index + num_insert
        new_num_points = num_before + num_insert + num_after

        p = data_types.make_points(new_num_points)
        p[:insert_index] = self.points[:insert_index]
        p[insert_index:new_after_index] = boundary.get_points()
        p[new_after_index:] = self.points[old_after_index:]
        self.points = p

        # ring adjacency needs the exact same substitution
        r = data_types.make_ring_adjacency_array(new_num_points)
        r[:insert_index] = self.ring_adjacency[:insert_index]
        r[insert_index]['point_flag'] = -num_insert
        r[new_after_index - 1]['point_flag'] = 2
        r[insert_index]['state'] = 0
        if num_insert > 0:
            r[insert_index + 1]['state'] = self.ring_adjacency[insert_index + 1]['state']
        if num_insert > 1:
            r[insert_index + 2]['state'] = self.ring_adjacency[insert_index + 2]['state']
        r[new_after_index:] = self.ring_adjacency[old_after_index:]
        self.ring_adjacency = r

        # print(f"after: points={self.points} rings={self.ring_adjacency}")

        # Force rebuild
        self.create_rings()
        self.rebuild_needed = True

    def check_for_problems(self, window):
        pass

    def rebuild_renderer(self, renderer, in_place=False):
        log.debug("rebuilding polygon2 {self.name}")
        if self.rings is None:
            self.create_rings()
        projection = self.manager.project.layer_canvas.projection
        projected_point_data = data_types.compute_projected_point_data(self.points, projection)
        renderer.set_points(projected_point_data, self.points.z, self.points.color.copy().view(dtype=np.uint8))
        # renderer.set_rings(self.ring_adjacency)
        renderer.set_polygons(self.rings, self.point_adjacency_array)

    def can_render_for_picker(self, renderer):
        return renderer.canvas.project.layer_tree_control.is_edit_layer(self)

    def render_projected(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        log.log(5, "Rendering polygon folder layer!!! pick=%s" % (picker))
        if picker.is_active and not self.can_render_for_picker(renderer):
            return
        if self.rebuild_needed:
            self.rebuild_renderer(renderer)
        if layer_visibility["polygons"]:
            edit_layer = self.manager.find_transient_layer()
            if edit_layer is not None and hasattr(edit_layer, 'parent_layer') and edit_layer.parent_layer == self:
                ring_index = edit_layer.ring_index
            else:
                ring_index = None
            renderer.draw_polygons(self, picker, self.rings.color, color_floats_to_int(0, 0, 0, 1.0), 1, editing_polygon_index=ring_index)

    ##### User interface

    def calc_context_menu_actions(self, object_type, object_index, world_point):
        """Return actions that are appropriate when the right mouse button
        context menu is displayed over a particular object within the layer.
        """
        from ..actions import EditLayerAction

        if object_index is not None:
            return [EditLayerAction]
