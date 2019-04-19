import numpy as np

import shapely.geometry as sg

from ..errors import PointsError
from ..library.Boundary import Boundaries
from ..renderer import color_floats_to_int, data_types
from ..library.shapefile_utils import GeomInfo, parse_from_old_json
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

    layer_info_panel = ["Point count", "Line segment count", "Color", "Save polygon", "Cancel polygon"]

    selection_info_panel = ["Selected points", "Point index", "Point latitude", "Point longitude", "Area"]

    draw_on_top_when_selected = True

    transient_edit_layer = True

    def __init__(self, manager, parent_layer, object_type, feature_code):
        """NOTE: feature_code applies to all polygons being edited
        """
        super().__init__(manager)
        self.parent_layer = parent_layer
        self.object_type = object_type
        self.feature_code = feature_code
        self.ring_fill_color = 0
        self.ring_indexes = []
        self.feature_name = ""

    def calc_initial_style(self):
        style = self.manager.get_default_style_for(self)
        style.line_color = style.default_highlight_color
        log.debug("calc_initial_style for %s: %s" % (self.type, str(style)))
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

    def get_info_panel_text(self, prop):
        if prop == "Area":
            return str(self.area)
        return LineLayer.get_info_panel_text(self, prop)

    def verify_winding(self, positive_area=True):
        area = self.area
        cw = self.is_clockwise
        log.debug(f"area={area} cw={cw}, should be cw={positive_area}")
        if (positive_area and not cw) or (not positive_area and cw):
            self.reverse_line_direction()
            log.debug(f"reversed: area={area} cw={self.is_clockwise}")

    def get_undo_info(self):
        return (self.copy_points(), self.copy_bounds(), self.copy_lines(), list(self.ring_indexes))

    def restore_undo_info(self, info):
        self.points = info[0]
        self.bounds = info[1]
        self.line_segment_indexes = info[2]
        self.ring_indexes = info[3]

    def set_data_from_parent_points(self, parent_points, index, count, feature_code, feature_name):
        parent_point_map = np.arange(index, index + count, dtype=np.uint32)
        self.points = parent_points[parent_point_map]  # Copy!
        log.debug(f"polygon point index={index} count={count}")
        self.points.color = self.style.line_color
        self.points.state = 0
        lsi = data_types.make_line_segment_indexes(count)
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

    # def append_from_parent_points(self, parent_points, index, count, feature_code, feature_name):
    #     parent_point_map = np.arange(index, index + count, dtype=np.uint32)
    #     points = parent_points[parent_point_map]  # Copy!
    #     count, lines = self.calc_simple_data(points)
    #     self.append_data(points, 0.0, lines)

    def set_data_from_geometry(self, points, ring_index):
        self.set_simple_data(points)
        if ring_index is not None:
            self.ring_indexes.append(ring_index)

    def add_polygon_from_parent_layer(self, ring_index):
        geom, ident = self.parent_layer.get_geometry_from_object_index(ring_index, 0, 0)
        count, lines = self.calc_simple_data(geom)
        self.append_data(geom, 0.0, lines)
        self.ring_indexes.append(ring_index)
        self.update_bounds()

    def update_transient_layer(self, command):
        return None

    ##### User interface

    def calc_context_menu_desc(self, object_type, object_index, world_point):
        """Return actions that are appropriate when the right mouse button
        context menu is displayed over a particular object within the layer.
        """
        from .. import actions as a

        desc = ["save_ring_edit", "cancel_ring_edit"]
        if object_index is not None:
            log.debug(f"object type {object_type} index {object_index}")
            if not self.parent_layer.is_hole(object_index) and object_index not in self.ring_indexes:
                _, _, count, _, feature_code, _ = self.parent_layer.get_ring_state(object_index)
                desc.extend([None, "add_polygon_to_edit_layer"])
                a.add_polygon_to_edit_layer.name = f"Add Polygon to Edit Layer ({count} points, id={object_index})"
        return desc


class PolygonParentLayer(PointLayer):
    """Parent folder for group of polygons. Direct children will be
    PolygonBoundaryLayer objects (with grandchildren will be HoleLayers) or
    PointLayer objects.

    """
    name = "Polygon"

    type = "shapefile"

    mouse_mode_toolbar = "PolygonLayerToolBar"

    visibility_items = ["points", "lines", "labels"]

    layer_info_panel = ["Point count", "Polygon count"]

    selection_info_panel = ["Selected points", "Point index", "Point latitude", "Point longitude"]

    def __init__(self, manager):
        super().__init__(manager)
        self.rebuild_needed = False
        self.geometry_list = []
        _, ring_adjacency = data_types.compute_rings([], [], feature_code_to_color)
        self.ring_adjacency = ring_adjacency
        self.rings = []
        self.point_adjacency_array = data_types.make_point_adjacency_array(0)
        self.points = data_types.make_points(0)

    def calc_initial_style(self):
        style = self.manager.get_default_style_for(self)
        style.use_next_default_color()
        log.debug("calc_initial_style for %s: %s" % (self.type, str(style)))
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

    # JSON Serialization

    def ring_adjacency_to_json(self):
        if self.ring_adjacency is not None:
            return self.ring_adjacency.tolist()

    def ring_adjacency_from_json(self, json_data):
        jd = json_data['ring_adjacency']
        if jd is not None:
            self.ring_adjacency = np.array([tuple(i) for i in jd], data_types.RING_ADJACENCY_DTYPE).view(np.recarray)
        else:
            raise RuntimeError("Missing adjacency array from old save file")

    def geometry_list_to_json(self):
        if self.geometry_list is not None:
            return self.geometry_list

    def geometry_list_from_json(self, json_data):
        jd = json_data['geometry_list']
        if jd is not None:
            self.geometry_list = [GeomInfo(*i) for i in jd]

    def geometry_from_json(self, json_data):
        _, geometry_list, points = parse_from_old_json(json_data['geometry'])
        self.set_geometry(points, geometry_list)

    def copy_ring(self, start_index, points, feature_code, color):
        count = len(points)
        if count < 0:
            return
        end = start_index + count
        self.points.view(data_types.POINT_XY_VIEW_DTYPE).xy[start_index:end] = points[:]
        self.ring_adjacency[start_index]['point_flag'] = -count
        self.ring_adjacency[start_index + count - 1]['point_flag'] = 2
        self.ring_adjacency[start_index]['state'] = 0
        if count > 1:
            self.ring_adjacency[start_index + 1]['state'] = feature_code
        if count > 2:
            self.ring_adjacency[start_index + 2]['state'] = color

    def create_first_ring(self, points, feature_code, color):
        num_new_points = len(points)
        self.points = data_types.make_points(num_new_points)
        self.ring_adjacency = data_types.make_ring_adjacency_array(num_new_points)
        feature_code = 1
        self.copy_ring(0, points, feature_code, feature_code_to_color[feature_code])
        g = GeomInfo(start_index=0, count=num_new_points, name="New Polygon", feature_code=feature_code, feature_name="None")
        self.geometry_list = [g]
        self.create_rings()
        self.rebuild_needed = True

    def set_data_from_boundaries(self, boundaries):
        num_new_points = 0
        for i, b in enumerate(boundaries):
            points = b.get_xy_points()
            num_new_points += np.alen(points)

        self.points = data_types.make_points(num_new_points)
        self.ring_adjacency = data_types.make_ring_adjacency_array(num_new_points)
        index = 0
        for i, b in enumerate(boundaries):
            points = b.get_xy_points()
            self.copy_ring(index, points, 1, feature_code_to_color[1])
            index += len(points)

        self.create_rings()

    def get_undo_info(self):
        return (self.copy_points(), self.ring_adjacency.copy())

    def restore_undo_info(self, info):
        self.points = info[0]
        self.update_bounds()
        self.ring_adjacency = info[1]
        self.create_rings()
        self.rebuild_needed = True

    def get_ring_start_end(self, ring_index):
        r = self.rings[ring_index]
        start = r['start']
        return start, start + r['count']

    def get_ring_points(self, ring_index):
        start, end = self.get_ring_start_end(ring_index)
        p = self.points.view(data_types.POINT_XY_VIEW_DTYPE).xy[start:end]
        return p

    def get_ring_state(self, ring_index):
        if ring_index is None:
            raise IndexError
        start, end = self.get_ring_start_end(ring_index)
        count = -self.ring_adjacency[start]['point_flag']
        state = self.ring_adjacency[start]['state']
        if count > 1:
            feature_code = self.ring_adjacency[start + 1]['state']
            if count > 2:
                color = self.ring_adjacency[start + 2]['state']
            else:
                color = None
        else:
            feature_code = None
            color = None
        return start, end, count, state, feature_code, color

    def get_feature_code(self, ring_index):
        _, _, _, _, feature_code, _ = self.get_ring_state(ring_index)
        return feature_code

    def get_ring_and_holes_start_end(self, ring_index):
        start, end, _, _, feature_code, _ = self.get_ring_state(ring_index)
        log.debug(f"found ring {ring_index}: {start}, {end}, {feature_code}")
        count = 1
        if feature_code >= 0:
            while True:
                try:
                    _, possible, _, _, feature_code, _ = self.get_ring_state(ring_index + 1)
                except IndexError:
                    # no more rings
                    break
                log.debug(f"checking ring {ring_index}: {start}, {end}, {feature_code}")
                if feature_code >= 0:
                    break
                end = possible
                log.debug(f"found hole: {start}, {end}, {feature_code}")
                count += 1
                ring_index += 1

        return start, end, count

    def get_geometry_from_object_index(self, object_index, sub_index, ring_index):
        points = self.get_ring_points(object_index)
        feature_code = self.get_feature_code(object_index)
        return points, feature_code

    def is_hole(self, ring_index):
        feature_code = self.get_feature_code(ring_index)
        return feature_code < 0

    def get_shapely_polygon(self, start, end, debug=False):
        p = self.points
        points = np.c_[p.x[start:end], p.y[start:end]]
        points = np.require(points, np.float64, ["C", "OWNDATA"])
        if np.alen(points) > 2:
            poly = sg.Polygon(points)
        else:
            poly = sg.LineString(points)
        if debug: print(("points tuples:", points))
        if debug: print(("numpy:", points.__array_interface__, points.shape, id(points), points.flags))
        if debug: print(("shapely polygon:", poly.bounds))
        return poly

    def iter_rings(self):
        for i in range(len(self.rings)):
            geom = self.geometry_list[i]
            state = self.get_ring_state(i)
            yield geom, state

    def crop_rectangle(self, w_r):
        log.debug("crop_rectangle: to %s" % str(w_r))

        crop_rect = sg.box(w_r[0][0], w_r[1][1], w_r[1][0], w_r[0][1])

        cropped_list = []
        new_point_count = 0
        for geom, state in self.iter_rings():
            start = state[0]
            end = state[1]
            poly = self.get_shapely_polygon(start, end)
            try:
                cropped_poly = crop_rect.intersection(poly)
            except Exception as e:
                log.error("Shapely intersection exception: %s\npoly=%s\nvalid=%s" % (e, poly, poly.is_valid))
                raise

            if not cropped_poly.is_empty:
                num_points = len(cropped_poly.exterior.coords.xy)
                new_point_count += num_points
                cropped_list.append((geom, state, num_points, cropped_poly))

        log.debug(f"{len(cropped_list)} cropped polygons")
        p = data_types.make_points(new_point_count)
        p = data_types.make_ring_adjacency_array(new_point_count)

        # FIXME: WIP, not working yet!
        total_points = 0
        def add_polygon(shapely_poly, geom, state):
            points = np.require(cropped_poly.exterior.coords.xy, np.float64, ["C", "OWNDATA"])
            num_points = points.shape[1]
            p[total_points:, :] = points.T

        for geom, state, num_points, cropped_poly in cropped_list:
            points = np.require(cropped_poly.exterior.coords.xy, np.float64, ["C", "OWNDATA"])
            count = points.shape[1]
            if cropped_poly.geom_type == "MultiPolygon":
                for i, p in enumerate(cropped_poly):
                    add_polygon(p, geom, state)
                continue
            elif not hasattr(cropped_poly, 'exterior'):
                log.debug("Temporarily skipping %s" % cropped_poly.geom_type)
                continue
            add_polygon(cropped_poly, geom, state)

        old_state = self.get_undo_info()
        self.points = p
        self.ring_adjacency = r
        self.create_rings()

        undo = UndoInfo()
        undo.data = old_state
        undo.flags.refresh_needed = True
        undo.flags.items_moved = True
        lf = undo.flags.add_layer_flags(self)
        lf.layer_contents_deleted = True
        return undo

    def create_rings(self):
        log.debug(f"creating rings from {self.ring_adjacency}")
        polygon_starts = np.where(self.ring_adjacency['point_flag'] < 0)[0]
        log.debug(f"polygon_starts {polygon_starts}")
        polygon_counts = -self.ring_adjacency[polygon_starts]['point_flag']
        log.debug(f"polygon counts {polygon_counts}")
        polys = data_types.make_polygons(len(polygon_counts))
        paa = data_types.make_point_adjacency_array(len(self.points))
        group_index = 0
        ring_index = 0
        for ring_index, (start, count) in enumerate(zip(polygon_starts, polygon_counts)):
            end = start + count
            try:
                log.debug(f"ring[{ring_index}]: {start, end} geom={self.geometry_list[ring_index]}")
            except IndexError:
                log.warning(f"ring[{ring_index}]: {start, end} geometry is missing!")
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
        # print(paa)
        # print(polys)
        self.rings = polys
        self.point_adjacency_array = paa
        log.debug(f"created {ring_index} rings; geometry_list={len(self.geometry_list)}")

    def set_geometry(self, point_list, geom_list):
        self.set_data(point_list)
        log.debug(f"points {self.points}")
        self.geometry_list, self.ring_adjacency = data_types.compute_rings(point_list, geom_list, feature_code_to_color)
        log.debug(f"adjacency {self.ring_adjacency}")
        self.create_rings()

    def dup_geometry_list_entry(self, ring_index_to_copy):
        g = self.geometry_list[ring_index_to_copy]
        self.geometry_list[ring_index_to_copy:ring_index_to_copy] = [g]

    def delete_geometry_list_entries(self, start_ring_index, count):
        self.geometry_list[start_ring_index:start_ring_index + count] = []

    def commit_editing_layer(self, layer):
        log.debug(f"commiting layer {layer}, ring_indexes={layer.ring_indexes if layer is not None else -1}")
        if layer is None:
            return
        boundaries = Boundaries(layer, allow_branches=False, allow_self_crossing=False, allow_points_outside_boundary=True, allow_polylines=False)
        boundaries.check_errors(True)
        feature_code = layer.feature_code
        log.debug(f"boundaries: {boundaries}")
        for boundary in boundaries:
            try:
                ring_index = layer.ring_indexes.pop()
                new_boundary = False
            except IndexError:
                ring_index = 0
                new_boundary = True
            self.replace_ring(ring_index, boundary.get_points(), feature_code, new_boundary)
        for ring_index in layer.ring_indexes:
            self.delete_ring(ring_index)

    def replace_ring(self, ring_index, points, feature_code=None, new_boundary=False):
        log.debug(f"replacing ring {ring_index}")
        if feature_code is None:
            feature_code = self.get_feature_code(ring_index)
        try:
            insert_index, old_after_index = self.get_ring_start_end(ring_index)
        except IndexError:
            self.create_first_ring(points.view(data_types.POINT_XY_VIEW_DTYPE).xy, 1, feature_code_to_color[1])
            log.debug(points)
            log.debug(points.view(data_types.POINT_XY_VIEW_DTYPE).xy[:])
            return
        log.debug(f"insert_index={insert_index}, old_after_index={old_after_index}, points={points}")
        if new_boundary:
            if feature_code < 0:
                # insert after indicated polygon so it becomes a hole of that one
                insert_index = old_after_index
                self.dup_geometry_list_entry(ring_index)
            else:
                # arbitrarily insert at beginning
                old_after_index = insert_index
                self.dup_geometry_list_entry(0)
        old_num_points = len(self.points)

        insert_points = points
        num_insert = len(insert_points)

        # import pdb; pdb.set_trace()
        num_before = insert_index
        num_replace = old_after_index - insert_index
        num_after = old_num_points - old_after_index

        new_after_index = insert_index + num_insert
        new_num_points = num_before + num_insert + num_after

        p = data_types.make_points(new_num_points)
        p[:insert_index] = self.points[:insert_index]
        try:
            p[insert_index:new_after_index] = insert_points
        except ValueError:
            # maybe they are just x,y values, not POINT_DTYPE records
            p.view(data_types.POINT_XY_VIEW_DTYPE).xy[insert_index:new_after_index] = insert_points
        p[new_after_index:] = self.points[old_after_index:]
        self.points = p

        # ring adjacency needs the exact same substitution
        r = data_types.make_ring_adjacency_array(new_num_points)
        r[:insert_index] = self.ring_adjacency[:insert_index]
        r[insert_index]['point_flag'] = -num_insert
        r[new_after_index - 1]['point_flag'] = 2  # flag to signal end of polygon
        r[insert_index]['state'] = 0
        if num_insert > 0:
            r[insert_index + 1]['state'] = feature_code
        if num_insert > 1:
            # color
            r[insert_index + 2]['state'] = self.ring_adjacency[insert_index + 2]['state']
        r[new_after_index:] = self.ring_adjacency[old_after_index:]
        self.ring_adjacency = r

        # Force rebuild
        self.create_rings()
        self.rebuild_needed = True

    def delete_ring(self, ring_index):
        start, end, num_rings = self.get_ring_and_holes_start_end(ring_index)
        log.debug(f"deleting rings {ring_index} - {ring_index + num_rings}")
        old_num_points = len(self.points)
        new_num_points = old_num_points - end + start
        p = data_types.make_points(new_num_points)
        p[:start] = self.points[:start]
        p[start:] = self.points[end:]
        self.points = p

        r = data_types.make_ring_adjacency_array(new_num_points)
        r[:start] = self.ring_adjacency[:start]
        r[start:] = self.ring_adjacency[end:]
        self.ring_adjacency = r

        self.delete_geometry_list_entries(ring_index, num_rings)

        # Force rebuild
        self.create_rings()
        self.rebuild_needed = True

    def check_for_problems(self):
        pass

    def rebuild_renderer(self, renderer, in_place=False):
        log.debug(f"rebuilding polygon2 {self.name}")
        if self.rings is None or len(self.rings) == 0:
            self.create_rings()
        projection = renderer.canvas.projection
        projected_point_data = data_types.compute_projected_point_data(self.points, projection)
        renderer.set_points(projected_point_data, self.points.z, self.points.color.copy().view(dtype=np.uint8))
        # renderer.set_rings(self.ring_adjacency)
        renderer.set_polygons(self.rings, self.point_adjacency_array)
        self.rebuild_needed = False

    def can_render_for_picker(self, renderer):
        if renderer.canvas.project.layer_tree_control.is_edit_layer(self):
            return True
        edit_layer = self.manager.find_transient_layer()
        return edit_layer is not None and hasattr(edit_layer, 'parent_layer') and edit_layer.parent_layer == self

    def render_projected(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        log.log(5, "Rendering polygon folder layer!!! pick=%s" % (picker))
        if picker.is_active and not self.can_render_for_picker(renderer):
            return
        if self.rebuild_needed:
            self.rebuild_renderer(renderer)
        if layer_visibility["polygons"]:
            edit_layer = self.manager.find_transient_layer()
            if edit_layer is not None and hasattr(edit_layer, 'parent_layer') and edit_layer.parent_layer == self and edit_layer.ring_indexes:
                ring_indexes = edit_layer.ring_indexes
            else:
                ring_indexes = []
            renderer.draw_polygons(self, picker, self.rings.color, color_floats_to_int(0, 0, 0, 1.0), 1, editing_polygon_indexes=ring_indexes)

    ##### User interface

    def calc_context_menu_desc(self, object_type, object_index, world_point):
        """Return actions that are appropriate when the right mouse button
        context menu is displayed over a particular object within the layer.
        """
        from .. import actions as a

        desc = []
        if object_index is not None:
            desc = ["edit_layer"]
            log.debug(f"object type {object_type} index {object_index}")
            _, _, count, _, feature_code, _ = self.get_ring_state(object_index)
            if self.is_hole(object_index):
                a.edit_layer.name = f"Edit Hole ({count} points, id={object_index})"
            else:
                a.edit_layer.name = f"Edit Polygon ({count} points, id={object_index})"
                desc.append("add_polygon_hole")
            desc.extend([None, "simplify_polygon", None, "delete_polygon"])
        desc.append("add_polygon_boundary")
        return desc
