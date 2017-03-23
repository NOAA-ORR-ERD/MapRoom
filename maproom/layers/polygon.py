import os
import os.path
import time
import sys
import numpy as np
from cStringIO import StringIO

from shapely.geometry import box, Polygon, MultiPolygon, LineString, MultiLineString

# Enthought library imports.
from traits.api import Int, Unicode, Any, Str, Float, Enum, Property

from ..library import rect
from ..library.projection import Projection
from ..library.Boundary import Boundary, Boundaries, PointsError
from ..library.shapely_utils import shapely_to_polygon
from ..renderer import color_floats_to_int, data_types
from ..command import UndoInfo

from point import PointLayer
from constants import *

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


class PolygonLayer(PointLayer):
    """Layer for polygons.
    
    """
    type = Str("polygon")

    mouse_mode_toolbar = Str("PolygonLayerToolBar")

    rings = Any

    point_adjacency_array = Any  # parallels the points array

    ring_identifiers = Any

    visibility_items = ["points", "polygons"]

    layer_info_panel = ["Layer name", "Polygon count"]

    selection_info_panel = []

    def __str__(self):
        try:
            points = len(self.points)
        except:
            points = 0
        try:
            rings = len(self.rings)
        except:
            rings = 0
        return "PolygonLayer %s: %d points, %d rings" % (self.name, points, rings)

    def get_info_panel_text(self, prop):
        if prop == "Polygon count":
            if self.rings is not None:
                return str(len(self.rings))
            return "0"
        return PointLayer.get_info_panel_text(self, prop)

    def empty(self):
        """
        We shouldn't allow saving of a layer with no content, so we use this method
        to determine if we can save this layer.
        """
        no_points = (self.points is None or len(self.points) == 0)
        no_rings = (self.rings is None or len(self.rings) == 0)

        return no_points and no_rings

    def visibility_item_exists(self, label):
        """Return keys for visibility dict lookups that currently exist in this layer
        """
        if label == "points":
            return self.points is not None
        if label == "polygons":
            return self.rings is not None
        raise RuntimeError("Unknown label %s for %s" % (label, self.name))

    def set_data(self, f_ring_points, f_ring_starts, f_ring_counts,
                 f_ring_identifiers, f_ring_groups=None, style=None):
        if style is None:
            self.set_layer_style_defaults()
        else:
            self.style = style
        n_points = np.alen(f_ring_points)
        self.points = self.make_points(n_points)
        if (n_points > 0):
            n_rings = np.alen(f_ring_starts)
            self.points.view(data_types.POINT_XY_VIEW_DTYPE).xy[
                0: n_points
            ] = f_ring_points
            self.rings = data_types.make_polygons(n_rings)
            self.rings.start[
                0: n_rings
            ] = f_ring_starts
            self.rings.count[
                0: n_rings
            ] = f_ring_counts
            if f_ring_groups is None:
                # if not otherwise specified, each polygon is in its own group
                self.rings.group = np.arange(n_rings)
            else:
                # grouping of rings allows for holes: the first polygon is
                # the outer boundary and subsequent rings in the group are
                # the holes
                self.rings.group = np.asarray(f_ring_groups, dtype=np.uint32)
            self.point_adjacency_array = data_types.make_point_adjacency_array(n_points)

            # set up feature code to color map
            green = color_floats_to_int(0.25, 0.5, 0, 0.75)
            blue = color_floats_to_int(0.0, 0.0, 0.5, 0.75)
            gray = color_floats_to_int(0.5, 0.5, 0.5, 0.75)
            mapbounds = color_floats_to_int(0.9, 0.9, 0.9, 0.15)
            spillable = color_floats_to_int(0.0, 0.2, 0.5, 0.15)
            color_array = np.array((0, green, blue, gray, mapbounds, spillable), dtype=np.uint32)

            total = 0
            for p in xrange(n_rings):
                c = self.rings.count[p]
                self.point_adjacency_array.ring_index[total: total + c] = p
                self.point_adjacency_array.next[total: total + c] = np.arange(total + 1, total + c + 1)
                self.point_adjacency_array.next[total + c - 1] = total
                total += c
                self.rings.color[p] = color_array[np.clip(f_ring_identifiers[p]['feature_code'], 1, len(color_array) - 1)]

            self.ring_identifiers = list(f_ring_identifiers)
            self.points.state = 0
        self.update_bounds()

    def set_data_from_boundaries(self, boundaries):
        all_points = None
        starts = []
        counts = []
        identifiers = []
        total_points = 0
        for i, b in enumerate(boundaries):
            points = b.get_xy_points()
            num_points = np.alen(points)
            if all_points is None:
                all_points = points
            else:
                all_points = np.vstack([all_points, points])
            starts.append(total_points)
            counts.append(num_points)
            total_points += num_points
            identifiers.append(
                {'name': "boundary %d" % i,
                 'feature_code': 1}
                )
        self.set_data(all_points, starts, counts, identifiers)

    def set_data_from_geometry(self, geom):
        self.load_error_string, points, starts, counts, identifiers, groups = shapely_to_polygon(geom)
        log.debug("New geometry: for %s: %s" % (self, str(points)))
        self.set_data(points, starts, counts, identifiers, groups)

    def has_boundaries(self):
        return True

    def get_all_boundaries(self):
        boundaries = []
        points = self.points.view(data_types.POINT_XY_VIEW_DTYPE).xy
        for index in range(np.alen(self.rings)):
            start = self.rings.start[index]
            count = self.rings.count[index]
            indexes = np.arange(start, start + count, dtype=np.uint32)
            b = Boundary(points, indexes, 0.0)
            boundaries.append(b)
        return boundaries

    def get_points_lines(self):
        points = self.points.view(data_types.POINT_XY_VIEW_DTYPE).xy
        all_lines = np.empty((0, 2), dtype=np.uint32)
        for index in range(np.alen(self.rings)):
            start = self.rings.start[index]
            count = self.rings.count[index]
            lines = np.empty((count, 2), dtype=np.uint32)
            lines[:,0] = np.arange(start, start + count, dtype=np.uint32)
            lines[:,1] = np.arange(start + 1, start + count + 1, dtype=np.uint32)
            lines[count - 1,1] = start
            all_lines = np.vstack([all_lines, lines])
        return points, all_lines

    def can_save_as(self):
        return True

    def polygons_to_json(self):
        return self.rings.tolist()

    def polygons_from_json(self, json_data):
        self.rings = np.array([tuple(i) for i in json_data["polygons"]], data_types.POLYGON_DTYPE).view(np.recarray)

    def adjacency_to_json(self):
        return self.point_adjacency_array.tolist()

    def adjacency_from_json(self, json_data):
        self.point_adjacency_array = np.array([tuple(i) for i in json_data['adjacency']], data_types.POLYGON_ADJACENCY_DTYPE).view(np.recarray)

    def identifiers_to_json(self):
        return self.ring_identifiers

    def identifiers_from_json(self, json_data):
        self.ring_identifiers = json_data['identifiers']

    def check_for_problems(self, window):
        problems = []
        # record log messages from the shapely package
        templog = logging.getLogger("shapely.geos")
        buf = StringIO()
        handler = logging.StreamHandler(buf)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        templog.addHandler(handler)
        for n in range(np.alen(self.rings)):
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
        if errors:
            raise PointsError(errors)

    def clear_all_ring_selections(self, mark_type=STATE_SELECTED):
        if (self.rings is not None):
            self.rings.state = self.rings.state & (0xFFFFFFFF ^ mark_type)
            self.increment_change_count()

    def select_ring(self, polygon_index, mark_type=STATE_SELECTED):
        self.rings.state[polygon_index] = self.rings.state[polygon_index] | mark_type
        self.increment_change_count()

    def deselect_ring(self, polygon_index, mark_type=STATE_SELECTED):
        self.rings.state[polygon_index] = self.rings.state[polygon_index] & (0xFFFFFFFF ^ mark_type)
        self.increment_change_count()

    def is_ring_selected(self, polygon_index, mark_type=STATE_SELECTED):
        return self.rings is not None and (self.rings.state[polygon_index] & mark_type) != 0

    def get_selected_ring_indexes(self, mark_type=STATE_SELECTED):
        if (self.rings is None):
            return []
        #
        return np.where((self.rings.state & mark_type) != 0)[0]

    def insert_line_segment(self, point_index_1, point_index_2):
        raise RuntimeError("Not implemented yet for polygon layer!")

    def can_crop(self):
        return True

    def get_ring(self, index):
        start = self.rings.start[index]
        count = self.rings.count[index]
        boundary = self.points
        points = np.c_[boundary.x[start:start + count], boundary.y[start:start + count]]
        points = np.require(points, np.float64, ["C", "OWNDATA"])
        return points, self.ring_identifiers[index]

    def iter_rings(self):
        for n in range(np.alen(self.rings)):
            poly = self.get_ring(n)
            yield poly

    def get_shapely_polygon(self, index, debug=False):
        points, ident = self.get_ring(index)
        if np.alen(points) > 2:
            poly = Polygon(points)
        else:
            poly = LineString(points)
        if debug:
            print "points tuples:", points
            print "numpy:", points.__array_interface__, points.shape, id(points), points.flags
            print "shapely polygon:", poly.bounds
        return poly

    def crop_rectangle(self, w_r):
        print "Cropping to %s" % str(w_r)

        crop_rect = box(w_r[0][0], w_r[1][1], w_r[1][0], w_r[0][1])

        class AccumulatePolygons(object):
            """Helper class to store results from stepping through the clipped
            rings
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
        for n in range(np.alen(self.rings)):
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
                            'name': '%s (cropped part #%d)' % (self.ring_identifiers[n]['name'], i + 1),
                            'feature_code': self.ring_identifiers[n]['feature_code'],
                            })
                        new_polys.add_polygon(p, ident)
                    continue
                elif not hasattr(cropped_poly, 'exterior'):
                    print "Temporarily skipping %s" % cropped_poly.geom_type
                    continue
                new_polys.add_polygon(cropped_poly, self.ring_identifiers[n])

        old_state = self.get_restore_state()
        self.set_data(new_polys.p_points,
                      np.asarray(new_polys.p_starts, dtype=np.uint32),
                      np.asarray(new_polys.p_counts, dtype=np.uint32),
                      new_polys.p_identifiers)

        undo = UndoInfo()
        undo.data = old_state
        undo.flags.refresh_needed = True
        undo.flags.items_moved = True
        lf = undo.flags.add_layer_flags(self)
        lf.layer_contents_deleted = True
        return undo

    def get_restore_state(self):
        return self.points.copy(), self.rings.copy(), self.point_adjacency_array.copy(), list(self.ring_identifiers)

    def set_state(self, params):
        self.points, self.rings, self.point_adjacency_array, self.ring_identifiers = params
        undo = UndoInfo()
        undo.flags.refresh_needed = True
        undo.flags.items_moved = True
        # Don't know if items were added or deleted, so mark both
        lf = undo.flags.add_layer_flags(self)
        lf.layer_contents_added = True
        lf.layer_contents_deleted = True
        return undo

    def rebuild_renderer(self, renderer, in_place=False):
        projected_point_data = self.compute_projected_point_data()
        renderer.set_points(projected_point_data, self.points.z, self.points.color.copy().view(dtype=np.uint8))
        renderer.set_polygons(self.rings, self.point_adjacency_array)

    def render_projected(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        log.log(5, "Rendering polygon layer!!! pick=%s" % (picker))
        # the rings
        if layer_visibility["polygons"]:
            renderer.draw_polygons(self, picker,
                                        self.rings.color,
                                        color_floats_to_int(0, 0, 0, 1.0),
                                        1)


class RNCLoaderLayer(PolygonLayer):
    """Layer for selecting RNC maps
    
    """
    type = Str("rncloader")

    mouse_mode_toolbar = Str("RNCToolBar")

    layer_info_panel = ["Polygon count"]

    def can_highlight_clickable_object(self, canvas, object_type, object_index):
        return canvas.picker.is_polygon_fill_type(object_type)

    def get_highlight_lines(self, object_type, object_index):
        points, polygon_id = self.get_ring(object_index)
        # add starting point again so the outline will be closed
        boundary = np.vstack((points, points[0]))
        return [boundary]
