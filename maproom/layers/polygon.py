import os
import os.path
import time
import sys
import numpy as np

from shapely.geometry import box, Polygon, MultiPolygon, LineString, MultiLineString

# Enthought library imports.
from traits.api import Int, Unicode, Any, Str, Float, Enum, Property

from ..library import rect
from ..library.Projection import Projection
from ..library.Boundary import Boundaries, PointsError
from ..renderer import color_to_int, data_types
from ..layer_undo import *

from point import PointLayer
from constants import *

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


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
