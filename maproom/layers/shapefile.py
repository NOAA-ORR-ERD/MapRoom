
from shapely.wkt import loads

# Enthought library imports.
from traits.api import List
from traits.api import Str

from ..library import rect
from ..library.shapely_utils import add_maproom_attributes_to_shapely_geom
from ..library.shapely_utils import rebuild_geometry_list

from polygon import PolygonLayer
import state

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


class PolygonShapefileLayer(PolygonLayer):
    """Layer for shapely objects rendered as rings.
    
    """
    type = Str("shapefile")

    geometry = List

    layer_info_panel = ["Layer name", "Shapefile Objects", "Polygon count"]

    def __str__(self):
        num = len(self.geometry)
        return "ShapefileLayer %s: %d objects" % (self.name, num)

    def get_info_panel_text(self, prop):
        if prop == "Shapefile Objects":
            return str(len(self.geometry))
        return PolygonLayer.get_info_panel_text(self, prop)

    def empty(self):
        """
        We shouldn't allow saving of a layer with no content, so we use this method
        to determine if we can save this layer.
        """
        return len(self.geometry) == 0

    def set_layer_style_defaults(self):
        self.style.use_next_default_color()
        self.style.line_width = 1

    def compute_bounding_rect(self, mark_type=state.CLEAR):
        bounds = rect.NONE_RECT

        if (len(self.geometry) > 0):
            for o in self.geometry:
                l, b, r, t = o.bounds
                bounds = rect.accumulate_rect(bounds, ((l, b), (r, t)))

        return bounds

    def points_to_json(self):
        # dummy value; geometry is used to restore polygon data
        return False

    def points_from_json(self, json_data):
        pass

    def polygons_to_json(self):
        # dummy value; geometry is used to restore polygon data
        return False

    def polygons_from_json(self, json_data):
        pass

    def geometry_to_json(self):
        wkt = []
        for geom in self.geometry:
            entry = ("v2", geom.maproom_name, geom.maproom_feature_code, geom.wkt)
            wkt.append(entry)
        return wkt

    def geometry_from_json(self, json_data):
        geom_list = []
        for entry in json_data['geometry']:
            if entry[0] == "v2":
                name = entry[1]
                feature_code = entry[2]
                wkt = entry[3]
            else:
                name = ""
                feature_code = 0
                wkt = entry
            geom = loads(wkt)
            add_maproom_attributes_to_shapely_geom(geom, name, feature_code)
            geom_list.append(geom)
        self.set_geometry(geom_list)

    def set_geometry(self, geom):
        self.geometry = geom
        self.set_data_from_geometry(geom)

    def get_geometry_from_object_index(self, object_index, sub_index, ring_index):
        """Get the Shapely geometry given the polygon object index from the
        PolygonLayer metadata
        """
        # print "obj_index=%d" % object_index, self.ring_identifiers
        ident = self.ring_identifiers[object_index]
        gi = ident['geom_index']
        possible = [i for i in self.ring_identifiers if i['geom_index'] == gi]
        # print "possible matching geom_index %d" % gi, possible
        found = None
        for ident in possible:
            if sub_index == ident['sub_index'] and ring_index == ident['ring_index']:
                found = ident
        if not found:
            raise IndexError("Can't find sub_index %d, ring_index %d" % (sub_index, ring_index))

        return self.geometry[gi], found

    def get_polygons(self, object_index):
        poly, ident = self.get_geometry_from_object_index(object_index, 0, 0)
        return poly.exterior.coords, poly.interiors

    def can_highlight_clickable_object(self, canvas, object_type, object_index):
        return canvas.picker.is_polygon_fill_type(object_type)

    def get_highlight_lines(self, object_type, object_index):
        points, holes = self.get_polygons(object_index)
        boundaries = []
        boundaries.append(points)
        for hole in holes:
            boundaries.append(hole.coords)
        return boundaries

    def rebuild_geometry_from_points(self, object_type, object_index, new_points):
        new_geoms = rebuild_geometry_list(self.geometry, new_points)
        self.set_geometry(new_geoms)
