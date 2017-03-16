import os
import math
import time

from fs.opener import fsopen
import numpy as np
from osgeo import gdal, gdal_array, osr
import pyproj
from shapely.geometry import Polygon, LineString

from maproom.library.accumulator import accumulator
from maproom.library.shapely_utils import add_maproom_attributes_to_shapely_geom
from maproom.layers import PolygonLayer, RNCLoaderLayer, PolygonShapefileLayer

from common import BaseLayerLoader

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")

class BNALoader(BaseLayerLoader):
    mime = "application/x-maproom-bna-old"
    
    layer_types = ["polygon"]
    
    extensions = [".bna"]
    
    name = "BNA"
    
    layer_class = PolygonLayer

    def load_layers(self, metadata, manager):
        layer = self.layer_class(manager=manager)
        
        (layer.load_error_string,
         f_ring_points,
         f_ring_starts,
         f_ring_counts,
         f_ring_identifiers) = load_bna_file(metadata.uri)
        progress_log.info("Creating layer...")
        if (layer.load_error_string == ""):
            layer.set_data(f_ring_points, f_ring_starts, f_ring_counts,
                           f_ring_identifiers)
            layer.file_path = metadata.uri
            layer.name = os.path.split(layer.file_path)[1]
            layer.mime = self.mime
        return [layer]
    
    def save_to_fh(self, fh, layer):
        save_bna_file(fh, layer)

class RNCLoader(BNALoader):
    mime = "application/x-maproom-rncloader"
    
    layer_types = ["rncloader"]
    
    extensions = [".bna"]
    
    name = "RNCLoader"
    
    layer_class = RNCLoaderLayer


class BNAShapefileLoader(BaseLayerLoader):
    mime = "application/x-maproom-bna"
    
    layer_types = ["shapefile"]
    
    extensions = [".shp", ".kml", ".json", ".geojson"]
    
    extension_desc = {
        ".shp": "ESRI Shapefile",
        ".kml": "KML",
        ".geojson": "GeoJSON",
        ".json": "GeoJSON",
    }

    name = "BNA"
    
    layer_class = PolygonShapefileLayer

    def load_layers(self, metadata, manager):
        layer = self.layer_class(manager=manager)
        
        try:
            layer.load_error_string, geometry_list, ring_identifiers = load_bna_as_shapely(metadata.uri)
        except RuntimeError, e:
            layer.load_error_string = str(e)
        progress_log.info("Creating layer...")
        if (layer.load_error_string == ""):
            layer.set_geometry(geometry_list)
            layer.file_path = metadata.uri
            layer.name = os.path.split(layer.file_path)[1]
            layer.mime = self.mime
        return [layer]
    
    def save_to_local_file(self, filename, layer):
        _, ext = os.path.splitext(filename)
        desc = self.extension_desc[ext]
        write_layer_as_shapefile(filename, layer, desc)

def parse_bna_file(uri):
    f = fsopen(uri, "r")
    s = f.read()
    f.close()
    lines = s.splitlines()
    items = []

    update_every = 1000
    total_points = 0
    i = 0
    num_lines = len(lines)
    progress_log.info("TICKS=%d" % num_lines)
    progress_log.info("Loading BNA...")
    while True:
        if (i >= num_lines):
            break
        if (i % update_every) == 0:
            progress_log.info("TICK=%d" % i)
        line = lines[i].strip()
        i += 1
        if (len(line) == 0):
            continue

        # fixme -- this will break if there are commas in any of the fields!
        pieces = line.split(",")
        if len(pieces) != 3:
            raise RuntimeError("Error at line {0}. Expecting line with 3 items: primary name, secondary name & point count.".format(i))
        try:
            feature_code = int(pieces[1].strip('"'))
        except ValueError:
            feature_code = 0
        name = pieces[0].strip('"')
        if name.lower() in ['map bounds', 'mapbounds']:
            feature_code = 4
        elif name.lower() in ['spillable area', 'spillablearea']:
            feature_code = 5
        feature_type = pieces[1].strip('"')
        num_points = int(pieces[2])

        # A negative num_points value indicates that this is a line
        # rather than a polygon. And if a "polygon" only has 1 or 2
        # points, it's not a polygon.
        is_polygon = False
        if num_points < 3:
            num_points = abs(num_points)
        else:
            is_polygon = True

        polygon_points = np.zeros((num_points, 2), dtype=np.float64)
        first_point = ()
        for j in xrange(num_points):
            line = lines[i].strip()
            i += 1
            pieces = line.split(",")
            if len(pieces) != 2:
                raise RuntimeError("Error at line {0}. Expecting line with 2 items: longitude, latitude.".format(i))
            p = (float(pieces[0]), float(pieces[1]))
            if (j == 0):
                first_point = tuple(p)
            # if the last point is a duplicate of the first point, remove it
            if (j == (num_points - 1) and p[0] == first_point[0] and p[1] == first_point[1]):
                num_points -= 1
                continue
            polygon_points[j,:] = p
        total_points += num_points

        item = [name, feature_type, feature_code, num_points, is_polygon, polygon_points]
        items.append(item)
    progress_log.info("TICK=%d" % num_lines)
    return items, total_points


def load_bna_file(uri):
    """
    used by the code below, to separate reading the file from creating the special maproom objects.
    reads the data in the file, and returns:
    
    ( load_error_string, polygon_points, polygon_starts, polygon_counts, polygon_types, ring_identifiers )
    
    where:
        load_error_string = string descripting the loading error, or "" if there was no error
        polygon_points = numpy array (type = 2 x np.float64)
        polygon_starts = numpy array (type = 1 x np.uint32)
        polygon_counts = numpy array (type = 1 x np.uint32)
        polygon_types = numpy array (type = 1 x np.uint32) (these are the BNA feature codes)
        ring_identifiers = list
    """
    items, total_points = parse_bna_file(uri)
    num_polygons = len(items)
    all_polygon_points = np.zeros((total_points, 2), dtype=np.float64)
    polygon_starts = np.zeros((num_polygons,), dtype=np.uint32)
    polygon_counts = np.zeros((num_polygons,), dtype=np.uint32)
    ring_identifiers = []

    polygon_index = 0
    start_index = 0
    for name, feature_type, feature_code, num_points, is_polygon, item_points in items:
        ring_identifiers.append(
            {'name': name,
             'feature_code': feature_code}
            )
        last_index = start_index + num_points
        all_polygon_points[start_index:last_index,:] = item_points[0:num_points,:]
        polygon_starts[polygon_index] = start_index
        polygon_counts[polygon_index] = num_points
        polygon_index += 1
        total_points += num_points
        start_index = last_index

    return ("",
            all_polygon_points,
            polygon_starts,
            polygon_counts,
            ring_identifiers)


def save_bna_file(f, layer):
    update_every = 1000
    ticks = 0
    progress_log.info("TICKS=%d" % np.alen(layer.points))
    progress_log.info("Saving BNA...")
    for i, p in enumerate(layer.iter_rings()):
        print "polygon #%d" % i
        polygon = p[0]
        count = np.alen(polygon)
        ident = p[1]
        print ident
        f.write('"%s","%s", %d\n' % (ident['name'], ident['feature_code'], count))
        for j in range(count):
            f.write("%s,%s\n" % (polygon[j][0], polygon[j][1]))
            ticks += 1
                
            if (ticks % update_every) == 0:
                progress_log.info("TICK=%d" % ticks)
    progress_log.info("TICK=%d" % ticks)
    progress_log.info("Saved BNA")

def load_bna_as_shapely(uri):
    """
    used by the code below, to separate reading the file from creating the special maproom objects.
    reads the data in the file, and returns:
    
    ( load_error_string, geometry_list )
    
    where:
        load_error_string = string descripting the loading error, or "" if there was no error
        geometry_list = list of shapely objects
    """

    items, total_points = parse_bna_file(uri)

    geometry_list = []
    ring_identifiers = []

    start_index = 0
    for name, feature_type, feature_code, num_points, is_polygon, item_points in items:
        ring_identifiers.append(
            {'name': name,
             'feature_code': feature_code}
            )
        if is_polygon:
            geom = Polygon(item_points[0:num_points])
        else:
            geom = LineString(item_points[0:num_points])
        add_maproom_attributes_to_shapely_geom(geom, name, feature_code)
        geometry_list.append(geom)

    return "", geometry_list, ring_identifiers
