import collections

import numpy as np
from sawx.filesystem import fsopen as open

from shapely.geometry import Polygon
from shapely.geometry import shape
from shapely.wkt import loads
from osgeo import ogr, osr
import pyproj

from .shapely_utils import DriverLoadFailure, get_fiona, get_dataset, add_maproom_attributes_to_shapely_geom
from .accumulator import accumulator

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


GeomInfo = collections.namedtuple('GeomInfo', 'start_index count name feature_code feature_name')

# A feature_list is a list of items, where each item is itself a list
# containing a string identifier and one or more GeomInfo objects.
#
# For example, this feature_list contains 2 entries: a polygon and a polygon
# with a hole.
#
# [
#    ['Polygon', GeomInfo(start_index=0, count=5, name='', feature_code=1, feature_name='1')],
#    ['Polygon', GeomInfo(start_index=5, count=4, name='', feature_code=1, feature_name='1'), GeomInfo(start_index=9, count=4, name='', feature_code=-1, feature_name='1')],
# ]

def calc_feature_from_geom(geom, point_list):
    item = None
    name = ""
    feature_code = 1
    feature_name = "1"
    if geom.geom_type == 'MultiPolygon':
        item = [geom.geom_type]
        for poly in geom.geoms:
            points = np.array(poly.exterior.coords[:-1], dtype=np.float64)
            index = len(point_list)
            point_list.extend(points[:,0:2])
            sub_item = [poly.geom_type, GeomInfo(index, len(points), name, feature_code, feature_name)]
            for hole in poly.interiors:
                points = np.asarray(hole.coords[:-1], dtype=np.float64)
                index = len(point_list)
                point_list.extend(points[:,0:2])
                sub_item.append(GeomInfo(index, len(points), name, -feature_code, feature_name))
            item.append(sub_item)
    elif geom.geom_type == 'Polygon':
        points = np.array(geom.exterior.coords[:-1], dtype=np.float64)
        index = len(point_list)
        point_list.extend(points[:,0:2])
        item = [geom.geom_type, GeomInfo(index, len(points), name, feature_code, feature_name)]
        for hole in geom.interiors:
            points = np.asarray(hole.coords[:-1], dtype=np.float64)
            index = len(point_list)
            point_list.extend(points[:,0:2])
            item.append(GeomInfo(index, len(points), name, -feature_code, feature_name))
    elif geom.geom_type == 'LineString':
        points = np.asarray(geom.coords, dtype=np.float64)
        index = len(point_list)
        point_list.extend(points[:,0:2])
        item = [geom.geom_type, GeomInfo(index, len(points), name, feature_code, feature_name)]
    elif geom.geom_type == 'Point':
        points = np.asarray(geom.coords, dtype=np.float64)
        index = len(point_list)
        point_list.extend(points[:,0:2])
        item = [geom.geom_type, GeomInfo(index, len(points), name, feature_code, feature_name)]
    else:
        log.warning(f"Unsupported geometry type {geom.geom_type}")
    return item


def parse_fiona(source, point_list):
    # Note: all coordinate points are converted from
    # shapely.coords.CoordinateSequence to normal numpy array

    feature_list = []
    for f in source:
        geom = shape(f['geometry'])
        item = calc_feature_from_geom(geom, point_list)
        if geom is not None:
            feature_list.append(item)
    return feature_list

def parse_ogr(dataset, point_list):
    feature_list = []
    count = dataset.GetLayerCount()
    log.debug(f"parse_ogr: {count} layers")
    for layer_index in range(count):
        layer = dataset.GetLayer(layer_index)
        log.debug(f"parse_ogr: {len(layer)} features")
        for feature in layer:
            ogr_geom = feature.GetGeometryRef()
            if ogr_geom is None:
                continue
            wkt = ogr_geom.ExportToWkt()
            geom = loads(wkt)
            item = calc_feature_from_geom(geom, point_list)
            if item is not None:
                feature_list.append(item)
    log.debug(f"parse_ogr: found {len(point_list)} points, {len(feature_list)} geom entries")
    return feature_list

def parse_from_old_json(json_data):
    feature_list = []
    point_list = accumulator(block_shape=(2,), dtype=np.float64)
    for entry in json_data:
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
        item = calc_feature_from_geom(geom, point_list)
        if item is not None:
            feature_list.append(item)
    points = np.asarray(point_list)
    return ("", feature_list, points)


def load_shapefile(uri):
    feature_list = []
    point_list = accumulator(block_shape=(2,), dtype=np.float64)
    source = None
    try:
        # Try fiona first
        error, source = get_fiona(uri)
        feature_list = parse_fiona(source, point_list)
    except (DriverLoadFailure, ImportError):
        # use GDAL instead
        error, dataset = get_dataset(uri)
        if not error:
            layer = dataset.GetLayer()
            sref = layer.GetSpatialRef()
            if sref is not None:
                try:
                    source = pyproj.Proj(sref.ExportToProj4())
                    log.debug(f"load_shapefile: source projection {source.srs}")
                except RuntimeError as e:
                    log.error(f"error in source projection: {e}")
                target = pyproj.Proj(init='epsg:4326')
                log.debug(f"load_shapefile: target projection: {target.srs}")
            else:
                log.warning(f"load_shapefile: no projection found in {uri}")
            try:
                feature_list = parse_ogr(dataset, point_list)
            except ValueError as e:
                error = str(e)
                import traceback
                print(traceback.format_exc())

    if error:
        return (error, None, None)
    points = np.asarray(point_list)
    log.debug(f"load_shapefile: {len(points)} points, {len(feature_list)} features")
    if source is not None:
        tx, ty = pyproj.transform(source, target, points[:,0], points[:,1])
        # Re-create (n,2) coordinates
        points = np.dstack([tx, ty])[0]

    return ("", feature_list, points)


def parse_bna_to_feature_list(uri, points_accumulator):
    f = open(uri, "r")
    s = f.read()
    f.close()
    lines = s.splitlines()
    feature_list = []

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
            feature_code = int(pieces[1].strip('" ,'))
        except ValueError:
            feature_code = 0
        name = pieces[0].strip('" ,')
        if name.lower() in ['map bounds', 'mapbounds']:
            feature_code = 4
        elif name.lower() in ['spillable area', 'spillablearea']:
            feature_code = 5
        feature_name = pieces[1].strip('" ,')
        num_points = int(pieces[2])

        start_index = len(points_accumulator)
        polygon_points = np.zeros((num_points, 2), dtype=np.float64)
        first_point = ()
        for j in range(num_points):
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
            polygon_points[j, :] = p
        points_accumulator.extend(polygon_points[:num_points])

        # A negative num_points value indicates that this is a line
        # rather than a polygon. And if a "polygon" only has 1 or 2
        # points, it's not a polygon.
        if num_points < 3:
            item = ['LineString', GeomInfo(start_index, num_points, name, feature_code, feature_name)]
        else:
            item = ['Polygon', GeomInfo(start_index, num_points, name, feature_code, feature_name)]
        feature_list.append(item)

    progress_log.info("TICK=%d" % num_lines)
    return feature_list


def load_bna_items(uri):
    point_list = accumulator(block_shape=(2,), dtype=np.float64)
    feature_list = parse_bna_to_feature_list(uri, point_list)

    return ("", feature_list, np.asarray(point_list))

