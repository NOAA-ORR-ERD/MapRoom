import os
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


ext_to_driver_name = {
    ".shp": "ESRI Shapefile",
    ".json": "GeoJSON",
    ".geojson": "GeoJSON",
    ".kml": "KML",
}

shape_restriction = set(["ESRI Shapefile"])


def write_feature_list_as_shapefile(filename, feature_list, points_per_tick=1000):
    # with help from http://www.digital-geography.com/create-and-edit-shapefiles-with-python-only/
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS("WGS84")

    _, ext = os.path.splitext(filename)
    try:
        driver_name = ext_to_driver_name[ext]
    except KeyError:
        raise RuntimeError(f"Unknown shapefile extension '{ext}'")

    single_shape = driver_name in shape_restriction

    driver = ogr.GetDriverByName(driver_name)
    shapefile = driver.CreateDataSource(filename)
    log.debug(f"writing {filename}, driver={driver}, srs={srs}")

    file_point_index = 0

    def fill_ring(dest_ring, points, geom_info, dup_first_point=True):
        nonlocal file_point_index
        if geom_info.count == "boundary":
            # using a Boundary object
            boundary = geom_info.start_index
            index_iter = boundary.point_indexes
            first_index = index_iter[0]
            if dup_first_point and boundary.is_closed:
                dup_first_point = False
            # temporarily shadow x & y with boundary points
            rx, ry = boundary.points.x, boundary.points.y
        else:
            rx, ry = points.x, points.y
            if geom_info.count == "indexed":
                # using a list of point indexes
                index_iter = geom_info.start_index
                first_index = index_iter[0]
            else:
                first_index = geom_info.start_index
                index_iter = range(first_index, first_index + geom_info.count)
        # print(first_index, geom_info.count)
        for index in index_iter:
            # print(index)
            dest_ring.AddPoint(rx[index], ry[index])

            file_point_index += 1
            if file_point_index % points_per_tick == 0:
                progress_log.info("Saved %d points" % file_point_index)
        if dup_first_point:
            dest_ring.AddPoint(rx[first_index], ry[first_index])

    last_geom_type = None
    shapefile_layer = None
    for feature_index, feature in enumerate(feature_list):
        geom_type = feature[0]
        points = feature[1]
        if last_geom_type is None:
            last_geom_type = geom_type
        elif single_shape and last_geom_type != geom_type:
            raise RuntimeError(f"Only one geometry type may be saved to a {driver_name}. Starting writing {last_geom_type}, found {geom_type}")
        log.debug(f"writing: {geom_type}, {feature[1:]}")
        if geom_type == "Polygon":
            if shapefile_layer is None:
                shapefile_layer = shapefile.CreateLayer("test", srs, ogr.wkbPolygon)
            poly = ogr.Geometry(ogr.wkbPolygon)
            for geom_info in feature[2:]:
                dest_ring = ogr.Geometry(ogr.wkbLinearRing)
                fill_ring(dest_ring, points, geom_info)
                poly.AddGeometry(dest_ring)

            layer_defn = shapefile_layer.GetLayerDefn()
            f = ogr.Feature(layer_defn)
            f.SetGeometry(poly)
            f.SetFID(feature_index)
            shapefile_layer.CreateFeature(f)
            f = None
            poly = None
        elif geom_type == "LineString":
            if shapefile_layer is None:
                shapefile_layer = shapefile.CreateLayer("test", srs, ogr.wkbLineString)
            poly = ogr.Geometry(ogr.wkbLineString)
            geom_info = feature[2]
            fill_ring(poly, points, geom_info, False)
            layer_defn = shapefile_layer.GetLayerDefn()
            f = ogr.Feature(layer_defn)
            f.SetGeometry(poly)
            f.SetFID(feature_index)
            shapefile_layer.CreateFeature(f)
            f = None
            poly = None
        elif geom_type == "Point":
            raise RuntimeError("Points should be saved as particle layers")

    # ## lets add now a second point with different coordinates:
    # point.AddPoint(474598, 5429281)
    # feature_index = 1
    # feature = osgeo.ogr.Feature(layer_defn)
    # feature.SetGeometry(point)
    # feature.SetFID(feature_index)
    # layer.CreateFeature(feature)
    shapefile = None  # garbage collection = save
