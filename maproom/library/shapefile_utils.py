import numpy as np
from fs.opener import opener, fsopen

from shapely.geometry import Polygon
from shapely.geometry import shape
from shapely.wkt import loads
from osgeo import ogr

from .shapely_utils import DriverLoadFailure, get_fiona, get_dataset
from .accumulator import accumulator

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


def parse_geom(geom, point_list):
    item = None
    name = ""
    feature_code = 1
    feature_name = "1"
    if geom.geom_type == 'MultiPolygon':
        if True:
            return None
        item = [geom.geom_type]
        for poly in geom.geoms:
            points = np.array(poly.exterior.coords[:-1], dtype=np.float64)
            index = len(point_list)
            point_list.extend(points)
            sub_item = [poly.geom_type, (index, len(points), name, feature_code, feature_name)]
            for hole in poly.interiors:
                points = np.asarray(hole.coords[:-1], dtype=np.float64)
                index = len(point_list)
                point_list.extend(points)
                sub_item.append((index, len(points), name, feature_code, feature_name))
            item.append(sub_item)
    elif geom.geom_type == 'Polygon':
        points = np.array(geom.exterior.coords[:-1], dtype=np.float64)
        index = len(point_list)
        point_list.extend(points)
        item = [geom.geom_type, (index, len(points), name, feature_code, feature_name)]
        for hole in geom.interiors:
            points = np.asarray(hole.coords[:-1], dtype=np.float64)
            index = len(point_list)
            point_list.extend(points)
            item.append((index, len(points), name, feature_code, feature_name))
    elif geom.geom_type == 'LineString':
        points = np.asarray(geom.coords, dtype=np.float64)
        index = len(point_list)
        point_list.extend(points)
        item = [geom.geom_type, (index, len(points), name, feature_code, feature_name)]
    elif geom.geom_type == 'Point':
        points = np.asarray(geom.coords, dtype=np.float64)
        index = len(point_list)
        point_list.extend(points)
        item = [geom.geom_type, (index, len(points), name, feature_code, feature_name)]
    else:
        log.warning(f"Unsupported geometry type {geom.geom_type}")
    return item


def parse_fiona(source, point_list):
    # Note: all coordinate points are converted from
    # shapely.coords.CoordinateSequence to normal numpy array

    geometry_list = []
    for f in source:
        geom = shape(f['geometry'])
        item = parse_geom(geom, point_list)
        if geom is not None:
            geometry_list.append(item)
    return geometry_list

def parse_ogr(dataset, point_list):
    geometry_list = []
    layer = dataset.GetLayer()
    for feature in layer:
        ogr_geom = feature.GetGeometryRef()
        if ogr_geom is None:
            continue
        wkt = ogr_geom.ExportToWkt()
        geom = loads(wkt)
        item = parse_geom(geom, point_list)
        if item is not None:
            geometry_list.append(item)
    return geometry_list


def load_shapefile(uri):
    geometry_list = []
    point_list = accumulator(block_shape=(2,), dtype=np.float64)
    point_list.append((np.nan, np.nan))  # zeroth point is a NaN so it can be used for deleted points
    try:
        # Try fiona first
        error, source = get_fiona(uri)
        geometry_list = parse_fiona(source, point_list)
    except (DriverLoadFailure, ImportError):
        # use GDAL instead
        source = None
        error, dataset = get_dataset(uri)
        if not error:
            try:
                geometry_list = parse_ogr(dataset, point_list)
            except ValueError as e:
                error = str(e)

    if error:
        return (error, None, None)

    # extra 1000 pts at the end to prevent resizing too often
    extra_space_for_new_points = np.full((1000,2), np.nan)
    point_list.extend(extra_space_for_new_points)

    return ("", geometry_list, np.asarray(point_list))


def parse_bna_file2(uri, points_accumulator, regime=0):
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
        feature_name = pieces[1].strip('"')
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
        polygon_points[:,0] += regime
        points_accumulator.extend(polygon_points[:num_points])

        # A negative num_points value indicates that this is a line
        # rather than a polygon. And if a "polygon" only has 1 or 2
        # points, it's not a polygon.
        if num_points < 3:
            item = ['LineString', (start_index, num_points, name, feature_code, feature_name)]
        else:
            item = ['Polygon', (start_index, num_points, name, feature_code, feature_name)]
        items.append(item)

    progress_log.info("TICK=%d" % num_lines)
    return items


def load_bna_items(uri):
    point_list = accumulator(block_shape=(2,), dtype=np.float64)
    point_list.append((np.nan, np.nan))  # zeroth point is a NaN so it can be used for deleted points
    item_list = parse_bna_file2(uri, point_list)

    # extra 1000 pts at the end to prevent resizing too often
    extra_space_for_new_points = np.full((1000,2), np.nan)
    point_list.extend(extra_space_for_new_points)

    return ("", item_list, np.asarray(point_list))

