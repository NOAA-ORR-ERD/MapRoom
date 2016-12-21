import numpy as np
from fs.opener import opener

import fiona
from shapely.geometry import shape
from shapely.wkt import loads
from osgeo import ogr, osr

from accumulator import accumulator

def get_dataset(uri):
    """Get OGR Dataset, performing URI to filename conversion since OGR
    doesn't support URIs, only files on the local filesystem
    """

    fs, relpath = opener.parse(uri)
    print "OGR:", relpath
    print "OGR:", fs
    if not fs.hassyspath(relpath):
        raise RuntimeError("Only file URIs are supported for OGR: %s" % metadata.uri)
    file_path = fs.getsyspath(relpath)
    if file_path.startswith("\\\\?\\"):  # OGR doesn't support extended filenames
        file_path = file_path[4:]
    dataset = ogr.Open(str(file_path))

    if (dataset is None):
        return ("Unable to load the shapefile " + file_path, None)

    if (dataset.GetLayerCount() < 1):
        return ("No vector layers in shapefile " + file_path, None)

    return "", dataset

def convert_dataset(dataset):
    geometry_list = []
    layer = dataset.GetLayer()
    for feature in layer:
        geom = feature.GetGeometryRef()
        print geom
        wkt = geom.ExportToWkt()
        g = loads(wkt)
        print g
        geometry_list.append(g)
    return geometry_list


def get_fiona(uri):
    """Get fiona Dataset, performing URI to filename conversion since OGR
    doesn't support URIs, only files on the local filesystem
    """

    fs, relpath = opener.parse(uri)
    print "fiona:", relpath
    print "fiona:", fs
    if not fs.hassyspath(relpath):
        raise RuntimeError("Only file URIs are supported for OGR: %s" % metadata.uri)
    file_path = fs.getsyspath(relpath)
    if file_path.startswith("\\\\?\\"):  # OGR doesn't support extended filenames
        file_path = file_path[4:]
    source = fiona.open(str(file_path), 'r')
    print source

    if (source is None):
        return ("Unable to load the shapefile " + file_path, None)

    return "", source


def load_shapely(uri):
    geometry_list = []
    try:
        error, source = get_fiona(uri)
        for f in source:
            print f
            g = shape(f['geometry'])
            print g.geom_type, g
            geometry_list.append(g)
    except fiona.errors.DriverError, e:
        source = None
        error, dataset = get_dataset(uri)
        if not error:
            geometry_list = convert_dataset(dataset)

    if error:
        return (error, None)

    return ("", geometry_list)

def shapely_to_polygon(geom_list):
    polygon_points = accumulator(block_shape=(2,), dtype=np.float64)
    polygon_starts = accumulator(block_shape=(1,), dtype = np.uint32)
    polygon_counts = accumulator(block_shape=(1,), dtype = np.uint32)
    polygon_identifiers = []
    polygon_groups = []
    scoping_hack = [0]

    def add_polygon(geom, points, name, feature_code, group):
        if len(points) < 1:
            return
        example = points[0]
        if len(example) > 2:
            points = [(p[0], p[1]) for p in points]
        num_points = len(points)
        polygon_points.extend(points)
        polygon_starts.append(scoping_hack[0])
        polygon_counts.append(num_points)
        scoping_hack[0] += num_points
        if hasattr(geom, "polygon_identifiers"):
            pi = geom.polygon_identifiers
        else:
            pi = {
                'name': name,
                'feature_code': feature_code,
                }
        pi['geom'] = geom
        print pi
        polygon_identifiers.append(pi)
        polygon_groups.append(group)

    group = 0
    for geom in geom_list:
        feature_code = 0
        name = "shapefile"
        group += 1

        if geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                add_polygon(poly, poly.exterior.coords, poly.geom_type, feature_code, group)
                for hole in poly.interiors:
                    add_polygon(poly, hole.coords, poly.geom_type, feature_code, group)
                group += 1
        elif geom.geom_type == 'Polygon':
            add_polygon(geom, geom.exterior.coords, geom.geom_type, feature_code, group)
            for hole in geom.interiors:
                add_polygon(geom, hole.coords, geom.geom_type, feature_code, group)
        elif geom.geom_type == 'LineString':
            # polygon layer doesn't currently support lines, so fake it by
            # reversing the points and taking the line back on itself
            points = list(geom.coords)
            backwards = reversed(list(geom.coords))
            points.extend(backwards)
            add_polygon(geom, points, geom.geom_type, feature_code, group)
        elif geom.geom_type == 'Point':
            # polygon layer doesn't currently support points, so fake it by
            # creating tiny little triangles for each point
            x, y = geom.coords[0]
            polygon = [(x, y), (x + 0.0005, y + .001), (x + 0.001, y)]
            add_polygon(geom, polygon, geom.geom_type, feature_code, group)
        else:
            print 'unknown type: ', geom.geom_type

    return ("",
            np.asarray(polygon_points),
            np.asarray(polygon_starts)[:, 0],
            np.asarray(polygon_counts)[:, 0],
            polygon_identifiers, polygon_groups)
