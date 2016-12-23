import numpy as np
from fs.opener import opener

import fiona
from shapely.geometry import shape, Polygon, MultiPolygon, LineString, Point
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
        add_maproom_attributes_to_shapely_geom(g)
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
            add_maproom_attributes_to_shapely_geom(g)
            geometry_list.append(g)
    except fiona.errors.DriverError, e:
        source = None
        error, dataset = get_dataset(uri)
        if not error:
            geometry_list = convert_dataset(dataset)

    if error:
        return (error, None)

    return ("", geometry_list)

def add_maproom_attributes_to_shapely_geom(geom, name="", feature_code=0):
    if not name:
        name = geom.geom_type
    geom.maproom_name = name
    geom.maproom_feature_code = feature_code

def copy_maproom_attributes(geom, source):
    geom.maproom_name = source.maproom_name
    geom.maproom_feature_code = source.maproom_feature_code

def shapely_to_polygon(geom_list):
    polygon_points = accumulator(block_shape=(2,), dtype=np.float64)
    polygon_starts = accumulator(block_shape=(1,), dtype = np.uint32)
    polygon_counts = accumulator(block_shape=(1,), dtype = np.uint32)
    polygon_identifiers = []
    polygon_groups = []
    total_points_scoping_hack = [0]
    ring_index_scoping_hack = [0]

    def add_polygon(geom_index, sub_index, points, name, feature_code, group):
        if len(points) < 1:
            return

        # we're only interested in 2D points
        example = points[0]
        if len(example) > 2:
            points = [(p[0], p[1]) for p in points]

        # shapely/OGR geometries need the last point to be the same as the
        # first point to complete the polygon, but OpenGL doesn't want that
        num_points = len(points) - 1
        start_index = total_points_scoping_hack[0]
        polygon_points.extend(points[:-1])
        polygon_starts.append(total_points_scoping_hack[0])
        polygon_counts.append(num_points)
        total_points_scoping_hack[0] += num_points
        source_geom = geom_list[geom_index]
        pi = {
            'name': source_geom.maproom_name,
            'feature_code': source_geom.maproom_feature_code,
            'geom_index': geom_index,
            'sub_index': sub_index,  # index of polygon inside of multipolygon
            'ring_index': int(ring_index_scoping_hack[0]),
            'point_start_index': start_index,
            'num_points': num_points
            }

        ring_index_scoping_hack[0] += 1
        polygon_identifiers.append(pi)
        polygon_groups.append(group)

    group = 0
    for geom_index, geom in enumerate(geom_list):
        feature_code = 0
        name = "shapefile"
        group += 1

        ring_index_scoping_hack[0] = 0
        if geom.geom_type == 'MultiPolygon':
            start_group = group
            for poly in geom.geoms:
                add_polygon(geom_index, group - start_group, poly.exterior.coords, poly.geom_type, feature_code, group)
                for hole in poly.interiors:
                    add_polygon(geom_index, group - start_group, hole.coords, poly.geom_type, feature_code, group)
                group += 1
        elif geom.geom_type == 'Polygon':
            add_polygon(geom_index, 0, geom.exterior.coords, geom.geom_type, feature_code, group)
            for hole in geom.interiors:
                add_polygon(geom_index, 0, hole.coords, geom.geom_type, feature_code, group)
        elif geom.geom_type == 'LineString':
            # polygon layer doesn't currently support lines, so fake it by
            # reversing the points and taking the line back on itself
            points = list(geom.coords)
            backwards = reversed(list(geom.coords))
            points.extend(backwards)
            add_polygon(geom_index, 0, points, geom.geom_type, feature_code, group)
        elif geom.geom_type == 'Point':
            # polygon layer doesn't currently support points, so fake it by
            # creating tiny little triangles for each point
            x, y = geom.coords[0]
            polygon = [(x, y), (x + 0.0005, y + .001), (x + 0.001, y)]
            add_polygon(geom_index, 0, polygon, geom.geom_type, feature_code, group)
        else:
            print 'unknown type: ', geom.geom_type

    return ("",
            np.asarray(polygon_points),
            np.asarray(polygon_starts)[:, 0],
            np.asarray(polygon_counts)[:, 0],
            polygon_identifiers, polygon_groups)

def rebuild_geometry_list(geometry_list, changes):
    """Shapely geometries are immutable, so we have to replace an object
    with a new instance when a polygon is edited.
    """
    for ident, points in changes:
        gi = ident['geom_index']
        geom = geometry_list[gi]
        print ident
        sub_index = ident['sub_index']
        ring_index = ident['ring_index']

        if geom.geom_type == 'Polygon':
            # handle single polygon with holes here
            points = list(points)
            points.append(points[0])
            exterior = geom.exterior.coords
            holes = [hole.coords for hole in geom.interiors]
            if ring_index == 0:
                exterior = points
            else:
                holes[ring_index - 1] = points
            new_geom = Polygon(exterior, holes)
        elif geom.geom_type == 'MultiPolygon':
            # handle multiple polygons, which means multiple outer boundaries
            pass
        else:
            new_geom = geom
        if new_geom != geom:
            copy_maproom_attributes(new_geom, geom)
        geometry_list[gi] = new_geom
    return geometry_list
