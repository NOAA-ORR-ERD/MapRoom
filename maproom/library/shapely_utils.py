import numpy as np
from fs.opener import opener

from shapely.geometry import Polygon
from shapely.geometry import shape
from shapely.wkt import loads
from osgeo import ogr

from .accumulator import accumulator

import logging
log = logging.getLogger(__name__)


class DriverLoadFailure(RuntimeError):
    pass


def get_dataset(uri):
    """Get OGR Dataset, performing URI to filename conversion since OGR
    doesn't support URIs, only files on the local filesystem
    """

    fs, relpath = opener.parse(uri)
    if not fs.hassyspath(relpath):
        raise RuntimeError("Only file URIs are supported for OGR: %s" % uri)
    file_path = fs.getsyspath(relpath)
    if file_path.startswith("\\\\?\\"):  # OGR doesn't support extended filenames
        file_path = file_path[4:]
    dataset = ogr.Open(str(file_path))

    if (dataset is None):
        return ("Unable to load the shapefile " + file_path, None)

    if (dataset.GetLayerCount() < 1):
        return ("No vector layers in shapefile " + file_path, None)

    return "", dataset


def convert_dataset(dataset, point_list):
    geometry_list = []
    layer = dataset.GetLayer()
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None:
            continue
        wkt = geom.ExportToWkt()
        g = loads(wkt)
        if g.geom_type == "Point":
            points = list(g.coords)
            example = points[0]
            if len(example) > 2:
                points = [(p[0], p[1]) for p in points]

            point_list.extend(points)
        else:
            add_maproom_attributes_to_shapely_geom(g)
            geometry_list.append(g)
    if not geometry_list:
        raise ValueError("No GNOME particle data found.")
    return geometry_list


def get_fiona(uri):
    """Get fiona Dataset, performing URI to filename conversion since OGR
    doesn't support URIs, only files on the local filesystem
    """
    if False:
        # Still disabling fiona support until working on windows. Disabling it
        # this way rather than commenting it out means thatlinters will still
        # see fiona even though the code can never be executed
        import fiona

    if True:
        raise ImportError("fiona not found")
    fs, relpath = opener.parse(uri)
    if not fs.hassyspath(relpath):
        raise RuntimeError("Only file URIs are supported for OGR: %s" % uri)
    file_path = fs.getsyspath(relpath)
    if file_path.startswith("\\\\?\\"):  # OGR doesn't support extended filenames
        file_path = file_path[4:]
    try:
        source = fiona.open(str(file_path), 'r')
    except fiona.errors.DriverError as e:
        raise DriverLoadFailure(e)

    if (source is None):
        return ("Unable to load the shapefile " + file_path, None)

    return "", source


def load_shapely(uri):
    geometry_list = []
    point_list = accumulator(block_shape=(2,), dtype=np.float64)
    try:
        # Try fiona first
        error, source = get_fiona(uri)
        for f in source:
            g = shape(f['geometry'])
            if g.geom_type == "Point":
                points = list(geom.coords)
                example = points[0]
                if len(example) > 2:
                    points = [(p[0], p[1]) for p in points]

                point_list.extend(points)
            else:
                add_maproom_attributes_to_shapely_geom(g)
                geometry_list.append(g)
    except (DriverLoadFailure, ImportError):
        # use GDAL instead
        source = None
        error, dataset = get_dataset(uri)
        if not error:
            try:
                geometry_list = convert_dataset(dataset, point_list)
            except ValueError as e:
                error = str(e)

    if error:
        return (error, None, None)

    return ("", geometry_list, np.asarray(point_list))


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
    polygon_starts = accumulator(block_shape=(1,), dtype=np.uint32)
    polygon_counts = accumulator(block_shape=(1,), dtype=np.uint32)
    point_points = accumulator(block_shape=(2,), dtype=np.float64)
    ring_identifiers = []
    ring_groups = []
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
        ring_identifiers.append(pi)
        ring_groups.append(group)

    def add_points(points, feature_code, group):
        if len(points) < 1:
            return

        # we're only interested in 2D points
        example = points[0]
        if len(example) > 2:
            points = [(p[0], p[1]) for p in points]

        point_points.extend(points)

    group = 0
    geom_index = 0
    for geom in geom_list:
        feature_code = 0
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
            # x, y = geom.coords[0]
            # polygon = [(x, y), (x + 0.0000005, y + .000001), (x + 0.000001, y), (x, y)]
            # add_polygon(geom_index, 0, polygon, geom.geom_type, feature_code, group)
            points = list(geom.coords)
            #print "point", points
            #add_point(points, feature_code, group)
            group -= 1  # points are not in a group, so skip them
            geom_index -= 1  # skip points in the geometry index
        else:
            log.error("unknown type: %s" % str(geom.geom_type))

        geom_index += 1

    return ("",
            np.asarray(polygon_points),
            np.asarray(polygon_starts)[:,0],
            np.asarray(polygon_counts)[:,0],
            ring_identifiers, ring_groups,
            )


def rebuild_geometry_list(geometry_list, changes):
    """Shapely geometries are immutable, so we have to replace an object
    with a new instance when a polygon is edited.
    """
    for ident, points in changes:
        gi = ident['geom_index']
        geom = geometry_list[gi]
        log.debug("rebuild_geometry_list: ident=%s geom=%s" % (ident, geom))
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


def parse_geom(geom):
    item = None
    if geom.geom_type == 'MultiPolygon':
        if True:
            return None
        item = [geom.geom_type]
        for poly in geom.geoms:
            points = np.array(poly.exterior.coords[:-1], dtype=np.float64)
            sub_item = [poly.geom_type, points]
            for hole in poly.interiors:
                points = np.asarray(hole.coords[:-1], dtype=np.float64)
                sub_item.append(points)
            item.append(sub_item)
    elif geom.geom_type == 'Polygon':
        points = np.array(geom.exterior.coords[:-1], dtype=np.float64)
        item = [geom.geom_type, points]
        for hole in geom.interiors:
            points = np.asarray(hole.coords[:-1], dtype=np.float64)
            item.append(points)
    elif geom.geom_type == 'LineString':
        points = np.asarray(geom.coords, dtype=np.float64)
        item = [geom.geom_type, points]
    elif geom.geom_type == 'Point':
        points = np.asarray(geom.coords, dtype=np.float64)
        if points.shape[1] > 2:
            points = points[:,0:2].copy()
        item = [geom.geom_type, points]
    else:
        log.warning(f"Unsupported geometry type {geom.geom_type}")
    return item


def parse_fiona(source):
    # Note: all coordinate points are converted from
    # shapely.coords.CoordinateSequence to normal numpy array

    geometry_list = []
    for f in source:
        geom = shape(f['geometry'])
        item = parse_geom(geom)
        if geom is not None:
            geometry_list.append(item)
    return geometry_list

def parse_ogr(dataset):
    geometry_list = []
    layer = dataset.GetLayer()
    for feature in layer:
        ogr_geom = feature.GetGeometryRef()
        if ogr_geom is None:
            continue
        wkt = ogr_geom.ExportToWkt()
        geom = loads(wkt)
        item = parse_geom(geom)
        if item is not None:
            geometry_list.append(item)
    return geometry_list

def load_shapely2(uri):
    geometry_list = []
    try:
        # Try fiona first
        error, source = get_fiona(uri)
        geometry_list = parse_fiona(source)
    except (DriverLoadFailure, ImportError):
        # use GDAL instead
        source = None
        error, dataset = get_dataset(uri)
        if not error:
            try:
                geometry_list = parse_ogr(dataset)
            except ValueError as e:
                error = str(e)

    if error:
        return (error, None)

    return ("", geometry_list)
