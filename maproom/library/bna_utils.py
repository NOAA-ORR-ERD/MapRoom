import os

from sawx.filesystem import fsopen as open

import numpy as np

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


def parse_bna_file(uri):
    f = open(uri, "r")
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
        total_points += num_points

        item = [name, feature_type, feature_code, num_points, is_polygon, polygon_points]
        items.append(item)
    progress_log.info("TICK=%d" % num_lines)
    return items, total_points


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


def load_bna_file(uri, regimes=None):
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

    if regimes is None:
        regimes = [0]
    total_points *= len(regimes)
    num_polygons *= len(regimes)
    all_polygon_points = np.zeros((total_points, 2), dtype=np.float64)
    polygon_starts = np.zeros((num_polygons,), dtype=np.uint32)
    polygon_counts = np.zeros((num_polygons,), dtype=np.uint32)
    ring_identifiers = []

    polygon_index = 0
    start_index = 0
    for regime in regimes:
        for name, feature_type, feature_code, num_points, is_polygon, item_points in items:
            ring_identifiers.append(
                {'name': name,
                 'feature_code': feature_code}
            )
            last_index = start_index + num_points
            p = item_points[0:num_points]
            p[:,0] += regime
            all_polygon_points[start_index:last_index, :] = p
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
        polygon = p[0]
        count = np.alen(polygon)
        ident = p[1]
        f.write('"%s","%s", %d\n' % (ident['name'], ident['feature_code'], count + 1))  # extra point for closed polygon
        for j in range(count):
            f.write("%s,%s\n" % (polygon[j][0], polygon[j][1]))
            ticks += 1

            if (ticks % update_every) == 0:
                progress_log.info("TICK=%d" % ticks)
        # duplicate first point to create a closed polygon
        f.write("%s,%s\n" % (polygon[0][0], polygon[0][1]))
    progress_log.info("TICK=%d" % ticks)
    progress_log.info("Saved BNA")


def write_feature_list_as_bna(filename, feature_list, projection, points_per_tick=1000):
    update_every = 1000
    ticks = 0

    def write(feature_list, fh=None):
        file_point_index = 0

        def write_poly(points, geom_info, dup_first_point=True):
            nonlocal file_point_index

            if geom_info.count == "boundary":
                # using a Boundary object
                boundary = geom_info.start_index
                count = len(boundary)
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
                    count = len(index_iter)
                    first_index = index_iter[0]
                else:
                    first_index = geom_info.start_index
                    count = geom_info.count
                    index_iter = range(first_index, first_index + count)
            if dup_first_point:
                count += 1  # extra point for closed polygon

            if fh is None:
                file_point_index += count
            else:
                fh.write(f'"{geom_info.name}","{geom_info.feature_name}", {count}\n')

                # print(first_index, geom_info.count)
                for index in index_iter:
                    fh.write(f"{rx[index]},{ry[index]}\n")
                    file_point_index += 1
                    if file_point_index % points_per_tick == 0:
                        progress_log.info("Saved %d points" % file_point_index)

                # duplicate first point to create a closed polygon
                if dup_first_point:
                    fh.write(f"{rx[first_index]},{ry[first_index]}\n")

        for feature_index, feature in enumerate(feature_list):
            geom_type = feature[0]
            points = feature[1]
            if fh is not None:
                log.debug(f"writing: {geom_type}, {feature[2:]}")
            if geom_type == "Polygon":
                for geom_info in feature[2:]:
                    write_poly(points, geom_info)
            else:
                geom_info = feature[2]
                write_poly(points, geom_info, False)

        return file_point_index

    total = write(feature_list)
    progress_log.info(f"TICKS={total}")
    progress_log.info("Saving BNA...")
    with open(filename, "w") as fh:
        write(feature_list, fh)
