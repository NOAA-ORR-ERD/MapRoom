import os
import numpy as np
import re

from sawx.filesystem import fsopen as open

from maproom.library.accumulator import accumulator
from maproom.library.Boundary import Boundaries, PointsError

import logging
progress_log = logging.getLogger("progress")

WHITESPACE_PATTERN = re.compile("\s+")


def load_verdat_file(uri, points_per_tick=1000):
    """
    Load data from a DOGS-style verdat file. Returns:

    ( load_error_string, points, depths, line_segment_indexes, depth_unit )

    where:
        load_error_string = string descripting the loading error, or "" if there was no error
        points = numpy array (type = 2 x np.float64)
        depths = numpy array (type = 1 x np.float32)
        line_segment_indexes = numpy array (type = 2 x np.uint32)
        depth_unit = string
    """

    points = accumulator(block_shape=(2,), dtype=np.float64)
    depths = accumulator(dtype=np.float32)
    line_segment_indexes = accumulator(block_shape=(2,), dtype=np.uint32)

    in_file = open(uri, "r")
    line_num = 0
    error = ""
    warning = ""

    def get_line():
        nonlocal line_num
        line_num += 1
        return in_file.readline().strip()

    header_line = get_line()
    header = WHITESPACE_PATTERN.split(header_line)
    if (header[0] == "DOGS"):
        not_actually_header = None
    else:
        not_actually_header = header_line

    if len(header) == 2:
        depth_unit = header[1].lower()
    else:
        depth_unit = "unknown"

    # a .verdat file lists the points first, and then the boundary offsets; it is assumed that
    # the points are listed such that the first boundary polygon is defined by points 0 through i - 1,
    # the second boundary polygon is defined by points i through j - 1, and so on; and then there
    # are some number of non-boundary-polygon points at the end of the list; in this way, the boundary
    # rings can be specified simply by giving the indexes i, j, etc.

    # read the points
    try:
        while True:
            line_str = not_actually_header or get_line()
            not_actually_header = None
            line = line_str.split(",")

            data = tuple(map(float, line))
            if data == (0, 0, 0, 0):
                break
            if len(data) != 4:
                return ("The .verdat file {0} is invalid.".format(uri), None, None, None, "", None, None)

            (index, longitude, latitude, depth) = data

            points.append((longitude, latitude))
            depths.append(depth)
            if (index % line_num) == 0:
                progress_log.info("Loaded %s points" % int(index))
    except Exception as e:
        raise ValueError(str(e) + f" at line {line_num}")

    # read the boundary polygon indexes

    try:
        boundary_count = int(get_line())
    except ValueError:
        warning = "Missing polygon specifiers. Assuming all points belong to one polygon."
        boundary_indexes = [len(points)]
    else:
        boundary_indexes = []
        for i in range(boundary_count):
            try:
                end_point_number = int(get_line())
            except Exception as e:
                error = str(e) + f" at line {line_num}. Expecting {boundary_count} polygon end points, found {i}"
            else:
                boundary_indexes.append(end_point_number)

    point_index = 0
    start_point_index = 0

    for end_point_index in boundary_indexes:
        # -1 to make zero-indexed
        end_point_index -= 1
        point_index = start_point_index

        # Skip "boundaries" that are only one or two points.
        if end_point_index - start_point_index + 1 < 3:
            start_point_index = end_point_index + 1
            continue

        while point_index < end_point_index:
            line_segment_indexes.append((point_index, point_index + 1))
            point_index += 1

        # Close the boundary by connecting the first point to the last.
        line_segment_indexes.append((point_index, start_point_index))

        start_point_index = end_point_index + 1

    in_file.close()

    # import code; code.interact( local = locals() )

    return (error, warning,
            np.asarray(points),
            np.asarray(depths),
            np.asarray(line_segment_indexes),
            depth_unit)


def write_layer_as_verdat(f, layer, points_per_tick=1000):
    boundaries = Boundaries(layer, allow_branches=False)
    errors, error_points = boundaries.check_errors()
    if errors:
        raise PointsError("Layer can't be saved as Verdat:\n\n%s" % "\n\n".join(errors), error_points)

    points = layer.points

    f.write("DOGS")
    if layer.depth_unit is not None and layer.depth_unit != "unknown":
        f.write("\t{0}\n".format(layer.depth_unit.upper()))
    else:
        f.write("\n")

    boundary_endpoints = []
    POINT_FORMAT = "%3d, %4.6f, %4.6f, %3.3f\n"
    file_point_index = 1  # one-based instead of zero-based

    ticks = (boundaries.num_points() / points_per_tick) + 1
    progress_log.info("TICKS=%d" % ticks)

    # write all boundary points to the file
    # print "writing boundaries"
    for (boundary_index, boundary) in enumerate(boundaries):
        # if the outer boundary's area is positive, then reverse its
        # points so that they're wound counter-clockwise
        # print "index:", boundary_index, "area:", area, "len( boundary ):", len( boundary )
        if boundary_index == 0:
            if boundary.area > 0.0:
                boundary = reversed(boundary)
        # if any other boundary has a negative area, then reverse its
        # points so that they're wound clockwise
        elif boundary.area < 0.0:
            boundary = reversed(boundary)

        for point_index in boundary:
            f.write(POINT_FORMAT % (
                file_point_index,
                points.x[point_index],
                points.y[point_index],
                points.z[point_index],
            ))
            file_point_index += 1

            if file_point_index % VerdatLoader.points_per_tick == 0:
                progress_log.info("Saved %d points" % file_point_index)

        boundary_endpoints.append(file_point_index - 1)

    # Write non-boundary points to file.
    for point_index in boundaries.non_boundary_points:
        x = points.x[point_index]
        if np.isnan(x):
            continue

        y = points.y[point_index]
        z = points.z[point_index]

        f.write(POINT_FORMAT % (
            file_point_index,
            x, y, z,
        ))
        file_point_index += 1

        if file_point_index % VerdatLoader.points_per_tick == 0:
            progress_log.info("Saved %d points" % file_point_index)

    # zero record signals the end of the points section
    f.write(POINT_FORMAT % (0, 0.0, 0.0, 0.0))

    # write the number of boundaries, followed by each boundary endpoint index
    f.write("%d\n" % len(boundary_endpoints))

    for endpoint in boundary_endpoints:
        f.write("{0}\n".format(endpoint))

    progress_log.info("Saved verdat")
