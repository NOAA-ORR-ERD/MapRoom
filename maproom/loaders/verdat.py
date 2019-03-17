import os
import numpy as np
import re

from sawx.filesystem import fsopen as open
from sawx.utils.textutil import guessBinary

from maproom.library.accumulator import accumulator
from maproom.library.Boundary import Boundaries, PointsError

from .common import BaseLayerLoader
from maproom.layers import LineLayer

import logging
progress_log = logging.getLogger("progress")

WHITESPACE_PATTERN = re.compile("\s+")


def identify_mime(header, fh):
    is_binary = guessBinary(header)
    if not is_binary:
        if header.startswith(b"DOGS"):
            mime = "application/x-maproom-verdat"
            return dict(mime=mime, loader=VerdatLoader)


class VerdatLoader(BaseLayerLoader):
    mime = "application/x-maproom-verdat"

    layer_types = ["line"]

    extensions = [".verdat", ".dat"]

    name = "Verdat"

    points_per_tick = 5000

    def load_layers(self, metadata, manager, **kwargs):
        layer = LineLayer(manager=manager)

        progress_log.info("Loading from %s" % metadata.uri)
        (layer.load_error_string,
         f_points,
         f_depths,
         f_line_segment_indexes,
         layer.depth_unit) = load_verdat_file(metadata.uri)
        if (layer.load_error_string == ""):
            progress_log.info("Finished loading %s" % metadata.uri)
            layer.set_data(f_points, f_depths, f_line_segment_indexes)
            layer.file_path = metadata.uri
            layer.name = os.path.split(layer.file_path)[1]
            layer.mime = self.mime
        return [layer]

    def save_to_fh(self, fh, layer):
        return write_layer_as_verdat(fh, layer)


def load_verdat_file(uri):
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

    header_line = in_file.readline().strip()
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
    while True:
        line_str = not_actually_header or in_file.readline().strip()
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
        if (index % VerdatLoader.points_per_tick) == 0:
            progress_log.info("Loaded %s points" % int(index))

    # read the boundary polygon indexes

    boundary_count = int(in_file.readline())
    point_index = 0
    start_point_index = 0

    for boundary_index in range(boundary_count):
        # -1 to make zero-indexed
        end_point_index = int(in_file.readline()) - 1
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

    return ("",
            np.asarray(points),
            np.asarray(depths),
            np.asarray(line_segment_indexes),
            depth_unit)


def write_layer_as_verdat(f, layer):
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

    ticks = (boundaries.num_points() / VerdatLoader.points_per_tick) + 1
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
