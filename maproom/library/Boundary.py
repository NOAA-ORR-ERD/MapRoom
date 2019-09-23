import time
import random
import numpy as np

from ..errors import PointsError
from libmaproom.Shape import point_in_polygon, points_outside_polygon

import logging
progress_log = logging.getLogger("progress")


class Polyline:
    # max number of iterations for finding inside/outside hole points
    MAX_SEARCH_COUNT = 10000

    def __init__(self, points, indexes):
        self.points = points
        self.point_indexes = indexes
        self.length = self.calc_length()
        self.area = 0

    def calc_length(self, indexes=None):
        if indexes is None:
            indexes = self.point_indexes
        dx = np.diff(self.points.x[indexes])
        dy = np.diff(self.points.y[indexes])
        dist = np.sqrt(dx*dx + dy*dy)
        return np.sum(dist)

    def __str__(self):
        return f"{len(self.point_indexes)} points, length={self.length}"

    def __len__(self):
        return len(self.point_indexes)

    def __getitem__(self, index):
        return self.point_indexes[index]

    @property
    def is_closed(self):
        return self.point_indexes[0] == self.point_indexes[-1]

    def get_xy_point_tuples(self):
        points = self.points
        return [(points.x[i], points.y[i]) for i in self.point_indexes]

    def get_xy_points(self):
        points = self.points
        view = np.c_[points.x[self.point_indexes], points.y[self.point_indexes]]
        return view

    def get_xy_point_float64(self):
        points = self.points
        view = np.c_[points.x[self.point_indexes], points.y[self.point_indexes]]
        return view.astype(np.float64)
        return view

    def get_points(self):
        return self.points[self.point_indexes]

    def check_boundary_self_crossing(self):
        t0 = time.clock()
        progress_log.info("Checking for boundary self-crossing...")

        error_points = set()

        # test for boundary self-crossings (i.e. making a non-simple polygon)
        boundary_points = self.get_points_list()
        intersecting_segments = self_intersection_check(boundary_points)
        if len(intersecting_segments) > 0:
            # Get all the point indexes in the boundary point array:
            # >>> s = [(((1.1, 2.2), (2.2, 3.3), 15, 16), ((2.1, 3.2), (1.3, 2.3), 14, 13))]
            # >>> {segment for segment in s}
            # set([(((1.1, 2.2), (2.2, 3.3), 15, 16), ((2.1, 3.2), (1.3, 2.3), 14, 13))])
            # >>> {item for segment in s for item in segment}
            # set([((1.1, 2.2), (2.2, 3.3), 15, 16), ((2.1, 3.2), (1.3, 2.3), 14, 13)])
            # >>> {point for segment in s for item in segment for point in item}
            # set([(2.1, 3.2), (1.1, 2.2), 13, 14, 15, 16, (2.2, 3.3), (1.3, 2.3)])
            # >>> {point for segment in s for item in segment for point in item[2:]}
            # set([16, 13, 14, 15])

            error_points.update({self[point] for segment in intersecting_segments for item in segment for point in item[2:]})

        t = time.clock() - t0
        print("DONE WITH BOUNDARY SELF-CROSSING CHECK! %f" % t)

        return tuple(error_points)

    def get_points_list(self):
        points = self.points
        return [(points.x[i], points.y[i]) for i in self]


class Boundary(Polyline):
    def __init__(self, points, indexes, area=None):
        super().__init__(points, indexes)
        if area is None:
            area = self.calc_area()
        self.area = area

    def calc_length(self):
        indexes = np.empty(len(self.point_indexes) + 1, dtype=np.int32)
        indexes[0:-1] = self.point_indexes
        indexes[-1] = self.point_indexes[0]
        return Polyline.calc_length(self, indexes)

    def calc_area(self):
        # See http://alienryderflex.com/polygon_area/
        connected_indexes = np.roll(self.point_indexes, -1)
        p = self.points
        # print(f"point_indexes: {self.point_indexes}; connected {connected_indexes}")
        # x = p.x[self.point_indexes]
        # yroll = p.y[connected_indexes]
        # print(x)
        # print(yroll)
        # print(x * yroll)

        # y = p.y[self.point_indexes]
        # xroll = p.x[connected_indexes]
        # print(xroll)
        # print(y)
        # print(y * xroll)
        return 0.5 * np.dot(p.x[self.point_indexes], p.y[connected_indexes]) - np.dot(p.y[self.point_indexes], p.x[connected_indexes])

    def __str__(self):
        return f"{len(self.point_indexes)} points, area={self.area} circumference={self.length}"

    def generate_inside_hole_point(self):
        """
            bounday = a boundary point index list as returned from find_boundaries() above
            points = a numpy array of points with at least .x and .y fields
        """
        points = self.points
        inside = False
        search_count = 0

        while inside is False:
            # pick three random boundary points and take the average of their coordinates
            p = np.random.choice(self.point_indexes, 3)

            candidate_x = (points.x[p[0]] + points.x[p[1]] + points.x[p[2]]) / 3.0
            candidate_y = (points.y[p[0]] + points.y[p[1]] + points.y[p[2]]) / 3.0

            inside = point_in_polygon(
                points_x=points.x,
                points_y=points.y,
                point_count=len(points),
                polygon=np.array(self.point_indexes, np.uint32),
                x=candidate_x,
                y=candidate_y
            )

            search_count += 1
            if search_count > self.MAX_SEARCH_COUNT:
                raise PointsError("Cannot find an inner boundary hole for triangulation.")

        return (candidate_x, candidate_y)

    def generate_outside_hole_point(self):
        """
            bounday = a boundary point index list as returned from find_boundaries() above
            points = a numpy array of points with at least .x and .y fields
        """
        points = self.points
        boundary_size = len(self)
        inside = True
        search_count = 0

        while inside is True:
            # pick two consecutive boundary points, take the average of their coordinates, then perturb randomly
            point1 = random.randint(0, boundary_size - 1)
            point2 = (point1 + 1) % boundary_size

            candidate_x = (points.x[point1] + points.x[point2]) / 2.0
            candidate_y = (points.y[point1] + points.y[point2]) / 2.0

            candidate_x *= random.random() + 0.5
            candidate_y *= random.random() + 0.5

            inside = point_in_polygon(
                points_x=points.x,
                points_y=points.y,
                point_count=len(points),
                polygon=np.array(self.point_indexes, np.uint32),
                x=candidate_x,
                y=candidate_y
            )

            search_count += 1
            if search_count > self.MAX_SEARCH_COUNT:
                raise PointsError("Cannot find an outer boundary hole for triangulation.")

        return (candidate_x, candidate_y)


class Boundaries(object):
    def __init__(self, layer, allow_branches=True, allow_self_crossing=True, allow_points_outside_boundary=False, allow_polylines=True):
        self.points = layer.points
        self.point_count = len(layer.points)
        self.lines = layer.line_segment_indexes
        self.line_count = len(layer.line_segment_indexes)

        self.allow_branches = allow_branches
        self.allow_self_crossing = allow_self_crossing
        self.allow_points_outside_boundary = allow_points_outside_boundary
        self.allow_polylines = allow_polylines
        self.branch_points = []
        self.boundaries = []
        self.polylines = []
        self.non_boundary_points = []
        self.find_boundaries()

    def __str__(self):
        lines = [f"{len(self.boundaries)} boundaries ({self.point_count} points, {len(self.polylines)} polylines, {len(self.non_boundary_points)} non-boundary points, {len(self.branch_points)} branch points)"]
        lines.extend([f"  boundary {i}: {b}" for i, b in enumerate(self.boundaries)])
        lines.extend([f"  polyline {i}: {b}" for i, b in enumerate(self.polylines)])
        return "\n".join(lines)

    def __len__(self):
        return len(self.boundaries)

    def __getitem__(self, index):
        return self.boundaries[index]

    def has_branches(self):
        return len(self.branch_points) > 0

    def num_boundaries(self):
        return len(self.boundaries)

    def num_points(self):
        points = 0
        for boundary in self.boundaries:
            points += len(boundary)
        return points + len(self.non_boundary_points)

    def get_outer_boundary(self):
        if len(self) > 0:
            return self.boundaries[0]
        return None

    def extract_polylines(self, point1, point2):
        scratch = np.zeros(len(self.points), dtype=np.int32)
        connection_count = np.zeros(len(self.points), dtype=np.int32)
        self.polylines = []
        self.branch_points = []
        self.non_boundary_points = []
        first_time = True
        while True:
            connection_count[:] = 0

            # this doesn't handle duplicated point indexes
            # connection_count[lines.point1] += 1
            # connection_count[lines.point2] += 1

            # this works
            c = np.bincount(point1[point1 >= 0], minlength=(len(connection_count))).astype(np.int32)
            connection_count += c
            c = np.bincount(point2[point2 >= 0], minlength=(len(connection_count))).astype(np.int32)
            connection_count += c
            print(f"loop: connection_count={connection_count}")

            if first_time:
                self.branch_points = np.where(connection_count > 2)[0].astype(np.int32)
                self.non_boundary_points = np.where(connection_count == 0)[0].astype(np.int32)
                first_time = False

            endpoints = np.where(connection_count == 1)[0]
            print(endpoints)
            if len(endpoints) == 0:
                print(f"No more polyline endpoints")
                break

            # polylines will start at an endpoint and end when the point has 1
            # connection (ends at another endpoint) or has more than 2 connections
            # (ends at a branch point)
            for endpoint in endpoints:
                if connection_count[endpoint] == 0:
                    print(f"already checked endpoint {endpoint}")
                    continue
                else:
                    print(f"starting from {endpoint}")
                polyline_len = 0
                scratch[polyline_len] = endpoint
                polyline_len += 1
                while True:
                    # each time through this loop, we assume there's only one line
                    # segment that matches this endpoint because we start at
                    # segment that has only one other endpoint and continue until
                    # it stops or there's a branch.
                    line_index = np.where(point1 == endpoint)[0]
                    if len(line_index) > 0:
                        other_end = point2[line_index[0]]
                        used = "point1"
                    else:
                        line_index = np.where(point2 == endpoint)[0]
                        other_end = point1[line_index[0]]
                        used = "point2"
                    print(f"connecting to {other_end}")

                    scratch[polyline_len] = other_end
                    polyline_len += 1

                    # check connectivity of the other end
                    line_index1 = np.where(point1 == other_end)[0]
                    line_index2 = np.where(point2 == other_end)[0]
                    count = len(line_index1) + len(line_index2)

                    # remove line segments as we process them
                    point1[line_index] = point2[line_index] = -1
                    if count == 0:
                        # something bad happened
                        raise IndexError(f"missing endpoint connected to {endpoint}")
                    elif count == 1 or count > 2:
                        # end of the line when there's no more points to
                        # connect to or it hits a branch point
                        connection_count[other_end] -= 1
                        p = Polyline(self.points, np.array(scratch[0:polyline_len]))
                        self.polylines.append(p)
                        break
                    endpoint = other_end
                print(f"found polyline {self.polylines[-1]})")

        print(F"after polyline extraction: connection_count {connection_count}")

    def extract_polygons(self, point1, point2):
        scratch = np.zeros(len(self.lines), dtype=np.int32)
        connection_count = np.zeros(len(self.points), dtype=np.int32)
        self.boundaries = []
        while True:
            connection_count[:] = 0

            # this doesn't handle duplicated point indexes
            # connection_count[lines.point1] += 1
            # connection_count[lines.point2] += 1

            # this works
            c = np.bincount(point1[point1 >= 0], minlength=(len(connection_count))).astype(np.int32)
            connection_count += c
            c = np.bincount(point2[point2 >= 0], minlength=(len(connection_count))).astype(np.int32)
            connection_count += c
            print(f"loop: connection_count={connection_count}")

            connected_points = np.where(connection_count > 0)[0]
            print(connected_points)
            if len(connected_points) == 0:
                print(f"No more polygons")
                break

            # polygons by definition are closed, so it doesn't matter where we
            # start to trace them. Arbitrarily use the first one we find.
            start_index = connected_points[0]
            polyline_len = 0
            scratch[polyline_len] = start_index
            polyline_len += 1
            endpoint = start_index
            print(f"polygon starting at {start_index}")
            while True:
                line_index = np.where(point1 == endpoint)[0]
                if len(line_index) > 0:
                    other_end = point2[line_index[0]]
                    used = "point1"
                else:
                    line_index = np.where(point2 == endpoint)[0]
                    if len(line_index) > 0:
                        other_end = point1[line_index[0]]
                        used = "point2"
                    else:
                        print(F"no more lines; assume this polygon is closed")
                        break

                # remove line segments as we process them
                point1[line_index] = point2[line_index] = -1

                if other_end == start_index:
                    print(F"found polygon end")
                    break
                print(f"connecting to {other_end}")

                scratch[polyline_len] = other_end
                polyline_len += 1
                endpoint = other_end
            p = Boundary(self.points, np.array(scratch[0:polyline_len]))
            self.boundaries.append(p)
            print(f"found boundary {self.boundaries[-1]})")

        print(F"after polygon extraction: connection_count {connection_count}")

        # Find the outer boundary that contains all the other boundaries.
        # Determine this by simply selecting the boundary with the biggest
        # interior area.
        outer_boundary = None
        outer_boundary_index = None

        for (index, boundary) in enumerate(self.boundaries):
            if outer_boundary is None or abs(boundary.area) > abs(outer_boundary.area):
                outer_boundary = boundary
                outer_boundary_index = index

        # Make the outer boundary first in the list of boundaries if it's not
        # there already.
        if outer_boundary_index is not None and outer_boundary_index != 0:
            del(self.boundaries[outer_boundary_index])
            self.boundaries.insert(0, outer_boundary)

    def find_boundaries(self):
        # need copies of points array because it must be updated as polylines
        # and polygons are discovered.
        point1 = self.lines.point1[:].astype(np.int32)
        point2 = self.lines.point2[:].astype(np.int32)
        self.extract_polylines(point1, point2)
        self.extract_polygons(point1, point2)

    def check_boundary_crossings(self):
        t0 = time.clock()
        progress_log.info("Checking for boundary crossings...")

        error_points = set()

        point_indexes = []
        boundary_points = []
        start_index = 0
        boundary_min_max = []
        for boundary_id, boundary in enumerate(self.boundaries):
            points = boundary.get_points_list()
            point_indexes.extend(boundary.point_indexes)
            num_points = len(points)
            print(f"crossings: {num_points} points={points}\n  indexes={point_indexes}")

            # points must be set up for each closed-loop boundary so they don't
            # point into another boundary. The special case for i==0 exists
            # because indexes for each boundary must wrap around only within
            # that boundary.
            points = (
                [(tuple(points[num_points - 1]), tuple(points[0]), start_index, boundary_id),
                 (tuple(points[0]), tuple(points[num_points - 1]), start_index, boundary_id)] +
                [(tuple(points[i - 1]), tuple(points[i]), i + start_index, boundary_id) for i in range(1, num_points)] +
                [(tuple(points[i]), tuple(points[i - 1]), i + start_index, boundary_id) for i in range(1, num_points)]
            )
            boundary_points.extend(points)
            boundary_min_max.append((start_index, start_index + num_points - 1))
            start_index += num_points

        intersecting_segments = general_intersection_check(boundary_points, boundary_min_max)
        if len(intersecting_segments) > 0:
            # Get all the point indexes in the boundary point array:
            #>>> s = [(((1.1, 2.2), (2.2, 3.3), 15, 16), ((2.1, 3.2), (1.3, 2.3), 14, 13))]
            #>>> {segment for segment in s}
            #set([(((1.1, 2.2), (2.2, 3.3), 15, 16), ((2.1, 3.2), (1.3, 2.3), 14, 13))])
            #>>> {item for segment in s for item in segment}
            #set([((1.1, 2.2), (2.2, 3.3), 15, 16), ((2.1, 3.2), (1.3, 2.3), 14, 13)])
            #>>> {point for segment in s for item in segment for point in item}
            #set([(2.1, 3.2), (1.1, 2.2), 13, 14, 15, 16, (2.2, 3.3), (1.3, 2.3)])
            #>>> {point for segment in s for item in segment for point in item[2:]}
            #set([16, 13, 14, 15])

            error_points.update({point_indexes[index] for segment in intersecting_segments for item in segment for index in item[2:]})

        t = time.clock() - t0
        print("DONE WITH BOUNDARY CROSSING CHECK! %f" % t)
        progress_log.info("TIME_DELTA=Boundary crossing")

        return tuple(error_points)

    def check_outside_outer_boundary(self):
        t0 = time.clock()
        progress_log.info("Checking for points outside outer boundary...")

        if len(self) == 0:
            return []

        points = self.points
        outer_boundary = self.boundaries[0]

        # ensure that all points are within (or on) the outer boundary
        outside_point_indices = points_outside_polygon(
            points.x,
            points.y,
            point_count=len(points),
            polygon=np.array(outer_boundary, np.uint32)
        )

        t = time.clock() - t0
        print("DONE WITH OUTSIDE BOUNDARY CHECK! %f" % t)
        progress_log.info("TIME_DELTA=Points outside boundary")
        return outside_point_indices

    def check_errors(self, throw_exception=False):
        errors = set()
        error_points = set()

        progress_log.info("Checking for branching boundaries...")

        if len(self.branch_points) > 0 and not self.allow_branches:
            errors.add("Branching boundaries.")
            error_points.update(self.branch_points)

        if len(self.polylines) > 0 and not self.allow_polylines:
            errors.add("Unclosed polygons.")
            for p in self.polylines:
                error_points.update(p.point_indexes)

        if not self.allow_points_outside_boundary:
            point_indexes = self.check_outside_outer_boundary()
            if len(point_indexes) > 0:
                errors.add("Points occur outside the outer boundary.")
                error_points.update(point_indexes)

        if not self.allow_self_crossing:
            point_indexes = self.check_boundary_crossings()
            if len(point_indexes) > 0:
                errors.add("Boundary crossings.")
                error_points.update(point_indexes)

        if errors and throw_exception:
            # print "error points: %s" % sorted(list(error_points))
            raise PointsError(
                "\n\n".join(errors),
                error_points=tuple(error_points)
            )

        return errors, error_points


# from Planar, 2D geometery library: http://pypi.python.org/pypi/planar/
#############################################################################
# Copyright (c) 2010 by Casey Duncan
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name(s) of the copyright holders nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AS IS AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#############################################################################

def segments_intersect(a, b, c, d):
    """Return True if the line segment a->b intersects with
    line segment c->d
    """
    dir1 = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
    dir2 = (b[0] - a[0]) * (d[1] - a[1]) - (d[0] - a[0]) * (b[1] - a[1])
    if (dir1 > 0.0) != (dir2 > 0.0) or (not dir1) != (not dir2):
        dir1 = (d[0] - c[0]) * (a[1] - c[1]) - (a[0] - c[0]) * (d[1] - c[1])
        dir2 = (d[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (d[1] - c[1])
        return ((dir1 > 0.0) != (dir2 > 0.0) or (not dir1) != (not dir2))
    return False


def self_intersection_check(points):
    """Check the polygon for self-intersection and cache the result

    We use a simplified plane sweep algorithm. Worst case, it still takes
    O(n^2) time like a brute force intersection test, but it will typically
    be O(n log n) for common simple non-convex polygons. It should
    also quickly identify self-intersecting polygons in most cases,
    although it is slower for severely self-intersecting cases due to
    its startup cost.

    :returns: list of intersecting segments, where each entry contains
    two tuples each representing one of the intersecting segments.  Each
    tuple contains 4 items representingthe intersecting segment: a tuple
    of coordinates for the start point of the segment, a tuple containing
    coordinates of the endpoint of the segment, and the indexes into :point:
    for both the start and end point in the segment.
    """
    indices = list(range(len(points)))
    points = ([(tuple(points[i - 1]), tuple(points[i]), i, 0) for i in indices] + [(tuple(points[i]), tuple(points[i - 1]), i, 0) for i in indices])
    return general_intersection_check(points, [(0, len(indices) - 1)])


def general_intersection_check(points, boundary_min_max):
    """Check the list of polygons for intersecting lines

    We use a simplified plane sweep algorithm. Worst case, it still takes
    O(n^2) time like a brute force intersection test, but it will typically
    be O(n log n) for common simple non-convex polygons. It should
    also quickly identify self-intersecting polygons in most cases,
    although it is slower for severely self-intersecting cases due to
    its startup cost.

    :param points: list of points and line segment numbers set up in 4-tuple
    form: tuple of first coord in line, tuple of second coord of line, index
    number, and boundary id number. The boundary id number is an index into
    the boundary_min_max array below.

    :param boundary_min_max: list of tuples defining a closed-loop boundary,
    each tuple contains the min and max index for each boundary.

    :returns: list of intersecting segments, where each entry contains
    two tuples each representing one of the intersecting segments.  Each
    tuple contains 4 items representingthe intersecting segment: a tuple
    of coordinates for the start point of the segment, a tuple containing
    coordinates of the endpoint of the segment, and the indexes into :point:
    for both the start and end point in the segment.
    """
    progress_log.info("PULSE")
    points.sort()  # lexicographical sort
    open_segments = {}

    intersecting_segments = []

    count = 0
    for point in points:
        seg_start, seg_end, index, seg_id = point
        if index not in open_segments:
            # Segment start point
            for open_start, open_end, open_index, open_id in list(open_segments.values()):
                # ignore adjacent edges (note that the index number wraps
                # around within each closed-loop boundary)
                if (((seg_id != open_id) or ((boundary_min_max[seg_id][1] - boundary_min_max[seg_id][0]) > abs(index - open_index) > 1)) and segments_intersect(seg_start, seg_end, open_start, open_end)):
                    seg_prev_index = index - 1
                    if seg_prev_index < boundary_min_max[seg_id][0]:
                        seg_prev_index = boundary_min_max[seg_id][1]
                    open_prev_index = open_index - 1
                    if open_prev_index < boundary_min_max[open_id][0]:
                        open_prev_index = boundary_min_max[open_id][1]
                    intersecting_segments.append(((seg_start, seg_end, seg_prev_index, index), (open_start, open_end, open_index, open_prev_index)))
            open_segments[index] = point
        else:
            # Segment end point
            del open_segments[index]

        count += 1
        if count > 500:
            count = 0
            progress_log.info("PULSE")
    return intersecting_segments
