import time
import random
import numpy as np
from maproom.library.Shape import point_in_polygon, points_outside_polygon

import logging
progress_log = logging.getLogger("progress")


class PointsError(Exception):
    def __init__(self, message, points=None):
        Exception.__init__(self, message)
        self.points = points


class Boundary(object):
    # max number of iterations for finding inside/outside hole points
    MAX_SEARCH_COUNT = 10000

    def __init__(self, points, indexes, area):
        self.points = points
        self.point_indexes = indexes
        self.area = area

    def __len__(self):
        return len(self.point_indexes)

    def __getitem__(self, index):
        return self.point_indexes[index]

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

    def generate_inside_hole_point(self):
        """
            bounday = a boundary point index list as returned from find_boundaries() above
            points = a numpy array of points with at least .x and .y fields
        """
        points = self.points
        boundary_size = len(self)
        inside = False
        search_count = 0

        while inside is False:
            # pick three random boundary points and take the average of their coordinates
            (point1, point2, point3) = random.sample(self.point_indexes, 3)

            candidate_x = (points.x[point1] + points.x[point2] + points.x[point3]) / 3.0
            candidate_y = (points.y[point1] + points.y[point2] + points.y[point3]) / 3.0

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
                raise Find_boundaries_error("Cannot find an inner boundary hole for triangulation.")

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
                raise Find_boundaries_error("Cannot find an outer boundary hole for triangulation.")

        return (candidate_x, candidate_y)

    def check_boundary_self_crossing(self):
        t0 = time.clock()
        progress_log.info("Checking for boundary self-crossing...")

        error_points = set()

        # test for boundary self-crossings (i.e. making a non-simple polygon)
        boundary_points = self.get_points_list()
        intersecting_segments = self_intersection_check(boundary_points)
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

            error_points.update({self[point] for segment in intersecting_segments for item in segment for point in item[2:]})

        t = time.clock() - t0
        print "DONE WITH BOUNDARY SELF-CROSSING CHECK! %f" % t

        return tuple(error_points)

    def get_points_list(self):
        points = self.points
        return [(points.x[i], points.y[i]) for i in self]


class Boundaries(object):
    def __init__(self, layer, allow_branches=True, allow_self_crossing=True):
        self.points = layer.points
        self.point_count = len(layer.points)
        self.lines = layer.line_segment_indexes
        self.line_count = len(layer.line_segment_indexes)

        self.allow_branches = allow_branches
        self.allow_self_crossing = allow_self_crossing
        self.branch_points = []
        self.boundaries = []
        self.non_boundary_points = []
        self.find_boundaries()

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

    def raise_errors(self):
        if self.has_branches():
            raise Find_boundaries_error("Branching boundaries are not supported in Verdat files.",
                                        points=tuple(self.branch_points))

    def find_boundaries(self):
        """
            points = a numpy array of points with at least .x and .y fields
            point_count = number of points to consider
            lines = numpy array of point-to-point line segments with at least .point1 and point2 fields
                    (where point1 and point2 are indexes into the points array)
            line_count = number of line segments to consider
            
            output: ( boundaries, non_boundary_points )
                        where:
                          boundaries = a python list of ( boundary, area )
                            where:
                              boundary = a python list of ordered point indexes for the boundary
                              area = the area of the boundary (in coordinate space)
                          non_boundary_points = a python set of the points non included in a boundary
        """
        t0 = time.clock()

        progress_log.info("Determining boundaries...")
        points = self.points
        lines = self.lines

        adjacency_map = {}
        non_boundary_points = set(xrange(0, self.point_count))

        # Build up a map of adjacency lists: point index -> list of indexes
        # of adjacent points connected by line segments.
        for line_index in xrange(self.line_count):
            point1 = lines.point1[line_index]
            point2 = lines.point2[line_index]

            if point1 == point2:
                continue
            if np.isnan(points.x[point1]) or np.isnan(points.x[point2]):
                continue

            adjacent1 = adjacency_map.setdefault(point1, [])
            adjacent2 = adjacency_map.setdefault(point2, [])

            if point2 not in adjacent1:
                adjacent1.append(point2)
            if point1 not in adjacent2:
                adjacent2.append(point1)
            non_boundary_points.discard(point1)
            non_boundary_points.discard(point2)

        progress_log.info("PULSE")

        branch_points = set()
        for point, adjacent in adjacency_map.iteritems():
            if len(adjacent) > 2:
                branch_points.add(point)
                for a in adjacent:
                    branch_points.add(a)
        self.branch_points = sorted(list(branch_points))

        # find any endpoints of jetties and segments not connected to the boundary
        endpoints = []
        for point, adjacent in adjacency_map.iteritems():
            if len(adjacent) == 1:
                endpoints.append(point)
        while len(endpoints) > 0:
            endpoint = endpoints.pop()
    #        print "BEFORE REMOVING ENDPOINT %d: " % endpoint
    #        for point, adjacent in adjacency_map.iteritems():
    #            print "  point: %d  adjacent: %s" % (point, adjacent)

            # check if other points are connected to this point, otherwise we have
            # found the other end of the segment and can skip to the next endpoint
            if endpoint in adjacency_map:
                adjacent = adjacency_map[endpoint]
                other_end = adjacent[0]
                del(adjacency_map[endpoint])
                adjacent = adjacency_map[other_end]
                adjacent.remove(endpoint)
                if len(adjacent) == 0:
                    # found end of line segment
                    del(adjacency_map[other_end])
                elif len(adjacent) == 1:
                    # creating new segment end
                    endpoints.append(other_end)

        progress_log.info("PULSE")

    #    print "FINISHED REMOVING ENDPOINTS!"
    #    for point, adjacent in adjacency_map.iteritems():
    #        print "  point: %d  adjacent: %s" % (point, adjacent)

        # Walk the adjacency map to create a list of line boundaries.
        boundaries = []  # ( boundary point index list, boundary area )

        while len(adjacency_map) > 0:
            boundary = []
            area = 0.0
            previous_point = None

            # Start from an arbitrary point.
            (point, adjacent) = adjacency_map.iteritems().next()
            boundary.append(point)
            del(adjacency_map[point])

            count = 0

            while True:
                # If the first adjacent point is not the previous point, add it
                # to the boundary. Otherwise, try adding the second adjacent
                # point. If there isn't one, then the boundary isn't closed.
                if len(adjacent) == 1:
                    raise Find_boundaries_error(
                        "Only closed boundaries are supported.",
                        points=(boundary[-2], boundary[-1], )
                        if len(boundary) >= 2
                        else (boundary[0], adjacent[0]),
                    )
                elif adjacent[0] != previous_point:
                    adjacent_point = adjacent[0]
                elif adjacent[1] != previous_point:
                    adjacent_point = adjacent[1]
                elif adjacent[0] == adjacent[1]:
                    adjacent_point = adjacent[0]
                else:
                    raise Find_boundaries_error(
                        "Two points are connected by multiple line segments.",
                        points=(previous_point, ) + tuple(adjacent),
                    )

                previous_point = boundary[-1]

                if adjacent_point != boundary[0]:
                    # Delete the map as we walk through it so we know when we're
                    # done.
                    boundary.append(adjacent_point)
                    try:
                        adjacent = adjacency_map.pop(adjacent_point)
                    except KeyError:
                        # already removed point due to branch connecting
                        # multiple boundaries, so it's been handled before and
                        # we can skip the area calculation
                        break

                # See http://alienryderflex.com/polygon_area/
                area += \
                    ( points.x[ previous_point ] + points.x[ adjacent_point ] ) * \
                    (points.y[previous_point] - points.y[adjacent_point])

                # If the adjacent point is the first point in the boundary,
                # the boundary is now closed and we're done with it.
                if adjacent_point == boundary[0]:
                    break

                count += 1
                if count > 500:
                    count = 0
                    progress_log.info("PULSE")

            boundaries.append(Boundary(self.points, boundary, 0.5 * area))

        # Find the outer boundary that contains all the other boundaries.
        # Determine this by simply selecting the boundary with the biggest
        # interior area.
        outer_boundary = None
        outer_boundary_index = None

        for (index, boundary) in enumerate(boundaries):
            if outer_boundary is None or abs(boundary.area) > abs(outer_boundary.area):
                outer_boundary = boundary
                outer_boundary_index = index

        # Make the outer boundary first in the list of boundaries if it's not
        # there already.
        if outer_boundary_index is not None and outer_boundary_index != 0:
            del(boundaries[outer_boundary_index])
            boundaries.insert(0, outer_boundary)

        self.boundaries = boundaries
        self.non_boundary_points = non_boundary_points

        t = time.clock() - t0
        print "DONE WITH BOUNDARY GENERATION! %f" % t
        progress_log.info("TIME_DELTA=Boundary generation")

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

            # points must be set up for each closed-loop boundary so they don't
            # point into another boundary. The special case for i==0 exists
            # because indexes for each boundary must wrap around only within
            # that boundary.
            points = (
                [(tuple(points[num_points - 1]), tuple(points[0]), start_index, boundary_id),
                 (tuple(points[0]), tuple(points[num_points - 1]), start_index, boundary_id)]
                + [(tuple(points[i - 1]), tuple(points[i]), i + start_index, boundary_id) for i in range(1, num_points)]
                + [(tuple(points[i]), tuple(points[i - 1]), i + start_index, boundary_id) for i in range(1, num_points)]
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
        print "DONE WITH BOUNDARY CROSSING CHECK! %f" % t
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
        print "DONE WITH OUTSIDE BOUNDARY CHECK! %f" % t
        progress_log.info("TIME_DELTA=Points outside boundary")
        return outside_point_indices

    def check_errors(self, throw_exception=False):
        errors = set()
        error_points = set()

        t0 = time.clock()

        progress_log.info("Checking for branching boundaries...")

        if len(self.branch_points) > 0 and self.allow_branches == False:
            errors.add("Branching boundaries.")
            error_points.update(self.branch_points)

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
            print "error points: %s" % sorted(list(error_points))
            raise PointsError(
                "\n\n".join(errors),
                points=tuple(error_points)
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
    dir1 = (b[0] - a[0])*(c[1] - a[1]) - (c[0] - a[0])*(b[1] - a[1])
    dir2 = (b[0] - a[0])*(d[1] - a[1]) - (d[0] - a[0])*(b[1] - a[1])
    if (dir1 > 0.0) != (dir2 > 0.0) or (not dir1) != (not dir2):
        dir1 = (d[0] - c[0])*(a[1] - c[1]) - (a[0] - c[0])*(d[1] - c[1])
        dir2 = (d[0] - c[0])*(b[1] - c[1]) - (b[0] - c[0])*(d[1] - c[1])
        return ((dir1 > 0.0) != (dir2 > 0.0)
            or (not dir1) != (not dir2))
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
    indices = range(len(points))
    points = ([(tuple(points[i - 1]), tuple(points[i]), i, 0) for i in indices]
        + [(tuple(points[i]), tuple(points[i - 1]), i, 0) for i in indices])
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
    print len(points)
    progress_log.info("PULSE")
    points.sort() # lexicographical sort
    open_segments = {}

    intersecting_segments = []

    count = 0
    for point in points:
        seg_start, seg_end, index, seg_id = point
        if index not in open_segments:
            # Segment start point
            for open_start, open_end, open_index, open_id in open_segments.values():
                # ignore adjacent edges (note that the index number wraps
                # around within each closed-loop boundary)
                if (((seg_id != open_id) or ((boundary_min_max[seg_id][1] - boundary_min_max[seg_id][0]) > abs(index - open_index) > 1))
                    and segments_intersect(seg_start, seg_end, open_start, open_end)):
                    seg_prev_index = index-1
                    if seg_prev_index < boundary_min_max[seg_id][0]:
                        seg_prev_index = boundary_min_max[seg_id][1]
                    open_prev_index = open_index-1
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
