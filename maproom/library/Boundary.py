import random
import numpy as np
from maproom.library.Shape import point_in_polygon


class Find_boundaries_error(Exception):

    def __init__(self, message, points=None):
        Exception.__init__(self, message)
        self.points = points


def find_boundaries(points, point_count, lines, line_count, allow_branches=True):
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

    adjacency_map = {}
    non_boundary_points = set(xrange(0, point_count))

    # Build up a map of adjacency lists: point index -> list of indexes
    # of adjacent points connected by line segments.
    for line_index in xrange(line_count):
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
    
    if not allow_branches:
        branch_points = set()
        for point, adjacent in adjacency_map.iteritems():
            if len(adjacent) > 2:
                branch_points.add(point)
                for a in adjacent:
                    branch_points.add(a)
        if len(branch_points) > 0:
            raise Find_boundaries_error("Branching boundaries are not supported in Verdat files.",
                                        points=tuple(branch_points))
    
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
                adjacent = adjacency_map.pop(adjacent_point)

            # See http://alienryderflex.com/polygon_area/
            area += \
                ( points.x[ previous_point ] + points.x[ adjacent_point ] ) * \
                (points.y[previous_point] - points.y[adjacent_point])

            # If the adjacent point is the first point in the boundary,
            # the boundary is now closed and we're done with it.
            if adjacent_point == boundary[0]:
                break

        boundaries.append((boundary, 0.5 * area))

    # Find the outer boundary that contains all the other boundaries.
    # Determine this by simply selecting the boundary with the biggest
    # interior area.
    outer_boundary = None
    outer_boundary_index = None
    outer_boundary_area = None

    for (index, (boundary, area)) in enumerate(boundaries):
        if outer_boundary_area is None or abs(area) > abs(outer_boundary_area):
            outer_boundary = boundary
            outer_boundary_index = index
            outer_boundary_area = area

    # Make the outer boundary first in the list of boundaries if it's not
    # there already.
    if outer_boundary_index is not None and outer_boundary_index != 0:
        del(boundaries[outer_boundary_index])
        boundaries.insert(0, (outer_boundary, outer_boundary_area))

    return (boundaries, non_boundary_points)

MAX_SEARCH_COUNT = 10000


def generate_inside_hole_point(boundary, points):
    """
        bounday = a boundary point index list as returned from find_boundaries() above
        points = a numpy array of points with at least .x and .y fields
    """
    boundary_size = len(boundary)
    inside = False
    search_count = 0

    while inside is False:
        # pick three random boundary points and take the average of their coordinates
        (point1, point2, point3) = random.sample(boundary, 3)

        candidate_x = (points.x[point1] + points.x[point2] + points.x[point3]) / 3.0
        candidate_y = (points.y[point1] + points.y[point2] + points.y[point3]) / 3.0

        inside = point_in_polygon(
            points_x=points.x,
            points_y=points.y,
            point_count=len(points),
            polygon=np.array(boundary, np.uint32),
            x=candidate_x,
            y=candidate_y
        )

        search_count += 1
        if search_count > MAX_SEARCH_COUNT:
            raise Find_boundaries_error("Cannot find an inner boundary hole for triangulation.")

    return (candidate_x, candidate_y)


def generate_outside_hole_point(boundary, points):
    """
        bounday = a boundary point index list as returned from find_boundaries() above
        points = a numpy array of points with at least .x and .y fields
    """
    boundary_size = len(boundary)
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
            polygon=np.array(boundary, np.uint32),
            x=candidate_x,
            y=candidate_y
        )

        search_count += 1
        if search_count > MAX_SEARCH_COUNT:
            raise Find_boundaries_error("Cannot find an outer boundary hole for triangulation.")

    return (candidate_x, candidate_y)

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
    last_index = len(points) - 1
    indices = range(len(points))
    points = ([(tuple(points[i - 1]), tuple(points[i]), i) for i in indices] 
        + [(tuple(points[i]), tuple(points[i - 1]), i) for i in indices])
#    points = ([(tuple(points.x[i - 1], points.y[i-1]),
#                tuple(points.x[i], points.y[i]),
#                i) for i in indices] +
#              [(tuple(points.x[i], points.y[i]),
#                tuple(points.x[i - 1], points.y[i-1]),
#                i) for i in indices])
#    points = ([(tuple(self[i - 1]), tuple(self[i]), i) for i in indices] 
#        + [(tuple(self[i]), tuple(self[i - 1]), i) for i in indices])
    points.sort() # lexicographical sort
    open_segments = {}
    
    intersecting_segments = []

    for point in points:
        seg_start, seg_end, index = point
        if index not in open_segments:
            # Segment start point
            for open_start, open_end, open_index in open_segments.values():
                # ignore adjacent edges
                if (last_index > abs(index - open_index) > 1
                    and segments_intersect(seg_start, seg_end, open_start, open_end)):
                    intersecting_segments.append(((seg_start, seg_end, index-1, index), (open_start, open_end, open_index, open_index-1)))
            open_segments[index] = point
        else:
            # Segment end point
            del open_segments[index]
    return intersecting_segments
