import os

import numpy as np

from mock import *

from maproom.library.Boundary import Boundaries
from maproom.renderer import data_types


class TestVerdatPolygonCrossing(object):
    def setup(self):
        self.project = MockProject()

    def test_verdat_is_self_intersecting(self):
        layer = self.project.raw_load_first_layer("../TestData/Verdat/simple-boundary-crossing.verdat", "application/x-maproom-verdat")
        boundary = self.project.get_outer_boundary(layer)
        error_points = boundary.check_boundary_self_crossing()
        print(error_points)
        assert set(error_points) == set((0, 4, 5, 6))

    def test_verdat_is_overlapping(self):
        layer = self.project.raw_load_first_layer("../TestData/Verdat/separate-boundary-crossings.verdat", "application/x-maproom-verdat")
        boundaries = Boundaries(layer)
        error_points = boundaries.check_boundary_crossings()
        print(error_points)
        assert set(error_points) == set((7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20))

    @slow
    def test_large_verdat_is_self_intersecting(self):
        layer = self.project.raw_load_first_layer("../TestData/Verdat/large-self-intersecting.verdat", "application/x-maproom-verdat")
        boundary = self.project.get_outer_boundary(layer)
        error_points = boundary.check_boundary_self_crossing()
        print(error_points)
        assert set(error_points) == set((16, 13, 14, 15))

    @slow
    def test_large_verdat_is_overlapping(self):
        layer = self.project.raw_load_first_layer("../TestData/Verdat/GreenBay2222016.verdat", "application/x-maproom-verdat")
        boundaries = Boundaries(layer)
        error_points = boundaries.check_boundary_crossings()
        print(error_points)
        assert set(error_points) == set((14829, 14830, 14831, 5979, 5980, 5981, 5982, 14823, 14824, 14825, 9898, 9899, 9901, 9902, 9903, 9907, 9908, 9909, 9910, 9911, 9912, 14826))


class TestBoundary(object):
    def setup(self):
        self.points = data_types.make_points(30)
        self.points['x'] = 0.0
        self.points['y'] = 0.0
        self.points['z'] = 0.0

    def test_complete_polygons(self):
        line_indexes = list(range(len(self.points) - 1))
        rot = line_indexes[1:] + line_indexes[:1]
        lines = data_types.make_line_segment_indexes(len(line_indexes))
        i = 0
        for start, end in zip(line_indexes, rot):
            lines[i].point1 = start
            lines[i].point2 = end
            i += 1
        print(lines)
        layer = Layer()
        layer.points = self.points
        layer.line_segment_indexes = lines

        # make two polygons by creating a break
        lines[4].point2 = 0
        lines[8].point2 = 5

        # make polyline by pointing last line to unused point
        lines[-1].point2 = len(self.points) - 1
        print(lines)

        boundaries = Boundaries(layer)
        print(boundaries)

    def test_polylines(self):
        line_indexes = [
            0,1,

            1,2,
            2,3,
            1,5,
            5,6,
            6,7,
            7,3,

            7,11,
            11,15,
            8,9,
            9,10,
            13,14,
            14,11,
        ]
        lines = data_types.make_line_segment_indexes(len(line_indexes)//2)
        i = 0
        for start, end in zip(line_indexes[::2], line_indexes[1::2]):
            lines[i].point1 = start
            lines[i].point2 = end
            i += 1
        print(lines)
        layer = Layer()
        layer.points = self.points
        layer.line_segment_indexes = lines

        boundaries = Boundaries(layer)
        print(boundaries)

        # need copies of points array because it must be modified when
        # branching points are found
        point1 = lines.point1[:].astype(np.int32)
        point2 = lines.point2[:].astype(np.int32)
        scratch = np.zeros(len(lines), dtype=np.int32)
        connection_count = np.zeros(len(self.points), dtype=np.int32)
        polylines = []
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
                        # end of the line; the other end only points to the
                        # starting point
                        connection_count[other_end] -= 1
                        polylines.append(np.array(scratch[0:polyline_len]))
                        break
                    endpoint = other_end
                print(f"found polyline {polylines[-1]})")

        print(F"polyline summary: {polylines}")
        lines = data_types.make_line_segment_indexes(len(point1))
        lines.point1 = point1
        lines.point2 = point2
        layer.line_segment_indexes = lines

        boundaries = Boundaries(layer)
        print(boundaries)



if __name__ == "__main__":
    import time
    
    t = TestBoundary()
    t.setup()
    t.test_polylines()

    # t = TestVerdatPolygonCrossing()
    # t.setup()
    # t.test_verdat_is_self_intersecting()
    # t.test_verdat_is_overlapping()
    # t.test_large_verdat_is_self_intersecting()
    # t.test_large_verdat_is_overlapping()
        
#    iterations = 1
#    
#    t0 = time.clock()
#    for i in range(iterations):
#        print "loop %d" % i
#        t.test_verdat_is_self_intersecting()
#        #t.test_large_verdat_is_self_intersecting()
#    elapsed = time.clock() - t0
#    print "%d custom loops: %f" % (iterations, elapsed)
#    
#    t0 = time.clock()
#    for i in range(iterations):
#        print "loop %d" % i
#        t.test_large_verdat_is_self_intersecting()
#    elapsed = time.clock() - t0
#    print "%d custom loops: %f" % (iterations, elapsed)
    
#    t0 = time.clock()
#    for i in range(iterations):
#        print "loop %d" % i
#        t.test_large_verdat_is_simple_slow()
#    elapsed = time.clock() - t0
#    print "%d slow loops: %f" % (iterations, elapsed)
#    
#    t0 = time.clock()
#    for i in range(iterations):
#        print "loop %d" % i
#        t.test_large_verdat_is_simple_fast()
#    elapsed = time.clock() - t0
#    print "%d fast loops: %f" % (iterations, elapsed)
    
