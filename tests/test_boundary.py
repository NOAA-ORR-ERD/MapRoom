import os

import numpy as np

from mock import *

from maproom.library import Boundary as b
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
        boundaries = b.Boundaries(layer)
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
        boundaries = b.Boundaries(layer)
        error_points = boundaries.check_boundary_crossings()
        print(error_points)
        assert set(error_points) == set((14829, 14830, 14831, 5979, 5980, 5981, 5982, 14823, 14824, 14825, 9898, 9899, 9901, 9902, 9903, 9907, 9908, 9909, 9910, 9911, 9912, 14826))


class TestBoundary(object):
    def setup(self):
        self.points = data_types.make_points(24)
        x=np.linspace(0.,3.,4)
        y=np.linspace(0.,5,6)
        xv,yv = np.meshgrid(x,y)
        self.points['x'] = xv.ravel()
        self.points['y'] = yv.ravel()
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

        boundaries = b.Boundaries(layer)
        print(boundaries)
        assert len(boundaries.boundaries) == 2
        assert len(boundaries.polylines) == 1
        outer = boundaries.get_outer_boundary().point_indexes
        print(outer)
        assert list(outer) == [5, 6, 7, 8]


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

        boundaries = b.Boundaries(layer)
        print(boundaries)
        print(boundaries.get_outer_boundary().point_indexes)
        assert len(boundaries.boundaries) == 1
        assert len(boundaries.polylines) == 4
        assert list(boundaries.get_outer_boundary().point_indexes) == [1,2,3,7,6,5]


if __name__ == "__main__":
    import time
    
    t = TestBoundary()
    t.setup()
    # t.test_polylines()
    t.test_complete_polygons()

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
    
