import os

import numpy as np

from mock import *

from maproom.library.Boundary import Boundaries


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



if __name__ == "__main__":
    import time
    
    t = TestVerdatPolygonCrossing()
    t.setup()
    t.test_verdat_is_self_intersecting()
    t.test_verdat_is_overlapping()
    t.test_large_verdat_is_self_intersecting()
    t.test_large_verdat_is_overlapping()
        
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
    
