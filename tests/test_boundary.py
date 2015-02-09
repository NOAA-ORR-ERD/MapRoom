import os

import unittest
from nose.tools import *

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
        print error_points
        assert len(error_points) == 4

    def test_large_verdat_is_self_intersecting(self):
        layer = self.project.raw_load_first_layer("../TestData/Verdat/large-self-intersecting.verdat", "application/x-maproom-verdat")
        boundary = self.project.get_outer_boundary(layer)
        error_points = boundary.check_boundary_self_crossing()
        print error_points
        assert len(error_points) == 4



if __name__ == "__main__":
    #unittest.main()
    import time
    
    t = TestVerdatPolygonCrossing()
    t.setup()
    iterations = 1
    
    t0 = time.clock()
    for i in range(iterations):
        print "loop %d" % i
        t.test_verdat_is_self_intersecting()
        #t.test_large_verdat_is_self_intersecting()
    elapsed = time.clock() - t0
    print "%d custom loops: %f" % (iterations, elapsed)
    
    t0 = time.clock()
    for i in range(iterations):
        print "loop %d" % i
        t.test_large_verdat_is_self_intersecting()
    elapsed = time.clock() - t0
    print "%d custom loops: %f" % (iterations, elapsed)
    
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
    
