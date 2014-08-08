import os

import unittest
from nose.tools import *

import numpy as np

from maproom.library.Boundary import Boundaries
from planar.polygon import Polygon
from planar.c import Polygon as cPolygon

from mock import *

class PolygonBaseTestCase(object):
    def setup(self):
        pass

    def test_triangle_is_known_simple(self):
        poly = self.Polygon([(-1,0), (1,1), (0,0)])
        assert poly.is_simple_known
        assert poly.is_simple

    def test_is_simple(self):
        poly = self.Polygon([(0,0), (-1,-1), (-2, 0), (-1, 1)])
        assert not poly.is_simple_known
        assert poly.is_simple
        assert poly.is_simple_known
        assert poly.is_simple # test cached value

    def test_not_is_simple(self):
        poly = self.Polygon([(0,0), (-1,1), (1,1), (-1,0)])
        assert not poly.is_simple_known
        assert not poly.is_simple
        assert poly.is_simple_known
        assert not poly.is_simple # test cached value

class PyPolygonTestCase(PolygonBaseTestCase, unittest.TestCase):
    from planar.vector import Vec2, Seq2
    from planar.transform import Affine
    from planar.box import BoundingBox
    from planar.polygon import Polygon

class CPolygonTestCase(PolygonBaseTestCase, unittest.TestCase):
    from planar.c import Vec2, Seq2, Affine, BoundingBox
    from planar.c import Polygon


class TestVerdatPolygonCrossing(object):
    def setup(self):
        self.manager = MockManager()

    def test_verdat_is_simple(self):
        layer = self.manager.load_first_layer("../TestData/Verdat/002385pts.verdat", "application/x-maproom-verdat")
        boundary = self.manager.get_outer_boundary(layer)
        poly = Polygon(boundary.get_xy_point_tuples())
        assert poly.is_simple

    def test_verdat_is_not_simple(self):
        layer = self.manager.load_first_layer("../TestData/Verdat/simple-boundary-crossing.verdat", "application/x-maproom-verdat")
        boundary = self.manager.get_outer_boundary(layer)
        print boundary
        poly = Polygon(boundary.get_xy_point_tuples())
        assert not poly.is_simple

    def test_verdat_is_self_intersecting(self):
        layer = self.manager.load_first_layer("../TestData/Verdat/simple-boundary-crossing.verdat", "application/x-maproom-verdat")
        boundary = self.manager.get_outer_boundary(layer)
        error_points = boundary.check_boundary_self_crossing()
        print error_points
        assert len(error_points) == 4

    def test_large_verdat_is_simple_slow(self):
        layer = self.manager.load_first_layer("../TestData/Verdat/158328pts.verdat", "application/x-maproom-verdat")
        boundary = self.manager.get_outer_boundary(layer)
        poly = Polygon(boundary.get_xy_point_tuples())
        assert poly.is_simple

    def test_large_verdat_is_simple_fast(self):
        layer = self.manager.load_first_layer("../TestData/Verdat/158328pts.verdat", "application/x-maproom-verdat")
        boundary = self.manager.get_outer_boundary(layer)
        poly = cPolygon(boundary.get_xy_point_tuples())
        assert poly.is_simple

    def test_large_verdat_is_self_intersecting(self):
        layer = self.manager.load_first_layer("../large-self-intersecting.verdat", "application/x-maproom-verdat")
        boundary = self.manager.get_outer_boundary(layer)
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
    
