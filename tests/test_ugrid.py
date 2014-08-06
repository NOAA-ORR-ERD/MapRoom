import os

from nose.tools import *

import numpy as np

from pyugrid.ugrid import UGrid

from maproom.layers import loaders, TriangleLayer
from maproom.library.Boundary import find_boundaries

from mock import *

class TestVerdatConversion(object):
    def setup(self):
        self.manager = MockManager()
        self.verdat = self.manager.load_first_layer("../TestData/Verdat/negative-depth-triangles.verdat", "application/x-maproom-verdat")

    def test_simple(self):
        print self.verdat
        eq_(23, np.alen(self.verdat.points))
        print self.verdat.points
        
        tris = TriangleLayer(manager=self.manager)
        tris.triangulate_from_layer(self.verdat, None, None)
        
        loaders.save_layer(tris, "negative-depth-triangles.nc")
        
        t2 = self.manager.load_first_layer("negative-depth-triangles.nc", "application/x-hdf")
        print t2.points

    def test_jetty(self):
        layer = self.verdat
        eq_(16, np.alen(layer.line_segment_indexes))
        layer.insert_line_segment(2, 17)
        eq_(17, np.alen(layer.line_segment_indexes))
        
        (boundaries, non_boundary_points) = find_boundaries(
            points=layer.points,
            point_count=len(layer.points),
            lines=layer.line_segment_indexes,
            line_count=len(layer.line_segment_indexes))

class TestJetty(object):
    def setup(self):
        self.manager = MockManager()
        self.verdat = self.manager.load_first_layer("../TestData/Verdat/jetty.verdat", "application/x-maproom-verdat")
    
    def add_segments(self, point_list):
        start = point_list[0]
        for end in point_list[1:]:
            self.verdat.insert_line_segment(start, end)
            start = end

    def create_jetty(self):
        layer = self.verdat
        segments = [
            (2, 5, 6, 7, 8, 9),
            (7, 10, 11, 12),
            (6, 13, 14, 15),
            ]
        for segment in segments:
            self.add_segments(segment)

    def test_jetty(self):
        layer = self.verdat
        eq_(5, np.alen(layer.line_segment_indexes))
        self.create_jetty()
        eq_(16, np.alen(layer.line_segment_indexes))
        
        (boundaries, non_boundary_points) = find_boundaries(
            points=layer.points,
            point_count=len(layer.points),
            lines=layer.line_segment_indexes,
            line_count=len(layer.line_segment_indexes))

    def test_jetty_save_as_ugrid(self):
        self.create_jetty()
        uri = "tmp.jetty.nc"
        loaders.save_layer(self.verdat, uri)
        layer = self.manager.load_first_layer(uri, "application/x-hdf")
        eq_(16, np.alen(layer.line_segment_indexes))

    def create_channel(self):
        layer = self.verdat
        segments = [
            (5, 6, 7, 8, 9),
            (7, 10, 11, 12),
            (6, 13, 14, 15),
            ]
        for segment in segments:
            self.add_segments(segment)

    def test_channel(self):
        layer = self.verdat
        eq_(5, np.alen(layer.line_segment_indexes))
        self.create_channel()
        eq_(15, np.alen(layer.line_segment_indexes))
        
        (boundaries, non_boundary_points) = find_boundaries(
            points=layer.points,
            point_count=len(layer.points),
            lines=layer.line_segment_indexes,
            line_count=len(layer.line_segment_indexes))

    def test_channel_save_as_ugrid(self):
        self.create_channel()
        uri = "tmp.channel.nc"
        loaders.save_layer(self.verdat, uri)
        layer = self.manager.load_first_layer(uri, "application/x-hdf")
        eq_(15, np.alen(layer.line_segment_indexes))


class TestUGrid(object):
    def setup(self):
        self.manager = MockManager()
        self.layers = self.manager.load_all_layers("../TestData/UGrid/2_triangles.nc", "application/x-hdf")
    
    def test_load(self):
        eq_(2, len(self.layers))
        layer = self.layers[0]
        eq_('line', layer.type)
        layer = self.layers[1]
        eq_('triangle', layer.type)

if __name__ == "__main__":
    t = TestVerdatConversion()
    t.setup()
    t.test_simple()
    t.test_jetty()
    t = TestJetty()
    t.setup()
    t.test_channel()
    t.setup()
    t.test_jetty()
    t = TestUGrid()
    t.setup()
    t.test_load()
