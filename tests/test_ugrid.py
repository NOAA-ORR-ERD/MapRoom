import os

import numpy as np

from pyugrid.ugrid import UGrid

from mock import *

from maproom.layers import loaders, TriangleLayer
from maproom.library.Boundary import Boundaries, PointsError

class TestVerdatConversion(object):
    def setup(self):
        self.project = MockProject()
        self.project.load_file("../TestData/Verdat/negative-depth-triangles.verdat", "application/x-maproom-verdat")
        self.verdat = self.project.layer_manager.get_layer_by_invariant(1)

    def test_simple(self):
        print self.verdat
        assert 23 == np.alen(self.verdat.points)
        print self.verdat.points
        
        tris = TriangleLayer(manager=self.project.layer_manager)
        tris.triangulate_from_layer(self.verdat, None, None)
        
        loaders.save_layer(tris, "negative-depth-triangles.nc")
        
        t2 = self.project.raw_load_first_layer("negative-depth-triangles.nc", "application/x-nc_ugrid")
        print t2.points

    def test_jetty(self):
        layer = self.verdat
        assert 16 == np.alen(layer.line_segment_indexes)
        layer.insert_line_segment(2, 17)
        assert 17 == np.alen(layer.line_segment_indexes)
        
        b = Boundaries(layer, allow_branches=False)
        with pytest.raises(PointsError):
            b.check_errors(True)

class TestJetty(object):
    def setup(self):
        self.project = MockProject()
        self.project.load_file("../TestData/Verdat/jetty.verdat", "application/x-maproom-verdat")
        self.verdat = self.project.layer_manager.get_layer_by_invariant(1)
    
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
        assert 5 == np.alen(layer.line_segment_indexes)
        self.create_jetty()
        assert 16 == np.alen(layer.line_segment_indexes)
        
        b = Boundaries(layer, allow_branches=False)
        with pytest.raises(PointsError):
            b.check_errors(True)

    def test_jetty_save_as_ugrid(self):
        self.create_jetty()
        uri = "tmp.jetty.nc"
        loaders.save_layer(self.verdat, uri)
        layer = self.project.raw_load_first_layer(uri, "application/x-nc_ugrid")
        assert 16 == np.alen(layer.line_segment_indexes)

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
        assert 5 == np.alen(layer.line_segment_indexes)
        self.create_channel()
        assert 15 == np.alen(layer.line_segment_indexes)
        
        b = Boundaries(layer, allow_branches=False)
        with pytest.raises(PointsError):
            b.check_errors(True)

    def test_channel_save_as_ugrid(self):
        self.create_channel()
        uri = "tmp.channel.nc"
        loaders.save_layer(self.verdat, uri)
        layer = self.project.raw_load_first_layer(uri, "application/x-nc_ugrid")
        assert 15 == np.alen(layer.line_segment_indexes)


class TestUGrid(object):
    def setup(self):
        self.project = MockProject()
        self.layers = self.project.raw_load_all_layers("../TestData/UGrid/2_triangles.nc", "application/x-nc_ugrid")
    
    def test_load(self):
        assert 2 == len(self.layers)
        layer = self.layers[0]
        assert 'line' == layer.type
        layer = self.layers[1]
        assert 'triangle' == layer.type
    
    def test_problems(self):
        layer = self.layers[0]
        with pytest.raises(PointsError):
            layer.check_for_problems(None)
        layer = self.layers[1]
        layer.check_for_problems(None)

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
    t.test_problems()
