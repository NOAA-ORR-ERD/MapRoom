import os

from nose.tools import *

import numpy as np

from pyugrid.ugrid import UGrid


#def test_simple_read():
#	""" can it be read at all """
#	ug = UGrid.from_ncfile('files/ElevenPoints_UGRIDv0.9.nc')
#
#	assert True
#
#def test_read_nodes():
#	""" Do we get the right nodes array? """
#	ug = UGrid.from_ncfile('files/ElevenPoints_UGRIDv0.9.nc')
#
#	assert ug.nodes.shape == (11,2)
#
#	# not ideal to pull specific values out, but how else to test?
#	assert np.array_equal( ug.nodes[0,:],	 (-62.242, 12.774999) )
#	assert np.array_equal( ug.nodes[-1,:],	 (-34.911235,  29.29379) )
#
### no edge data in test file at this point
## def test_read_edges():
## 	""" Do we get the right edges array? """
## 	ug = UGrid.from_ncfile('files/ElevenPoints_UGRIDv0.9.nc')
#
## 	print ug.edges
#
## 	assert False
#
#def test_read_face_node_connectivity():
#	""" Do we get the right connectivity array? """
#	ug = UGrid.from_ncfile('files/ElevenPoints_UGRIDv0.9.nc')
#
#	assert ug.faces.shape == (13, 3)
#
#	# # not ideal to pull specific values out, but how else to test?
#	## note: file is 1-indexed, so these values are adjusted
#	assert np.array_equal( ug.faces[0,:],	 (2, 3, 10) )
#	assert np.array_equal( ug.faces[-1,:],	 (10, 5, 6) )
#
## def test_simple_read():
## 	ug = UGrid.from_ncfile('files/two_triangles.nc')
#
## 	assert False
#

from peppy2.utils.file_guess import FileGuess
from maproom.layers import loaders, TriangleLayer
from maproom.library.Boundary import find_boundaries

class MockControl(object):
    def __init__(self):
        self.projection_is_identity = True

class MockProject(object):
    def __init__(self):
        self.control = MockControl()

class MockManager(object):
    def __init__(self):
        self.project = MockProject()
    
    def dispatch_event(self, event, value=True):
        pass
    
    def add_undo_operation_to_operation_batch(self, op, layer, index, values):
        pass

class TestVerdatConversion(object):
    def setup(self):
        self.manager = MockManager()
        guess = FileGuess("../TestData/Verdat/negative-depth-triangles.verdat")
        guess.metadata.mime = "application/x-maproom-verdat"
        print guess
        print guess.metadata
        self.verdat = loaders.load_layer(guess.metadata, manager=self.manager)

    def test_simple(self):
        eq_(23, np.alen(self.verdat.points))
        print self.verdat.points
        
        tris = TriangleLayer(manager=self.manager)
        tris.triangulate_from_layer(self.verdat, None, None)
        
        loaders.save_layer(tris, "negative-depth-triangles.nc")
        
        guess = FileGuess("negative-depth-triangles.nc")
        guess.metadata.mime = "application/x-hdf"
        t2 = loaders.load_layer(guess.metadata, self.manager)
        
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
        guess = FileGuess("../TestData/Verdat/jetty.verdat")
        guess.metadata.mime = "application/x-maproom-verdat"
        print guess
        print guess.metadata
        self.verdat = loaders.load_layer(guess.metadata, manager=self.manager)
    
    def add_segments(self, point_list):
        start = point_list[0]
        for end in point_list[1:]:
            self.verdat.insert_line_segment(start, end)
            start = end

    def test_jetty(self):
        layer = self.verdat
        eq_(5, np.alen(layer.line_segment_indexes))
        segments = [
            (4, 5, 6, 7, 8, 9),
            (7, 10, 11, 12),
            (6, 13, 14, 15),
            ]
        for segment in segments:
            self.add_segments(segment)
        eq_(16, np.alen(layer.line_segment_indexes))
        
        (boundaries, non_boundary_points) = find_boundaries(
            points=layer.points,
            point_count=len(layer.points),
            lines=layer.line_segment_indexes,
            line_count=len(layer.line_segment_indexes))

    def test_channel(self):
        layer = self.verdat
        eq_(5, np.alen(layer.line_segment_indexes))
        segments = [
            (5, 6, 7, 8, 9),
            (7, 10, 11, 12),
            (6, 13, 14, 15),
            ]
        for segment in segments:
            self.add_segments(segment)
        eq_(15, np.alen(layer.line_segment_indexes))
        
        (boundaries, non_boundary_points) = find_boundaries(
            points=layer.points,
            point_count=len(layer.points),
            lines=layer.line_segment_indexes,
            line_count=len(layer.line_segment_indexes))


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
