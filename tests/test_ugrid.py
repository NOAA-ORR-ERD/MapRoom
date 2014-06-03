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



if __name__ == "__main__":
    t = TestVerdatConversion()
    t.setup()
    t.test_simple()
