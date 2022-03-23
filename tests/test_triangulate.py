import os

import numpy as np

from pyugrid.ugrid import UGrid

from mock import *
from maproom import loaders
from maproom.layers import TriangleLayer
from maproom.library.Boundary import Boundaries, PointsError


class TestTriangulate(object):
    def setup(self):
        pass

    def load_verdat(self, uri):
        self.project = MockProject()
        self.project.load_file(uri, "application/x-maproom-verdat")
        self.verdat = self.project.layer_manager.get_nth_oldest_layer_of_type("line", 1)

    def test_simple(self):
        self.load_verdat("../TestData/Verdat/negative-depth-triangles.verdat")
        print(self.verdat)
        assert 23 == np.alen(self.verdat.points)
        print(self.verdat.points)

        tris = TriangleLayer(manager=self.project.layer_manager)
        tris.triangulate_from_layer(self.verdat, None, None)

        uri = os.path.join(os.getcwd(), "negative-depth-triangles.nc")
        loaders.save_layer(tris, uri)

        t2 = self.project.raw_load_first_layer("negative-depth-triangles.nc", "application/x-nc_ugrid")
        print(t2.points)
        assert 23 == np.alen(t2.points)

    def test_large(self):
        self.load_verdat("../TestData/Verdat/011795pts.verdat")
        print(self.verdat)
        assert 11795 == np.alen(self.verdat.points)

        tris = TriangleLayer(manager=self.project.layer_manager)
        tris.triangulate_from_layer(self.verdat, None, None)

        uri = os.path.join(os.getcwd(), "test011795.nc")
        loaders.save_layer(tris, uri)

        t2 = self.project.raw_load_first_layer(uri, "application/x-nc_ugrid")
        print(t2)
        assert 11795 == np.alen(t2.points)

if __name__ == "__main__":
    t = TestTriangulate()
    t.setup()
    t.test_simple()
    t.test_large()
