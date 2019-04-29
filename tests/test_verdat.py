import os

import numpy as np

from mock import *

from maproom import loaders
from maproom.library.Boundary import Boundaries, PointsError

class TestVerdat(object):
    def setup(self):
        self.project = MockProject()
        self.project.load_file("../TestData/Verdat/negative-depth-triangles.verdat", "application/x-maproom-verdat")
        self.verdat = self.project.layer_manager.get_nth_oldest_layer_of_type("line", 1)

    def test_simple(self):
        layer = self.verdat
        assert 23 == np.alen(layer.points)
        print(layer.points)
        layer.check_for_problems()
    
    def test_save(self):
        uri = os.path.join(os.getcwd(), "test.verdat")
        loaders.save_layer(self.verdat, uri)
        self.project.load_file("test.verdat", "application/x-maproom-verdat")
        self.orig = self.verdat
        self.verdat = self.project.layer_manager.get_nth_oldest_layer_of_type("line", 2)
        assert self.orig != self.verdat
        self.test_simple()

if __name__ == "__main__":
    t = TestVerdat()
    t.setup()
    t.test_simple()
    t.test_save()
