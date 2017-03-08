import os

import numpy as np

from mock import *

from maproom.layers import loaders, TriangleLayer
from maproom.library.Boundary import Boundaries, PointsError

class TestBNA(object):
    def setup(self):
        self.project = MockProject()
        self.project.load_file("../TestData/BNA/00003polys_000035pts.bna", "application/x-maproom-bna")
        self.bna = self.project.layer_manager.get_layer_by_invariant(1)

    def test_simple(self):
        layer = self.bna
        assert 33 == len(layer.points)
        assert 3 == len(layer.rings)
        print layer.points
        layer.check_for_problems(None)
    
    def test_save(self):
        loaders.save_layer(self.bna, "test.bna")
        self.project.load_file("test.bna", "application/x-maproom-bna")
        self.orig = self.bna
        self.bna = self.project.layer_manager.get_layer_by_invariant(2)
        assert self.orig != self.bna
        self.test_simple()

if __name__ == "__main__":
    t = TestBNA()
    t.setup()
    t.test_simple()
