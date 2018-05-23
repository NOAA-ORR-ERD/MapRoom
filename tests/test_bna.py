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
        print(layer.points)
        layer.check_for_problems(None)
    
    def test_save(self):
        loaders.save_layer(self.bna, "test.bna")
        self.project.load_file("test.bna", "application/x-maproom-bna")
        self.orig = self.bna
        self.bna = self.project.layer_manager.get_layer_by_invariant(2)
        assert self.orig != self.bna
        self.test_simple()

class TestBNAFailures(object):
    def setup(self):
        self.project = MockProject()

    def test_bad_missing_points(self):
        cmd = self.project.load_file("../TestData/BNA/bad--missing_polygon_points.bna", "application/x-maproom-bna")
        bna = self.project.layer_manager.get_layer_by_invariant(1)
        assert bna == None
        assert "line with 2 items" in cmd.undo_info.flags.errors[0]
        assert "scale" in str(self.project.layer_manager.flatten()).lower()
        assert "polygon" not in str(self.project.layer_manager.flatten()).lower()

    def test_extra_points(self):
        cmd = self.project.load_file("../TestData/BNA/bad--extra_polygon_points.bna", "application/x-maproom-bna")
        bna = self.project.layer_manager.get_layer_by_invariant(1)
        assert bna == None
        assert "line with 3 items" in cmd.undo_info.flags.errors[0]
        assert "scale" in str(self.project.layer_manager.flatten()).lower()
        assert "polygon" not in str(self.project.layer_manager.flatten()).lower()



if __name__ == "__main__":
    t = TestBNA()
    t.setup()
    t.test_simple()
