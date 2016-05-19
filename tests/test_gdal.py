import os

import numpy as np

from mock import *

from maproom.layers import loaders, TriangleLayer
from maproom.library.Boundary import Boundaries, PointsError

class TestGDAL(object):
    def setup(self):
        self.project = MockProject()
        self.project.load_file("../TestData/ChartsAndImages/12205_7.KAP", "image/x-gdal")
        self.gdal = self.project.layer_manager.get_layer_by_invariant(1)

    def test_simple(self):
        layer = self.gdal
        assert not layer.empty()
        assert layer.get_allowable_visibility_items() == ['images']
        assert layer.visibility_item_exists('images')

if __name__ == "__main__":
    t = TestGDAL()
    t.setup()
    t.test_simple()
