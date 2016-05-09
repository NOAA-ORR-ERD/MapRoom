import os

import numpy as np

from mock import *

from maproom.layers import loaders, TriangleLayer
from maproom.library.Boundary import Boundaries, PointsError

class TestVerdat(object):
    def setup(self):
        self.project = MockProject()
        self.project.load_file("../TestData/Verdat/negative-depth-triangles.verdat", "application/x-maproom-verdat")
        self.verdat = self.project.layer_manager.get_layer_by_invariant(1)

    def test_simple(self):
        layer = self.verdat
        assert 23 == np.alen(layer.points)
        print layer.points
        layer.check_for_problems(None)

if __name__ == "__main__":
    t = TestVerdat()
    t.setup()
    t.test_simple()
