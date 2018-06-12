import os

import numpy as np

from pyugrid.ugrid import UGrid

from mock import *

from maproom.layers import loaders, TriangleLayer
from maproom.library.Boundary import Boundaries, PointsError

class TestBNA(object):
    def setup(self):
        self.project = MockProject()
        self.project.load_file("../TestData/BNA/00003polys_000035pts.bna", "application/x-maproom-bna")
        self.bna = self.project.layer_manager.get_layer_by_invariant(1)
        self.bna.create_rings()

    def test_simple(self):
        print(self.bna)
        print(self.bna.points)
        print(len(self.bna.points))
        assert 1034 == np.alen(self.bna.points)
        
        uri = "../tests/tmp.3polys.bna"
        loaders.save_layer(self.bna, uri)
        # layer = self.project.raw_load_first_layer(uri, "application/x-maproom-bna")
        # assert 16 == np.alen(layer.line_segment_indexes)


if __name__ == "__main__":
    t = TestBNA()
    t.setup()
    t.test_simple()
