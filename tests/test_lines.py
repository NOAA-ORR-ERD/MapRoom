import os

import numpy as np

from mock import *

from maproom.layers import loaders, TriangleLayer
from maproom.library.Boundary import Boundaries, PointsError

class TestVerdat(object):
    def setup(self):
        self.project = MockProject()
        uri = os.path.normpath(os.getcwd() + "/../TestData/Verdat/000026pts.verdat")
        self.project.load_file(uri, "application/x-maproom-verdat")
        self.verdat = self.project.layer_manager.get_nth_oldest_layer_of_type("line", 1)

    def test_simple(self):
        print(self.verdat)
        assert 26 == np.alen(self.verdat.points)
        assert 16 == np.alen(self.verdat.line_segment_indexes)

    def test_append(self):
        uri = os.path.normpath(os.getcwd() + "/../TestData/Verdat/000011pts.verdat")
        self.project.load_file(uri, "application/x-maproom-verdat")
        layer = self.project.layer_manager.get_nth_oldest_layer_of_type("line", 2)
        self.verdat.append_points_and_lines(layer.points, layer.line_segment_indexes)
        print(self.verdat)
        assert 37 == np.alen(self.verdat.points)
        assert 25 == np.alen(self.verdat.line_segment_indexes)


if __name__ == "__main__":
    t = TestVerdat()
    t.setup()
    t.test_simple()
    t.test_append()
