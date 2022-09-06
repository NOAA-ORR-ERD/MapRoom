import os

import numpy as np

from mock import *

from maproom.library.point_utils import create_convex_hull

from maproom import loaders


class TestConvexHull(object):
    def setup(self):
        pass

    def load_verdat(self, uri):
        self.project = MockProject()
        self.project.load_file(uri, "application/x-maproom-verdat")
        self.verdat = self.project.layer_manager.get_nth_oldest_layer_of_type("line", 1)

    def test_simple(self):
        self.load_verdat("../TestData/Verdat/000689pts.verdat")
        print(self.verdat)
        assert 689 == len(self.verdat.points)
        print(self.verdat.points)

        layer, err = create_convex_hull([self.verdat], self.project.layer_manager)
        print(layer)

if __name__ == "__main__":
    t = TestConvexHull()
    t.setup()
    t.test_simple()
