import os

import numpy as np

from mock import *

from maproom.library.Boundary import Boundaries
from maproom.layers import *


class TestVerdatToBNA(object):
    def setup(self):
        self.project = MockProject()
        self.lm = self.project.layer_manager

    def test_sample_shoreline(self):
        layer = self.project.raw_load_first_layer("../TestData/Verdat/sample shoreline.verdat", "application/x-maproom-verdat")
        layer_bounds = Boundaries(layer, True, True)
        print(layer_bounds.boundaries)
        assert 1 == len(layer_bounds.boundaries)
        assert 0 == len(layer_bounds.non_boundary_points)
        for b in layer_bounds.boundaries:
            print(b.point_indexes)
            print(b.area)
            print(b.get_xy_points())

        p = PolygonLayer(manager=self.lm)
        p.set_data_from_boundaries(layer_bounds.boundaries)

    def test_hole(self):
        layer = self.project.raw_load_first_layer("../TestData/Verdat/000026pts.verdat", "application/x-maproom-verdat")
        boundaries = layer.get_all_boundaries()
        assert 2 == len(boundaries)

        p = PolygonLayer(manager=self.lm)
        p.set_data_from_boundaries(boundaries)

    def test_to_verdat(self):
        layer = self.project.raw_load_first_layer("../TestData/Verdat/000026pts.verdat", "application/x-maproom-verdat")
        boundaries = layer.get_all_boundaries()
        assert 2 == len(boundaries)

        p = PolygonLayer(manager=self.lm)
        p.set_data_from_boundaries(boundaries)

        v = LineLayer(manager=self.lm)
        points, lines = p.get_points_lines()
        print(points)
        print(lines)
        v.set_data(points, 0.0, lines)
        assert len(points) == len(v.points)
        assert len(lines) == len(v.line_segment_indexes)



if __name__ == "__main__":
    import time

    t = TestVerdatToBNA()
    t.setup()
    t.test_sample_shoreline()
    t.test_hole()
    t.test_to_verdat()
