import os

import unittest
from nose.tools import *
from nose.plugins.attrib import attr

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
        print layer_bounds.boundaries
        eq_(1, len(layer_bounds.boundaries))
        eq_(0, len(layer_bounds.non_boundary_points))
        for b in layer_bounds.boundaries:
            print b.point_indexes
            print b.area
            print b.get_xy_points()
            
        p = PolygonLayer(manager=self.lm)
        p.set_data_from_boundaries(layer_bounds.boundaries)

    def test_hole(self):
        layer = self.project.raw_load_first_layer("../TestData/Verdat/000026pts.verdat", "application/x-maproom-verdat")
        layer_bounds = layer.get_all_boundaries()
        eq_(2, len(layer_bounds.boundaries))
        eq_(10, len(layer_bounds.non_boundary_points))
            
        p = PolygonLayer(manager=self.lm)
        p.set_data_from_boundaries(layer_bounds.boundaries)



if __name__ == "__main__":
    #unittest.main()
    import time
    
    t = TestVerdatToBNA()
    t.setup()
    t.test_sample_shoreline()
    t.test_hole()
