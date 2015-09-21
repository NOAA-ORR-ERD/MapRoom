import os

import unittest
from nose.tools import *
from nose.plugins.attrib import attr

import numpy as np

from mock import *

from maproom.renderer import BaseCanvas


class TestZoom(object):
    def setup(self):
        self.project = MockProject()
        self.canvas = BaseCanvas(self.project.layer_manager, self.project)

    def test_zoom_int(self):
        c = self.canvas
        for n in range(16):
            upp = c.get_units_per_pixel_from_zoom(n)
            zoom1 = c.get_zoom_level(upp)
            assert_almost_equals(n, zoom1, places=5)

    def test_zoom1(self):
        c = self.canvas
        zoom1 = c.get_zoom_level(c.projected_units_per_pixel)
        upp = c.get_units_per_pixel_from_zoom(zoom1)
        assert_almost_equals(c.projected_units_per_pixel, upp, places=5)
        zoom2 = c.get_zoom_level(upp)
        assert_almost_equals(zoom1, zoom2, places=5)

    def test_zoom2(self):
        c = self.canvas
        zoom = c.get_zoom_level(3459, round=.25)
        assert_almost_equals(zoom, 5.5, places=5)
        upp = c.get_units_per_pixel_from_zoom(zoom)
        assert_almost_equals(3460, upp, places=3)



if __name__ == "__main__":
    #unittest.main()
    import time
    
    t = TestZoom()
    t.setup()
    t.test_zoom_int()
    t.test_zoom1()
    t.test_zoom2()
    