import os

import numpy as np

from mock import *

from maproom.renderer import BaseCanvas


class TestZoom(object):
    def setup(self):
        self.project = MockProject()
        self.canvas = BaseCanvas(self.project)

    def test_zoom_int(self):
        c = self.canvas
        for n in range(16):
            upp = c.get_units_per_pixel_from_zoom(n)
            zoom1 = c.get_zoom_level(upp)
            assert_almost_equal(n, zoom1, decimal=5)

    def test_zoom1(self):
        c = self.canvas
        zoom1 = c.get_zoom_level(c.projected_units_per_pixel)
        upp = c.get_units_per_pixel_from_zoom(zoom1)
        assert_almost_equal(c.projected_units_per_pixel, upp, decimal=5)
        zoom2 = c.get_zoom_level(upp)
        assert_almost_equal(zoom1, zoom2, decimal=5)

    def test_zoom2(self):
        c = self.canvas
        zoom = c.get_zoom_level(3459, round=.25)
        assert_almost_equal(zoom, 5.5, decimal=5)
        upp = c.get_units_per_pixel_from_zoom(zoom)
        assert_almost_equal(3459.145, upp, decimal=3)



if __name__ == "__main__":
    import time
    
    t = TestZoom()
    t.setup()
    t.test_zoom_int()
    t.test_zoom1()
    t.test_zoom2()
    