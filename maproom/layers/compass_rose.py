# coding=utf8
import math
import bisect
import numpy as np

# Enthought library imports.
from traits.api import Unicode, Str

import glsvg

from base import ScreenLayer

import logging
log = logging.getLogger(__name__)

class CompassRose(ScreenLayer):
    """Compass Rose layer
    
    Shows a compass rose or north-up arrow as a graphic overlay
    """
    name = Unicode("Compass Rose")
    
    type = Str("compass_rose")
    
    skip_on_insert = True
    
    x_offset = 10
    y_offset = 50
    
    def render_screen(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        if picker.is_active:
            return
        log.log(5, "Rendering scale!!! pick=%s" % (picker))
        render_window = renderer.canvas

        x = s_r[0][0] + self.x_offset
        y = s_r[1][1] - self.y_offset

        renderer.draw_screen_lines([(x, y - 10), (x, y), (x + 100, y), (x + 100, y - 10)], width=4)
