import os
import os.path
import time
import sys
import numpy as np

# Enthought library imports.
from traits.api import Unicode, Str, Any, Float
from pyface.api import YES

from ..library import rect

from ..renderer import color_to_int, data_types

from base import Layer, ScreenLayer
from constants import *

import logging
log = logging.getLogger(__name__)

class AnnotationLayer(ScreenLayer):
    """Layer for raster annotation image
    
    """
    name = Unicode("Annotation Layer")

    type = Str("annotation")

    image_data = Any
    
    alpha = Float(1.0)
    
    layer_info_panel = ["Layer name", "Transparency"]
    
    selection_info_panel = []

    def has_alpha(self):
        return True

    def empty(self):
        """
        We shouldn't allow saving of a layer with no content, so we use this method
        to determine if we can save this layer.
        """
        return self.image_data is not None
    
    def get_allowable_visibility_items(self):
        """Return allowable keys for visibility dict lookups for this layer
        """
        return ["annotations"]
    
    def visibility_item_exists(self, label):
        if label == "annotations":
            return self.image_data is not None

    def compute_bounding_rect(self, mark_type=STATE_NONE):
        bounds = rect.NONE_RECT

        if self.image_data is not None:
            bounds = self.image_data.get_bounds()

        return bounds
    
    def get_image(self, w, h):
        self.image = wx.EmptyBitmap(w, h)
        bg = (255, 255, 255, 0)
        fg = (255, 0, 0, 255)
        DC = wx.MemoryDC()
        DC.SelectObject(self.image)
        DC = wx.GCDC(DC)
        DC.SetBackground(wx.Brush(bg))
        DC.SetBrush(wx.Brush(fg))
        DC.Clear()
        
        DC.DrawLine(0, 0, w, h)
        DC.DrawLine(0, h, w, 0) 

    def rebuild_renderer(self, in_place=False):
        """Update renderer
        
        """
        pass

    def render_screen(self, w_r, p_r, s_r, layer_visibility, layer_index_base, picker):
        log.log(5, "Rendering annotation layer!!! visible=%s, pick=%s" % (layer_visibility["layer"], picker))
        if (not layer_visibility["layer"] or picker.is_active):
            return
        print "Rendering annotation for screen size: ", s_r

        self.renderer.draw_screen_line((s_r[0][0], s_r[0][1]), (s_r[1][0], s_r[1][1]))
        self.renderer.draw_screen_line((s_r[0][0], s_r[1][1]), (s_r[1][0], s_r[0][1]))
