import os
import os.path
import time
import sys
import numpy as np

import wx

# Enthought library imports.
from traits.api import Unicode, Str, Any, Float
from pyface.api import YES

from ..library import rect
from ..library.floatcanvas import FloatCanvas as FC
from ..library.floatcanvas.Utilities import BBox

from ..renderer import color_to_int, data_types
from ..renderer import ImageScreenData, SubImageLoader

from base import Layer, ScreenLayer
from constants import *

import logging
log = logging.getLogger(__name__)


class CanvasSubImageLoader(SubImageLoader):
    def __init__(self, canvas):
        self.image = canvas.GetNumpyArray(wx.WHITE)
    
    def load(self, origin, size, w_r):
        # mirror top for bottom as OpenGL origin is bottom left, but wxPython
        # image is top left
        h = self.image.shape[0]
        return self.image[h - origin[1]:h - origin[1] - size[1]:-1,
                          origin[0]:origin[0] + size[0]]


class AnnotationLayer(ScreenLayer):
    """Layer for raster annotation image
    
    """
    name = Unicode("Annotation Layer")

    type = Str("annotation")

    canvas = Any

    image_data = Any
    
    alpha = Float(1.0)
    
    layer_info_panel = ["Layer name", "Transparency"]
    
    selection_info_panel = []
    
    texture_size = 256

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
        if not self.canvas:
            self.canvas = FC.OffScreenFloatCanvas((1,1),
                                          ProjectionFun = None,
                                          Debug = 0,
                                          BackgroundColor = wx.WHITE,
                                          )
        
        size = self.manager.project.layer_canvas.get_screen_size()
        self.canvas.InitializePanelSize(size)
        self.canvas.MakeNewBuffers()
        for lat in range(-80, 90, 10):
            line = FC.Line([(lat, 0), (lat, 90)], LineWidth=3, LineColor="Yellow")
            self.canvas.AddObject(line)
        for lon in range(0, 90, 10):
            line = FC.Line([(-80, lon), (80, lon)], LineWidth=3, LineColor="Green")
            self.canvas.AddObject(line)
        projection = self.manager.project.layer_canvas.projection
        self.canvas.SetProjectionFun('FlatEarth')
        self.canvas.ZoomToBB()
#        self.canvas.SaveAsImage("annotation1.png", transparent_color=wx.WHITE)
        
        self.image_data = ImageScreenData(size[0], size[1])
        loader = CanvasSubImageLoader(self.canvas)
        self.image_data.load_texture_data(self.texture_size, loader)
        self.renderer.use_world_rects_as_screen_rects(self.image_data)

    def render_screen(self, w_r, p_r, s_r, layer_visibility, layer_index_base, picker):
        log.log(5, "Rendering annotation layer!!! visible=%s, pick=%s" % (layer_visibility["layer"], picker))
        if (not layer_visibility["layer"] or picker.is_active):
            return
        print "Rendering annotation for screen size: ", s_r

        self.image_data = ImageScreenData(s_r[1][0], s_r[1][1])
        loader = CanvasSubImageLoader(self.canvas)
        self.image_data.load_texture_data(self.texture_size, loader)
        self.renderer.use_world_rects_as_screen_rects(self.image_data)
        bbox = BBox.asBBox(w_r)
        self.canvas.ZoomToBB(bbox, margin_adjust=1.0)
        self.canvas.SaveAsImage("/tmp/annotation1.png", transparent_color=wx.WHITE)
        print "  center", self.canvas.ViewPortCenter, "world center:", (w_r[0][0] + w_r[1][0])/2, (w_r[0][1] + w_r[1][1])/2
        self.renderer.draw_screen_line((s_r[0][0], s_r[0][1]), (s_r[1][0], s_r[1][1]))
        self.renderer.draw_image(self.alpha)
        
