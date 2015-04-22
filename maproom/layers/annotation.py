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

from base import Layer, ProjectedLayer
from constants import *

import logging
log = logging.getLogger(__name__)


class AnnotationLayer(ProjectedLayer):
    """Layer for vector annotation image
    
    """
    name = Unicode("Annotation Layer")

    type = Str("annotation")

    canvas = Any
    
    layer_info_panel = ["Layer name"]
    
    selection_info_panel = []

    def empty(self):
        """
        We shouldn't allow saving of a layer with no content, so we use this method
        to determine if we can save this layer.
        """
        if self.canvas is not None:
            return self.canvas.IsEmpty()
        return True

    def load_fc_json(self, text):
        if not self.canvas:
            self.canvas = FC.PyProjFloatCanvas((1,1),
                                               ProjectionFun = None,
                                               Debug = 0,
                                               BackgroundColor = wx.WHITE,
                                               )
        self.canvas.ClearAll()
        self.canvas.Unserialize(text)
        self.update_bounds()

    def compute_bounding_rect(self, mark_type=STATE_NONE):
        bounds = rect.NONE_RECT

        if self.canvas is not None:
            self.canvas._ResetBoundingBox()
            b = self.canvas.BoundingBox
            if not b.IsNull():
                # convert from FloatCanvas BBox object to MapRoom tuple
                bounds = tuple((tuple(b[0]), tuple(b[1])))

        return bounds
    
    def rebuild_renderer(self, in_place=False):
        """Update renderer
        
        """
        if not self.canvas:
            self.canvas = FC.PyProjFloatCanvas((1,1),
                                               ProjectionFun = None,
                                               Debug = 0,
                                               BackgroundColor = wx.WHITE,
                                               )
        
        size = self.manager.project.layer_canvas.get_screen_size()
        self.canvas.InitializePanelSize(size)
        self.canvas.MakeNewBuffers()

        projection = self.manager.project.layer_canvas.projection
        self.canvas.SetProjectionFun(projection)

    def render_projected(self, w_r, p_r, s_r, layer_visibility, layer_index_base, picker):
        log.log(5, "Rendering annotation layer!!! visible=%s, pick=%s" % (layer_visibility["layer"], picker))
        if (not layer_visibility["layer"] or picker.is_active):
            return
        print "Rendering annotation for screen size: ", s_r

        dc =  self.renderer.get_emulated_dc()
        self.canvas.DrawToDC(w_r, dc)
