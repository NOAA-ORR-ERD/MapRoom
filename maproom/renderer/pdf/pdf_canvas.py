import os
import time

import math
import numpy as np

from renderer import ReportLabRenderer
import maproom.library.rect as rect

from .. import NullPicker, BaseCanvas

import logging
log = logging.getLogger(__name__)


class PDFCanvas(BaseCanvas):
    """PDF Rendering canvas for MapRoom using ReportLab.
    """
    
    def __init__(self, *args, **kwargs):
        BaseCanvas.__init__(self, *args, **kwargs)

    def get_picker(self):
        return NullPicker()
    
    def get_overlay_renderer(self):
        return ReportLabRenderer(self, None)
    
    def get_renderer(self, layer):
        return ReportLabRenderer(self, layer)
    
    def prepare_screen_viewport(self):
        pass

    def set_screen_rendering_attributes(self):
        pass

    def is_screen_ready(self):
        return True

    def get_screen_rect(self):
        # FIXME: need real coords
        return ((0, 0), (1024, 768))
