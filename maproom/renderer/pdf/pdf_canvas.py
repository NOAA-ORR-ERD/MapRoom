import os
import time

import math
import numpy as np

import wx

from reportlab.pdfgen import canvas
import reportlab.lib.pagesizes as pagesizes

from renderer import ReportLabRenderer
import maproom.library.rect as rect

from .. import NullPicker, BaseCanvas

import logging
log = logging.getLogger(__name__)

paper_size_map = {
    wx.PAPER_LETTER: pagesizes.letter,
    wx.PAPER_A4: pagesizes.A4,
    }


class PDFCanvas(BaseCanvas):
    """PDF Rendering canvas for MapRoom using ReportLab.
    """
    
    def __init__(self, *args, **kwargs):
        BaseCanvas.__init__(self, *args, **kwargs)

    def new_picker(self):
        return NullPicker()
    
    def new_renderer(self, layer):
        return ReportLabRenderer(self, layer)
    
    def get_page_size(self):
        print_data = self.project.task.print_data
        id = print_data.GetPaperId()
        size = paper_size_map[id]
        if print_data.GetOrientation() == wx.LANDSCAPE:
            size = (size[1], size[0])
        return size
    
    def prepare_screen_viewport(self):
        print self.screen_rect
        pagesize = self.get_page_size()
        print pagesize
        c = canvas.Canvas("maproom.pdf", pagesize=pagesize)
        c.drawString(100,100,"Hello MapRoom!")
        c.showPage()
        c.save()

    def set_screen_rendering_attributes(self):
        pass

    def is_screen_ready(self):
        return True

    def get_screen_rect(self):
        # FIXME: need real coords
        return ((0, 0), (1024, 768))
