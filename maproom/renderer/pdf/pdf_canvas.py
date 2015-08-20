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
    
    def get_page_margins(self):
        return (50, 50)
    
    def set_default_viewport(self):
        self.pdf.resetTransforms()
        w, h = self.screen_rect[1]
        ar = w * 1.0 / h
        pagesize = self.get_page_size()
        margins = self.get_page_margins()
        self.pdf.translate(*margins)
        
        drawing_area = (pagesize[0] - (2 * margins[0]), pagesize[1] - (2 * margins[1]))
        if ar > 1.0:
            scale = (drawing_area[0] / w, drawing_area[1] / w / ar)
        else:
            scale = (drawing_area[0] / h, drawing_area[1] / h / ar)
        print w, h, ar, pagesize, drawing_area, scale
        self.pdf.scale(*scale)
    
    def debug_boundingbox(self):
        w, h = self.screen_rect[1]
        self.pdf.rect(0, 0, w, h, stroke=1, fill=0)
        self.pdf.drawString(0, 0, "Hello MapRoom!!!!")
        
        print self.layer_renderers
    
    def prepare_screen_viewport(self):
        pagesize = self.get_page_size()
        self.pdf = canvas.Canvas("maproom.pdf", pagesize=pagesize)
        self.set_default_viewport()
        
        self.debug_boundingbox()

    def finalize_rendering_screen(self):
        self.pdf.showPage()
        self.pdf.save()

    def set_screen_rendering_attributes(self):
        pass

    def is_screen_ready(self):
        return True

    def get_screen_rect(self):
        # FIXME: need real coords
        return ((0, 0), (1024, 768))
