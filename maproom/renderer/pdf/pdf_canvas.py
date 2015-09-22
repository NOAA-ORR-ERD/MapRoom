import os
import time

import math
import numpy as np

import wx

from reportlab.pdfgen import canvas
import reportlab.lib.pagesizes as pagesizes
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab import rl_settings  # to allow py2exe bundling of this dynamic import

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
        self.path = kwargs.pop('path')
        BaseCanvas.__init__(self, *args, **kwargs)
        self.screen_size = (1600, 900)
        self.viewport_scale = 1.0
        
        # linewidth_factor is an empircal scale factor gauged to make the
        # linewidths on the printed output match the relative sizes on the
        # screen. < 1 reduces width in printed output
        self.linewidth_factor = 0.6

        self.font_name = "Courier"
        self.font_size = 5
        self.font_scale = 0

    def new_picker(self):
        return NullPicker()

    def is_canvas_pickable(self):
        return False
    
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
    
    def get_font_metrics(self, text):
        return stringWidth(text, self.font_name, self.font_scale), self.font_scale
    
    def set_viewport_rect(self, rect):
        self.pdf.resetTransforms()
        x1, y1 = rect[0]
        x2, y2 = rect[1]
        self.current_viewport = ((x1, y1), (x2, y2))
        w, h = (x2 - x1, y2 - y1)
        ar = w * 1.0 / h
        pagesize = self.get_page_size()
        margins = self.get_page_margins()
        self.pdf.translate(*margins)
        
        drawing_area = (pagesize[0] - (2 * margins[0]), pagesize[1] - (2 * margins[1]))
        if ar > 1.0:
            scale = drawing_area[0] / w
        else:
            scale = drawing_area[0] / h
        self.viewport_scale = scale
        print "viewport:", x1, y1, w, h, ar, pagesize, drawing_area, scale
        self.pdf.scale(scale, scale)
        self.pdf.translate(-x1, -y1)
        p = self.pdf.beginPath()
        p.rect(x1, y1, w, h)
        self.pdf.clipPath(p, fill=0, stroke=0)
        self.font_scale = self.font_size / self.viewport_scale
        self.pdf.setFont(self.font_name, self.font_scale)
    
    def is_onscreen(self, x, y, w, h):
        r = ((x, y), (x + w, y + h))
        return rect.intersects(r, self.current_viewport)
    
    def set_screen_viewport(self):
        print "screen rect!", self.screen_rect
        self.set_viewport_rect(self.screen_rect)
    
    def set_projected_viewport(self):
        print "proj rect!", self.projected_rect
        self.set_viewport_rect(self.projected_rect)
    
    def debug_boundingbox(self):
        w, h = self.screen_rect[1]
        self.pdf.rect(0, 0, w, h, stroke=1, fill=0)
        self.pdf.drawString(0, 0, "Hello MapRoom!!!!")
        
        print self.layer_renderers
    
    def prepare_screen_viewport(self):
        pagesize = self.get_page_size()
        self.pdf = canvas.Canvas(self.path, pagesize=pagesize)

    def finalize_rendering_screen(self):
        self.pdf.showPage()
        self.pdf.save()

    def set_screen_rendering_attributes(self):
        pass

    def is_screen_ready(self):
        return True

    def get_screen_rect(self):
        w, h = self.screen_size
        return ((0, 0), (w, h))
    
    def set_screen_size(self, size):
        self.screen_size = size
