import os
import wx
from collections import OrderedDict

import numpy as np

from constants import *

from ..renderer import color_to_int

import logging
log = logging.getLogger(__name__)


fill_50 = np.empty((32, 4), dtype=np.uint8)
fill_50[::2] = 0xaa
fill_50[1::2] = 0x55

hatched = np.empty((32, 4), dtype=np.uint8)
hatched[:] = 0x80
hatched[::8] = 0xff


class LayerStyle(object):
    """Style data for drawing layer objects
    
    """
    
    default_line_color = color_to_int(0,.5,.3,1.0)
    
    default_fill_color = color_to_int(0,.8,.7,1.0)

    default_colors = [
        color_to_int(0, 0, 1.0, 1),
        color_to_int(0, 0.75, 0, 1),
        color_to_int(0.5, 0, 1.0, 1),
        color_to_int(1.0, 0.5, 0, 1),
        color_to_int(0.5, 0.5, 0, 1),
    ]
    default_color_index = 0
    
    line_styles = [
        ("No Line", 0x0000, wx.TRANSPARENT),
        ("Solid", 0xffff, wx.SOLID),
        ("Dashed", 0xcccc, wx.LONG_DASH),
        ("Dotted", 0xaaaa, wx.DOT),
        ]
    
    fill_styles = OrderedDict([
        (0, ("No Fill", None,)),
        (1, ("Solid Color", None,)),
        (2, ("50%", fill_50)),
        (3, ("Hatched", hatched)),
        ])

    v1_serialization_order = [
        'line_color', 'line_stipple', 'line_stipple_factor',
        'line_width', 'fill_color',
        ]

    v2_serialization_order = [
        'line_color', 'line_stipple', 'line_stipple_factor',
        'line_width', 'line_start_symbol', 'line_end_symbol',
        'fill_color', 'fill_style',
        ]
    
    valid = set(v2_serialization_order)
    
    def __init__(self, **kwargs):
        if len(kwargs):
            # If kwargs are specified, only the style elements specified are
            # added to the object and the unspecified elements are left as
            # None.  This is used to when changing styles to only specify the
            # elements that have changed from some baseline style.
            for k in self.valid:
                v = kwargs.pop(k, None)
                setattr(self, k, v)
            if len(kwargs):
                # Any extra kwargs are invalid names
                raise KeyError("Invalid style names: %s" % ",".join(kwargs.keys()))
        else:
            self.line_color = self.default_line_color  # 4 byte including alpha
            self.line_stipple = 0xffff  # 32 bit stipple pattern
            self.line_stipple_factor = 2  # OpenGL scale factro
            self.line_width = 2  # in pixels
            self.line_start_symbol = 0
            self.line_end_symbol = 0
            self.fill_color = self.default_fill_color  # 4 byte including alpha
            self.fill_style = 1
    
    def __str__(self):
        args = [self.get_str(i) for i in self.v2_serialization_order]
        return "stylev2:%s" % ",".join(args)
    
    def get_str(self, k):
        v = getattr(self, k)
        if v is None:
            return "-"
        return "%x" % v
    
    def parse(self, txt):
        try:
            version, info = txt.split(":", 1)
            method = getattr(self, "parse_%s" % version)
            method(info)
        except Exception, e:
            raise
    
    def parse_stylev1(self, txt):
        vals = txt.split(",")
        print vals
        for k in self.v1_serialization_order:
            v = vals.pop(0)
            if v == "-":
                v = None
            else:
                v = int(v, 16)
            setattr(self, k, v)
    
    def parse_stylev2(self, txt):
        vals = txt.split(",")
        print vals
        for k in self.v2_serialization_order:
            v = vals.pop(0)
            if v == "-":
                v = None
            else:
                v = int(v, 16)
            setattr(self, k, v)
    
    def use_next_default_color(self):
        index = self.__class__.default_color_index
        self.line_color = self.default_colors[index]
        index += 1
        self.__class__.default_color_index = index % len(self.default_colors)
    
    def copy_from(self, other_style):
        """Copy the valid entries (i.e.  those that aren't None) from another
        style to this one
        
        """
        for k in self.valid:
            v = getattr(other_style, k)
            if v is not None:
                setattr(self, k, v)
    
    def get_copy(self):
        """Copy the valid entries (i.e.  those that aren't None) from another
        style to this one
        
        """
        style = self.__class__()
        style.copy_from(self)
        return style

    def get_line_style_from_stipple(self, stipple):
        for i, s in enumerate(self.line_styles):
            if stipple == s[1]:
                return i, s
        return 1, self.line_styles[1] # default to solid

    def get_current_line_style(self):
        return self.get_line_style_from_stipple(self.line_stipple)
    
    def get_fill_stipple(self):
        return self.fill_styles[self.fill_style][1]

    def get_current_fill_style(self):
        return self.fill_style, self.fill_styles[self.fill_style]
