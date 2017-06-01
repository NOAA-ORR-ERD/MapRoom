import wx
from collections import OrderedDict

import numpy as np

from ..renderer import color_floats_to_int, color_ints_to_int, int_to_color_uint8

from maproom.library.marplot_icons import get_numpy_bitmap

import logging
log = logging.getLogger(__name__)


fill_50 = np.empty((32, 4), dtype=np.uint8)
fill_50[::2] = 0xaa
fill_50[1::2] = 0x55

hatched = np.empty((32, 4), dtype=np.uint8)
hatched[:] = 0x80
hatched[::8] = 0xff

marker_arrow = np.array(((-8, 0), (-10, -4), (0, 0), (-10, 4)), dtype=np.float32)


class LayerStyle(object):
    """Style data for drawing layer objects

    """

    default_line_color = color_floats_to_int(0, 0, 0, 1.0)

    default_fill_color = color_floats_to_int(1.0, 1.0, 1.0, 0.75)

    default_text_color = color_floats_to_int(0, 0, 0, 1.0)

    default_outline_color = color_floats_to_int(1.0, 0.5, 1.0, 0.75)

    default_colors = [
        color_floats_to_int(0, 0, 1.0, 1),
        color_floats_to_int(0, 0.75, 0, 1),
        color_floats_to_int(0.5, 0, 1.0, 1),
        color_floats_to_int(1.0, 0.5, 0, 1),
        color_floats_to_int(0.5, 0.5, 0, 1),
        color_ints_to_int(31, 119, 180),
        color_ints_to_int(22, 120, 22),
        color_ints_to_int(214, 39, 40),
        color_ints_to_int(148, 103, 189),
        color_ints_to_int(140, 86, 75),
        color_ints_to_int(227, 119, 194),
        color_ints_to_int(127, 127, 127),
        color_ints_to_int(188, 189, 34),
        color_ints_to_int(23, 190, 207),
    ]
    default_color_index = 0

    line_styles = [
        ("No Line", 0x0000, wx.TRANSPARENT),
        ("Solid", 0xffff, wx.SOLID),
        ("Dashed", 0xcccc, wx.LONG_DASH),
        ("Dotted", 0xaaaa, wx.DOT),
    ]

    marker_styles = [
        ("None", None, False),
        ("Arrow", marker_arrow, False),
        ("Filled Arrow", marker_arrow, True),
    ]

    fill_styles = OrderedDict([
        (0, ("No Fill", None,)),
        (1, ("Solid Color", None,)),
        (2, ("50%", fill_50)),
        (3, ("Hatched", hatched)),
    ])

    text_format_styles = [
        ("Plain Text", None),
        ("HTML", None),
        ("RST Markup", None),
        ("Markdown", None),
    ]

    fonts = None  # Will be initialized in call to get_font_names

    standard_font_sizes = [4, 6, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 48, 56, 64, 72, 144]
    default_font_size = 12
    default_font_index = standard_font_sizes.index(default_font_size)

    standard_line_widths = [1, 2, 3, 4, 5, 6, 7, 8]
    default_line_width = 2
    default_line_width_index = standard_line_widths.index(default_line_width)

    stylev1_serialization_order = [
        'line_color', 'line_stipple', 'line_stipple_factor',
        'line_width', 'fill_color',
    ]

    stylev2_serialization_order = [
        'line_color', 'line_stipple', 'line_stipple_factor',
        'line_width', 'line_start_marker', 'line_end_marker',
        'fill_color', 'fill_style',
    ]

    stylev3_serialization_order = [
        'line_color', 'line_stipple', 'line_stipple_factor',
        'line_width', 'line_start_marker', 'line_end_marker',
        'fill_color', 'fill_style', 'font', 'font_size'
    ]

    stylev4_serialization_order = [
        'line_color', 'line_stipple', 'line_stipple_factor',
        'line_width', 'line_start_marker', 'line_end_marker',
        'fill_color', 'fill_style', 'font', 'font_size',
        'text_format'
    ]

    stylev5_serialization_order = [
        'line_color', 'line_stipple', 'line_stipple_factor',
        'line_width', 'line_start_marker', 'line_end_marker',
        'fill_color', 'fill_style', 'font', 'font_size',
        'text_format', 'icon_marker'
    ]

    stylev6_serialization_order = [
        'line_color', 'line_stipple', 'line_stipple_factor',
        'line_width', 'line_start_marker', 'line_end_marker',
        'fill_color', 'fill_style',
        'text_color', 'font', 'font_size', 'text_format',
        'icon_marker'
    ]

    stylev7_serialization_order = [
        'line_color', 'line_stipple', 'line_stipple_factor',
        'line_width', 'line_start_marker', 'line_end_marker',
        'fill_color', 'fill_style', 'outline_color',
        'text_color', 'font', 'font_size', 'text_format',
        'icon_marker'
    ]

    valid = set(stylev7_serialization_order)

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
            self.line_width = self.default_line_width  # in pixels
            self.line_start_marker = 0
            self.line_end_marker = 0
            self.fill_color = self.default_fill_color  # 4 byte including alpha
            self.fill_style = 1
            self.outline_color = self.default_outline_color  # 4 byte including alpha
            self.text_color = self.default_text_color  # 4 byte including alpha
            self.font = ""
            self.font_size = self.default_font_size
            self.text_format = 1
            self.icon_marker = 320  # Shapes/Cross

    def __str__(self):
        args = [self.get_str(i) for i in self.stylev7_serialization_order]
        return "stylev7:%s" % ",".join(args)

    def get_str(self, k):
        v = getattr(self, k)
        if v is None:
            return "-"
        if k == 'font':
            return v.encode('utf-8')
        return "%x" % v

    def has_same_keywords(self, other_style):
        for k in self.valid:
            v1 = getattr(self, k)
            v2 = getattr(other_style, k)
            if (v1 is None and v2 is not None) or (v1 is not None and v2 is None):
                return False
        return True

    def parse(self, txt):
        try:
            version, info = txt.split(":", 1)
            order = getattr(self, "%s_serialization_order" % version)
            self.parse_style(order, info)
        except Exception:
            raise

    def parse_style(self, order, txt):
        vals = txt.split(",")
        for k in order:
            v = vals.pop(0)
            if v == "-":
                v = None
            elif k == 'font':
                v = v.decode('utf-8')
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
        return 1, self.line_styles[1]  # default to solid

    def get_current_line_style(self):
        return self.get_line_style_from_stipple(self.line_stipple)

    def get_fill_stipple(self):
        return self.fill_styles[self.fill_style][1]

    def get_current_fill_style(self):
        return self.fill_style, self.fill_styles[self.fill_style]

    def get_marker_data(self, marker):
        m = self.marker_styles[marker]
        return m[1], m[2]

    def get_current_font(self):
        for i, f in enumerate(self.fonts):
            if self.font == f:
                return i, f
        return 0, self.fonts[0]  # default to system default

    def get_current_font_size(self):
        for i, f in enumerate(self.standard_font_sizes):
            if self.font_size == f:
                return i, f
        return self.default_font_index, self.default_font_size

    def get_current_line_width(self):
        for i, w in enumerate(self.standard_line_widths):
            if self.line_width == w:
                return i, w
        return self.default_line_width_index, self.default_line_width

    @classmethod
    def get_font_name(cls, index):
        fonts = cls.get_font_names()
        return fonts[index]

    @classmethod
    def get_font_names(cls):
        if cls.fonts is None:
            # NOTE: The list of face names from wx are in unicode
            fonts = wx.FontEnumerator()
            fonts.EnumerateFacenames()
            fonts = fonts.GetFacenames()
            fonts.sort()
            fonts[0:0] = [u"default"]
            cls.fonts = fonts
        return cls.fonts

    def get_numpy_image_from_icon(self):
        arr = get_numpy_bitmap(self.icon_marker)
        r, g, b, a = int_to_color_uint8(self.line_color)
        red, green, blue = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        mask = (red == 255) & (green == 255) & (blue == 255)
        arr[:,:,:3][mask] = [r, g, b]
        return arr

def parse_styles_from_json(sdict):
    d = {}
    for name, style_str in sdict.iteritems():
        style = LayerStyle()
        style.parse(style_str)
        d[name] = style
    return d

def styles_to_json(style_dict):
    j = {}
    for name, style in style_dict.iteritems():
        j[name] = str(style)
    return j

