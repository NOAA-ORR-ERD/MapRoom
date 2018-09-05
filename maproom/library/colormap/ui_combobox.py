import collections
import re

import wx
import wx.lib.scrolledpanel
import wx.lib.splitter
import wx.adv
import numpy as np
import numpy.random as rand

from omnivore.utils.wx.dialogs import SimplePromptDialog

from . import builtin_discrete_colormaps, builtin_continuous_colormaps, get_colormap, DiscreteColormap, ListedBoundedColormap
from ...ui.buttons import ColorSelectButton, EVT_COLORSELECT, prompt_for_rgba

import logging
log = logging.getLogger(__name__)


class FloatEntryDialog(SimplePromptDialog):
    """Simple subclass of wx.TextEntryDialog to convert text result from
    floating point if necessary.
    """

    def convert_text(self, text, return_error=False, default_base="dec", **kwargs):
        try:
            value = float(text)
            error = ""
        except (ValueError, TypeError) as e:
            value = None
            error = str(e)
        if return_error:
            return value, error
        else:
            return value


def prompt_for_float(parent, message, title, default=None, return_error=False):
    if default is not None:
        default = str(float(default))
    else:
        default = ""
    d = FloatEntryDialog(parent, message, title, default)
    if default:
        d.SetValue(default)
    return d.show_and_get_value(return_error=return_error)


class ColormapImage(object):
    def __init__(self, width, height):
        self.arrow_width = 20
        self.height = height
        self.create_image_array(width)

    def create_image_array(self, width):
        self.width = width
        self.array = np.empty((self.height, width, 3), dtype='uint8')
        self.image = wx.ImageFromBuffer(width, self.height, self.array)

    def calc_bitmap(self, colormap, width=None, bgcolor=(255, 255, 255)):
        if width is not None:
            if width != self.width:
                self.create_image_array(width)
        if colormap.is_discrete:
            bgcolor = bgcolor[0:3]
            middle_width = self.width - 2 * self.arrow_width
            middle = colormap.calc_rgb_texture(middle_width)
            self.array[:,self.arrow_width:-self.arrow_width,:] = middle
            self.array[:,0:self.arrow_width,:] = np.asarray(colormap.under_rgba[0:3], dtype=np.float32) * 255
            self.array[:,-self.arrow_width:,:] = np.asarray(colormap.over_rgba[0:3], dtype=np.float32) * 255
            half_height = self.height // 2
            for h in range(half_height):
                w = self.arrow_width * h * 2 // self.height
                self.array[half_height - h - 1,0:w,:] = bgcolor
                self.array[self.height - half_height + h,0:w,:] = bgcolor
                self.array[half_height - h - 1,self.width-w:,:] = bgcolor
                self.array[self.height - half_height + h,self.width-w:,:] = bgcolor
        else:
            colors = colormap.calc_rgb_texture(self.width)
            self.array[:,:,:] = colors
        return wx.Bitmap(self.image)


class ColormapComboBox(wx.adv.OwnerDrawnComboBox):
    def __init__(self, *args, **kwargs):
        # self.known_colormaps = dict(builtin_discrete_colormaps)
        popup_width = kwargs.pop('popup_width', 300)
        self.custom = kwargs.pop('custom_colormaps', {})  # name:colormap
        # if self.custom:
        #     self.known_colormaps.update(self.custom)
        # print self.known_colormaps
        self.recalc_order()
        kwargs['choices'] = self.colormap_name_order
        kwargs['style'] = wx.CB_READONLY
        # self.start_builtin_index = len(self.custom)
        wx.adv.OwnerDrawnComboBox.__init__(self, *args, **kwargs)

        self.height = 20
        self.control_image = ColormapImage(popup_width, self.height)
        self.dropdown_image = ColormapImage(popup_width, self.height)
        dc = wx.MemoryDC()
        self.char_height = dc.GetCharHeight()
        self.internal_spacing = 2
        self.item_height = self.height + self.char_height + 2 * self.internal_spacing
        self.bitmap_x = self.internal_spacing
        self.bitmap_y = self.char_height + self.internal_spacing

        self.SetPopupMinWidth(popup_width)

    def recalc_order(self):
        self.colormap_name_order = sorted(builtin_discrete_colormaps.keys()) + sorted(builtin_continuous_colormaps.keys())
        self.discrete_start = 0
        self.continuous_start = len(builtin_discrete_colormaps)

    def rebuild_colormap_list(self):
        self.recalc_order()

    def index_of(self, name):
        return self.colormap_name_order.index(name)

    def set_selection_by_name(self, name):
        try:
            index = self.index_of(name)
        except ValueError:
            return
        self.SetSelection(index)
        self.override_control_colormap = None

    def get_selected_name(self):
        index = self.GetSelection()
        return self.colormap_name_order[index]

    # Overridden from OwnerDrawnComboBox, called to draw each
    # item in the list
    def OnDrawItem(self, dc, rect, item, flags):
        if item == wx.NOT_FOUND:
            # painting the control, but there is no valid item selected yet
            return

        r = wx.Rect(*rect)  # make a copy

        name = self.colormap_name_order[item]
        c = get_colormap(name)
        if flags & wx.adv.ODCB_PAINTING_CONTROL:
            width = r.width - 2 * self.internal_spacing
            x = self.internal_spacing
            y = (r.height - self.height) // 2
            b = self.control_image.calc_bitmap(c, width, self.GetParent().GetBackgroundColour())
            dc.DrawBitmap(b, r.x + x, r.y + y)
        else:
            width = r.width - 2 * self.internal_spacing
            b = self.dropdown_image.calc_bitmap(c, width, self.GetParent().GetBackgroundColour())
            dc.DrawBitmap(b, r.x + self.bitmap_x, r.y + self.bitmap_y)
            dc.DrawText(c.name, r.x + self.bitmap_x, r.y)

    # Overridden from OwnerDrawnComboBox, called for drawing the
    # background area of each item.
    def OnDrawBackground(self, dc, rect, item, flags):
        # If the item is selected, or its item # iseven, or we are painting the
        # combo control itself, then use the default rendering.
        if (flags & wx.adv.ODCB_PAINTING_CONTROL):
            flags = flags & ~(wx.adv.ODCB_PAINTING_CONTROL | wx.adv.ODCB_PAINTING_SELECTED)
        wx.adv.OwnerDrawnComboBox.OnDrawBackground(self, dc, rect, item, flags)

    # Overridden from OwnerDrawnComboBox, should return the height
    # needed to display an item in the popup, or -1 for default
    def OnMeasureItem(self, item):
        return self.item_height

    # Overridden from OwnerDrawnComboBox.  Callback for item width, or
    # -1 for default/undetermined
    def OnMeasureItemWidth(self, item):
        return self.item_width


class DiscreteOnlyColormapComboBox(ColormapComboBox):
    def recalc_order(self):
        self.colormap_name_order = sorted(builtin_discrete_colormaps.keys())


class MultiSlider(wx.Panel):
    def __init__(self, parent, id=-1, *args, **kwargs):
        wx.Panel.__init__(self, parent, id, *args, **kwargs)
        self.separators = []
        self.rectangles = []
        self.label_width = []

        self.drag_cursor = wx.Cursor(wx.CURSOR_SIZEWE)
        self.separator_pen = wx.Pen(wx.WHITE, 3)
        self.background_brush = wx.Brush(wx.WHITE)
        self.text_color = wx.BLACK
        self.text_background = wx.WHITE
        attr = self.GetDefaultAttributes()
        self.SetBackgroundColour(attr.colBg)
        if wx.Platform == "__WXMSW__":
            fontsize = 8
        else:
            fontsize = 10
        self.text_font = wx.Font(fontsize, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.label_border = 3
        self.set_defaults()

        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_MOUSE_EVENTS, self.on_mouse)

    def set_defaults(self):
        dc = wx.MemoryDC()
        dc.SetFont(self.text_font)
        self.label_height = dc.GetCharHeight() + 3 * self.label_border
        self.bar_height = 0
        self.active_width = 0

    def on_mouse(self, evt):
        if not self.separators:
            evt.Skip()
            return

        x, y = evt.GetPosition()

        if evt.LeftDown():
            self.CaptureMouse()
            self.dragging, _, _ = self.hit_test(x, y)
            print(f"DRAGGING {self.dragging}")
            if self.dragging is not None:
                self.SetCursor(self.drag_cursor)
        elif evt.Dragging() and self.dragging is not None:
            self.move_border(x)
        elif evt.LeftUp():
            if self.HasCapture():
                self.ReleaseMouse()
            self.SetCursor(wx.NullCursor)
            if self.dragging:
                self.dragging = None
            else:
                _, color_index, value_index = self.hit_test(x, y)
                if color_index is not None:
                    self.change_color(color_index)
                elif value_index is not None:
                    self.change_value(value_index)
        elif evt.Moving():
            is_over_bin, _, _ = self.hit_test(x, y)
            if is_over_bin is not None:
                self.SetCursor(self.drag_cursor)
            else:
                self.SetCursor(wx.NullCursor)

        evt.Skip()

    def hit_test(self, x, y):
        drag = value = color = None
        if y < self.bar_height:
            for i, sx in enumerate(self.separators[:-1]):
                if abs(x - sx) < self.label_border:
                    drag = i
                    break
            if drag is None:
                color = 1
                for i, sx in enumerate(self.separators):
                    print(x, sx)
                    if sx >= x:
                        break
                    color += 1
                else:
                    # out of range
                    color = None
        elif y > self.bar_height + self.label_border and y < self.bar_height + self.label_border + self.label_height:
            for i, sx in enumerate(self.separators[:-1]):
                if abs(x - sx) < self.label_width[i]:
                    value = i
                    break
        return drag, color, value

    def move_border(self, x):
        d = self.dragging
        value = self.x_to_value(x)
        self.set_border_value(d, value)

    def set_border_value(self, d, value):
        lo = self.separators[d - 1] if d > 0 else self.label_border
        hi = self.separators[d + 1] if d < len(self.separators) - 1 else self.full_width - self.label_border
        if value > self.x_to_value(lo) and value < self.x_to_value(hi):
            p = self.GetParent()
            p.bin_borders[d + 2] = value
            self.update_borders()
            return True
        return False

    def change_color(self, color_index):
        p = self.GetParent()
        color = p.bin_colors[color_index]
        new_color = prompt_for_rgba(self, color, use_float=True)
        print(f"index={color_index} old={color} new={new_color}")
        if new_color is not None:
            p.bin_colors[color_index] = np.asarray(new_color, dtype=np.float32)
            self.update_borders()

    def change_value(self, value_index):
        p = self.GetParent()
        value = self.x_to_value(self.separators[value_index])
        new_value = prompt_for_float(self, "New bin", "Edit Bin Value", value)
        print(f"index={value_index} old={value} new={new_value}")
        if not self.set_border_value(value_index, new_value):
            print(f"value out of range")

    def on_size(self, evt):
        full = self.GetClientRect()
        self.bar_height = full.height - self.label_height
        self.full_height = full.height
        self.full_width = full.width
        self.active_width = full.width - 2 * self.label_border + 1
        self.update_borders()

    def on_paint(self, evt):
        dc = wx.PaintDC(self)
        self.draw(dc)

    def draw(self, dc):
        b = self.label_border
        label_top = self.bar_height + b
        dc.SetPen(wx.TRANSPARENT_PEN)
        dc.SetBrush(wx.Brush(self.GetBackgroundColour()))
        dc.DrawRectangle(0, 0, self.full_width, self.full_height)
        dc.SetBrush(self.background_brush)
        dc.DrawRectangle(b, 0, self.active_width, self.bar_height)
        for x1, x2, color in self.rectangles:
            print(f"drawing rect: {x1}->{x2},{self.bar_height} in {color}")
            dc.SetBrush(wx.Brush(color))
            dc.DrawRectangle(x1, 0, x2, self.bar_height)
        dc.SetPen(self.separator_pen)
        dc.SetFont(self.text_font)
        dc.SetTextBackground(self.text_background)
        dc.SetTextForeground(self.text_color)
        dc.SetBrush(wx.Brush(self.text_background))

        def draw_label(x):
            value = self.x_to_value(x)
            label = "%.3f" % value
            dc.DrawLine(x, 0, x, label_top)
            width, _ = dc.GetTextExtent(label)
            text_x = x - width//2
            if text_x + width > self.full_width:
                text_x = self.full_width - width - b
            elif text_x - b < 0:
                text_x = b
            dc.DrawRectangle(text_x - b, label_top, width + 2*b, label_top + self.label_height + 2*b)
            dc.DrawText(label, text_x, self.bar_height + 2*b)
            self.label_width.append(width)

        self.label_width = []
        draw_label(self.label_border)
        for sx in self.separators:
            draw_label(sx)

    def update_borders(self):
        p = self.GetParent()
        lo, hi = p.min_max
        print(f"bin_colors:{p.bin_colors}")
        print(f"bin_borders:{p.bin_borders}")
        print(f"min/max: {lo} {hi}")
        width, height = self.GetSize()
        last_pixel_pos = self.label_border
        r = []
        s = []
        if len(p.bin_colors) > 1:
            for c, v in zip(p.bin_colors[1:-1], p.bin_borders[2:]):
                print(f"c={c}, v={v}")
                color = [int(z*255) for z in c[0:3]]
                perc = (v - lo) / (hi - lo)
                pixel_pos = (perc * self.active_width) + self.label_border
                r.append((last_pixel_pos, pixel_pos, color))
                s.append(pixel_pos)
                print(f"splitter at {pixel_pos}")
                last_pixel_pos = pixel_pos
        self.rectangles = r
        self.separators = s
        self.Refresh()

    def x_to_value(self, x):
        perc = (x - self.label_border) / self.active_width
        return self.GetParent().perc_to_value(perc)




class GnomeColormapDialog(wx.Dialog):
    def __init__(self, parent, current_colormap, values_min_max):
        wx.Dialog.__init__(self, parent, -1, "Edit Discrete Colormap", size=(500, -1))
        self.bitmap_width = 300
        self.bitmap_height = 30
        self.working_copy = None
        self.values_min_max = values_min_max

        lsizer = wx.BoxSizer(wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        s = wx.StaticText(self, -1, "Color Scheme:")
        hsizer.Add(s, 0, wx.EXPAND|wx.ALL, 10)
        self.colormap_list = DiscreteOnlyColormapComboBox(self, -1, "colormap_list", popup_width=300)
        self.colormap_list.Bind(wx.EVT_COMBOBOX, self.colormap_changed)
        self.colormap_list.SetSelection(0)
        hsizer.Add(self.colormap_list, 1, wx.EXPAND, 0)
        lsizer.Add(hsizer, 0, wx.EXPAND|wx.ALL, 10)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        s = wx.StaticText(self, -1, "Add/Remove:")
        hsizer.Add(s, 0, wx.EXPAND|wx.ALL, 10)
        b = wx.Button(self, wx.ID_ADD, " + ")
        b.Bind(wx.EVT_BUTTON, self.add_bin)
        hsizer.Add(b, 0, wx.EXPAND, 0)
        b = wx.Button(self, wx.ID_REMOVE, " - ")
        b.Bind(wx.EVT_BUTTON, self.remove_bin)
        hsizer.Add(b, 0, wx.EXPAND, 0)
        lsizer.Add(hsizer, 0, wx.EXPAND|wx.ALL, 10)

        self.splitter = MultiSlider(self, size=(800,50))
        lsizer.Add(self.splitter, 1, wx.EXPAND | wx.ALL, 5)

        # self.autoscale_button = wx.CheckBox(self, -1, "Automatically scale bins when switching view\nto new type of data")
        # self.autoscale_button.Bind(wx.EVT_CHECKBOX, self.on_autoscale)
        # lsizer.Add(self.autoscale_button, 0, wx.ALL|wx.CENTER|wx.TOP, 10)

        btnsizer = wx.StdDialogButtonSizer()
        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()
        btnsizer.AddButton(btn)
        btn = wx.Button(self, wx.ID_CANCEL)
        btnsizer.AddButton(btn)
        btnsizer.Realize()

        lsizer.AddStretchSpacer(1)
        lsizer.Add(btnsizer, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        self.SetSizer(lsizer)

        self.populate_panel(current_colormap.name)
        self.Fit()
        self.Layout()

    @property
    def min_max(self):
        try:
            lo, hi = self.values_min_max
        except TypeError:
            lo = 0.0
            hi = 100.0
        if lo == hi:
            hi += 1.0
        return lo, hi

    def add_bin(self, evt):
        b = self.bin_borders
        new_border = (b[-2] + b[-1]) / 2
        b[-1:-1] = [new_border]
        index = len(b) - 2
        c = self.bin_colors
        try:
            new_color = self.color_scheme_copy.bin_colors[index]
        except IndexError:
            new_color = self.color_scheme_copy.bin_colors[-1]
        self.bin_colors[-1:-1] = [new_color]
        self.splitter.update_borders()

    def remove_bin(self, evt):
        print(self.bin_borders)
        print(self.bin_colors)
        if len(self.bin_borders) < 4:
            return
        self.bin_borders[-2:-1] = []
        self.bin_colors[-2:-1] = []
        self.splitter.update_borders()

    def get_colormap(self, name):
        try:
            d = get_colormap(name, True).copy()
        except KeyError:
            # not a valid discrete colormap name; start from the first item
            name = self.colormap_list.colormap_name_order[0]
            d = builtin_discrete_colormaps[name].copy()
        return d

    def populate_panel(self, name):
        self.colormap_list.set_selection_by_name(name)
        d = self.working_copy = self.get_colormap(name)
        self.color_scheme_copy = self.working_copy.copy()
        self.bin_borders = list(d.bin_borders)
        self.bin_colors = list(d.bin_colors)
        self.bin_borders[0:0] = [None]  # alternates color, val, color, val ... color
        self.scale_data()
        self.update_panel_controls()

    def regenerate_colormap(self):
        print("bin_borders(%d):%s" % (len(self.bin_borders), str(self.bin_borders)))
        print("bin_colors(%d):%s" % (len(self.bin_colors), str(self.bin_colors)))
        name = "custom"
        cmap = ListedBoundedColormap(self.bin_colors, name)
        values = self.bin_borders[1:]
        d = DiscreteColormap(name, cmap)
        d.set_values(values)
        self.working_copy = d

    def update_bitmap(self):
        self.regenerate_colormap()

    def update_panel_controls(self):
        self.update_bitmap()
        self.splitter.update_borders()

    def colormap_changed(self, evt):
        name = self.colormap_list.get_selected_name()
        self.color_scheme_copy = self.get_colormap(name)

    def get_edited_colormap(self):
        self.regenerate_colormap()
        return self.working_copy

    def boundary_changed(self, entry_num, val):
        if entry_num == 0:
            raise ValueError("How are you changing the hidden value?")
        if entry_num < len(self.bin_borders) - 1 and val >= self.bin_borders[entry_num + 1]:
            raise ValueError("%f is larger than next larger bin value %f" % (val, self.bin_borders[entry_num + 1]))
        if entry_num > 1 and val <= self.bin_borders[entry_num - 1]:
            raise ValueError("%f is larger than next smaller bin value %f" % (val, self.bin_borders[entry_num - 1]))
        self.bin_borders[entry_num] = val
        wx.CallAfter(self.update_bitmap)

    def set_color(self, entry_num, color):
        self.bin_colors[entry_num] = color
        wx.CallAfter(self.update_bitmap)

    def calc_percentages_of_bins(self):
        if self.bin_borders[0] is None:
            bins = self.bin_borders[1:]
        else:
            bins = self.bin_borders
        lo = min(bins)
        hi = max(bins)
        delta = hi - lo
        if delta == 0.0:
            hi = lo + 1.0
            delta = 1.0
        new_bins = [(v - lo) / delta for v in bins]
        return new_bins

    def perc_to_value(self, perc):
        lo, hi = self.min_max
        value = lo + (hi - lo) * perc
        return value

    def scale_data(self):
        # Rescale current colormap to match data instead of autoscaling when
        # map is loaded
        if self.values_min_max is None:
            log.debug("no min/max values specified when creating dialog box")
        else:
            temp = self.calc_percentages_of_bins()
            lo, hi = self.min_max
            delta = hi - lo
            self.bin_borders = [(v * delta) + lo for v in temp]
            self.bin_borders[0:0] = [None]  # Insert first dummy value
