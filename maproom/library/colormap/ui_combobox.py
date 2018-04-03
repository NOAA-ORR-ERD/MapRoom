import collections

import wx
import wx.adv
import numpy as np
import numpy.random as rand

from . import builtin_discrete_colormaps, builtin_continuous_colormaps, get_colormap
from ...ui.buttons import ColorSelectButton


class ColormapImage(object):
    def __init__(self, width, height):
        self.height = height
        self.create_image_array(width)

    def create_image_array(self, width):
        self.width = width
        self.array = np.empty((self.height, width, 3), dtype='uint8')
        self.image = wx.ImageFromBuffer(width, self.height, self.array)

    def calc_bitmap(self, colormap, width=None):
        if width is not None:
            if width != self.width:
                self.create_image_array(width)
        colors = colormap.calc_rgb_texture(self.width)
        self.array[:,:,:] = colors
        return wx.BitmapFromImage(self.image)


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

    def index_of(self, name):
        return self.colormap_name_order.index(name)

    def set_selection_by_name(self, name):
        try:
            index = self.index_of(name)
        except ValueError:
            return
        self.SetSelection(index)

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
            b = self.control_image.calc_bitmap(c, width)
            dc.DrawBitmap(b, r.x + x, r.y + y)
        else:
            width = r.width - 2 * self.internal_spacing
            b = self.dropdown_image.calc_bitmap(c, width)
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


class ColormapEntry(wx.Panel):
    def __init__(self, parent, num):
        wx.Panel.__init__(self, parent, -1, name="panel#%d" % num)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)

        self.text_box = wx.Panel(self, -1)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        t = wx.StaticText(self.text_box, -1, "Bin Boundary")
        hbox.Add(t, 0, wx.ALIGN_CENTER_VERTICAL)
        self.text = wx.TextCtrl(self.text_box, -1, size=(100, 20))
        hbox.Add(self.text, 1, wx.EXPAND)
        self.text_box.SetSizer(hbox)
        self.text_box.Layout()
        self.sizer.Add(self.text_box, 0, wx.EXPAND)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.color = ColorSelectButton(self, -1, "", (0, 0, 0), size=(100, 20))
        hbox.Add(self.color, 1, wx.EXPAND)
        self.sizer.Add(hbox, 0, wx.EXPAND)

        self.Layout()

    def set_values(self, value, color):
        if value is None:
            if self.sizer.GetItem(self.text_box):
                self.sizer.Detach(self.text_box)
            self.text_box.Hide()
        else:
            self.text.SetValue(str(value))
            if not self.sizer.GetItem(self.text_box):
                self.sizer.Insert(0, self.text_box, 0, wx.ALIGN_CENTER_VERTICAL)
            self.text_box.Show()
        int_colors = list((color * 255.0).astype(np.uint8))
        print("color", int_colors)
        self.color.SetColour(int_colors)
        self.sizer.Layout()
        self.Fit()


class DiscreteColormapDialog(wx.Dialog):
    displayed_style_types = ["Line style", "Line width", "Line color", "Start marker", "End marker", "Line transparency", "Fill style", "Fill color", "Fill transparency", "Text color", "Font", "Font size", "Text transparency", "Outline color", "Outline transparency", "Marplot icon"]

    def __init__(self, parent, current):
        wx.Dialog.__init__(self, parent, -1, "Edit Discrete Colormaps", size=(500, -1))

        sizer = wx.BoxSizer(wx.VERTICAL)

        self.colormap_list = DiscreteOnlyColormapComboBox(self, -1, "colormap_list", popup_width=300)
        self.colormap_list.Bind(wx.EVT_COMBOBOX, self.colormap_changed)
        self.colormap_list.SetSelection(0)
        sizer.Add(self.colormap_list, 0, wx.EXPAND, 0)

        self.add_hi = wx.Button(self, -1, "Add Bin Above")
        sizer.Add(self.add_hi, 0, wx.ALL|wx.CENTER, 5)

        self.param_panel = wx.Panel(self, -1, name="param_panel")
        self.param_panel.SetSizer(wx.BoxSizer(wx.VERTICAL))
        sizer.Add(self.param_panel, 1, wx.EXPAND, 0)

        self.add_lo = wx.Button(self, -1, "Add Bin Below")
        sizer.Add(self.add_lo, 0, wx.ALL|wx.CENTER, 5)

        self.panel_controls = []

        btnsizer = wx.StdDialogButtonSizer()
        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()
        btnsizer.AddButton(btn)
        btn = wx.Button(self, wx.ID_CANCEL)
        btnsizer.AddButton(btn)
        btnsizer.Realize()

        sizer.Add(btnsizer, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.SetSizer(sizer)

        self.populate_panel(current.name)

    def populate_panel(self, name):
        self.colormap_list.set_selection_by_name(name)
        d = builtin_discrete_colormaps[name]
        self.bin_borders = d.bin_borders
        self.bin_colors = d.bin_colors
        self.bin_borders.append(None)  # alternates color, val, color, val ... color
        self.create_panel_controls()
        self.param_panel.GetSizer().Layout()
        self.Fit()
        self.Layout()

    def create_panel_controls(self):
        for i, (val, color) in enumerate(reversed(zip(self.bin_borders, self.bin_colors))):
            print("creating entry %d for %s,%s" % (i, val, color))
            try:
                c = self.panel_controls[i]
            except IndexError:
                c = ColormapEntry(self.param_panel, i)
                self.param_panel.GetSizer().Add(c, 1, wx.EXPAND, 0)
                self.panel_controls.append(c)
            c.set_values(val, color)
            c.Show()
        i += 1
        while i < len(self.panel_controls):
            self.panel_controls[i].Hide()
            i += 1

    def colormap_changed(self, evt):
        name = self.colormap_list.get_selected_name()
        self.populate_panel(name)

    def get_colormap(self):
        return self.colormap_list.get_selected_name()
