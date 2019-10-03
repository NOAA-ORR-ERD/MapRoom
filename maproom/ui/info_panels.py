import sys
import wx
import wx.adv
from wx.lib.expando import ExpandoTextCtrl

from sawx import art

from . import sliders
from . import dialogs
from . import buttons
from .. import servers
from ..styles import LayerStyle
from ..layers import state, valid_legend_types
from ..library import coordinates
from ..library import colormap
from ..library.colormap.ui_combobox import ColormapComboBox, GnomeColormapDialog
from ..library.textparse import parse_int_string, int_list_to_string
from ..mouse_commands import StyleChangeCommand, StatusCodeColorCommand, SetAnchorCommand, ChangeDepthCommand, MovePointsCommand, TextCommand, BorderWidthCommand
from .. import menu_commands as mec
from ..vector_object_commands import MoveControlPointCommand
from ..renderer import color_floats_to_int, int_to_color_floats, int_to_wx_colour
from ..library.marplot_icons import marplot_icon_id_to_name

import logging
log = logging.getLogger(__name__)


class InfoField(object):
    same_line = False
    display_label = True

    # wx.Sizer proportion of the main control (not the label).  See the
    # wx.Sizer docs, but basically 0 will fix vertical size to initial size, >
    # 0 will fill available space based on the total proportion in the sizer.
    vertical_proportion = 0

    default_width = 100

    popup_width = 300

    def __init__(self, panel, field_name):
        self.field_name = field_name
        self.panel = panel
        self.create()

    def is_displayed(self, layer):
        return True

    def show(self, state=True):
        self.parent.Show(state)

    def hide(self):
        self.show(False)

    def create(self):
        self.parent = wx.Window(self.panel)
        self.box = wx.BoxSizer(wx.VERTICAL)
        self.parent.SetSizer(self.box)
        if self.display_label:
            self.label = wx.StaticText(self.parent, label=self.field_name, style=wx.ST_ELLIPSIZE_END)
            bold_font = self.parent.GetFont()
            bold_font.SetWeight(weight=wx.FONTWEIGHT_BOLD)
            self.label.SetFont(bold_font)
        self.create_all_controls()
        if self.same_line and self.display_label:
            hbox = wx.BoxSizer(wx.HORIZONTAL)
            hbox.Add(self.label, 99, wx.ALIGN_CENTER)
            hbox.AddStretchSpacer(1)
            hbox.Add(self.ctrl, 0, wx.ALIGN_CENTER)
            for extra in self.extra_ctrls:
                hbox.Add(extra, 0, wx.ALIGN_CENTER)
            self.box.Add(hbox, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, self.panel.SIDE_SPACING)
        else:
            if self.display_label:
                self.box.Add(self.label, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, self.panel.SIDE_SPACING)
                self.box.AddSpacer(self.panel.LABEL_SPACING)
            self.box.Add(self.ctrl, self.vertical_proportion, wx.EXPAND | wx.LEFT | wx.RIGHT, self.panel.SIDE_SPACING)
            for extra in self.extra_ctrls:
                self.box.Add(extra, 0, wx.ALIGN_CENTER)
        self.box.AddSpacer(self.panel.VALUE_SPACING)

    def create_all_controls(self):
        self.ctrl = self.create_control()
        if sys.platform.startswith("win"):
            self.ctrl.Bind(wx.EVT_MOUSEWHEEL, self.on_mouse_wheel_scroll)
        self.extra_ctrls = self.create_extra_controls()
        self.create_extra_event_handlers()

    def is_editable_control(self, ctrl):
        return ctrl == self.ctrl

    def create_extra_controls(self):
        return []

    def create_extra_event_handlers(self):
        pass

    def add_to_parent(self):
        self.panel.sizer.Add(self.parent, self.vertical_proportion, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.ALIGN_TOP, 0)
        self.show(True)

    def fill_data(self, layer):
        raise NotImplementedError

    def is_valid(self):
        return True

    def wants_focus(self):
        return False

    def set_focus(self):
        pass

    def process_command(self, cmd):
        # Override the normal refreshing of the InfoPanel when editing the
        # properties here because refreshing them messes up the text editing.
        self.panel.project.process_command(cmd, override_editable_properties_changed=False)

    def on_mouse_wheel_scroll(self, event):
        screen_point = event.GetPosition()
        size = self.ctrl.GetSize()
        if screen_point.x < 0 or screen_point.y < 0 or screen_point.x > size.x or screen_point.y > size.y:
            # print "Mouse not over info panel %s: trying map!" % self
            self.panel.project.layer_canvas.on_mouse_wheel_scroll(event)
            return

        event.Skip()


class LabelField(InfoField):
    same_line = True
    alignment_style = wx.ALIGN_RIGHT

    def create_control(self):
        c = wx.StaticText(self.parent, style=self.alignment_style)
        return c


class SimplePropertyField(LabelField):
    def fill_data(self, layer):
        text = layer.get_info_panel_text(self.field_name)
        self.ctrl.SetLabel(text)


class WholeLinePropertyField(SimplePropertyField):
    same_line = False


class MultiLinePropertyField(InfoField):
    same_line = False

    def fill_data(self, layer):
        text = layer.get_info_panel_text(self.field_name)
        self.ctrl.SetValue(str(text))

    def create_control(self):
        c = ExpandoTextCtrl(self.parent, style=wx.ALIGN_LEFT | wx.TE_READONLY | wx.NO_BORDER)
        return c


class BooleanLabelField(SimplePropertyField):
    def create_extra_controls(self):
        b = buttons.GenBitmapToggleButton(self.parent, -1, None, style=wx.BORDER_NONE | wx.BU_EXACTFIT)  # BU_EXACTFIT removes padding
        b.Bind(wx.EVT_BUTTON, self.on_toggled)
        bitmap = art.find_bitmap('closed-eye')
        b.SetBitmapLabel(bitmap)
        bitmap = art.find_bitmap('eye')
        b.SetBitmapSelected(bitmap)
        b.SetInitialSize()
        self.toggle = b
        return [self.toggle]

    def fill_data(self, layer):
        SimplePropertyField.fill_data(self, layer)
        vis = self.get_visibility(layer)
        self.toggle.SetToggle(vis)

    def on_toggled(self, evt):
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        self.set_visibility(layer, evt.GetIsDown())


class VisibilityField(BooleanLabelField):
    visibility_name = 'DEFINE IN SUBCLASS!'

    def get_visibility(self, layer):
        return self.panel.project.layer_visibility[layer][self.visibility_name]

    def set_visibility(self, layer, state):
        self.panel.project.layer_visibility[layer][self.visibility_name] = state
        self.panel.project.refresh()


class PointVisibilityField(VisibilityField):
    visibility_name = 'points'


class LineVisibilityField(VisibilityField):
    visibility_name = 'lines'


class DepthVisibilityField(VisibilityField):
    visibility_name = 'labels'

    def fill_data(self, layer):
        self.ctrl.SetLabel("")


class TriangleShadingVisibilityField(VisibilityField):
    visibility_name = 'triangles'

    def fill_data(self, layer):
        self.ctrl.SetLabel("")
        vis = self.get_visibility(layer)
        self.toggle.SetToggle(vis)

    def set_visibility(self, layer, state):
        self.panel.project.layer_visibility[layer][self.visibility_name] = state
        self.panel.project.refresh()


class TextEditField(InfoField):
    def create_control(self):
        c = wx.TextCtrl(self.parent)
        c.Bind(wx.EVT_TEXT, self.on_text_changed)
        c.SetEditable(True)
        return c

    def create_extra_event_handlers(self):
        self.ctrl.Bind(wx.EVT_KILL_FOCUS, self.lose_focus)

    def lose_focus(self, evt):
        if self.ctrl.IsShownOnScreen():
            layer = self.panel.project.current_layer
            self.fill_data(layer)
        evt.Skip()

    def fill_data(self, layer):
        try:
            text = self.get_value(layer)
            self.ctrl.Enable(True)
        except IndexError:
            text = ""
            self.ctrl.Enable(False)
        self.ctrl.ChangeValue(text)
        self.is_valid()

    def is_valid(self):
        c = self.ctrl
        c.SetBackgroundColour("#FFFFFF")
        try:
            self.parse_from_string()
            valid = True
        except Exception:
            c.SetBackgroundColour("#FF8080")
            valid = False
        self.ctrl.Refresh()
        return valid

    def parse_from_string(self):
        return self.ctrl.GetValue()

    def on_text_changed(self, evt):
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        self.process_text_change(layer)

    def process_text_change(self, layer):
        layer.name = self.ctrl.GetValue()
        self.panel.ignore_next_update = True
        # a side effect of select_layer() is to make sure the layer name is up-to-date
        self.panel.project.layer_tree_control.set_edit_layer(layer)

    def initial_text_input(self, text):
        self.ctrl.SetValue(text)
        self.ctrl.SetInsertionPointEnd()  # (self.ctrl.GetLastPosition())


class LayerNameField(TextEditField):
    def get_value(self, layer):
        if (layer is None):
            text = ""
        else:
            text = layer.name
        return text

    def process_text_change(self, layer):
        name = self.ctrl.GetValue()
        cmd = mec.RenameLayerCommand(layer, name)
        self.process_command(cmd)
        self.ctrl.SetFocus()


class DefaultDepthField(TextEditField):
    same_line = True

    def get_value(self, layer):
        if (layer is None):
            text = ""
        else:
            text = str(layer.default_depth)
        return text

    def parse_from_string(self):
        return float(self.ctrl.GetValue())

    def process_text_change(self, layer):
        if self.is_valid():
            layer.default_depth = self.parse_from_string()


class PointDepthField(TextEditField):
    same_line = True

    def is_displayed(self, layer):
        return layer.get_num_points_selected() > 0

    def wants_focus(self):
        return True

    def set_focus(self):
        self.ctrl.SetSelection(-1, -1)
        self.ctrl.SetFocus()

    def get_value(self, layer):
        if (layer is None):
            return ""
        conflict = False
        depth = -1
        selected_point_indexes = layer.get_selected_point_indexes()
        for i in selected_point_indexes:
            d = layer.points.z[i]
            if (d != depth):
                if (depth != -1):
                    conflict = True
                    break
                else:
                    depth = d
        s = ""
        if (not conflict):
            s = str(depth)
        return s

    def parse_from_string(self):
        text = self.ctrl.GetValue()
        if text == "":
            return None
        return float(text)

    def process_text_change(self, layer):
        if self.is_valid():
            depth = self.parse_from_string()
            if depth is not None:
                selected_point_indexes = layer.get_selected_point_indexes()
                if len(selected_point_indexes > 0):
                    cmd = ChangeDepthCommand(layer, selected_point_indexes, depth)
                    self.process_command(cmd)


class PointCoordinateField(TextEditField):
    def is_displayed(self, layer):
        return layer.get_num_points_selected() > 0

    def get_point_index(self, layer):
        if (layer is None):
            return -1
        indexes = layer.get_selected_point_indexes()
        if len(indexes) == 0:
            return -1
        return indexes[0]

    def get_value(self, layer):
        i = self.get_point_index(layer)
        if i < 0:
            return ""
        prefs = self.panel.project.preferences
        coords_text = coordinates.format_coords_for_display(layer.points.x[i], layer.points.y[i], prefs.coordinate_display_format)
        lat_text, long_text = coords_text.split(", ")
        if "long" in self.field_name:
            return long_text
        else:
            return lat_text

    def parse_from_string(self):
        text = self.ctrl.GetValue()
        c = coordinates.lat_or_lon_from_format_string(text)
        return c

    def get_command(self, layer, index, dx, dy):
        return MovePointsCommand(layer, [index], dx, dy)

    def process_text_change(self, layer):
        if self.is_valid():
            new_point = self.parse_from_string()
            index = self.get_point_index(layer)
            current_point = (layer.points.x[index], layer.points.y[index])
            if "long" in self.field_name:
                x_diff = new_point - current_point[0]
                y_diff = 0.0
            else:
                x_diff = 0.0
                y_diff = new_point - current_point[1]
            cmd = self.get_command(layer, index, x_diff, y_diff)
            self.process_command(cmd)


class AnchorCoordinateField(PointCoordinateField):
    def is_displayed(self, layer):
        return True

    def get_point_index(self, layer):
        if (layer is None):
            return -1
        indexes = layer.get_selected_point_indexes()
        if len(indexes) == 0:
            return layer.center_point_index
        return indexes[0]

    def get_command(self, layer, index, dx, dy):
        return MoveControlPointCommand(layer, index, index, dx, dy, None, None)


class AnchorPointField(InfoField):
    same_line = True

    def fill_data(self, layer):
        self.ctrl.Set(layer.control_point_names)
        self.ctrl.SetSelection(layer.anchor_point_index)

    def create_control(self):
        names = [str(s) for s in range(9)]
        c = wx.ComboBox(self.parent, -1, "",
                        size=(self.default_width, -1), choices=names, style=wx.CB_READONLY)
        c.Bind(wx.EVT_COMBOBOX, self.anchor_changed)
        return c

    def anchor_changed(self, event):
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        item = event.GetSelection()
        if layer.is_folder():
            layer = layer.get_layer_of_anchor()
        cmd = SetAnchorCommand(layer, item)
        self.process_command(cmd)


class PointIndexesField(TextEditField):
    def is_displayed(self, layer):
        return layer.get_num_points_selected() > 0

    def get_value(self, layer):
        if (layer is None):
            return ""
        selected_point_indexes = layer.get_selected_point_indexes()
        if len(selected_point_indexes) > 0:
            values = [x + 1 for x in selected_point_indexes]
            s = int_list_to_string(values)
        else:
            s = ""
        return s

    def process_text_change(self, layer):
        c = self.ctrl
        c.SetBackgroundColour("#FFFFFF")
        try:
            one_based_values, error = parse_int_string(c.GetValue())
            values = [x - 1 for x in one_based_values]
            self.panel.project.layer_canvas.do_select_points(layer, values)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            c.SetBackgroundColour("#FF8080")


class DropDownField(InfoField):
    def get_choices(self, layer):
        return []

    def get_value(self, layer):
        return ""

    def fill_data(self, layer):
        choices = self.get_choices(layer)
        self.ctrl.SetItems(choices)
        default_choice = self.get_value(layer)
        self.ctrl.SetSelection(choices.index(default_choice))

    def create_control(self):
        c = wx.Choice(self.parent, choices=[])
        c.Bind(wx.EVT_CHOICE, self.drop_down_changed)
        return c

    def drop_down_changed(self, event):
        pass


class DepthUnitField(DropDownField):
    same_line = True

    def get_choices(self, layer):
        return ["unknown", "meters", "feet", "fathoms"]

    def get_value(self, layer):
        return layer.depth_unit

    def drop_down_changed(self, event):
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        layer.depth_unit = self.ctrl.GetString(self.ctrl.GetSelection())


class FlaggedPointsField(DropDownField):
    same_line = True

    def is_displayed(self, layer):
        return layer.get_num_points_selected(state.FLAGGED) > 0

    def get_choices(self, layer):
        selected = ["Total: %d" % layer.get_num_points_selected(state.FLAGGED)]
        selected += [str(i + 1) for i in layer.get_selected_point_indexes(state.FLAGGED)]
        return selected

    def get_value(self, layer):
        return self.get_choices(layer)[0]

    def drop_down_changed(self, event):
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        c = self.ctrl
        item_num = c.GetSelection()
        if item_num > 0:
            point_index = int(c.GetString(item_num)) - 1
            self.panel.project.layer_canvas.do_select_points(layer, [point_index])


class FloatSliderField(InfoField):
    def get_value(self, layer):
        return 0

    def fill_data(self, layer):
        value = self.get_value(layer)
        self.ctrl.SetValue(value)

    def get_params(self):
        return 0, 100, 100

    def normalize(self, val):
        minval, maxval, steps = self.get_params()
        norm = (val - minval) / float(maxval - minval)
        return norm

    def scale(self, norm):
        minval, maxval, steps = self.get_params()
        scaled = norm * (maxval - minval) + minval
        return scaled

    def create_control(self):
        minval, maxval, steps = self.get_params()
        c = sliders.TextSlider(self.parent, -1, minval, minval, maxval, steps, num_digits=1)
        c.Bind(wx.EVT_SLIDER, self.slider_changed)
        c.Bind(wx.EVT_SPINCTRLDOUBLE, self.slider_changed)
        return c

    def slider_changed(self, event):
        pass


class XPercentageField(FloatSliderField):
    def get_value(self, layer):
        return self.scale(layer.x_percentage)

    def set_value(self, layer, val):
        layer.x_percentage = self.normalize(val)

    def slider_changed(self, event):
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        c = self.ctrl
        self.set_value(layer, c.GetValue())
        layer.manager.layer_contents_changed_event(layer)
        layer.manager.refresh_needed_event(None)

class YPercentageField(XPercentageField):
    def get_value(self, layer):
        return self.scale(layer.y_percentage)

    def set_value(self, layer, val):
        layer.y_percentage = self.normalize(val)

class MagnificationPercentageField(XPercentageField):
    def get_value(self, layer):
        return self.scale(layer.magnification)

    def set_value(self, layer, val):
        layer.magnification = self.normalize(val)


class PointSizeField(XPercentageField):
    def get_params(self):
        return 1, 16, 15

    def get_value(self, layer):
        return layer.point_size

    def set_value(self, layer, val):
        layer.point_size = val


class TransparencyField(FloatSliderField):
    # Objects use the line alpha value as the object tranparency
    def get_layer_color(self, layer):
        return layer.style.line_color

    def get_layer_style(self, layer, color):
        return LayerStyle(line_color=color)

    def get_style(self, layer, alpha):
        color = self.get_layer_color(layer)
        r, g, b, _ = int_to_color_floats(color)
        color = color_floats_to_int(r, g, b, alpha)
        return self.get_layer_style(layer, color)

    def get_value(self, layer):
        color = self.get_layer_color(layer)
        r, g, b, a = int_to_color_floats(color)
        t = int((1.0 - a) * 100)
        return t

    def slider_changed(self, event):
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        c = self.ctrl
        try:
            val = (100 - int(c.GetValue())) / 100.0
            style = self.get_style(layer, val)
            cmd = StyleChangeCommand(layer, style)
            wx.CallAfter(self.process_command, cmd)
            c.textCtrl.SetBackgroundColour("#FFFFFF")
        except Exception:
            c.textCtrl.SetBackgroundColour("#FF8080")


class IconSizeField(TransparencyField):
    def get_params(self):
        return 8, 64, 16

    def get_layer_style(self, layer, size):
        return LayerStyle(icon_pixel_size=size)

    def get_value(self, layer):
        return layer.style.icon_pixel_size

    def slider_changed(self, event):
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        c = self.ctrl
        style = self.get_layer_style(layer, int(c.GetValue()))
        cmd = StyleChangeCommand(layer, style)
        wx.CallAfter(self.process_command, cmd)



class ColorPickerField(InfoField):
    same_line = True

    default_width = 40

    def get_value(self, layer):
        return ""

    def fill_data(self, layer):
        rgba = int_to_wx_colour(self.get_value(layer))
        self.ctrl.SetColour(rgba)

    def create_control(self):
        color = (0, 0, 0)
        c = buttons.ColorSelectButton(self.parent, -1, "", color, size=(self.default_width, -1))
        c.Bind(buttons.EVT_COLORSELECT, self.color_changed)
        return c

    def color_changed(self, event):
        color = [float(c / 255.0) for c in event.GetValue()]
        int_color = color_floats_to_int(*color)
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        style = self.get_style(int_color)
        cmd = StyleChangeCommand(layer, style)
        self.process_command(cmd)


class ColorField(ColorPickerField):
    def get_style(self, color):
        return LayerStyle(line_color=color)

    def get_value(self, layer):
        return layer.style.line_color


class FillColorField(ColorPickerField):
    def get_style(self, color):
        return LayerStyle(fill_color=color)

    def get_value(self, layer):
        return layer.style.fill_color


class OutlineColorField(ColorPickerField):
    def get_style(self, color):
        return LayerStyle(outline_color=color)

    def get_value(self, layer):
        return layer.style.outline_color


class TextColorField(ColorPickerField):
    def get_style(self, color):
        return LayerStyle(text_color=color)

    def get_value(self, layer):
        return layer.style.text_color


class PenStyleComboBox(wx.adv.OwnerDrawnComboBox):

    # Overridden from OwnerDrawnComboBox, called to draw each
    # item in the list
    def OnDrawItem(self, dc, rect, item, flags):
        if item == wx.NOT_FOUND:
            # painting the control, but there is no valid item selected yet
            return

        r = wx.Rect(*rect)  # make a copy
        r.Deflate(3, 5)

        line_style = LayerStyle.line_styles[item]
        penStyle = line_style[2]
        pen = wx.Pen(dc.GetTextForeground(), 1, penStyle)
        dc.SetPen(pen)

        if flags & wx.adv.ODCB_PAINTING_CONTROL:
            # for painting the control itself
            dc.DrawLine(r.x + 5, r.y + r.height / 2, r.x + r.width - 5, r.y + r.height / 2)

        else:
            # for painting the items in the popup
            dc.DrawText(self.GetString(item),
                        r.x + 3,
                        (r.y + 0) + ((r.height / 2) - dc.GetCharHeight()) / 2
                        )
            dc.DrawLine(r.x + 5, r.y + ((r.height / 4) * 3) + 1, r.x + r.width - 5, r.y + ((r.height / 4) * 3) + 1)

    # Overridden from OwnerDrawnComboBox, called for drawing the
    # background area of each item.
    def OnDrawBackground(self, dc, rect, item, flags):
        # If the item is selected, or its item # iseven, or we are painting the
        # combo control itself, then use the default rendering.
        if (item & 1 == 0 or flags & (wx.adv.ODCB_PAINTING_CONTROL |
                                      wx.adv.ODCB_PAINTING_SELECTED)):
            wx.adv.OwnerDrawnComboBox.OnDrawBackground(self, dc, rect, item, flags)
            return

        # Otherwise, draw every other background with different colour.
        bgCol = wx.Colour(240, 240, 250)
        dc.SetBrush(wx.Brush(bgCol))
        dc.SetPen(wx.Pen(bgCol))
        dc.DrawRectangle(rect)

    # Overridden from OwnerDrawnComboBox, should return the height
    # needed to display an item in the popup, or -1 for default
    def OnMeasureItem(self, item):
        return 24

    # Overridden from OwnerDrawnComboBox.  Callback for item width, or
    # -1 for default/undetermined
    def OnMeasureItemWidth(self, item):
        return -1  # default - will be measured from text width


class LineStyleField(InfoField):
    same_line = True

    def fill_data(self, layer):
        index, style = layer.style.get_current_line_style()
        self.ctrl.SetSelection(index)

    def create_control(self):
        names = [s[0] for s in LayerStyle.line_styles]
        c = PenStyleComboBox(self.parent, -1, "", size=(self.default_width, -1), choices=names,
                             style=wx.CB_READONLY)
        c.Bind(wx.EVT_COMBOBOX, self.style_changed)
        return c

    def style_changed(self, event):
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        item = event.GetSelection()
        line_style = LayerStyle.line_styles[item]
        style = LayerStyle(line_stipple=line_style[1])
        cmd = StyleChangeCommand(layer, style)
        self.process_command(cmd)


class BorderWidthField(InfoField):
    same_line = True
    display_widths = [str(s) for s in range(21)]

    def fill_data(self, layer):
        try:
            index = self.display_widths.index(str(layer.border_width))
            self.ctrl.SetSelection(index)
        except ValueError:
            self.ctrl.ChangeValue(str(layer.border_width))
            pass

    def create_control(self):
        c = wx.ComboBox(self.parent, -1, "", size=(self.default_width, -1), choices=self.display_widths)
        c.Bind(wx.EVT_COMBOBOX, self.width_from_list)
        c.Bind(wx.EVT_TEXT, self.width_from_text)
        return c

    def width_from_list(self, event):
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        item = self.display_widths[event.GetSelection()]
        cmd = BorderWidthCommand(layer, int(item))
        self.process_command(cmd)

    def width_from_text(self, event):
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        try:
            width = int(self.ctrl.GetValue())
        except ValueError:
            width = 0
        cmd = BorderWidthCommand(layer, width)
        self.process_command(cmd)


class LineWidthField(InfoField):
    same_line = True

    def fill_data(self, layer):
        index, width = layer.style.get_current_line_width()
        self.ctrl.SetSelection(index)

    def create_control(self):
        names = [str(s) for s in LayerStyle.standard_line_widths]
        c = wx.ComboBox(self.parent, -1, "", size=(self.default_width, -1), choices=names,
                        style=wx.CB_READONLY)
        c.Bind(wx.EVT_COMBOBOX, self.style_changed)
        return c

    def style_changed(self, event):
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        item = event.GetSelection()
        line_width = LayerStyle.standard_line_widths[item]
        style = LayerStyle(line_width=line_width)
        cmd = StyleChangeCommand(layer, style)
        self.process_command(cmd)


class FillStyleField(InfoField):
    same_line = True

    def fill_data(self, layer):
        index, style = layer.style.get_current_fill_style()
        self.ctrl.SetSelection(index)

    def create_control(self):
        c = wx.adv.BitmapComboBox(self.parent, -1, "", size=(self.default_width, -1),
                                    style=wx.CB_READONLY)
        for i, s in LayerStyle.fill_styles.items():
            c.Append(s[0])

        c.Bind(wx.EVT_COMBOBOX, self.style_changed)
        return c

    def style_changed(self, event):
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        item = event.GetSelection()
        style = LayerStyle(fill_style=item)
        cmd = StyleChangeCommand(layer, style)
        self.process_command(cmd)


class MultiLineTextField(InfoField):
    def create_control(self):
        c = wx.TextCtrl(self.parent, style=wx.TE_MULTILINE, size=(-1,100))
        c.Bind(wx.EVT_TEXT, self.on_text_changed)
        c.SetEditable(True)
        return c

    def fill_data(self, layer):
        text = self.get_value(layer)
        current = self.ctrl.GetValue()
        # ChangeValue also sets the cursor to the end, so if the value in the
        # layer and the value in the control are the same, we assume that
        # we're in the middle of an edit (meaning fill_data has been called
        # by InfoPanel.set_fields) and we leave the cursor in place.
        if text != current:
            self.ctrl.ChangeValue(text)

    def on_text_changed(self, evt):
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        self.process_text_change(layer)

    def process_text_change(self, layer):
        pass

    def initial_text_input(self, text):
        self.ctrl.SetValue(text)
        self.ctrl.SetInsertionPointEnd()


class OverlayTextField(MultiLineTextField):
    vertical_proportion = 1

    def wants_focus(self):
        return True

    def set_focus(self):
        text = self.ctrl.GetValue()
        if text == "<b>New Label</b>":
            self.ctrl.SetSelection(-1, -1)
        self.ctrl.SetFocus()

    def get_value(self, layer):
        if (layer is None):
            return ""
        return layer.user_text

    def process_text_change(self, layer):
        text = self.ctrl.GetValue()
        layer = layer.get_text_box()  # it might be a sublayer
        cmd = TextCommand(layer, text)
        self.process_command(cmd)


class TextFormatField(InfoField):
    same_line = True

    def fill_data(self, layer):
        self.ctrl.SetSelection(layer.style.text_format)

    def create_control(self):
        names = [s[0] for s in LayerStyle.text_format_styles]
        c = wx.ComboBox(self.parent, -1, "", size=(self.default_width, -1), choices=names,
                        style=wx.CB_READONLY)
        c.Bind(wx.EVT_COMBOBOX, self.style_changed)
        return c

    def create_extra_controls(self):
        b = wx.Button(self.parent, -1, "?", style=wx.BU_EXACTFIT)
        b.Bind(wx.EVT_BUTTON, self.on_help)
        return [b]

    def on_help(self, event):
        item = self.ctrl.GetSelection()
        help_panes = [s[0].lower().replace(" ", "_") for s in LayerStyle.text_format_styles]
        task = self.panel.project.task

        # Show only the help text pane matching the current selection
        for index, name in enumerate(help_panes):
            pane = task.window.get_dock_pane('maproom.%s_help_pane' % name)
            if pane is not None:
                pane.visible = (index == item)

    def style_changed(self, event):
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        item = event.GetSelection()
        style = LayerStyle(text_format=item)
        cmd = StyleChangeCommand(layer, style)
        self.process_command(cmd)


class FontComboBox(wx.adv.OwnerDrawnComboBox):
    # Overridden from OwnerDrawnComboBox, called to draw each
    # item in the list
    def OnDrawItem(self, dc, rect, item, flags):
        if item == wx.NOT_FOUND:
            # painting the control, but there is no valid item selected yet
            return

        r = wx.Rect(*rect)  # make a copy
        r.Deflate(3, 5)

        face = LayerStyle.get_font_name(item)
        font = wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.NORMAL, False, face)
        dc.SetFont(font)

        if flags & wx.adv.ODCB_PAINTING_CONTROL:
            # for painting the control itself
            dc.DrawText(face, r.x + 5, (r.y + 5) + ((r.height / 2) - dc.GetCharHeight()) / 2)

        else:
            # for painting the items in the popup
            dc.DrawText(face,
                        r.x + 3,
                        (r.y + 5) + ((r.height / 2) - dc.GetCharHeight()) / 2
                        )

    # Overridden from OwnerDrawnComboBox, should return the height
    # needed to display an item in the popup, or -1 for default
    def OnMeasureItem(self, item):
        return 24

    # Overridden from OwnerDrawnComboBox.  Callback for item width, or
    # -1 for default/undetermined
    def OnMeasureItemWidth(self, item):
        return -1  # default - will be measured from text width


class FontStyleField(InfoField):
    same_line = True

    def fill_data(self, layer):
        index, style = layer.style.get_current_font()
        self.ctrl.SetSelection(index)

    def create_control(self):
        names = LayerStyle.get_font_names()
        c = FontComboBox(self.parent, -1, "", size=(self.default_width, -1), choices=names,
                         style=wx.CB_READONLY)
        c.Bind(wx.EVT_COMBOBOX, self.style_changed)
        return c

    def style_changed(self, event):
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        item = event.GetSelection()
        font = LayerStyle.get_font_name(item)
        style = LayerStyle(font=font)
        cmd = StyleChangeCommand(layer, style)
        self.process_command(cmd)


class FontSizeField(InfoField):
    same_line = True

    def fill_data(self, layer):
        index, style = layer.style.get_current_font_size()
        self.ctrl.SetSelection(index)

    def create_control(self):
        names = [str(s) for s in LayerStyle.standard_font_sizes]
        c = wx.ComboBox(self.parent, -1, str(LayerStyle.default_font_size),
                        size=(self.default_width, -1), choices=names, style=wx.CB_READONLY)
        c.Bind(wx.EVT_COMBOBOX, self.style_changed)
        return c

    def style_changed(self, event):
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        item = event.GetSelection()
        size = LayerStyle.standard_font_sizes[item]
        style = LayerStyle(font_size=size)
        cmd = StyleChangeCommand(layer, style)
        self.process_command(cmd)


class MarkerField(InfoField):
    same_line = True

    def get_marker(self, layer):
        raise NotImplementedError

    def get_style(self, marker):
        raise NotImplementedError

    def fill_data(self, layer):
        marker = self.get_marker(layer)
        self.ctrl.SetSelection(marker)

    def create_control(self):
        names = [m[0] for m in LayerStyle.marker_styles]
        c = wx.ComboBox(self.parent, -1, "", size=(self.default_width, -1), choices=names,
                        style=wx.CB_READONLY)
        c.Bind(wx.EVT_COMBOBOX, self.style_changed)
        return c

    def style_changed(self, event):
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        item = event.GetSelection()
        style = self.get_style(item)
        cmd = StyleChangeCommand(layer, style)
        self.process_command(cmd)


class StartMarkerField(MarkerField):
    def get_marker(self, layer):
        return layer.style.line_start_marker

    def get_style(self, marker):
        return LayerStyle(line_start_marker=marker)


class EndMarkerField(MarkerField):
    def get_marker(self, layer):
        return layer.style.line_end_marker

    def get_style(self, marker):
        return LayerStyle(line_end_marker=marker)


class MarplotIconField(InfoField):
    same_line = True

    def get_marker(self, layer):
        return layer.style.icon_marker

    def get_style(self, marker):
        return LayerStyle(icon_marker=marker)

    def fill_data(self, layer):
        marker = self.get_marker(layer)
        self.ctrl.SetLabel(marplot_icon_id_to_name[marker])

    def create_control(self):
        c = wx.Button(self.parent, -1, "none", size=(self.default_width, -1))
        c.Bind(wx.EVT_BUTTON, self.style_changed)
        return c

    def style_changed(self, event):
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        marker = self.get_marker(layer)
        d = dialogs.IconDialog(self.panel.project.control, marker)
        new_id = d.ShowModal()
        if new_id != wx.ID_CANCEL:
            style = self.get_style(new_id)
            cmd = StyleChangeCommand(layer, style)
            self.process_command(cmd)
            self.fill_data(layer)


class ListBoxComboPopup(wx.ComboPopup):
    """Popup for wx.adv.ComboCtrl, based on wxPython demo and converted to
    the ListBox rather than the full ListCtrl
    """

    def __init__(self):
        # Since we are using multiple inheritance, and don't know yet
        # which window is to be the parent, we'll do 2-phase create of
        # the ListCtrl instead, and call its Create method later in
        # our Create method.  (See Create below.)
#        self.PostCreate(wx.PreListBox())

        # Need to call this last so the ComboCtrl recognizes that this is of
        # type ComboPopup
        wx.ComboPopup.__init__(self)
        self.lc = None

    def OnMotion(self, evt):
        if evt.LeftIsDown():
            item = self.lc.HitTest(evt.GetPosition())
            if item >= 0:
                self.lc.Select(item)

    def OnLeftUp(self, evt):
        self.Dismiss()

    # This is called immediately after construction finishes.  You can
    # use self.GetCombo if needed to get to the ComboCtrl instance.
    def Init(self):
        self.value = -1
        self.curitem = -1

    # Create the popup child control.  Return true for success.
    def Create(self, parent):
        self.lc = wx.ListBox(parent, style=wx.LB_SINGLE | wx.SIMPLE_BORDER)
        self.lc.Bind(wx.EVT_MOTION, self.OnMotion)
        self.lc.Bind(wx.EVT_LEFT_DOWN, self.OnMotion)
        self.lc.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
        return True

    # Return the widget that is to be used for the popup
    def GetControl(self):
        return self.lc

    # Called just prior to displaying the popup, you can use it to
    # 'select' the current item.
    def SetStringValue(self, val):
        idx = self.lc.FindString(val)
        if idx != wx.NOT_FOUND:
            self.lc.Select(idx)

    # Return a string representation of the current item.
    def GetStringValue(self):
        if self.value >= 0:
            return self.lc.GetItemText(self.value)
        return ""

    # Called when popup is dismissed
    def OnDismiss(self):
        ev = wx.CommandEvent(wx.EVT_COMBOBOX.typeId)
        ev.SetInt(self.lc.GetSelection())
        self.lc.GetEventHandler().ProcessEvent(ev)
        wx.ComboPopup.OnDismiss(self)

    # Return final size of popup. Called on every popup, just prior to OnPopup.
    # minWidth = preferred minimum width for window
    # prefHeight = preferred height. Only applies if > 0,
    # maxHeight = max height for window, as limited by screen size
    #   and should only be rounded down, if necessary.
    def GetAdjustedSize(self, minWidth, prefHeight, maxHeight):
        return wx.ComboPopup.GetAdjustedSize(self, InfoField.popup_width, prefHeight, maxHeight)

    def SetItems(self, stuff):
        if stuff:
            self.lc.SetItems(stuff)
        else:
            self.lc.Clear()

    def SetSelection(self, index):
        self.lc.SetSelection(index)


class ParticleField(InfoField):
    same_line = True

    def get_valid_timestep_names(self, layer):
        children = layer.get_particle_layers()
        self.total_timesteps = len(children)
        names = [c.name for c in children]
        return names

    def get_timestep_index(self, layer):
        raise NotImplementedError

    def fill_data(self, layer):
        names = self.get_valid_timestep_names(layer)
        self.popup.SetItems(names)
        if names:
            selected = self.get_timestep_index(layer)
            self.popup.SetSelection(selected)
            self.ctrl.SetText(names[selected])
        else:
            self.ctrl.SetText("")

    def create_control(self):
        c = wx.ComboCtrl(self.parent, style=wx.CB_READONLY, size=(self.default_width, -1))
        self.popup = ListBoxComboPopup()
        c.SetPopupControl(self.popup)
        c.Bind(wx.EVT_COMBOBOX, self.timestep_changed)
        return c

    def timestep_changed(self, event):
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        index = event.GetSelection()
        self.highlight_timestep(layer, index)

    def highlight_timestep(self, layer, index):
        raise NotImplementedError


class ParticleStartField(ParticleField):
    def get_timestep_index(self, layer):
        i = layer.start_index
        if i < 0:
            i = 0
        return i

    def highlight_timestep(self, layer, index):
        layer.set_start_index(index)
        layer.update_timestep_visibility(self.panel.project)


class ParticleEndField(ParticleField):
    def get_timestep_index(self, layer):
        i = layer.end_index
        last_index = self.total_timesteps - 1
        if i > last_index:
            i = last_index
        return i

    def highlight_timestep(self, layer, index):
        layer.set_end_index(index)
        layer.update_timestep_visibility(self.panel.project)


class StatusCodeColorField(InfoField):
    same_line = False

    default_width = 40

    def is_displayed(self, layer):
        return not layer.is_using_colormap()

    def get_value(self, layer):
        pass

    def fill_data(self, layer):
        ctrls = {}
        code_map = layer.status_code_names
        code_colors = layer.status_code_colors
        codes = sorted(code_map.keys())
        sizer = self.ctrl.GetSizer()
        sizer.Clear(True)
        for code in codes:
            if layer.status_code_count[code] == 0:
                continue
            hbox = wx.BoxSizer(wx.HORIZONTAL)
            label = wx.StaticText(self.ctrl, label=layer.status_code_label(code), style=wx.ST_ELLIPSIZE_END)
            hbox.Add(label, 99, wx.ALIGN_CENTER)
            hbox.AddStretchSpacer(1)
            color = tuple(int(255 * c) for c in int_to_color_floats(code_colors[code])[0:3])
            c = buttons.ColorSelectButton(self.ctrl, -1, "", color, size=(self.default_width, -1))
            c.Bind(buttons.EVT_COLORSELECT, self.color_changed)
            hbox.Add(c, 0, wx.ALIGN_CENTER)
            sizer.Add(hbox, self.vertical_proportion, wx.EXPAND | wx.LEFT, self.panel.SIDE_SPACING)
            ctrls[id(c)] = code
        self.color_ctrls = ctrls
        self.ctrl.Fit()

    def create_control(self):
        panel = wx.Panel(self.parent)
        vbox = wx.BoxSizer(wx.VERTICAL)
        panel.SetSizer(vbox)
        return panel

    def color_changed(self, event):
        ctrl = event.GetEventObject()
        code = self.color_ctrls[id(ctrl)]
        color = [float(c / 255.0) for c in event.GetValue()]
        int_color = color_floats_to_int(*color)
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        layers = layer.get_particle_layers()
        cmd = StatusCodeColorCommand(layers, code, int_color)
        self.process_command(cmd)


class MapServerField(InfoField):
    same_line = False

    def fill_data(self, layer):
        names = layer.get_server_names()
        self.ctrl.SetItems(names)
        self.ctrl.SetSelection(layer.map_server_id)

    def create_control(self):
        c = wx.ComboBox(self.parent, -1, str(LayerStyle.default_font_size),
                        size=(self.default_width, -1), choices=[], style=wx.CB_READONLY)
        c.Bind(wx.EVT_COMBOBOX, self.style_changed)
        return c

    def style_changed(self, event):
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        item = event.GetSelection()
        layer.change_server_id(item, self.panel.project.layer_canvas)


class MapOverlayField(InfoField):
    same_line = False

    def fill_data(self, layer):
        downloader = layer.get_downloader(layer.map_server_id)
        server = downloader.get_server()
        titles = [str(n[1]) for n in server.get_layer_info()]
        self.ctrl.SetItems(titles)
        self.set_selected(layer, server)

    def set_selected(self, layer, server):
        names = [n[0] for n in server.get_layer_info()]
        selected = []
        if layer.map_layers is not None:
            for i, name in enumerate(names):
                if name in layer.map_layers:
                    selected.append(i)
        self.ctrl.SetChecked(selected)

    def create_control(self):
        names = []
        c = wx.CheckListBox(self.parent, -1, size=(self.default_width, -1), choices=names)
        #c.Bind(wx.EVT_LISTBOX, self.overlay_selected)
        c.Bind(wx.EVT_CHECKLISTBOX, self.overlay_selected)
        c.Bind(wx.EVT_RIGHT_DOWN, self.on_popup)
        self.select_id = wx.NewId()
        self.clear_id = wx.NewId()
        c.Bind(wx.EVT_MENU, self.on_menu)
        return c

    def overlay_selected(self, event):
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        item = event.GetSelection()
        downloader = layer.get_downloader(layer.map_server_id)
        server = downloader.get_server()
        name = server.get_layer_info()[item][0]
        if name in layer.map_layers:
            layer.map_layers.remove(name)
        else:
            layer.map_layers.add(name)
        self.set_selected(layer, server)
        layer.wms_rebuild(self.panel.project.layer_canvas)

    def on_popup(self, event):
        popup = wx.Menu()
        popup.Append(self.select_id, "Select All Layers")
        popup.Append(self.clear_id, "Clear All Selections")
        self.ctrl.PopupMenu(popup, event.GetPosition())

    def on_menu(self, event):
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        downloader = layer.get_downloader(layer.map_server_id)
        server = downloader.get_server()
        if event.GetId() == self.select_id:
            names = [n[0] for n in server.get_layer_info()]
            layer.map_layers = set(names)
        else:
            layer.map_layers = set()
        self.set_selected(layer, server)
        layer.wms_rebuild(self.panel.project.layer_canvas)


class ExpandableErrorField(InfoField):
    same_line = False

    def fill_data(self, layer):
        text, color = self.get_error_text(layer)
        if color is None:
            attr = self.panel.GetDefaultAttributes()
            color = attr.colBg
        self.ctrl.SetBackgroundColour(color)
        self.ctrl.SetValue(str(text))

    def get_error_text(layer):
        raise NotImplementedError

    def create_control(self):
        c = ExpandoTextCtrl(self.parent, style=wx.ALIGN_LEFT | wx.TE_READONLY | wx.NO_BORDER)
        return c


class ServerStatusField(ExpandableErrorField):
    same_line = False

    def get_error_text(self, layer):
        downloader = layer.get_downloader(layer.map_server_id)
        server = downloader.get_server()
        color = None
        if server.has_error():
            text = server.error
            color = "#FF8080"
        elif server.is_valid():
            text = "OK"
        else:
            text = "Initializing"
        return text, color


class ServerReloadField(InfoField):
    display_label = False

    def fill_data(self, layer):
        downloader = layer.get_downloader(layer.map_server_id)
        server = downloader.get_server()
        if server.has_error():
            self.ctrl.Show(True)
        else:
            self.ctrl.Show(False)

    def create_control(self):
        c = wx.Button(self.parent, -1, "Retry")
        c.Bind(wx.EVT_BUTTON, self.on_press)
        return c

    def on_press(self, event):
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        downloader = servers.get_threaded_wms_by_id(layer.map_server_id)
        downloader.get_server_config()
        layer.wms_rebuild(self.panel.project.layer_canvas)


class DownloadStatusField(ExpandableErrorField):
    same_line = False

    def get_error_text(self, layer):
        color = None
        if layer.download_status_text is not None:
            stype, text = layer.download_status_text
            if text is None:
                text = "OK"
            elif stype == "error":
                color = "#FF8080"
        else:
            text = "Waiting for server"
        return text, color

    def create_control(self):
        c = ExpandoTextCtrl(self.parent, style=wx.ALIGN_LEFT | wx.TE_READONLY | wx.NO_BORDER)
        return c


class SVGStatusField(ExpandableErrorField):
    same_line = False

    def get_error_text(self, layer):
        svg = layer.svg
        text = layer.svg_parse_error
        if svg is None or text is not None:
            color = "#FF8080"
        else:
            text = f"Size: {svg.width} x {svg.height}"
            color = None
        return text, color


class ScalarChoiceField(InfoField):
    same_line = True

    def get_variable_names(self, layer):
        names = layer.scalar_var_names
        names.discard("status_codes")
        names = sorted(names)
        names[0:0] = ["status codes"]
        return names

    def get_current_index(self, layer, names):
        current = layer.current_scalar_var
        if current is None or current not in names:
            selected = 0
        else:
            selected = names.index(current)
        return selected

    def fill_data(self, layer):
        self.choice_names = self.get_variable_names(layer)
        self.ctrl.SetItems(self.choice_names)
        if self.choice_names:
            selected = self.get_current_index(layer, self.choice_names)
            self.ctrl.SetSelection(selected)

    def create_control(self):
        c = wx.ComboBox(self.parent, -1, "", size=(self.default_width, -1), choices=[], style=wx.CB_READONLY)
        c.Bind(wx.EVT_COMBOBOX, self.variable_changed)
        return c

    def variable_changed(self, event):
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        index = event.GetSelection()
        var = self.choice_names[index]
        # self.change_variable(layer, var)
        wx.CallAfter(self.change_variable, layer, var)

    def change_variable(self, layer, var):
        layer.set_scalar_var(var)
        self.panel.project.update_info_panels(layer, True)


class ColormapField(InfoField):
    same_line = False

    def is_displayed(self, layer):
        return layer.is_using_colormap()

    def fill_data(self, layer):
        self.ctrl.rebuild_colormap_list()
        self.ctrl.set_selection_by_name(layer.colormap.name)

    def create_control(self):
        c = ColormapComboBox(self.parent, -1, "", size=(self.default_width, -1), popup_width=300)
        c.Bind(wx.EVT_COMBOBOX, self.style_changed)
        return c

    def style_changed(self, event):
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        name = self.ctrl.get_selected_name()
        wx.CallAfter(self.change_variable, layer, name)

    def change_variable(self, layer, name):
        cmap = colormap.get_colormap(name)
        layer.set_colormap(cmap)
        self.panel.project.update_info_panels(layer, True)


class DiscreteColormapField(InfoField):
    display_label = False

    def is_displayed(self, layer):
        return layer.is_using_colormap()

    def fill_data(self, layer):
        pass

    def create_control(self):
        b = wx.Button(self.parent, -1, "Edit Discrete Colormap")
        b.Bind(wx.EVT_BUTTON, self.on_edit_colormap)
        return b

    def on_edit_colormap(self, event):
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        d = GnomeColormapDialog(self.panel.project.control, layer.colormap, layer.current_min_max)
        ret = d.ShowModal()
        if ret != wx.ID_CANCEL:
            cmap = d.get_edited_colormap()
            wx.CallAfter(self.change_variable, layer, cmap)

    def change_variable(self, layer, cmap):
        layer.set_colormap(cmap)
        self.panel.project.update_info_panels(layer, True)


class LegendTypeField(InfoField):
    same_line = False

    def fill_data(self, layer):
        self.ctrl.Clear()
        self.ctrl.Set(valid_legend_types)
        index = self.ctrl.FindString(layer.legend_type)
        self.ctrl.SetSelection(index)

    def create_control(self):
        names = []
        c = wx.ComboBox(self.parent, -1, "", size=(self.default_width, -1), choices=names, style=wx.CB_READONLY)
        c.Bind(wx.EVT_COMBOBOX, self.style_changed)
        return c

    def style_changed(self, event):
        layer = self.panel.project.current_layer
        if (layer is None):
            return
        index = event.GetSelection()
        layer.legend_type = valid_legend_types[index]
        self.panel.project.refresh()


class LegendLabelsField(MultiLineTextField):
    same_line = False

    def get_value(self, layer):
        return layer.legend_labels

    def process_text_change(self, layer):
        layer.legend_labels = self.ctrl.GetValue()
        self.panel.project.refresh()


class ScalarSummaryField(WholeLinePropertyField):
    alignment_style = wx.ALIGN_LEFT

    def fill_data(self, layer):
        text = layer.calc_scalar_value_summary()
        self.ctrl.SetLabel(text)


class ScalarExpressionField(TextEditField):
    same_line = False

    def get_value(self, layer):
        if (layer is None):
            text = ""
        else:
            text = layer.scalar_subset_expression
        return text

    def process_text_change(self, layer):
        expression = self.parse_from_string()
        affected, error = layer.subset_using_logical_operation(expression)
        for layer in affected:
            self.panel.project.document.layer_contents_changed_event(layer)
        self.panel.project.refresh()
        c = self.ctrl
        if error:
            c.SetBackgroundColour("#FF8080")
            self.panel.project.frame.status_message(error)
        else:
            c.SetBackgroundColour("#FFFFFF")
            self.panel.project.frame.status_message("")
        c.Refresh()


class ButtonActionField(InfoField):
    display_label = False

    button_label = ""

    def fill_data(self, layer):
        pass

    def create_control(self):
        c = wx.Button(self.parent, -1, self.button_label)
        c.Bind(wx.EVT_BUTTON, self.button_pressed)
        return c

    def button_pressed(self, event):
        pass


class SaveChangesInPolygon(ButtonActionField):
    button_label = "Save Changes in Polygon"

    def button_pressed(self, event):
        layer = self.panel.project.current_layer
        cmd = mec.PolygonSaveEditLayerCommand(layer)
        self.panel.project.process_command(cmd)


class CancelEditInPolygon(ButtonActionField):
    button_label = "Cancel Edit"

    def button_pressed(self, event):
        layer = self.panel.project.current_layer
        cmd = mec.PolygonCancelEditLayerCommand(layer)
        self.panel.project.process_command(cmd)


class LoadSVG(ButtonActionField):
    button_label = "Load SVG"

    def button_pressed(self, event):
        layer = self.panel.project.current_layer
        svgfile = self.panel.project.frame.prompt_local_file_dialog("Load SVG", wildcard="*.svg")
        if svgfile is not None:
            cmd = mec.LoadSVGCommand(layer, svgfile)
            self.panel.project.process_command(cmd)


class MergeTrianglePoints(ButtonActionField):
    button_label = "Merge Points Into Source Layer"

    def button_pressed(self, event):
        layer = self.panel.project.current_layer
        cmd = mec.MergeGeneratedTrianglePointsCommand(layer)
        self.panel.project.process_command(cmd)


PANELTYPE = wx.lib.scrolledpanel.ScrolledPanel


class InfoPanel(PANELTYPE):

    """
    A panel for displaying and manipulating the properties of a layer.
    """
    LABEL_SPACING = 0
    VALUE_SPACING = 3
    SIDE_SPACING = 5

    window_name = "InfoPanel"

    def __init__(self, parent, project, size=(-1, -1)):
        self.project = project

        self.layer_name_control = None
        self.depth_unit_control = None
        self.default_depth_control = None
        self.point_index_control = None
        self.point_depth_control = None
        self.ignore_next_update = False
        self.current_layer_displayed = None
        self.current_layer_change_count = -1
        self.current_fields = []
        self.field_map = {}
        self.focus_on_input = None

        PANELTYPE.__init__(self, parent, name=self.window_name, size=size)

        # Mac/Win needs this, otherwise background color is black
        attr = self.GetDefaultAttributes()
        self.SetBackgroundColour(attr.colBg)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)

    def display_panel_for_layer(self, project, layer, force_selection_change=False, has_focus=None):
        self.project = project

        if (self.ignore_next_update):
            self.ignore_next_update = False

            return

        different_layer = True
        selection_changed = force_selection_change
        if (layer is None or self.current_layer_displayed is None):
            if (layer == self.current_layer_displayed):
                return
        elif self.current_layer_displayed == layer:
            if not selection_changed and self.current_layer_change_count == layer.change_count:
                return
            if layer.change_count > self.current_layer_change_count + 1:
                # NOTE: this is using an unobvious side effect to determine if
                # the selection has changed.  The selection change is needed
                # to determine whether to set new text in e.g.  the point
                # coordinates box, or leave the text as is so the user can
                # continue to edit.  The change count will always be increased
                # by least 2 when a new point or line is selected (or points
                # added to the selection etc.) in a point_base subclass.
                #
                # Vector object layers operate differently because the
                # selection doesn't apply, but the selection changed should
                # always remain false for these layers because processing a
                # command only increases the layer count by one.
                selection_changed = True
            different_layer = False

        self.current_layer_displayed = layer
        self.current_layer_change_count = -1
        if (layer is not None):
            self.current_layer_change_count = layer.change_count
            log.debug("%s: change count=%s, old count=%s, diff layer=%s, sel_changed=%s" % (layer.name, layer.change_count, self.current_layer_change_count, different_layer, selection_changed))
            fields = self.get_visible_fields(layer)
        else:
            fields = []
        self.display_fields(layer, fields, different_layer, selection_changed, has_focus)

    def display_fields(self, layer, fields, different_layer, selection_changed, has_focus):
        if self.current_fields == fields:
            log.debug("reusing current fields, sel_changed=%s layer=%d" % (layer, selection_changed))
            self.set_fields(layer, fields, different_layer, selection_changed, has_focus)
        else:
            log.debug("creating fields")
            self.create_fields(layer, fields)

    known_fields = {
        "Layer name": LayerNameField,
        "Depth unit": DepthUnitField,
        "Default depth": DefaultDepthField,
        "Point latitude": PointCoordinateField,
        "Point longitude": PointCoordinateField,
        "Point depth": PointDepthField,
        "Point index": PointIndexesField,
        "Point count": PointVisibilityField,
        "Line segment count": LineVisibilityField,
        "Show depth": DepthVisibilityField,
        "Show depth shading": TriangleShadingVisibilityField,
        "Flagged points": FlaggedPointsField,
        "Transparency": TransparencyField,
        "Color": ColorField,
        "Border width": BorderWidthField,
        "Line color": ColorField,
        "Line style": LineStyleField,
        "Line width": LineWidthField,
        "Start marker": StartMarkerField,
        "End marker": EndMarkerField,
        "Fill color": FillColorField,
        "Fill style": FillStyleField,
        "Outline color": OutlineColorField,
        "Text color": TextColorField,  # Same as Line Color except for the label
        "Font": FontStyleField,
        "Font size": FontSizeField,
        "Text": OverlayTextField,
        "Text format": TextFormatField,
        "Marplot icon": MarplotIconField,
        "Icon size": IconSizeField,
        "Start time": ParticleStartField,
        "End time": ParticleEndField,
        "Status Code Color": StatusCodeColorField,
        "Anchor latitude": AnchorCoordinateField,
        "Anchor longitude": AnchorCoordinateField,
        "Anchor point": AnchorPointField,
        "Map server": MapServerField,
        "Tile server": MapServerField,
        "Map layer": MapOverlayField,
        "Server status": ServerStatusField,
        "Server reload": ServerReloadField,
        "Map status": DownloadStatusField,
        "SVG status": SVGStatusField,
        "Path length": WholeLinePropertyField,
        "Width": WholeLinePropertyField,
        "Height": WholeLinePropertyField,
        "Radius": WholeLinePropertyField,
        "Circumference": WholeLinePropertyField,
        "Area": WholeLinePropertyField,
        "Scalar value": ScalarChoiceField,
        "Colormap": ColormapField,
        "Discrete colormap": DiscreteColormapField,
        "X location": XPercentageField,
        "Y location": YPercentageField,
        "Magnification": MagnificationPercentageField,
        "Point size": PointSizeField,
        "Scalar value expression": ScalarExpressionField,
        "Scalar value ranges": ScalarSummaryField,
        "Legend type": LegendTypeField,
        "Legend labels": LegendLabelsField,
        "Save polygon": SaveChangesInPolygon,
        "Cancel polygon": CancelEditInPolygon,
        "Load SVG": LoadSVG,
        "Merge created points": MergeTrianglePoints,
    }

    def create_fields(self, layer, fields):
        self.sizer.AddSpacer(self.LABEL_SPACING)
        self.layer_name_control = None
        self.depth_unit_control = None
        self.default_depth_control = None
        self.point_index_control = None
        self.point_depth_control = None

        self.Freeze()
        self.sizer.Clear(False)  # don't delete controls because we reuse them

        undisplayed = set(self.field_map.keys())
        focus = None
        for field_name in fields:
            if field_name not in self.field_map:
                if field_name not in self.known_fields:
                    value = layer.get_info_panel_text(field_name)
                    if value is not None:
                        fieldcls = SimplePropertyField
                    else:
                        # field not needed for this layer, so skip to next field
                        continue
                else:
                    fieldcls = self.known_fields[field_name]
                field = fieldcls(self, field_name)
                self.field_map[field_name] = field
            else:
                field = self.field_map[field_name]

            field.add_to_parent()
            if field.is_displayed(layer):
                field.fill_data(layer)
                if field_name in undisplayed:
                    undisplayed.remove(field_name)
                if field.wants_focus():
                    focus = field
            else:
                field.hide()

        # Hide fields that exist for some layer but not needed for this layer
        for field_name in undisplayed:
            field = self.field_map[field_name]
            field.hide()

        self.constrain_size(focus)

        self.Thaw()
        self.Update()
        self.Refresh()
        self.current_fields = list(fields)

    def set_fields(self, layer, fields, different_layer, selection_changed, has_focus):
        focus = None
        for field_name in fields:
            if field_name in self.field_map:
                field = self.field_map[field_name]
                if field.is_displayed(layer):
                    if different_layer or selection_changed:
                        if field.is_editable_control(has_focus):
                            log.debug("skipping field %s; has focus" % field_name)
                            field.is_valid()
                        else:
                            log.debug("updating field %s" % field_name)
                            field.fill_data(layer)
                        if field.wants_focus():
                            focus = field
                    field.show()
                else:
                    field.hide()
        self.constrain_size(focus)

    def constrain_size(self, focus=None):
        self.sizer.Layout()
        self.focus_on_input = focus
        if focus is not None:
            self.ScrollChildIntoView(focus.ctrl)
        self.SetupScrolling(scroll_x=False, scrollToTop=False, scrollIntoView=True)

    def process_initial_key(self, event, text):
        """ Uses keyboard input from another control to set the focus to the
        previously noted info field and process the text there.
        """
        if self.focus_on_input is not None:
            self.focus_on_input.set_focus()
            self.ScrollChildIntoView(self.focus_on_input.ctrl)
            self.focus_on_input.initial_text_input(text)
            return True
        return False


class LayerInfoPanel(InfoPanel):
    window_name = "Current Layer"

    def get_visible_fields(self, layer):
        fields = list(layer.layer_info_panel)
        return fields


class SelectionInfoPanel(InfoPanel):
    window_name = "Current Selection"

    def get_visible_fields(self, layer):
        fields = list(layer.selection_info_panel)
        return fields
