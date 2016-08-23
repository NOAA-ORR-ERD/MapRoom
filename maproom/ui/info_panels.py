import sys
import wx
import wx.combo
import wx.lib.buttons as buttons
from wx.lib.expando import ExpandoTextCtrl

from pyface.api import ImageResource

import sliders
import dialogs
from ..layers import constants, LayerStyle
from ..library import coordinates
from ..library.textparse import parse_int_string, int_list_to_string
from ..mouse_commands import *
from ..menu_commands import *
from ..vector_object_commands import MoveControlPointCommand
from ..renderer import color_floats_to_int, int_to_color_floats
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
        self.box =  wx.BoxSizer(wx.VERTICAL)
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
                hbox.Add(extra, 0, wx.ALIGN_CENTER)
        self.box.AddSpacer(self.panel.VALUE_SPACING)
    
    def create_all_controls(self):
        self.ctrl = self.create_control()
        if sys.platform.startswith("win"):
            self.ctrl.Bind(wx.EVT_MOUSEWHEEL, self.on_mouse_wheel_scroll)
        self.extra_ctrls = self.create_extra_controls()
    
    def is_editable_control(self, ctrl):
        return ctrl == self.ctrl
    
    def create_extra_controls(self):
        return []

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
#            print "Mouse not over info panel %s: trying map!" % self
            self.panel.project.control.on_mouse_wheel_scroll(event)
            return
        
        event.Skip()

class LabelField(InfoField):
    same_line = True
    
    def create_control(self):
        c = wx.StaticText(self.parent, style=wx.ALIGN_RIGHT)
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
        c = ExpandoTextCtrl(self.parent, style=wx.ALIGN_LEFT|wx.TE_READONLY|wx.NO_BORDER)
        return c
        
class BooleanLabelField(SimplePropertyField):
    def create_extra_controls(self):
        b = buttons.GenBitmapToggleButton(self.parent, -1, None, style=wx.BORDER_NONE|wx.BU_EXACTFIT)  # BU_EXACTFIT removes padding
        b.Bind(wx.EVT_BUTTON, self.on_toggled)
        image = ImageResource('eye-closed.png')
        bitmap = image.create_bitmap()
        b.SetBitmapLabel(bitmap)
        image = ImageResource('eye-open.png')
        b.SetBitmapSelected(image.create_bitmap())
        b.SetInitialSize()
        self.toggle = b
        return [self.toggle]
        
    def fill_data(self, layer):
        SimplePropertyField.fill_data(self, layer)
        vis = self.get_visibility(layer)
        self.toggle.SetToggle(vis)
    
    def on_toggled(self, evt):
        layer = self.panel.project.layer_tree_control.get_selected_layer()
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


class TextEditField(InfoField):
    def create_control(self):
        c = wx.TextCtrl(self.parent)
        c.Bind(wx.EVT_TEXT, self.on_text_changed)
        c.SetEditable(True)
        return c
    
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
        except Exception as e:
            c.SetBackgroundColour("#FF8080")
            valid = False
        self.ctrl.Refresh()
        return valid
    
    def parse_from_string(self):
        return self.ctrl.GetValue()
    
    def on_text_changed(self, evt):
        layer = self.panel.project.layer_tree_control.get_selected_layer()
        if (layer is None):
            return
        self.process_text_change(layer)
    
    def process_text_change(self, layer):
        layer.name = self.ctrl.GetValue()
        self.panel.ignore_next_update = True
        # a side effect of select_layer() is to make sure the layer name is up-to-date
        self.panel.project.layer_tree_control.select_layer(layer)
    
    def initial_text_input(self, text):
        self.ctrl.SetValue(text)
        self.ctrl.SetInsertionPointEnd()#(self.ctrl.GetLastPosition())
        
class LayerNameField(TextEditField):
    def get_value(self, layer):
        if (layer is None):
            text = ""
        else:
            text = layer.name
        return text
    
    def process_text_change(self, layer):
        name = self.ctrl.GetValue()
        cmd = RenameLayerCommand(layer, name)
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
        
class PointCoordinatesField(TextEditField):
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
        prefs = self.panel.project.task.get_preferences()
        coords_text = coordinates.format_coords_for_display(layer.points.x[i], layer.points.y[i], prefs.coordinate_display_format)
        return coords_text
    
    def parse_from_string(self):
        text = self.ctrl.GetValue()
        c = coordinates.lat_lon_from_format_string(text)
        return c

    def get_command(self, layer, index, dx, dy):
        return MovePointsCommand(layer, [index], dx, dy)

    def process_text_change(self, layer):
        if self.is_valid():
            new_point = self.parse_from_string()
            index = self.get_point_index(layer)
            current_point = (layer.points.x[index], layer.points.y[index])
            x_diff = new_point[0] - current_point[0]
            y_diff = new_point[1] - current_point[1]
            cmd = self.get_command(layer, index, x_diff, y_diff)
            self.process_command(cmd)

class AnchorCoordinatesField(PointCoordinatesField):
    def is_displayed(self, layer):
        return True
    
    def get_point_index(self, layer):
        if (layer is None):
            return -1
        return layer.center_point_index

    def get_command(self, layer, index, dx, dy):
        return MoveControlPointCommand(layer, index, index, dx, dy, None, None)

class AnchorPointField(InfoField):
    same_line = True
    
    def fill_data(self, layer):
        self.ctrl.SetSelection(layer.anchor_point_index)
    
    def create_control(self):
        names = [str(s) for s in range(9)]
        c = wx.ComboBox(self.parent, -1, "",
                        size=(self.default_width, -1), choices=names, style=wx.CB_READONLY)
        c.Bind(wx.EVT_COMBOBOX, self.anchor_changed)
        return c
        
    def anchor_changed(self, event):
        layer = self.panel.project.layer_tree_control.get_selected_layer()
        if (layer is None):
            return
        item = event.GetSelection()
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
            print traceback.format_exc(e)
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
        layer = self.panel.project.layer_tree_control.get_selected_layer()
        if (layer is None):
            return
        layer.depth_unit = self.ctrl.GetString(self.ctrl.GetSelection())
        
class FlaggedPointsField(DropDownField):
    same_line = True

    def is_displayed(self, layer):
        return layer.get_num_points_selected(constants.STATE_FLAGGED) > 0
    
    def get_choices(self, layer):
        selected = ["Total: %d" % layer.get_num_points_selected(constants.STATE_FLAGGED)]
        selected += [str(i + 1) for i in layer.get_selected_point_indexes(constants.STATE_FLAGGED)]
        return selected
    
    def get_value(self, layer):
        return self.get_choices(layer)[0]
        
    def drop_down_changed(self, event):
        layer = self.panel.project.layer_tree_control.get_selected_layer()
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
    
    def create_control(self):
        minval, maxval, steps = self.get_params()
        c = sliders.TextSlider(self.parent, -1, minval, minval, maxval, steps)
        c.Bind(wx.EVT_SLIDER, self.slider_changed)
        c.Bind(wx.EVT_SPINCTRLDOUBLE, self.slider_changed)
        return c
        
    def slider_changed(self, event):
        pass

class LineAlphaField(FloatSliderField):
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
        layer = self.panel.project.layer_tree_control.get_selected_layer()
        if (layer is None):
            return
        refresh = False
        c = self.ctrl
        try:
            val = (100 - int(c.GetValue())) / 100.0
            style = self.get_style(layer, val)
            cmd = StyleChangeCommand(layer, style)
            wx.CallAfter(self.process_command, cmd)
            c.textCtrl.SetBackgroundColour("#FFFFFF")
        except Exception as e:
            c.textCtrl.SetBackgroundColour("#FF8080")

class FillAlphaField(LineAlphaField):
    def get_layer_color(self, layer):
        return layer.style.fill_color
    
    def get_layer_style(self, layer, color):
        return LayerStyle(fill_color=color)

class TextAlphaField(LineAlphaField):
    def get_layer_color(self, layer):
        return layer.style.text_color
    
    def get_layer_style(self, layer, color):
        return LayerStyle(text_color=color)


class ColorPickerField(InfoField):
    same_line = True
    
    def get_value(self, layer):
        return ""
        
    def fill_data(self, layer):
        color = tuple(int(255 * c) for c in int_to_color_floats(self.get_value(layer))[0:3])
        self.ctrl.SetColour(color)
    
    def create_control(self):
        import wx.lib.colourselect as csel
        color = (0, 0, 0)
        c = csel.ColourSelect(self.parent, -1, "", color, size=(self.default_width,-1))
        c.Bind(csel.EVT_COLOURSELECT, self.color_changed)
        return c
        
    def color_changed(self, event):
        color = [float(c/255.0) for c in event.GetValue()]
        color.append(1.0)
        int_color = color_floats_to_int(*color)
        layer = self.panel.project.layer_tree_control.get_selected_layer()
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

class TextColorField(ColorPickerField):
    def get_style(self, color):
        return LayerStyle(text_color=color)
        
    def get_value(self, layer):
        return layer.style.text_color


class PenStyleComboBox(wx.combo.OwnerDrawnComboBox):

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

        if flags & wx.combo.ODCB_PAINTING_CONTROL:
            # for painting the control itself
            dc.DrawLine( r.x+5, r.y+r.height/2, r.x+r.width - 5, r.y+r.height/2 )

        else:
            # for painting the items in the popup
            dc.DrawText(self.GetString( item ),
                        r.x + 3,
                        (r.y + 0) + ( (r.height/2) - dc.GetCharHeight() )/2
                        )
            dc.DrawLine( r.x+5, r.y+((r.height/4)*3)+1, r.x+r.width - 5, r.y+((r.height/4)*3)+1 )

           
    # Overridden from OwnerDrawnComboBox, called for drawing the
    # background area of each item.
    def OnDrawBackground(self, dc, rect, item, flags):
        # If the item is selected, or its item # iseven, or we are painting the
        # combo control itself, then use the default rendering.
        if (item & 1 == 0 or flags & (wx.combo.ODCB_PAINTING_CONTROL |
                                      wx.combo.ODCB_PAINTING_SELECTED)):
            wx.combo.OwnerDrawnComboBox.OnDrawBackground(self, dc, rect, item, flags)
            return

        # Otherwise, draw every other background with different colour.
        bgCol = wx.Colour(240,240,250)
        dc.SetBrush(wx.Brush(bgCol))
        dc.SetPen(wx.Pen(bgCol))
        dc.DrawRectangleRect(rect);

    # Overridden from OwnerDrawnComboBox, should return the height
    # needed to display an item in the popup, or -1 for default
    def OnMeasureItem(self, item):
        return 24

    # Overridden from OwnerDrawnComboBox.  Callback for item width, or
    # -1 for default/undetermined
    def OnMeasureItemWidth(self, item):
        return -1; # default - will be measured from text width

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
        layer = self.panel.project.layer_tree_control.get_selected_layer()
        if (layer is None):
            return
        item = event.GetSelection()
        line_style = LayerStyle.line_styles[item]
        style = LayerStyle(line_stipple=line_style[1])
        cmd = StyleChangeCommand(layer, style)
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
        layer = self.panel.project.layer_tree_control.get_selected_layer()
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
        c = wx.combo.BitmapComboBox(self.parent, -1, "", size=(self.default_width, -1),
                             style=wx.CB_READONLY)
        for i, s in LayerStyle.fill_styles.iteritems():
            c.Append(s[0])

        c.Bind(wx.EVT_COMBOBOX, self.style_changed)
        return c
        
    def style_changed(self, event):
        layer = self.panel.project.layer_tree_control.get_selected_layer()
        if (layer is None):
            return
        item = event.GetSelection()
        style = LayerStyle(fill_style=item)
        cmd = StyleChangeCommand(layer, style)
        self.process_command(cmd)

        
class MultiLineTextField(InfoField):
    def create_control(self):
        c = wx.TextCtrl(self.parent, style=wx.TE_MULTILINE)
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
        layer = self.panel.project.layer_tree_control.get_selected_layer()
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
        layer = self.panel.project.layer_tree_control.get_selected_layer()
        if (layer is None):
            return
        item = event.GetSelection()
        style = LayerStyle(text_format=item)
        cmd = StyleChangeCommand(layer, style)
        self.process_command(cmd)


class FontComboBox(wx.combo.OwnerDrawnComboBox):
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
        
        if flags & wx.combo.ODCB_PAINTING_CONTROL:
            # for painting the control itself
            dc.DrawText(face, r.x+5, (r.y + 5) + ( (r.height/2) - dc.GetCharHeight() )/2)

        else:
            # for painting the items in the popup
            dc.DrawText(face,
                        r.x + 3,
                        (r.y + 5) + ( (r.height/2) - dc.GetCharHeight() )/2
                        )

    # Overridden from OwnerDrawnComboBox, should return the height
    # needed to display an item in the popup, or -1 for default
    def OnMeasureItem(self, item):
        return 24

    # Overridden from OwnerDrawnComboBox.  Callback for item width, or
    # -1 for default/undetermined
    def OnMeasureItemWidth(self, item):
        return -1; # default - will be measured from text width


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
        layer = self.panel.project.layer_tree_control.get_selected_layer()
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
        layer = self.panel.project.layer_tree_control.get_selected_layer()
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
        layer = self.panel.project.layer_tree_control.get_selected_layer()
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
        layer = self.panel.project.layer_tree_control.get_selected_layer()
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


class ListBoxComboPopup(wx.ListBox, wx.combo.ComboPopup):
    """Popup for wx.combo.ComboCtrl, based on wxPython demo and converted to
    the ListBox rather than the full ListCtrl
    """
    def __init__(self):
        # Since we are using multiple inheritance, and don't know yet
        # which window is to be the parent, we'll do 2-phase create of
        # the ListCtrl instead, and call its Create method later in
        # our Create method.  (See Create below.)
        self.PostCreate(wx.PreListBox())

        # Need to call this last so the ComboCtrl recognizes that this is of
        # type ComboPopup
        wx.combo.ComboPopup.__init__(self)

    def OnMotion(self, evt):
        if evt.LeftIsDown():
            item = self.HitTest(evt.GetPosition())
            if item >= 0:
                self.Select(item)

    def OnLeftUp(self, evt):
        self.Dismiss()

    # This is called immediately after construction finishes.  You can
    # use self.GetCombo if needed to get to the ComboCtrl instance.
    def Init(self):
        self.value = -1
        self.curitem = -1

    # Create the popup child control.  Return true for success.
    def Create(self, parent):
        wx.ListBox.Create(self, parent,
                          style=wx.LB_SINGLE|wx.SIMPLE_BORDER)
        self.Bind(wx.EVT_MOTION, self.OnMotion)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnMotion)
        self.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
        return True

    # Return the widget that is to be used for the popup
    def GetControl(self):
        return self

    # Called just prior to displaying the popup, you can use it to
    # 'select' the current item.
    def SetStringValue(self, val):
        idx = self.FindString(val)
        if idx != wx.NOT_FOUND:
            self.Select(idx)

    # Called when popup is dismissed
    def OnDismiss(self):
        ev = wx.CommandEvent(commandType=wx.EVT_COMBOBOX.typeId)
        ev.SetInt(self.GetSelection())
        self.GetEventHandler().ProcessEvent(ev)
        wx.combo.ComboPopup.OnDismiss(self)

    # Return final size of popup. Called on every popup, just prior to OnPopup.
    # minWidth = preferred minimum width for window
    # prefHeight = preferred height. Only applies if > 0,
    # maxHeight = max height for window, as limited by screen size
    #   and should only be rounded down, if necessary.
    def GetAdjustedSize(self, minWidth, prefHeight, maxHeight):
        return wx.combo.ComboPopup.GetAdjustedSize(self, InfoField.popup_width, prefHeight, maxHeight)

    SetItems = wx.ListBox.Set

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
        selected = self.get_timestep_index(layer)
        self.popup.SetSelection(selected)
        self.ctrl.SetText(names[selected])
    
    def create_control(self):
        names = ["all"]
        c = wx.combo.ComboCtrl(self.parent, style=wx.CB_READONLY, size=(self.default_width,-1))
        self.popup = ListBoxComboPopup()
        c.SetPopupControl(self.popup)
        c.Bind(wx.EVT_COMBOBOX, self.timestep_changed)
        return c
        
    def timestep_changed(self, event):
        layer = self.panel.project.layer_tree_control.get_selected_layer()
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

class ParticleColorField(ColorField):
    same_line = True
    
    def get_value(self, layer):
        return layer.get_particle_color(self.panel.project)
        
    def create_control(self):
        import wx.lib.colourselect as csel
        color = (0, 0, 0)
        c = csel.ColourSelect(self.parent, -1, "", color, size=(self.default_width,-1))
        c.Bind(csel.EVT_COLOURSELECT, self.color_changed)
        return c
        
    def color_changed(self, event):
        color = [float(c/255.0) for c in event.GetValue()]
        color.append(1.0)
        int_color = color_floats_to_int(*color)
        layer = self.panel.project.layer_tree_control.get_selected_layer()
        if (layer is None):
            return
        layers = layer.get_selected_particle_layers(self.panel.project)
        cmd = ParticleColorCommand(layers, int_color)
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
        layer = self.panel.project.layer_tree_control.get_selected_layer()
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
        layer = self.panel.project.layer_tree_control.get_selected_layer()
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
        layer = self.panel.project.layer_tree_control.get_selected_layer()
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
        c = ExpandoTextCtrl(self.parent, style=wx.ALIGN_LEFT|wx.TE_READONLY|wx.NO_BORDER)
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
        layer = self.panel.project.layer_tree_control.get_selected_layer()
        if (layer is None):
            return
        downloader = self.panel.project.task.get_threaded_wms_by_id(layer.map_server_id)
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
        if color is None:
            attr = self.panel.GetDefaultAttributes()
            color = attr.colBg
        return text, color
    
    def create_control(self):
        c = ExpandoTextCtrl(self.parent, style=wx.ALIGN_LEFT|wx.TE_READONLY|wx.NO_BORDER)
        return c


PANELTYPE = wx.lib.scrolledpanel.ScrolledPanel
class InfoPanel(PANELTYPE):

    """
    A panel for displaying and manipulating the properties of a layer.
    """
    LABEL_SPACING = 0
    VALUE_SPACING = 3
    SIDE_SPACING = 5

    def __init__(self, parent, project, size=(-1,-1)):
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

        PANELTYPE.__init__(self, parent, size=size)
        
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
            log.debug("reusing current fields")
            self.set_fields(layer, fields, different_layer, selection_changed, has_focus)
        else:
            log.debug("creating fields")
            self.create_fields(layer, fields)
    
    known_fields = {
        "Layer name": LayerNameField,
        "Depth unit": DepthUnitField,
        "Default depth": DefaultDepthField,
        "Point coordinates": PointCoordinatesField,
        "Point depth": PointDepthField,
        "Point index": PointIndexesField,
        "Point count": PointVisibilityField,
        "Line segment count": LineVisibilityField,
        "Show depth": DepthVisibilityField,
        "Flagged points": FlaggedPointsField,
        "Transparency": LineAlphaField,
        "Line transparency": LineAlphaField,
        "Fill transparency": FillAlphaField,
        "Color": ColorField,
        "Line color": ColorField,
        "Line style": LineStyleField,
        "Line width": LineWidthField,
        "Start marker": StartMarkerField,
        "End marker": EndMarkerField,
        "Fill color": FillColorField,
        "Fill style": FillStyleField,
        "Text color": TextColorField,  # Same as Line Color except for the label
        "Text transparency": TextAlphaField,
        "Font": FontStyleField,
        "Font size": FontSizeField,
        "Overlay text": OverlayTextField,
        "Text format": TextFormatField,
        "Marplot icon": MarplotIconField,
        "Start time": ParticleStartField,
        "End time": ParticleEndField,
        "Particle Color": ParticleColorField,
        "Anchor coordinates": AnchorCoordinatesField,
        "Anchor point": AnchorPointField,
        "Map server": MapServerField,
        "Tile server": MapServerField,
        "Map layer": MapOverlayField,
        "Server status": ServerStatusField,
        "Server reload": ServerReloadField,
        "Map status": DownloadStatusField,
        "Path length": WholeLinePropertyField,
        "Width": WholeLinePropertyField,
        "Height": WholeLinePropertyField,
        "Radius": WholeLinePropertyField,
        "Circumference": WholeLinePropertyField,
        "Area": WholeLinePropertyField,
        }
    
    def create_fields(self, layer, fields):
        self.sizer.AddSpacer(self.LABEL_SPACING)
        self.layer_name_control = None
        self.depth_unit_control = None
        self.default_depth_control = None
        self.point_index_control = None
        self.point_depth_control = None

        self.Freeze()
        self.sizer.Clear(False) # don't delete controls because we reuse them

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
                            field.is_valid()
                        else:
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
    def get_visible_fields(self, layer):
        fields = list(layer.layer_info_panel)
        return fields

class SelectionInfoPanel(InfoPanel):
    def get_visible_fields(self, layer):
        fields = list(layer.selection_info_panel)
        return fields
