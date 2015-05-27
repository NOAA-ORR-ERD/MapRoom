import sys
import wx
import wx.lib.buttons as buttons

from pyface.api import ImageResource

from ..layers import constants, LayerStyle
from ..library import coordinates
from ..library.textparse import parse_int_string, int_list_to_string
from ..mouse_commands import *
from ..menu_commands import *
from ..ui import sliders
from ..renderer import color_to_int, int_to_color


class InfoField(object):
    same_line = False
    
    # wx.Sizer proportion of the main control (not the label).  See the
    # wx.Sizer docs, but basically 0 will fix vertical size to initial size, >
    # 0 will fill available space based on the total proportion in the sizer.
    vertical_proportion = 0
    
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
        self.label = wx.StaticText(self.parent, label=self.field_name, style=wx.ST_ELLIPSIZE_END)
        bold_font = self.parent.GetFont()
        bold_font.SetWeight(weight=wx.FONTWEIGHT_BOLD)
        self.label.SetFont(bold_font)
        self.create_all_controls()
        if self.same_line:
            hbox = wx.BoxSizer(wx.HORIZONTAL)
            hbox.Add(self.label, 99, wx.ALIGN_CENTER)
            hbox.AddStretchSpacer(1)
            hbox.Add(self.ctrl, 0, wx.ALIGN_CENTER)
            for extra in self.extra_ctrls:
                hbox.Add(extra, 0, wx.ALIGN_CENTER)
            self.box.Add(hbox, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, self.panel.SIDE_SPACING)
        else:
            self.box.Add(self.label, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, self.panel.SIDE_SPACING)
            self.box.AddSpacer(self.panel.LABEL_SPACING)
            self.box.Add(self.ctrl, self.vertical_proportion, wx.EXPAND | wx.LEFT | wx.RIGHT, self.panel.SIDE_SPACING)
            for extra in self.extra_ctrls:
                hbox.Add(extra, 0, wx.ALIGN_CENTER)
        self.box.AddSpacer(self.panel.VALUE_SPACING)
    
    def create_all_controls(self):
        self.ctrl = self.create_control()
        self.extra_ctrls = self.create_extra_controls()
    
    def create_extra_controls(self):
        return []

    def add_to_parent(self):
        self.panel.sizer.Add(self.parent, self.vertical_proportion, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.ALIGN_TOP, 0)
        self.show(True)
    
    def fill_data(self):
        raise NotImplementedError
    
    def wants_focus(self):
        return False
    
    def set_focus(self):
        pass

class LabelField(InfoField):
    same_line = True
    
    def create_control(self):
        c = wx.StaticText(self.parent, style=wx.ALIGN_RIGHT)
        return c
        
class SimplePropertyField(LabelField):
    def fill_data(self, layer):
        text = layer.get_info_panel_text(self.field_name)
        self.ctrl.SetLabel(text)
        
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
        
class TextEditField(InfoField):
    def create_control(self):
        c = wx.TextCtrl(self.parent)
        c.Bind(wx.EVT_TEXT, self.on_text_changed)
        c.SetEditable(True)
        return c
        
    def fill_data(self, layer):
        text = self.get_value(layer)
        self.ctrl.ChangeValue(text)
    
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
        self.panel.project.process_command(cmd)
        
class DefaultDepthField(TextEditField):
    same_line = True

    def get_value(self, layer):
        if (layer is None):
            text = ""
        else:
            text = str(layer.default_depth)
        return text
    
    def process_text_change(self, layer):
        c = self.ctrl
        c.SetBackgroundColour("#FFFFFF")
        try:
            layer.default_depth = float(c.GetValue())
        except Exception as e:
            c.SetBackgroundColour("#FF8080")
        c.Refresh()
        
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
    
    def process_text_change(self, layer):
        cmd = None
        c = self.ctrl
        c.SetBackgroundColour("#FFFFFF")
        try:
            # layer.default_depth = float( c.GetValue() )
            depth = float(c.GetValue())
            selected_point_indexes = layer.get_selected_point_indexes()
            if len(selected_point_indexes > 0):
                cmd = ChangeDepthCommand(layer, selected_point_indexes, depth)
                self.panel.project.process_command(cmd)
        except Exception as e:
            print e
            c.SetBackgroundColour("#FF8080")
        c.Refresh()
        
class PointCoordinatesField(TextEditField):
    def is_displayed(self, layer):
        return layer.get_num_points_selected() > 0
    
    def get_value(self, layer):
        if (layer is None):
            return ""
        indexes = layer.get_selected_point_indexes()
        if len(indexes) == 0:
            return ""
        i = indexes[0]
        prefs = self.panel.project.task.get_preferences()
        coords_text = coordinates.format_coords_for_display(layer.points.x[i], layer.points.y[i], prefs.coordinate_display_format)
        return coords_text

    def process_text_change(self, layer):
        c = self.ctrl
        c.SetBackgroundColour("#FFFFFF")
        try:
            new_point = coordinates.lat_lon_from_format_string(c.GetValue())
            if new_point == (-1, -1):
                c.SetBackgroundColour("#FF8080")
                return
            index = layer.get_selected_point_indexes()[0]
            current_point = (layer.points.x[index], layer.points.y[index])
            x_diff = new_point[0] - current_point[0]
            y_diff = new_point[1] - current_point[1]
            cmd = MovePointsCommand(layer, [index], x_diff, y_diff)
            self.panel.project.process_command(cmd)
        except Exception as e:
            import traceback
            print traceback.format_exc(e)
            c.SetBackgroundColour("#FF8080")

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

class AlphaField(FloatSliderField):
    def get_value(self, layer):
        return int((1.0 - layer.alpha) * 100)

    def slider_changed(self, event):
        layer = self.panel.project.layer_tree_control.get_selected_layer()
        if (layer is None):
            return
        refresh = False
        c = self.ctrl
        try:
            val = 100 - int(c.GetValue())
            layer.alpha = float(val) / 100.0
            c.textCtrl.SetBackgroundColour("#FFFFFF")
            refresh = True
        except Exception as e:
            c.textCtrl.SetBackgroundColour("#FF8080")
        
        if refresh:
            self.panel.project.refresh()

class ColorPickerField(InfoField):
    same_line = True
    
    def get_value(self, layer):
        return ""
        
    def fill_data(self, layer):
        color = tuple(int(255 * c) for c in int_to_color(self.get_value(layer))[0:3])
        self.ctrl.SetColour(color)
    
    def create_control(self):
        import wx.lib.colourselect as csel
        color = (0, 0, 0)
        c = csel.ColourSelect(self.parent, -1, "", color, size=(100,-1))
        c.Bind(csel.EVT_COLOURSELECT, self.color_changed)
        return c
        
    def color_changed(self, event):
        color = [float(c/255.0) for c in event.GetValue()]
        color.append(1.0)
        int_color = color_to_int(*color)
        layer = self.panel.project.layer_tree_control.get_selected_layer()
        if (layer is None):
            return
        style = self.get_style(int_color)
        cmd = StyleChangeCommand(layer, style)
        self.panel.project.process_command(cmd)

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

import wx.combo
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
        c = PenStyleComboBox(self.parent, -1, "", size=(100, -1), choices=names,
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
        self.panel.project.process_command(cmd)


class FillStyleField(InfoField):
    same_line = True
    
    def fill_data(self, layer):
        index, style = layer.style.get_current_fill_style()
        self.ctrl.SetSelection(index)
    
    def create_control(self):
        c = wx.combo.BitmapComboBox(self.parent, -1, "", size=(100, -1),
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
        self.panel.project.process_command(cmd)

        
class MultiLineTextField(InfoField):
    def create_control(self):
        c = wx.TextCtrl(self.parent, style=wx.TE_MULTILINE)
        c.Bind(wx.EVT_TEXT, self.on_text_changed)
        c.SetEditable(True)
        return c
        
    def fill_data(self, layer):
        text = self.get_value(layer)
        self.ctrl.ChangeValue(text)
    
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
        
class OverlayTextField(MultiLineTextField):
    vertical_proportion = 1
    
    def wants_focus(self):
        return True
    
    def set_focus(self):
        text = self.ctrl.GetValue()
        if text == "<b>New Label</b>":
            self.ctrl.SetSelection(-1, -1)
        else:
            self.ctrl.SetInsertionPointEnd()
        self.ctrl.SetFocus()
    
    def get_value(self, layer):
        if (layer is None):
            return ""
        return layer.user_text

    def process_text_change(self, layer):
        text = self.ctrl.GetValue()
        cmd = TextCommand(layer, text)
        self.panel.project.process_command(cmd)


class FontComboBox(wx.combo.OwnerDrawnComboBox):
    # Overridden from OwnerDrawnComboBox, called to draw each
    # item in the list
    def OnDrawItem(self, dc, rect, item, flags):
        if item == wx.NOT_FOUND:
            # painting the control, but there is no valid item selected yet
            return

        r = wx.Rect(*rect)  # make a copy
        r.Deflate(3, 5)

        face = LayerStyle.fonts[item]
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
        names = LayerStyle.fonts
        c = FontComboBox(self.parent, -1, "", size=(100, -1), choices=names,
                             style=wx.CB_READONLY)
        c.Bind(wx.EVT_COMBOBOX, self.style_changed)
        return c
        
    def style_changed(self, event):
        layer = self.panel.project.layer_tree_control.get_selected_layer()
        if (layer is None):
            return
        item = event.GetSelection()
        font = LayerStyle.fonts[item]
        style = LayerStyle(font=font)
        cmd = StyleChangeCommand(layer, style)
        self.panel.project.process_command(cmd)


class FontSizeField(InfoField):
    same_line = True
    
    def fill_data(self, layer):
        index, style = layer.style.get_current_font_size()
        self.ctrl.SetSelection(index)
    
    def create_control(self):
        names = [str(s) for s in LayerStyle.standard_font_sizes]
        c = wx.ComboBox(self.parent, -1, str(LayerStyle.default_font_size),
                        size=(100, -1), choices=names, style=wx.CB_READONLY)
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
        self.panel.project.process_command(cmd)


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
        c = wx.ComboBox(self.parent, -1, "", size=(100, -1), choices=names,
                             style=wx.CB_READONLY)
        c.Bind(wx.EVT_COMBOBOX, self.style_changed)
        return c
        
    def style_changed(self, event):
        layer = self.panel.project.layer_tree_control.get_selected_layer()
        if (layer is None):
            return
        item = event.GetSelection()
        font = LayerStyle.fonts[item]
        style = self.get_style(item)
        cmd = StyleChangeCommand(layer, style)
        self.panel.project.process_command(cmd)

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


PANELTYPE = wx.lib.scrolledpanel.ScrolledPanel
class InfoPanel(PANELTYPE):

    """
    A panel for displaying and manipulating the properties of a layer.
    """
    LABEL_SPACING = 0
    VALUE_SPACING = 3
    SIDE_SPACING = 5

    def __init__(self, parent, project):
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

        PANELTYPE.__init__(self, parent)
        
        # Mac/Win needs this, otherwise background color is black
        attr = self.GetDefaultAttributes()
        self.SetBackgroundColour(attr.colBg)
        
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)
    
    def display_panel_for_layer(self, project, layer):
        self.project = project

        if (self.ignore_next_update):
            self.ignore_next_update = False

            return

        if (layer is None or self.current_layer_displayed is None):
            if (layer == self.current_layer_displayed):
                return
        else:
            if (self.current_layer_displayed == layer and self.current_layer_change_count == layer.change_count):
                return

        self.current_layer_displayed = layer
        self.current_layer_change_count = -1
        if (layer is not None):
            self.current_layer_change_count = layer.change_count

        fields = self.get_visible_fields(layer)
        self.display_fields(layer, fields)

    def display_fields(self, layer, fields):
        if self.current_fields == fields:
            self.set_fields(layer, fields)
        else:
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
        "Flagged points": FlaggedPointsField,
        "Transparency": AlphaField,
        "Color": ColorField,
        "Line Color": ColorField,
        "Line Style": LineStyleField,
        "Start Marker": StartMarkerField,
        "End Marker": EndMarkerField,
        "Fill Color": FillColorField,
        "Fill Style": FillStyleField,
        "Text Color": ColorField,  # Same as Line Color except for the label
        "Font": FontStyleField,
        "Font Size": FontSizeField,
        "Overlay Text": OverlayTextField,
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
    
    def set_fields(self, layer, fields):
        focus = None
        for field_name in fields:
            if field_name in self.field_map:
                field = self.field_map[field_name]
                if field.is_displayed(layer):
                    field.fill_data(layer)
                    field.show()
                    if field.wants_focus():
                        focus = field
                else:
                    field.hide()
        self.constrain_size(focus)
    
    def constrain_size(self, focus=None):
        self.sizer.Layout()
        if focus is not None:
            focus.set_focus()
            self.ScrollChildIntoView(focus.ctrl)
        self.SetupScrolling(scroll_x=False, scrollToTop=False, scrollIntoView=True)


class LayerInfoPanel(InfoPanel):
    def get_visible_fields(self, layer):
        fields = list(layer.layer_info_panel)
        return fields

class SelectionInfoPanel(InfoPanel):
    def get_visible_fields(self, layer):
        fields = list(layer.selection_info_panel)
        return fields
