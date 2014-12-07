import sys
import wx

from ..layers import constants
from ..library import coordinates
from ..library.textparse import parse_int_string, int_list_to_string
from ..mouse_commands import *
from ..ui import sliders

class InfoField(object):
    same_line = False
    
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
        self.label = wx.StaticText(self.parent, label=self.field_name)
        bold_font = self.parent.GetFont()
        bold_font.SetWeight(weight=wx.FONTWEIGHT_BOLD)
        self.label.SetFont(bold_font)
        self.ctrl = self.create_control()
        if self.same_line:
            hbox = wx.BoxSizer(wx.HORIZONTAL)
            hbox.Add(self.label, 1, wx.ALIGN_LEFT | wx.ALIGN_CENTER)
            hbox.AddStretchSpacer(100)
            hbox.Add(self.ctrl, 1, wx.ALIGN_RIGHT | wx.ALIGN_CENTER)
            self.box.Add(hbox, 1, wx.LEFT | wx.RIGHT | wx.ALIGN_TOP, self.panel.SIDE_SPACING)
        else:
            self.box.Add(self.label, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.ALIGN_TOP, self.panel.SIDE_SPACING)
            self.box.AddSpacer(self.panel.LABEL_SPACING)
            self.box.Add(self.ctrl, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.ALIGN_TOP, self.panel.SIDE_SPACING)
        self.box.AddSpacer(self.panel.VALUE_SPACING)

    def add_to_parent(self):
        self.panel.sizer.Add(self.parent, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.ALIGN_TOP, 0)
        self.show(True)
    
    def fill_data(self):
        raise NotImplementedError
    
    def set_focus(self):
        pass

class LabelField(InfoField):
    same_line = True
    
    def create_control(self):
        c = wx.StaticText(self.parent)
        return c
        
class SimplePropertyField(LabelField):
    def fill_data(self, layer):
        text = layer.get_info_panel_text(self.field_name)
        self.ctrl.SetLabel(text)
        
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
        layer.name = self.ctrl.GetValue()
        self.panel.ignore_next_update = True
        # a side effect of select_layer() is to make sure the layer name is up-to-date
        self.panel.project.layer_tree_control.select_layer(layer)
        
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
        return layer.get_selected_point_indexes() > 0
    
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
        return layer.get_selected_point_indexes() > 0
    
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
            # layer.default_depth = float( c.GetValue() )
            new_point = coordinates.lat_lon_from_format_string(c.GetValue())
            print "New point = %r" % (new_point,)
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
        return layer.get_selected_point_indexes() > 0
    
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
            self.panel.project.control.do_select_points(layer, values)
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
            print point_index
            self.panel.project.control.do_select_points(layer, [point_index])

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

class InfoPanel(wx.Panel):

    """
    A panel for displaying and manipulating the properties of a layer.
    """
    LABEL_SPACING = 2
    VALUE_SPACING = 10
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

        wx.Panel.__init__(self, parent)
        
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

        if (layer == None or self.current_layer_displayed == None):
            if (layer == self.current_layer_displayed):
                return
        else:
            if (self.current_layer_displayed == layer and self.current_layer_change_count == layer.change_count):
                return

        self.current_layer_displayed = layer
        self.current_layer_change_count = -1
        if (layer != None):
            self.current_layer_change_count = layer.change_count

        fields = self.get_visible_fields(layer)
        self.display_fields(layer, fields)

    def display_fields(self, layer, fields):
        if self.current_fields == fields:
            print "SAME FIELDS!!!!!!!!!!!!!!!!!!!" + str(fields)
            self.set_fields(layer, fields)
            #self.create_fields(layer, fields)
        else:
            print "DIFFERENT FIELDS!!!!!!!!!!!!!!!!!!!!!" + str(fields)
            self.create_fields(layer, fields)
    
    known_fields = {
        "Layer name": LayerNameField,
        "Depth unit": DepthUnitField,
        "Default depth": DefaultDepthField,
        "Point coordinates": PointCoordinatesField,
        "Point depth": PointDepthField,
        "Point index": PointIndexesField,
        "Flagged points": FlaggedPointsField,
        "Transparency": AlphaField,
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
        print "create_fields: BEFORE", undisplayed
        for field_name in fields:
            if field_name not in self.field_map:
                if field_name not in self.known_fields:
                    value = layer.get_info_panel_text(field_name)
                    if value is not None:
                        fieldcls = SimplePropertyField
                    else:
                        print "field %s not added yet" % field_name
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
                field.set_focus()
            else:
                field.hide()
        
        for field_name in undisplayed:
            print "hiding", field_name
            field = self.field_map[field_name]
            field.hide()

        undisplayed = set(self.field_map.keys())
        print "create_fields: AFTER", undisplayed

        self.sizer.Layout()

        # self.layer_name_control.SetSelection( -1, -1 )
        # self.layer_name_control.SetFocus()

        self.Thaw()
        self.Update()
        self.Refresh()
        self.current_fields = list(fields)
    
    def set_fields(self, layer, fields):
        for field_name in fields:
            if field_name in self.field_map:
                print "FOUND FIELD %s" % field_name
                field = self.field_map[field_name]
                if field.is_displayed(layer):
                    field.fill_data(layer)
                    field.set_focus()
                    field.show()
                else:
                    field.hide()
        self.sizer.Layout()


class LayerInfoPanel(InfoPanel):
    def get_visible_fields(self, layer):
        fields = list(layer.layer_info_panel)
        return fields

class SelectionInfoPanel(InfoPanel):
    def get_visible_fields(self, layer):
        fields = list(layer.selection_info_panel)
        return fields
