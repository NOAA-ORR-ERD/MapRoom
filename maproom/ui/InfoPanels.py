import sys
import wx

from ..layers import constants
from ..library import coordinates
from ..library.textparse import parse_int_string, int_list_to_string
from ..mouse_commands import *
from ..ui import sliders

class InfoField(object):
    same_line = False
    
    def __init__(self, parent, field_name):
        self.field_name = field_name
        self.parent = parent
        self.label = wx.StaticText(self.parent, label=field_name)
        bold_font = self.parent.GetFont()
        bold_font.SetWeight(weight=wx.FONTWEIGHT_BOLD)
        self.label.SetFont(bold_font)
        self.ctrl = self.create_control()
        self.box = None
        self.spacer1 = None
        self.spacer2 = None
    
    def is_displayed(self, layer):
        return True
    
    def show(self, state=True):
        if self.spacer1 is not None: self.spacer1.Show(state)
        if self.spacer2 is not None: self.spacer2.Show(state)
        if self.box is not None:
            self.box.Show(self.label, state)
            self.box.Show(self.ctrl, state)
            self.box.Layout()
        else:
            self.label.Show(state)
            self.ctrl.Show(state)
    
    def hide(self):
        self.show(False)
    
    def add_to_parent(self):
        if self.same_line:
            box = wx.BoxSizer(wx.HORIZONTAL)
            box.Add(self.label, 1, wx.ALIGN_LEFT | wx.ALIGN_CENTER)
            box.AddStretchSpacer(100)
            box.Add(self.ctrl, 1, wx.ALIGN_RIGHT | wx.ALIGN_CENTER)
            self.parent.sizer.Add(box, 0, wx.LEFT | wx.RIGHT | wx.ALIGN_TOP, self.parent.SIDE_SPACING)
            self.box = box
            self.spacer1 = self.parent.sizer.AddSpacer(self.parent.VALUE_SPACING)
        else:
            self.parent.sizer.Add(self.label, 0, wx.LEFT | wx.RIGHT | wx.ALIGN_TOP, self.parent.SIDE_SPACING)
            self.spacer1 = self.parent.sizer.AddSpacer(self.parent.LABEL_SPACING)
            self.parent.sizer.Add(self.ctrl, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.ALIGN_TOP, self.parent.SIDE_SPACING)
            self.spacer2 = self.parent.sizer.AddSpacer(self.parent.VALUE_SPACING)
        self.show(True)

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
        layer = self.parent.project.layer_tree_control.get_selected_layer()
        if (layer is None):
            return
        self.process_text_change(layer)
    
    def process_text_change(self, layer):
        layer.name = self.ctrl.GetValue()
        self.parent.ignore_next_update = True
        # a side effect of select_layer() is to make sure the layer name is up-to-date
        self.parent.project.layer_tree_control.select_layer(layer)
        
class LayerNameField(TextEditField):
    def get_value(self, layer):
        if (layer is None):
            text = ""
        else:
            text = layer.name
        return text
    
    def process_text_change(self, layer):
        layer.name = self.ctrl.GetValue()
        self.parent.ignore_next_update = True
        # a side effect of select_layer() is to make sure the layer name is up-to-date
        self.parent.project.layer_tree_control.select_layer(layer)
        
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
                self.parent.project.process_command(cmd)
        except Exception as e:
            print e
            c.SetBackgroundColour("#FF8080")
        c.Refresh()

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
        layer = self.parent.project.layer_tree_control.get_selected_layer()
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
        layer = self.parent.project.layer_tree_control.get_selected_layer()
        if (layer is None):
            return
        c = self.ctrl
        item_num = c.GetSelection()
        if item_num > 0:
            point_index = int(c.GetString(item_num)) - 1
            print point_index
            self.parent.project.control.do_select_points(layer, [point_index])
    
#                selected = ["Total flagged: %d" % layer.get_num_points_selected(constants.STATE_FLAGGED)]
#                selected += [str(i + 1) for i in layer.get_selected_point_indexes(constants.STATE_FLAGGED)]
#                self.flagged_points_control = self.add_drop_down(field, selected, selected[0], self.find_flagged_point)

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
        "Point depth": PointDepthField,
        "Flagged points": FlaggedPointsField,
#        "Line segment count": ('label', SAME_LINE),
#        "Flagged points": ('label', SAME_LINE),
#        "Default depth": ('text', 0),
#        "Depth unit": ,
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
            else:
                field.hide()
        
        for field_name in undisplayed:
            print "hiding", field_name
            field = self.field_map[field_name]
            field.hide()
#                    
#            if layer.has_points():
#                if (field == "Triangle count" and layer.triangles == None):
#                    continue
#                if (field == "Selected points" or field == "Point depth"):
#                    if (layer.get_num_points_selected() == 0):
#                        continue
#                if (field == "Flagged points"):
#                    if (layer.get_num_points_selected(constants.STATE_FLAGGED) == 0):
#                        continue
#            
#            
#            label = wx.StaticText(self, label=field)
#            label.SetFont(bold_font)
#            self.sizer.Add(label, 0, wx.LEFT | wx.RIGHT | wx.ALIGN_TOP, self.SIDE_SPACING)
#            self.sizer.AddSpacer(self.LABEL_SPACING)
#
#            if (field == "Layer name" or field == "Folder name"):
#                self.layer_name_control = self.add_text_field(field, layer.name, self.layer_name_changed, wx.EXPAND)
#                self.field_map[field] = (self.layer_name_control, lambda ctrl, layer: self.set_text_field(ctrl, layer.name))
#            if field == "Depth unit":
#                self.depth_unit_control = self.add_drop_down(field, ["meters", "feet", "fathoms", "unknown"], layer.depth_unit, self.depth_unit_changed)
#                self.field_map[field] = (self.depth_unit_control, lambda ctrl, layer: self.set_drop_down(ctrl, None, layer.depth_unit))
#            elif (field == "Default depth"):
#                self.default_depth_control = self.add_text_field(field, str(layer.default_depth), self.default_depth_changed)
#                self.field_map[field] = (self.default_depth_control, lambda ctrl, layer: self.set_text_field(ctrl, str(layer.default_depth)))
#            elif (field == "Point count"):
#                if (layer.points != None):
#                    self.add_static_text_field(field, str(len(layer.points)))
#            elif (field == "Line segment count"):
#                if (layer.line_segment_indexes != None):
#                    self.add_static_text_field(field, str(len(layer.line_segment_indexes)))
#            elif (field == "Triangle count"):
#                if (layer.triangles != None):
#                    self.add_static_text_field(field, str(len(layer.triangles)))
#            elif (field == "Selected points"):
#                self.add_static_text_field(field, str(layer.get_num_points_selected()))
#            elif (field == "Flagged points"):
#                selected = ["Total flagged: %d" % layer.get_num_points_selected(constants.STATE_FLAGGED)]
#                selected += [str(i + 1) for i in layer.get_selected_point_indexes(constants.STATE_FLAGGED)]
#                self.flagged_points_control = self.add_drop_down(field, selected, selected[0], self.find_flagged_point)
#            elif (field == "Point index"):
#                selected_point_indexes = layer.get_selected_point_indexes()
#                if len(selected_point_indexes) > 0:
#                    values = [x + 1 for x in selected_point_indexes]
#                    s = int_list_to_string(values)
#                else:
#                    s = ""
#                # FIXME: editing ability turned off for now due to the text
#                # selection box being reset to the point depth box after
#                # every update.  Maybe this can be changed when the panes are
#                # recoded to reuse existing controls rather than rebuilding
#                # the entire layout every time
#                self.point_index_control = self.add_text_field(field, s, self.point_indexes_changed, wx.EXPAND, enabled=False)
#                self.point_index_control.SetSelection(-1, -1)
#            elif (field == "Point depth"):
#                conflict = False
#                depth = -1
#                selected_point_indexes = layer.get_selected_point_indexes()
#                for i in selected_point_indexes:
#                    d = layer.points.z[i]
#                    if (d != depth):
#                        if (depth != -1):
#                            conflict = True
#                            break
#                        else:
#                            depth = d
#                s = ""
#                if (not conflict):
#                    s = str(depth)
#                self.point_depth_control = self.add_text_field(field, s, self.point_depth_changed)
#                self.point_depth_control.SetSelection(-1, -1)
#                self.point_depth_control.SetFocus()
#            elif field == "Point coordinates":
#                index = layer.get_selected_point_indexes()[0]
#                prefs = self.project.task.get_preferences()
#                coords_text = coordinates.format_coords_for_display(layer.points.x[i], layer.points.y[i], prefs.coordinate_display_format)
#                self.point_coord_control = self.add_text_field(field, coords_text, self.point_coords_changed, expand=wx.EXPAND)
#            elif field == "Transparency":
#                self.alpha_control = self.add_float_slider(field, int((1.0 - layer.alpha) * 100), 0, 100, 100, self.alpha_changed, expand=wx.EXPAND)
#
#        if (layer != None):
#            for field in layer.display_properties():
#                label = wx.StaticText(self, label=field)
#                label.SetFont(bold_font)
#                self.sizer.Add(label, 0, wx.LEFT | wx.RIGHT | wx.ALIGN_TOP, self.SIDE_SPACING)
#                self.sizer.AddSpacer(self.LABEL_SPACING)
#
#                self.add_static_text_field(field, layer.get_display_property(field))

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
                    field.show()
                else:
                    field.hide()
        self.sizer.Layout()

    def add_float_slider(self, field_name, value, minval, maxval, steps, changed_function, expand=0, enabled=True):
        c = sliders.TextSlider(self, -1, value, minval, maxval, steps)
        self.sizer.Add(c, 0, expand | wx.LEFT | wx.RIGHT | wx.ALIGN_TOP, self.SIDE_SPACING)
        self.sizer.AddSpacer(self.VALUE_SPACING)
        c.Bind(wx.EVT_SLIDER, changed_function)

        return c

    def add_text_field(self, field_name, default_text, changed_function, expand=0, enabled=True):
        c = wx.TextCtrl(self, value=default_text)
        self.sizer.Add(c, 0, expand | wx.LEFT | wx.RIGHT | wx.ALIGN_TOP, self.SIDE_SPACING)
        self.sizer.AddSpacer(self.VALUE_SPACING)
        c.Bind(wx.EVT_TEXT, changed_function)
        c.SetEditable(enabled)
        # c.Bind( wx.EVT_SET_FOCUS, self.text_field_focused )

        return c

    def set_text_field(self, ctrl, text):
        ctrl.ChangeValue(text)

    def add_drop_down(self, field_name, choices, default_choice, changed_function):
        c = wx.Choice(self, choices=choices)
        self.sizer.Add(c, 0, wx.LEFT | wx.RIGHT | wx.ALIGN_TOP, self.SIDE_SPACING)
        self.sizer.AddSpacer(self.VALUE_SPACING)
        c.SetSelection(choices.index(default_choice))
        c.Bind(wx.EVT_CHOICE, changed_function)
        # c.Bind( wx.EVT_SET_FOCUS, self.choice_focused )

        return c

    def set_drop_down(self, ctrl, choices, default_choice):
        if choices is not None:
            ctrl.SetItems(choices)
        else:
            choices = ctrl.GetItems()
        ctrl.SetSelection(choices.index(default_choice))

    def add_static_text_field(self, field_name, text):
        c = wx.StaticText(self, label=text)
        self.sizer.Add(c, 0, wx.LEFT | wx.RIGHT | wx.ALIGN_TOP, self.SIDE_SPACING)
        self.sizer.AddSpacer(self.VALUE_SPACING)

        return c

    def layer_name_changed(self, event):
        layer = self.project.layer_tree_control.get_selected_layer()
        if (layer == None or self.layer_name_control == None):
            return

        c = self.layer_name_control
        layer.name = c.GetValue()
        self.ignore_next_update = True
        # a side effect of select_layer() is to make sure the layer name is up-to-date
        self.project.layer_tree_control.select_layer(layer)

    def depth_unit_changed(self, event):
        layer = self.project.layer_tree_control.get_selected_layer()
        if (layer == None or self.depth_unit_control == None):
            return

        c = self.depth_unit_control
        layer.depth_unit = c.GetString(c.GetSelection())

    def default_depth_changed(self, event):
        layer = self.project.layer_tree_control.get_selected_layer()
        if (layer == None or self.depth_unit_control == None):
            return

        c = self.default_depth_control
        c.SetBackgroundColour("#FFFFFF")
        try:
            layer.default_depth = float(c.GetValue())
        except Exception as e:
            c.SetBackgroundColour("#FF8080")
        c.Refresh()

    def point_indexes_changed(self, event):
        layer = self.project.layer_tree_control.get_selected_layer()
        if layer == None:
            return

        c = self.point_index_control
        c.SetBackgroundColour("#FFFFFF")
        try:
            one_based_values, error = parse_int_string(c.GetValue())
            values = [x - 1 for x in one_based_values]
            self.project.control.do_select_points(layer, values)
        except Exception as e:
            import traceback
            print traceback.format_exc(e)
            c.SetBackgroundColour("#FF8080")

    def point_coords_changed(self, event):
        layer = self.project.layer_tree_control.get_selected_layer()
        if layer == None:
            return

        c = self.point_coord_control
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
            self.project.process_command(cmd)
        except Exception as e:
            import traceback
            print traceback.format_exc(e)
            c.SetBackgroundColour("#FF8080")
        # c.Refresh()

    def point_depth_changed(self, event):
        layer = self.project.layer_tree_control.get_selected_layer()
        if layer == None:
            return

        cmd = None
        c = self.point_depth_control
        c.SetBackgroundColour("#FFFFFF")
        try:
            # layer.default_depth = float( c.GetValue() )
            depth = float(c.GetValue())
            selected_point_indexes = layer.get_selected_point_indexes()
            if len(selected_point_indexes > 0):
                cmd = ChangeDepthCommand(layer, selected_point_indexes, depth)
                self.project.process_command(cmd)
        except Exception as e:
            print e
            c.SetBackgroundColour("#FF8080")
        c.Refresh()
        # self.ignore_next_update = True

    def alpha_changed(self, event):
        layer = self.project.layer_tree_control.get_selected_layer()
        if layer == None:
            return

        refresh = False
        c = self.alpha_control
        try:
            val = 100 - int(c.GetValue())
            layer.alpha = float(val) / 100.0
            c.textCtrl.SetBackgroundColour("#FFFFFF")
            refresh = True
        except Exception as e:
            c.textCtrl.SetBackgroundColour("#FF8080")
        
        if refresh:
            self.project.refresh()

    def find_flagged_point(self, event):
        layer = self.project.layer_tree_control.get_selected_layer()
        c = self.flagged_points_control
        if (layer == None or c == None):
            return
        item_num = c.GetSelection()
        if item_num > 0:
            point_index = int(c.GetString(item_num)) - 1
            print point_index
            self.project.control.do_select_points(layer, [point_index])

class LayerInfoPanel(InfoPanel):
    def get_visible_fields(self, layer):
        fields = list(layer.layer_info_panel)
        return fields

class SelectionInfoPanel(InfoPanel):
    def get_visible_fields(self, layer):
        fields = list(layer.selection_info_panel)
        return fields
