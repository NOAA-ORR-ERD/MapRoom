import sys
import wx

from ..layers import constants
from ..library import coordinates
from ..library.textparse import parse_int_string, int_list_to_string
from ..layer_undo import *
from ..mouse_commands import *
from ..ui import sliders

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
        self.layer_name_control = None
        self.depth_unit_control = None
        self.default_depth_control = None
        self.point_index_control = None
        self.point_depth_control = None

        self.Freeze()
        self.sizer.Clear(True)

        bold_font = self.GetFont()
        bold_font.SetWeight(weight=wx.FONTWEIGHT_BOLD)

        self.sizer.AddSpacer(self.LABEL_SPACING)
        layer_name_control = None
        for field in fields:
            if layer.has_points():
                if (field == "Triangle count" and layer.triangles == None):
                    continue
                if (field == "Selected points" or field == "Point depth"):
                    if (layer.get_num_points_selected() == 0):
                        continue
                if (field == "Flagged points"):
                    if (layer.get_num_points_selected(constants.STATE_FLAGGED) == 0):
                        continue
            label = wx.StaticText(self, label=field)
            label.SetFont(bold_font)
            self.sizer.Add(label, 0, wx.LEFT | wx.RIGHT | wx.ALIGN_TOP, self.SIDE_SPACING)
            self.sizer.AddSpacer(self.LABEL_SPACING)

            if (field == "Layer name" or field == "Folder name"):
                self.layer_name_control = self.add_text_field(field, layer.name, self.layer_name_changed, wx.EXPAND)
            if field == "Depth unit":
                self.depth_unit_control = self.add_drop_down(field, ["meters", "feet", "fathoms", "unknown"], layer.depth_unit, self.depth_unit_changed)
            elif (field == "Default depth"):
                self.default_depth_control = self.add_text_field(field, str(layer.default_depth), self.default_depth_changed)
            elif (field == "Point count"):
                if (layer.points != None):
                    self.add_static_text_field(field, str(len(layer.points)))
            elif (field == "Line segment count"):
                if (layer.line_segment_indexes != None):
                    self.add_static_text_field(field, str(len(layer.line_segment_indexes)))
            elif (field == "Triangle count"):
                if (layer.triangles != None):
                    self.add_static_text_field(field, str(len(layer.triangles)))
            elif (field == "Selected points"):
                self.add_static_text_field(field, str(layer.get_num_points_selected()))
            elif (field == "Flagged points"):
                selected = ["Total flagged: %d" % layer.get_num_points_selected(constants.STATE_FLAGGED)]
                selected += [str(i + 1) for i in layer.get_selected_point_indexes(constants.STATE_FLAGGED)]
                self.flagged_points_control = self.add_drop_down(field, selected, selected[0], self.find_flagged_point)
            elif (field == "Point index"):
                selected_point_indexes = layer.get_selected_point_indexes()
                if len(selected_point_indexes) > 0:
                    values = [x + 1 for x in selected_point_indexes]
                    s = int_list_to_string(values)
                else:
                    s = ""
                # FIXME: editing ability turned off for now due to the text
                # selection box being reset to the point depth box after
                # every update.  Maybe this can be changed when the panes are
                # recoded to reuse existing controls rather than rebuilding
                # the entire layout every time
                self.point_index_control = self.add_text_field(field, s, self.point_indexes_changed, wx.EXPAND, enabled=False)
                self.point_index_control.SetSelection(-1, -1)
            elif (field == "Point depth"):
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
                self.point_depth_control = self.add_text_field(field, s, self.point_depth_changed)
                self.point_depth_control.SetSelection(-1, -1)
                self.point_depth_control.SetFocus()
            elif field == "Point coordinates":
                index = layer.get_selected_point_indexes()[0]
                prefs = self.project.task.get_preferences()
                coords_text = coordinates.format_coords_for_display(layer.points.x[i], layer.points.y[i], prefs.coordinate_display_format)
                self.point_coord_control = self.add_text_field(field, coords_text, self.point_coords_changed, expand=wx.EXPAND)
            elif field == "Transparency":
                self.alpha_control = self.add_float_slider(field, int((1.0 - layer.alpha) * 100), 0, 100, 100, self.alpha_changed, expand=wx.EXPAND)

        if (layer != None):
            for field in layer.display_properties():
                label = wx.StaticText(self, label=field)
                label.SetFont(bold_font)
                self.sizer.Add(label, 0, wx.LEFT | wx.RIGHT | wx.ALIGN_TOP, self.SIDE_SPACING)
                self.sizer.AddSpacer(self.LABEL_SPACING)

                self.add_static_text_field(field, layer.get_display_property(field))

        self.sizer.Layout()

        # self.layer_name_control.SetSelection( -1, -1 )
        # self.layer_name_control.SetFocus()

        self.Thaw()
        self.Update()
        self.Refresh()

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

    def add_drop_down(self, field_name, choices, default_choice, changed_function):
        c = wx.Choice(self, choices=choices)
        self.sizer.Add(c, 0, wx.LEFT | wx.RIGHT | wx.ALIGN_TOP, self.SIDE_SPACING)
        self.sizer.AddSpacer(self.VALUE_SPACING)
        c.SetSelection(choices.index(default_choice))
        c.Bind(wx.EVT_CHOICE, changed_function)
        # c.Bind( wx.EVT_SET_FOCUS, self.choice_focused )

        return c

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
        fields = []

        if (layer != None and layer.type != "root"):
            if (layer.type == "folder"):
                fields = ["Folder name"]
            elif (layer.type == "polygon"):
                fields.extend(["Layer name", "Polygon count"])
            elif (layer.type == "particle"):
                fields.extend(["Layer name", ])    
            else: ## fixme --  shouldn't this be looping for LineLayer -- or whatever?
                fields.extend(["Layer name"])
                if layer.has_alpha():
                    fields.extend(["Transparency"])
                if layer.has_points():
                    fields.extend(["Default depth", "Depth unit", "Point count", "Flagged points"])
                    if layer.type == "triangle":
                        fields.extend(["Triangle count"])
                    elif layer.type == "line":
                        fields.extend(["Line segment count"])
                        
        return fields

class SelectionInfoPanel(InfoPanel):
    def get_visible_fields(self, layer):
        fields = []

        if (layer != None and layer.type != "root"):
            if layer.has_points() and layer.get_num_points_selected() > 0:
                fields.extend(["Selected points", "Point index", "Point depth"])
                if layer.get_num_points_selected() == 1:
                    fields.extend(["Point coordinates"])
        return fields
