import sys
import wx

from ..layers import constants
from ..library import coordinates
from ..layer_undo import *


class Properties_panel(wx.Panel):

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

        self.layer_name_control = None
        self.depth_unit_control = None
        self.default_depth_control = None
        self.point_depth_control = None

        self.Freeze()
        self.sizer.Clear(True)

        if (layer != None and layer.type != "root"):
            bold_font = self.GetFont()
            bold_font.SetWeight(weight=wx.FONTWEIGHT_BOLD)

            fields = []
            if (layer.type == "folder"):
                fields = ["Folder name"]

            elif (layer.type == ".bna"):
                fields.extend(["Layer name", "Polygon count"])
            else:
                if layer.has_points() and layer.get_num_points_selected() > 0:
                    fields.extend(["Depth unit",  "Selected points", "Point depth"])
                    if layer.get_num_points_selected() == 1:
                        fields.extend(["Point coordinates"])
                else:
                    fields.extend(["Layer name"])
                    if layer.has_points():
                        fields.extend(["Default depth", "Depth unit", "Point count", "Line segment count", "Triangle count", "Flagged points", "Selected points"])

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
                    if layer.get_num_points_selected() > 0:
                        self.add_static_text_field(field, layer.depth_unit)
                    else:
                        self.depth_unit_control = self.add_drop_down(field, ["meters", "feet", "fathoms", "unknown"], layer.depth_unit, self.depth_unit_changed)
                elif (field == "Default depth"):
                    self.default_depth_control = self.add_text_field(field, str(layer.default_depth), self.default_depth_changed)
                elif (field == "Point count"):
                    if (layer.points != None):
                        self.add_static_text_field(field, str(len(layer.points)))
                elif (field == "Line segment count"):
                    if (layer.line_segment_indexes != None):
                        self.add_static_text_field(field, str(len(layer.line_segment_indexes)))
                elif (field == "Selected points"):
                    self.add_static_text_field(field, str(layer.get_num_points_selected()))
                elif (field == "Flagged points"):
                    self.add_static_text_field(field, str(layer.get_num_points_selected(constants.STATE_FLAGGED)))
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

            self.sizer.Layout()

            # self.layer_name_control.SetSelection( -1, -1 )
            # self.layer_name_control.SetFocus()

        self.Thaw()
        self.Update()
        self.Refresh()

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
            params = (-x_diff, -y_diff)

            print "params = %r" % (params,)
            self.project.layer_manager.add_undo_operation_to_operation_batch(OP_MOVE_POINT, layer, index, params)
            layer.offset_point(index, x_diff, y_diff)
            layer.manager.dispatch_event('layer_contents_changed', layer)
            self.project.layer_manager.end_operation_batch()
        except Exception as e:
            import traceback
            print traceback.format_exc(e)
            c.SetBackgroundColour("#FF8080")
        # c.Refresh()
        self.project.refresh()

    def point_depth_changed(self, event):
        layer = self.project.layer_tree_control.get_selected_layer()
        if layer == None:
            return

        num_points_changed = 0
        c = self.point_depth_control
        c.SetBackgroundColour("#FFFFFF")
        try:
            # layer.default_depth = float( c.GetValue() )
            depth = float(c.GetValue())
            selected_point_indexes = layer.get_selected_point_indexes()
            for i in selected_point_indexes:
                params = (layer.points.z[i], depth)
                self.project.layer_manager.add_undo_operation_to_operation_batch(OP_CHANGE_POINT_DEPTH, layer, i, params)
                layer.points.z[i] = depth
                num_points_changed += 1
        except Exception as e:
            c.SetBackgroundColour("#FF8080")
        c.Refresh()
        # self.ignore_next_update = True
        if (num_points_changed > 0):
            self.project.layer_manager.end_operation_batch()
        self.project.refresh()
