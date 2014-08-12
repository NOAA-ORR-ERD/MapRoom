import sys
import math
import wx

from ..layers import constants
import sliders


class Distance_slider(wx.Panel):
    SPACING = 5
    MINIMUM = 0.0
    MAXIMUM = 60.0
    INITIAL_VALUE = 0.6
    SLIDER_STEPS = 1000.0
    LOG_BASE = 10000.0
    SECONDS_TO_METERS = 1852 / 60.0

    def __init__(self, parent):
        self.value = self.INITIAL_VALUE

        wx.Panel.__init__(self, parent, -1)
        self.Sizer = wx.BoxSizer(wx.VERTICAL)
        label = wx.StaticText(
            self, wx.ID_ANY, "Distance Tolerance for Duplicate Points (seconds latitude)",
        )
        self.Sizer.Add(label, 0, wx.ALL, self.SPACING)

        self.slider = sliders.TextSlider(
            self,
            wx.ID_ANY,
            value=self.INITIAL_VALUE,
            minValue=self.MINIMUM,
            maxValue=self.MAXIMUM,
            steps=self.SLIDER_STEPS,
            valueUnit="''",
            style=wx.SL_HORIZONTAL | wx.SL_LABELS
        )

        self.Sizer.Add(
            self.slider,
            0, wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT,
            border=self.SPACING
        )

        self.meters_label = sliders.SliderLabel(self, -1, self.seconds_to_meters(self.slider.GetValue()), self.seconds_to_meters(self.MINIMUM), self.seconds_to_meters(self.MAXIMUM), " m")
        self.Sizer.Add(
            self.meters_label,
            0, wx.EXPAND | wx.LEFT | wx.RIGHT,
            border=12
        )

        self.slider.Bind(wx.EVT_SLIDER, self.slider_moved)
        self.slider.textCtrl.Bind(wx.EVT_TEXT, self.OnTextChanged)

    def OnTextChanged(self, event):
        event.Skip()
        self.meters_label.SetValue("%s" % self.seconds_to_meters(float(event.String)))

    def slider_moved(self, event):
        self.value = self.slider.GetValue()
        self.meters_label.SetValue("%s" % self.seconds_to_meters(self.slider.GetValue()))

    @staticmethod
    def seconds_to_meters(value):
        return int(value * Distance_slider.SECONDS_TO_METERS)



class MergePointsPanel(wx.Panel):
    SPACING = 5
    SLIDER_MIN_WIDTH = 100
    NAME = "Merge Duplicate Points"

    layer = None
    duplicates = []
    list_contains_real_data = False
    dirty = False

    def __init__(self, parent, task):
        self.task = task
        wx.Panel.__init__(self, parent, wx.ID_ANY)
        
        # Mac/Win needs this, otherwise background color is black
        attr = self.GetDefaultAttributes()
        self.SetBackgroundColour(attr.colBg)

        self.sizer = wx.BoxSizer(wx.VERTICAL)

        self.distance_slider = Distance_slider(self)
        self.distance_slider.SetMinSize((self.SLIDER_MIN_WIDTH, -1))
        self.sizer.Add(
            self.distance_slider,
            0, wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT, border=self.SPACING
        )

        self.depth_check = wx.CheckBox(self, -1, "Enable Depth Tolerance Check for Duplicate Points")
        self.sizer.Add(self.depth_check, 0, wx.TOP | wx.LEFT | wx.RIGHT, border=self.SPACING)

        self.depth_slider = sliders.TextSlider(self, -1, 100, minValue=0, maxValue=1000, steps=1000, valueUnit="%", style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.depth_slider.Enable(False)

        self.sizer.Add(self.depth_slider, 0, wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT, self.SPACING)

        #self.depth_slider = Depth_slider( self )
        self.depth_slider.SetMinSize((self.SLIDER_MIN_WIDTH, -1))

        self.find_button_id = wx.NewId()
        self.find_button = wx.Button(
            self,
            self.find_button_id,
            "Step 1: Find Duplicates"
        )
        self.find_button.SetDefault()
        self.find_button_id = wx.NewId()
        self.find_button.SetToolTipString("Click Find Duplicates to display a list of possible duplicate points, grouped into pairs and displayed as point index numbers. Click on a pair to highlight its points on the map.")
        self.sizer.Add(
            self.find_button, 0,
            wx.ALIGN_LEFT | wx.ALL,
            border=self.SPACING
        )

        self.list_view = wx.ListView(self, wx.ID_ANY, style=wx.LC_LIST)
        self.list_view.SetMinSize((100, 150))

        self.sizer.Add(
            self.list_view, 1, wx.EXPAND | wx.ALL,
            border=self.SPACING
        )

        self.remove_button_id = wx.NewId()
        self.remove_button = wx.Button(
            self,
            self.remove_button_id,
            "Remove from Merge List"
        )
        self.remove_button.Enable(False)
        self.sizer.Add(
            self.remove_button, 0,
            wx.ALIGN_LEFT | wx.BOTTOM | wx.LEFT | wx.RIGHT,
            border=self.SPACING
        )

        self.button_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.merge_button_id = wx.NewId()
        self.merge_button = wx.Button(
            self,
            self.merge_button_id,
            "Step 2: Merge"
        )
        self.merge_button.SetToolTipString("Click Merge to merge each pair into a single point. Pairs that cannot be merged automatically are indicated in red and will be skipped during merging. (You can merge such points manually.)")
        self.merge_button.Enable(False)
        self.button_sizer.Add(self.merge_button, 0, wx.LEFT, border=self.SPACING)

        self.sizer.Add(
            self.button_sizer, 0, wx.ALIGN_RIGHT | wx.BOTTOM | wx.LEFT | wx.RIGHT,
            border=self.SPACING
        )

        self.list_view.Bind(wx.EVT_LIST_ITEM_SELECTED, self.update_selection_once)
        self.list_view.Bind(wx.EVT_LIST_ITEM_DESELECTED, self.update_selection_once)
        self.list_view.Bind(wx.EVT_LIST_KEY_DOWN, self.key_pressed)
        self.remove_button.Bind(wx.EVT_BUTTON, self.delete_selected_groups)

        self.Bind(wx.EVT_BUTTON, self.merge_clicked, id=self.merge_button_id)

        self.SetSizer(self.sizer)
        self.sizer.Layout()
        self.Fit()

        self.find_button.Bind(wx.EVT_BUTTON, self.find_duplicates)
        self.Bind(wx.EVT_CHECKBOX, self.on_depth_check)
    
    def set_task(self, task):
        self.task = task

    def on_points_deleted(self, layer):
        if layer == self.layer:
            self.clear_results()

    def on_depth_check(self, event):
        self.depth_slider.Enable(event.IsChecked())

    def find_duplicates(self, event):
        # at the time the button is pressed, we commit to a layer
        project = self.task.active_editor
        self.layer = project.layer_tree_control.get_selected_layer()
        if (self.layer == None or self.layer.points == None or len(self.layer.points) < 2):
            project.window.error("You must first select a layer with points in the layer tree.")
            self.layer = None

            return

        depth_value = -1
        if self.depth_check.IsChecked():
            depth_value = self.depth_slider.GetValue()

        self.duplicates = self.layer.find_duplicates(self.distance_slider.value / (60 * 60), depth_value)
        # print self.duplicates
#        self.create_results_area()
        self.display_results()

    def clear_results(self):
        self.list_view.ClearAll()
        self.list_view.InsertStringItem(0, "Click Find Duplicates to search.")

        project = self.task.active_editor
        for layer in project.layer_manager.flatten():
            layer.clear_all_selections(constants.STATE_FLAGGED)

        self.list_contains_real_data = False
        self.update_selection()

    def display_results(self):
        project = self.task.active_editor
        self.list_view.ClearAll()
        self.layer.clear_all_selections(constants.STATE_FLAGGED)

        MAX_POINTS_TO_LIST = 500

        pair_count = len(self.duplicates)

        self.merge_button.Enable(False)
        if (pair_count == 0):
            self.list_view.InsertStringItem(0, "No duplicate points found.")

            self.list_contains_real_data = False
            self.update_selection()
            project.refresh()

            return

        if (pair_count > MAX_POINTS_TO_LIST):
            self.list_view.InsertStringItem(
                0,
                "Too many duplicate points to display (%d pairs)." % pair_count
            )

            # self.list_view.SetItemData( 0, -1 )

            # self.list_points.extend( duplicates )

            self.list_contains_real_data = False
            self.update_selection()
            project.refresh()

            return
        self.merge_button.Enable(True)

        self.list_contains_real_data = True
        points_in_lines = self.layer.get_all_line_point_indexes()

        for (index, points) in enumerate(self.duplicates):
            # + 1 because points start from zero but users should see them
            # starting from one.
            formatted = ", ".join([str(p + 1) for p in points])
            self.list_view.InsertStringItem(index, formatted)

            # If each of the points in this group is within a line, then we
            # don't know how to merge it automatically. So distinguish it
            # from the groups we do know how to merge.
            for point in points:
                if point not in points_in_lines:
                    break
            else:
                self.list_view.SetItemTextColour(index, wx.RED)

        self.update_selection()
        self.merge_button.SetDefault()

    def update_selection_once(self, event):
        if not self.dirty:
            self.dirty = True
            wx.CallAfter(self.update_selection)
        event.Skip()

    def update_selection(self):
        if (not self.list_contains_real_data):
            return

        project = self.task.active_editor
        self.dirty = False

        points = []

        selected = self.list_view.GetFirstSelected()
        while selected != -1:
            """
            original_index = self.list.GetItem( selected ).GetData()
            if original_index != -1:
                points.extend( self.list_points[ original_index ] )
            """
            points.extend(self.duplicates[selected])
            selected = self.list_view.GetNextSelected(selected)

        if (len(points) == 0):
            self.remove_button.Enable(False)
            points = sum([list(d) for d in self.duplicates], [])
        else:
            self.remove_button.Enable(True)

        point_count = len(points)

        self.layer.clear_all_selections(constants.STATE_FLAGGED)

        if (point_count == 0):
            return

        self.layer.select_points(points, constants.STATE_FLAGGED)
        bounds = self.layer.compute_bounding_rect(constants.STATE_FLAGGED)
        project.control.zoom_to_include_world_rect(bounds)
        project.refresh()

    def key_pressed(self, event):
        key_code = event.GetKeyCode()
        if key_code == wx.WXK_DELETE:
            self.delete_selected_groups()
        else:
            event.Skip()

    def delete_selected_groups(self, event=None):
        if (not self.list_contains_real_data):
            return

        selected = self.list_view.GetFirstSelected()
        if (selected == -1):
            return

        to_delete = []

        while selected != -1:
            to_delete.append(selected)
            selected = self.list_view.GetNextSelected(selected)

        # Reversing is necessary so as not to mess up the indices of
        # subsequent selected groups when a previous group is deleted.
        for selected in reversed(to_delete):
            # original_index = self.list.GetItem( selected ).GetData()
            self.list_view.DeleteItem(selected)
            self.duplicates.pop(selected)
            # self.list_points[ original_index ] = []

        # If there's a focused group after the group deletions, then select
        # that group. This way the user can hit delete repeatedly to delete
        # a bunch of groups from the list.
        focused = self.list_view.GetFocusedItem()
        if focused != -1:
            self.list_view.Select(focused, on=True)

        self.update_selection()

    def merge_clicked(self, event):
        if (not self.list_contains_real_data):
            return

        points_in_lines = self.layer.get_all_line_point_indexes()
        self.layer.merge_duplicates(self.duplicates, points_in_lines)

        event.Skip()
        self.clear_results()
