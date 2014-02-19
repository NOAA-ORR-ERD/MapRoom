import sys
import math
import wx
from wx.lib.pubsub import pub

import Layer
import app_globals
import ui.controls.sliders as sliders


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


class Merge_duplicate_points_dialog(wx.Dialog):
    SPACING = 15
    SLIDER_MIN_WIDTH = 400
    NAME = "Merge Duplicate Points"

    layer = None
    duplicates = []
    list_contains_real_data = False
    dirty = False

    def __init__(self, parent):
        wx.Dialog.__init__(
            self, parent, wx.ID_ANY, self.NAME,
            style=wx.DEFAULT_DIALOG_STYLE, name=self.NAME
        )
        self.SetIcon(app_globals.application.frame.GetIcon())

        self.outer_sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer = wx.BoxSizer(wx.VERTICAL)

        self.panel = wx.Panel(self, wx.ID_ANY)
        self.outer_sizer.Add(self.panel, 1, wx.EXPAND)

        self.distance_slider = Distance_slider(self.panel)
        self.distance_slider.SetMinSize((self.SLIDER_MIN_WIDTH, -1))
        self.sizer.Add(
            self.distance_slider,
            0, wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT, border=self.SPACING
        )

        self.depth_check = wx.CheckBox(self.panel, -1, "Enable Depth Tolerance Check for Duplicate Points")
        self.sizer.Add(self.depth_check, 0, wx.TOP | wx.LEFT | wx.RIGHT, border=self.SPACING)

        self.depth_slider = sliders.TextSlider(self.panel, -1, 100, minValue=0, maxValue=1000, steps=1000, valueUnit="%", style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.depth_slider.Enable(False)

        self.sizer.Add(self.depth_slider, 0, wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT, self.SPACING)

        #self.depth_slider = Depth_slider( self.panel )
        self.depth_slider.SetMinSize((self.SLIDER_MIN_WIDTH, -1))

        self.find_button_id = wx.NewId()
        self.find_button = wx.Button(
            self.panel,
            self.find_button_id,
            "Find Duplicates"
        )
        self.find_button.SetDefault()
        self.find_button_id = wx.NewId()
        self.sizer.Add(
            self.find_button, 0,
            wx.ALIGN_LEFT | wx.ALL,
            border=self.SPACING
        )

        self.label = None

        self.panel.SetSizer(self.sizer)
        self.SetSizer(self.outer_sizer)

        self.sizer.Layout()
        self.Fit()
        self.Show()

        self.find_button.Bind(wx.EVT_BUTTON, self.find_duplicates)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.Bind(wx.EVT_CHECKBOX, self.on_depth_check)

        pub.subscribe(self.on_points_deleted, ('layer', 'points', 'deleted'))

    def on_points_deleted(self, layer):
        if layer == self.layer:
            self.clear_results()

    def on_depth_check(self, event):
        self.depth_slider.Enable(event.IsChecked())

    def on_close(self, event):
        for layer in app_globals.layer_manager.flatten():
            layer.clear_all_selections(Layer.STATE_FLAGGED)
        self.Destroy()

    def find_duplicates(self, event):
        # at the time the button is pressed, we commit to a layer
        self.layer = app_globals.application.layer_tree_control.get_selected_layer()
        if (self.layer == None or self.layer.points == None or len(self.layer.points) < 2):
            wx.MessageDialog(
                self,
                message="You must first select a layer with points in the layer tree.",
                caption="Error",
                style=wx.OK | wx.ICON_ERROR,
            ).ShowModal()
            self.layer = None

            return

        self.SetTitle("Merge Duplicate Points -- Layer '" + self.layer.name + "'")

        depth_value = -1
        if self.depth_check.IsChecked():
            depth_value = self.depth_slider.GetValue()

        self.duplicates = self.layer.find_duplicates(self.distance_slider.value / (60 * 60), depth_value)
        # print self.duplicates
        self.create_results_area()
        self.display_results()

    def create_results_area(self):
        if (self.label is not None):
            return

        self.Freeze()

        self.label_text = "Below is a list of possible duplicate points, " + \
                          "grouped into pairs and displayed as point index numbers. " + \
                          "Click on a pair to highlight its points on the map."

        self.label = wx.StaticText(
            self.panel,
            wx.ID_ANY,
            self.label_text,
            style=wx.ST_NO_AUTORESIZE
        )
        self.sizer.Add(
            self.label,
            0, wx.EXPAND | wx.LEFT | wx.RIGHT, border=self.SPACING
        )

        self.list_view = wx.ListView(self.panel, wx.ID_ANY, style=wx.LC_LIST)
        self.list_view.SetMinSize((-1, 150))

        self.sizer.Add(
            self.list_view, 1, wx.EXPAND | wx.ALL,
            border=self.SPACING
        )

        self.remove_button_id = wx.NewId()
        self.remove_button = wx.Button(
            self.panel,
            self.remove_button_id,
            "Remove from Merge List"
        )
        self.remove_button.Enable(False)
        self.sizer.Add(
            self.remove_button, 0,
            wx.ALIGN_LEFT | wx.BOTTOM | wx.LEFT | wx.RIGHT,
            border=self.SPACING
        )

        self.merge_label_text = \
            "Click Merge to merge each pair into a single point. " + \
            "Pairs that cannot be merged automatically are indicated " + \
            "in red and will be skipped during merging. (You can merge " + \
            "such points manually.)"

        self.merge_label = wx.StaticText(
            self.panel,
            wx.ID_ANY,
            self.merge_label_text,
            style=wx.ST_NO_AUTORESIZE
        )
        self.sizer.Add(
            self.merge_label, 0,
            wx.ALIGN_LEFT | wx.BOTTOM | wx.LEFT | wx.RIGHT,
            border=self.SPACING
        )

        self.button_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.merge_button_id = wx.NewId()
        self.merge_button = wx.Button(
            self.panel,
            self.merge_button_id,
            "Merge"
        )

        self.close_button_id = wx.NewId()
        self.close_button = wx.Button(
            self.panel,
            self.close_button_id,
            "Close"
        )

        # Dialog button ordering, by convention, is backwards on Windows.
        if sys.platform.startswith("win"):
            self.button_sizer.Add(self.merge_button, 0, wx.LEFT, border=self.SPACING)
            self.button_sizer.Add(self.close_button, 0, wx.LEFT, border=self.SPACING)
        else:
            self.button_sizer.Add(self.close_button, 0, wx.LEFT, border=self.SPACING)
            self.button_sizer.Add(self.merge_button, 0, wx.LEFT, border=self.SPACING)

        self.sizer.Add(
            self.button_sizer, 0, wx.ALIGN_RIGHT | wx.BOTTOM | wx.LEFT | wx.RIGHT,
            border=self.SPACING
        )

        self.label.Wrap(self.sizer.GetSize()[0] - self.SPACING * 2)
        self.merge_label.Wrap(self.sizer.GetSize()[0] - self.SPACING * 2)

        self.list_view.Bind(wx.EVT_LIST_ITEM_SELECTED, self.update_selection_once)
        self.list_view.Bind(wx.EVT_LIST_ITEM_DESELECTED, self.update_selection_once)
        self.list_view.Bind(wx.EVT_LIST_KEY_DOWN, self.key_pressed)
        self.remove_button.Bind(wx.EVT_BUTTON, self.delete_selected_groups)

        self.Bind(wx.EVT_BUTTON, self.merge_clicked, id=self.merge_button_id)
        self.Bind(wx.EVT_BUTTON, self.on_close, id=self.close_button_id)

        self.sizer.Layout()
        self.Fit()
        self.Thaw()

    def clear_results(self):
        if (self.label is None):
            return

        self.list_view.ClearAll()
        self.list_view.InsertStringItem(0, "Click Find Duplicates to search.")

        for layer in app_globals.layer_manager.flatten():
            layer.clear_all_selections(Layer.STATE_FLAGGED)

        self.list_contains_real_data = False
        self.update_selection()

    def display_results(self):
        self.list_view.ClearAll()
        # self.list_points = []
        MAX_POINTS_TO_LIST = 500

        pair_count = len(self.duplicates)

        if (pair_count == 0):
            self.list_view.InsertStringItem(0, "No duplicate points found.")

            for layer in app_globals.layer_manager.flatten():
                layer.clear_all_selections(Layer.STATE_FLAGGED)

            self.list_contains_real_data = False
            self.update_selection()

            return

        if (pair_count > MAX_POINTS_TO_LIST):
            self.list_view.InsertStringItem(
                0,
                "Too many duplicate points to display (%d pairs)." % pair_count
            )

            for layer in app_globals.layer_manager.flatten():
                layer.clear_all_selections(Layer.STATE_FLAGGED)

            # self.list_view.SetItemData( 0, -1 )

            # self.list_points.extend( duplicates )

            self.list_contains_real_data = False
            self.update_selection()

            return

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

        self.layer.clear_all_selections(Layer.STATE_FLAGGED)

        if (point_count == 0):
            return

        self.layer.select_points(points, Layer.STATE_FLAGGED)
        bounds = self.layer.compute_bounding_rect(Layer.STATE_FLAGGED)
        app_globals.application.renderer.zoom_to_include_world_rect(bounds)
        app_globals.application.refresh()

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
