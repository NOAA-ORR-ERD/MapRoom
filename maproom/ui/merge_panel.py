import wx

from ..layers import state
from . import sliders

import logging
log = logging.getLogger(__name__)


class DistanceCtrl(wx.Panel):
    SPACING = 5
    SECONDS_TO_METERS = 1852 / 60.0

    def __init__(self, parent, initial_value=0.6):
        wx.Panel.__init__(self, parent, -1)
        self.Sizer = wx.BoxSizer(wx.VERTICAL)
        label = wx.StaticText(self, wx.ID_ANY, "Distance Tolerance for Duplicate Points")
        self.Sizer.Add(label, 0, wx.ALL, 0)
        label = wx.StaticText(self, wx.ID_ANY, "(add m for meters, otherwise interpreted as \" lat)")
        self.Sizer.Add(label, 0, wx.ALL, 2)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.distance_ctrl = wx.TextCtrl(self, -1)
        hbox.Add(self.distance_ctrl, 1, wx.EXPAND, self.SPACING)
        self.other_distance = wx.StaticText(self, wx.ID_ANY, " =")
        hbox.Add(self.other_distance, 3, wx.ALIGN_CENTER_VERTICAL, self.SPACING)

        self.Sizer.Add(hbox, 0, wx.EXPAND, self.SPACING)

        self.is_valid(str(initial_value))
        self.distance_ctrl.SetValue(str(initial_value))
        self.distance_ctrl.Bind(wx.EVT_TEXT, self.on_text_changed)

    def is_valid(self, text):
        c = self.distance_ctrl
        c.SetBackgroundColour("#FFFFFF")
        try:
            is_m = self.parse_from_string(text)
            valid = True
        except Exception:
            c.SetBackgroundColour("#FF8080")
            valid = False
        c.Refresh()
        label = ""
        if valid:
            if is_m:
                label = " = %f\"" % (self.degrees * 3600.0)
            else:
                label = " = %fm" % self.meters
        self.other_distance.SetLabel(label)
        return valid

    def parse_from_string(self, text):
        text = text.strip()
        if not text:
            meters = 0.0
        elif text.endswith("m"):
            meters = float(text[:-1])
            is_m = True
        else:
            text.strip("\"")
            meters = float(text) * self.SECONDS_TO_METERS
            is_m = False
        self.meters = meters
        self.degrees = meters / self.SECONDS_TO_METERS / 60.0 / 60.0
        return is_m

    def on_text_changed(self, evt):
        evt.Skip()
        if self.is_valid(evt.String):
            self.GetParent().find_button.Enable(True)
        else:
            self.meters = -1
            self.degrees = -1
            self.GetParent().find_button.Enable(False)


class MergePointsPanel(wx.Panel):
    SPACING = 5
    SLIDER_MIN_WIDTH = 100
    NAME = "Merge Duplicate Points"

    layer = None
    duplicates = []
    list_contains_real_data = False
    dirty = False

    def __init__(self, parent, editor):
        self.editor = editor
        wx.Panel.__init__(self, parent, wx.ID_ANY, name="Merge Points")

        # Mac/Win needs this, otherwise background color is black
        attr = self.GetDefaultAttributes()
        self.SetBackgroundColour(attr.colBg)

        self.sizer = wx.BoxSizer(wx.VERTICAL)

        self.distance_ctrl = DistanceCtrl(self)
        self.sizer.Add(self.distance_ctrl, 0, wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT, self.SPACING)

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
        self.find_button.SetToolTip("Click Find Duplicates to display a list of possible duplicate points, grouped into pairs and displayed as point index numbers. Click on a pair to highlight its points on the map.")
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
        self.merge_button.SetToolTip("Click Merge to merge each pair into a single point. Pairs that cannot be merged automatically are indicated in red and will be skipped during merging. (You can merge such points manually.)")
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

    def on_points_deleted(self, layer):
        if layer == self.layer:
            self.clear_results()

    def on_depth_check(self, event):
        self.depth_slider.Enable(event.IsChecked())

    def find_duplicates(self, event):
        if not self.IsShownOnScreen():
            # Hack for OS X: default buttons can still get Return key presses
            # even when the panel is hidden, which is not what we want.
            event.Skip()
            return

        # at the time the button is pressed, we commit to a layer
        self.layer = self.editor.current_layer
        if (self.layer is None or not self.layer.has_points() or len(self.layer.points) < 2 or not hasattr(self.layer, "find_duplicates")):
            self.editor.frame.error("You must first select a layer with points in the layer tree.")
            self.layer = None

            return

        depth_value = -1
        if self.depth_check.IsChecked():
            depth_value = self.depth_slider.GetValue()

        self.duplicates = self.layer.find_duplicates(self.distance_ctrl.degrees, depth_value)
        # print self.duplicates
#        self.create_results_area()
        self.display_results()

    def clear_results(self):
        self.list_view.ClearAll()
        self.list_view.InsertStringItem(0, "Click Find Duplicates to search.")

        for layer in self.editor.layer_manager.flatten():
            layer.clear_all_selections(state.FLAGGED)

        self.list_contains_real_data = False
        self.update_selection()

    def display_results(self):
        self.list_view.ClearAll()
        self.layer.clear_all_selections(state.FLAGGED)

        MAX_POINTS_TO_LIST = 500

        pair_count = len(self.duplicates)

        self.merge_button.Enable(False)
        if (pair_count == 0):
            self.list_view.InsertStringItem(0, "No duplicate points found.")

            self.list_contains_real_data = False
            self.update_selection()
            self.editor.refresh()

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
            self.editor.refresh()

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
        try:
            self.merge_button.SetDefault()
        except Exception as e:
            # don't know why this is failing on a Mac. The default button is supposed to move
            # from the Find button to the Merge button, but it fails instead. Catching this
            # exception partially works around it, but leaves the Find button still highlighted.
            # Perhaps because the merge panel is reparented in the sidebar?
            #
            # Traceback (most recent call last):
            #  File "/Users/rob.mcmullen/src/refactor/maproom/maproom/ui/merge_panel.py", line 205, in find_duplicates
            #    self.display_results()
            #  File "/Users/rob.mcmullen/src/refactor/maproom/maproom/ui/merge_panel.py", line 271, in display_results
            #    self.merge_button.SetDefault()
            # wx._core.wxAssertionError: C++ assertion "tlw" failed at /Users/robind/projects/buildbots/macosx-vm6/dist-osx-py36/Phoenix/ext/wxWidgets/src/common/btncmn.cpp(104) in SetDefault(): button without top level window?
            log.error(f"Must be on a mac? {e}")

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

        self.layer.clear_all_selections(state.FLAGGED)

        if (point_count == 0):
            return

        self.layer.select_points(points, state.FLAGGED)
        bounds = self.layer.compute_bounding_rect(state.FLAGGED)
        self.editor.layer_canvas.zoom_to_include_world_rect(bounds)
        self.editor.refresh()

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
        cmd = self.layer.merge_duplicates(self.duplicates, points_in_lines)
        self.editor.process_command(cmd)

        event.Skip()
        self.clear_results()
