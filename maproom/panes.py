import os
import cgi
import time
import datetime
import calendar

import wx
from wx.lib.ClickableHtmlWindow import PyClickableHtmlWindow

# Enthought library imports.

from omnivore.framework.panes import FrameworkPane, FrameworkFixedPane
from omnivore.utils.wx.popuputil import SpringTabs
from omnivore.utils.wx.download_manager import DownloadControl
from omnivore.utils.wx.zoomruler import ZoomRuler
from omnivore.utils.textutil import pretty_seconds, parse_pretty_seconds

from layer_tree_control import LayerTreeControl
from ui.info_panels import LayerInfoPanel, SelectionInfoPanel
from ui.triangle_panel import TrianglePanel
from ui.merge_panel import MergePointsPanel
from ui.undo_panel import UndoHistoryPanel

import logging
log = logging.getLogger(__name__)


class TimelinePanel(ZoomRuler):
    def __init__(self, parent, task, **kwargs):
        ZoomRuler.__init__(self, parent, **kwargs)
        self.task = task
        self.editor = None
 
    def get_timeline_info(self):
        if self.editor is not None:
            layers, start, end = self.editor.layer_manager.get_timestamped_layers(self.editor.layer_visibility, False)
        else:
            layers = []
            start = end = 0.0
        if start == 0.0:
            today = datetime.datetime.today()
            noon = today.replace(hour=12, minute=0, second=0, tzinfo=None)
            start = calendar.timegm(noon.timetuple())
        if end < start:
            end = start

        today = datetime.datetime.fromtimestamp(start)
        noon = today.replace(hour=12, minute=0, second=0, tzinfo=None)
        day_before = noon - datetime.timedelta(days=1)
        start = calendar.timegm(day_before.timetuple())

        today = datetime.datetime.fromtimestamp(end)
        noon = today.replace(hour=12, minute=0, second=0, tzinfo=None)
        day_after = noon + datetime.timedelta(days=1)
        end = calendar.timegm(day_after.timetuple())
        # print "DATES:", start, day_before, end, day_after
        info = {
            "format": "month",
            "earliest_time": start,
            "latest_time": end,
            "marks": [(l.start_time, l.end_time, l) for l in layers],
        }
        # print info
        return info

    def marks_to_display_as_selected(self):
        sel_marks = self.marks_in_selection()
        for start, end, data in self._marks:
            if self.editor.layer_visibility[data]['layer']:
                sel_marks.add(data)
        return sel_marks

    def selection_finished_callback(self, selected_ranges):
        if self.editor is not None:
            selected_layers = self.editor.layer_manager.get_visible_layers_in_ranges(selected_ranges)
            self.editor.set_layer_visibility(selected_layers)
            self.editor.layer_tree_control.Refresh()

    def item_activation_callback(self, item):
        if self.editor is not None:
            layers = [item]
            self.editor.set_layer_visibility(layers)
            self.editor.layer_tree_control.Refresh()

    def over_item_callback(self, pos, item):
        if self.editor is not None:
            self.editor.status_message = str(item)

    def not_over_item_callback(self, pos):
        if self.editor is not None:
            self.editor.status_message = ""

    def selected_item_callback(self, item):
        if self.editor is not None:
            self.editor.layer_tree_control.set_edit_layer(item)

    def selection_cleared_callback(self):
        if self.editor is not None:
            self.Refresh()

    def playback_start_callback(self):
        self.GetParent().play.SetLabel("Pause")

    def playback_pause_callback(self):
        self.GetParent().play.SetLabel("Play")

    def playback_callback(self, current_time):
        log.debug("playback for time: %f" % current_time)
        selected_layers = self.editor.layer_manager.get_visible_layers_at_time(self.editor.layer_visibility, current_time)
        self.editor.set_layer_visibility(selected_layers)
        self.editor.layer_tree_control.Refresh()


wxEVT_TIME_STEP_MODIFIED = wx.NewEventType()
EVT_TIME_STEP_MODIFIED = wx.PyEventBinder(wxEVT_TIME_STEP_MODIFIED, 1)


class TimeStepEvent(wx.CommandEvent):
    def __init__(self, evtType, evtId, step=None, rate=None, **kwargs):
        wx.CommandEvent.__init__(self, evtType, evtId, **kwargs)
        self._step = step
        self._rate = rate

    def GetStep(self):
        return self._step

    def GetRate(self):
        return self._rate


class TimeStepPanelMixin(object):
    def setup_panel(self, step_value, step_rate):
        panel = wx.Panel(self)
        vsizer = wx.BoxSizer(wx.VERTICAL)

        self.add_header(panel, vsizer)

        sizer = wx.BoxSizer(wx.HORIZONTAL)

        step_values = ['10m', '20m', '30m', '40m', '45m', '60m', '90m', '120m', '3hr', '4hr', '5hr', '6hr', '8hr', '10hr', '12hr', '16h', '24hr', '36hr', '48hr', '3d', '4d', '5d', '6d', '7d', '2wk', '3wk', '4wk']
        step_values_as_seconds = [parse_pretty_seconds(a) for a in step_values]
        cb = wx.ComboBox(panel, 500, choices=step_values, style=wx.CB_DROPDOWN)
        try:
            i = step_values_as_seconds.index(step_value)
            cb.SetSelection(i)
        except ValueError:
            cb.ChangeValue(pretty_seconds(step_value))
        cb.Bind(wx.EVT_COMBOBOX, self.on_combo)
        cb.Bind(wx.EVT_TEXT, self.on_text)
        cb.Bind(wx.EVT_TEXT_ENTER, self.on_text_enter)
        sizer.Add(cb, 1, wx.ALL, 5)
        self.step_ctrl = cb

        st = wx.StaticText(panel, -1, "every")
        sizer.Add(st, 0, wx.ALL, 5)

        rate_values = ['100ms', '1s', '2s', '3s', '4s', '5s', '10s', '15s', '20s']
        rate_values_as_seconds = [parse_pretty_seconds(a) for a in rate_values]
        cb = wx.ComboBox(panel, 500, choices=rate_values, style=wx.CB_DROPDOWN)
        try:
            i = rate_values_as_seconds.index(step_rate)
            cb.SetSelection(i)
        except ValueError:
            cb.ChangeValue(pretty_seconds(rate_value))
        cb.Bind(wx.EVT_COMBOBOX, self.on_combo)
        cb.Bind(wx.EVT_TEXT, self.on_text)
        cb.Bind(wx.EVT_TEXT_ENTER, self.on_text_enter)
        sizer.Add(cb, 1, wx.ALL, 5)
        self.rate_ctrl = cb

        vsizer.Add(sizer, 1, wx.ALL, 0)

        self.add_footer(panel, vsizer)

        panel.SetSizer(vsizer)
        vsizer.Fit(panel)
        vsizer.Fit(self)
        self.Layout()

    def add_header(self, parent, sizer):
        pass

    def add_footer(self, parent, sizer):
        pass

    def on_combo(self, evt):
        # When the user selects something, we go here.
        self.verify_and_send_event(evt)

    # Capture events every time a user hits a key in the text entry field.
    def on_text(self, evt):
        self.verify_and_send_event(evt)
        evt.Skip()

    # Capture events when the user types something into the control then
    # hits ENTER.
    def on_text_enter(self, evt):
        self.verify_and_send_event(evt)
        evt.Skip()

    def verify_and_send_event(self, evt):
        cb = evt.GetEventObject()
        val = evt.GetString()
        try:
            if cb == self.step_ctrl:
                step = parse_pretty_seconds(val)
                rate = None
            else:
                step = None
                rate = parse_pretty_seconds(val)
        except ValueError:
            pass
        else:
            self.send_event(step, rate)

    def send_event(self, step, rate):
        e = TimeStepEvent(wxEVT_TIME_STEP_MODIFIED, self.GetId(), step, rate)
        e.SetEventObject(self)
        print "sending", e, "to", self.GetEventHandler()
        self.GetParent().GetEventHandler().ProcessEvent(e)


class TimeStepPopup(wx.PopupTransientWindow, TimeStepPanelMixin):
    def __init__(self, parent, step_value, step_rate, style):
        wx.PopupTransientWindow.__init__(self, parent, style)
        self.setup_panel(step_value, step_rate)

    def show_relative_to(self, btn):
        # Show the popup right below or above the button
        # depending on available screen space...
        pos = btn.ClientToScreen( (0,0) )
        sz =  btn.GetSize()
        self.Position(pos, (0, sz[1]))
        self.Popup()


class TimeStepDialog(wx.Dialog, TimeStepPanelMixin):
    def __init__(self, parent, step_value, step_rate, style):
        wx.Dialog.__init__(self, parent, -1, "Time Step Rate")
        self.setup_panel(step_value, step_rate)

    def add_header(self, parent, sizer):
        t = wx.StaticText(parent, -1, "Choose step rate")
        sizer.Add(t, 0, wx.ALL, 5)

    def add_footer(self, parent, sizer):
        btnsizer = wx.StdDialogButtonSizer()
        btn = wx.Button(parent, wx.ID_OK)
        btn.SetDefault()
        btnsizer.AddButton(btn)
        btn = wx.Button(parent, wx.ID_CANCEL)
        btnsizer.AddButton(btn)
        btnsizer.Realize()
        sizer.Add(btnsizer, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

    def on_combo(self, evt):
        evt.Skip()

    def on_text(self, evt):
        evt.Skip()

    def on_text_enter(self, evt):
        evt.Skip()

    def show_relative_to(self, btn):
        if self.ShowModal() == wx.ID_OK:
            step = parse_pretty_seconds(self.step_ctrl.GetValue())
            rate = parse_pretty_seconds(self.rate_ctrl.GetValue())
            wx.CallAfter(self.send_event, step, rate)


class TimelinePlaybackPanel(wx.Panel):
    def __init__(self, parent, task, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)

        sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.play = wx.Button(self, -1, "Play")
        self.play.Bind(wx.EVT_BUTTON, self.on_play)
        sizer.Add(self.play, 0, wx.EXPAND)

        self.steps = wx.Button(self, -1, "MM", style=wx.BU_EXACTFIT)
        self.steps.Bind(wx.EVT_BUTTON, self.on_steps)
        sizer.Add(self.steps, 0, wx.EXPAND)

        self.timeline = TimelinePanel(self, task)
        sizer.Add(self.timeline, 1, wx.EXPAND|wx.LEFT, 5)

        self.SetSizer(sizer)

        self.Bind(EVT_TIME_STEP_MODIFIED, self.on_set_steps)

    def serialize_json(self):
        return {
            "step_value": self.timeline.step_value,
            "step_rate": self.timeline.step_rate,
            "viewport": list(self.timeline.get_viewport()),
        }

    def unserialize_json(self, json):
        log.debug(str(json))
        self.timeline.step_value = json["step_value"]
        self.timeline.step_rate = json["step_rate"]
        info = json["viewport"]
        self.timeline.set_viewport(*info)

    def clear_marks(self):
        log.debug("timeline clear_marks")
        self.timeline.clear_marks()

    def recalc_view(self):
        log.debug("timeline recalc_view")
        self.timeline.editor = self.timeline.task.active_editor
        self.timeline.rebuild(self.timeline)
        self.update_ui()

    def refresh_view(self):
        log.debug("timeline refresh_view")
        editor = self.timeline.task.active_editor
        if editor is not None:
            if self.timeline.editor != editor:
                self.recalc_view()
            else:
                self.Refresh()
        self.update_ui()

    def update_ui(self):
        self.play.Enable(self.timeline.can_play)
        label = "%s / %s" % (pretty_seconds(self.timeline.step_value), pretty_seconds(self.timeline.step_rate))
        self.steps.SetLabel(label)

    def on_play(self, evt):
        log.debug("timeline play")
        if self.timeline.is_playing:
            self.timeline.pause_playback()
        else:
            self.timeline.start_playback()

    def on_steps(self, evt):
        btn = evt.GetEventObject()
        wx.CallAfter(self.on_steps_cb, btn)

    def on_steps_cb(self, btn):
        if True:
            win = TimeStepDialog(self, self.timeline.step_value, self.timeline.step_rate, wx.SIMPLE_BORDER)
        else:  # popup works on linux but not mac or windows
            win = TimeStepPopup(self, self.timeline.step_value, self.timeline.step_rate, wx.SIMPLE_BORDER)
        win.show_relative_to(btn)

    def on_set_steps(self, evt):
        step = evt.GetStep()
        rate = evt.GetRate()
        print("step=%s rate=%s" % (step, rate))
        if step is not None:
            self.timeline.step_value = step
        if rate is not None:
            self.timeline.step_rate = rate
        self.update_ui()


class TimelinePane(FrameworkPane):
    # TaskPane interface ###################################################

    id = 'maproom.timeline_pane'
    name = 'Timeline'

    caption_visible = False
    movable = False

    def create_contents(self, parent):
        control = TimelinePlaybackPanel(parent, self.task)
        return control

    def get_new_info(self):
        info = FrameworkPane.get_new_info(self)
        info.LeftDockable(False)
        info.RightDockable(False)
        info.DockFixed(True)
        return info


class LayerSelectionPane(FrameworkPane):
    # TaskPane interface ###################################################

    id = 'maproom.layer_selection_pane'
    name = 'Layers'

    def create_contents(self, parent):
        control = LayerTreeControl(parent, self.task.active_editor, size=(200, 300))
        return control


class LayerInfoPane(FrameworkPane):
    # TaskPane interface ###################################################

    id = 'maproom.layer_info_pane'
    name = 'Current Layer'

    def create_contents(self, parent):
        control = LayerInfoPanel(parent, self.task.active_editor, size=(200, 200))
        return control


class SelectionInfoPane(FrameworkPane):
    # TaskPane interface ###################################################

    id = 'maproom.selection_info_pane'
    name = 'Current Selection'

    def create_contents(self, parent):
        control = SelectionInfoPanel(parent, self.task.active_editor, size=(200, 200))
        return control


class TriangulatePane(FrameworkPane):
    # TaskPane interface ###################################################

    id = 'maproom.triangulate_pane'
    name = 'Triangulate'

    def create_contents(self, parent):
        control = TrianglePanel(parent, self.task)
        return control


class MergePointsPane(FrameworkPane):
    # TaskPane interface ###################################################

    id = 'maproom.merge_points_pane'
    name = 'Merge Points'

    def create_contents(self, parent):
        control = MergePointsPanel(parent, self.task)
        return control


class UndoHistoryPane(FrameworkPane):
    # TaskPane interface ###################################################

    id = 'maproom.undo_history_pane'
    name = 'Undo History'

    def create_contents(self, parent):
        control = UndoHistoryPanel(parent, self.task)
        return control

    # trait change handlers

    def _task_changed(self):
        log.debug("TASK CHANGED IN UNDOHISTORYPANE!!!! %s" % self.task)
        if self.control:
            self.control.set_task(self.task)


# class FlaggedPointPanel(wx.Panel):
#    def __init__(self, parent, task, **kwargs):
#        self.task = task
#        self.editor = None
#        wx.Panel.__init__(self, parent, wx.ID_ANY, **kwargs)
#
#        # Mac/Win needs this, otherwise background color is black
#        attr = self.GetDefaultAttributes()
#        self.SetBackgroundColour(attr.colBg)
#
#        self.sizer = wx.BoxSizer(wx.VERTICAL)
#
#        self.points = wx.ListBox(self, size=(100, -1))
#
#        self.sizer.Add(self.points, 1, wx.EXPAND)
#
#        self.SetSizer(self.sizer)
#        self.sizer.Layout()
#        self.Fit()

class FlaggedPointPanel(wx.ListBox):
    def __init__(self, parent, task, **kwargs):
        self.task = task
        self.editor = None
        self.point_indexes = []
        wx.ListBox.__init__(self, parent, wx.ID_ANY, **kwargs)
        self.Bind(wx.EVT_LEFT_DOWN, self.on_click)

        # Return key not sent through to EVT_CHAR, EVT_CHAR_HOOK or
        # EVT_KEY_DOWN events in a ListBox. This is the only event handler
        # that catches a Return.
        self.Bind(wx.EVT_KEY_UP, self.on_char)

    def DoGetBestSize(self):
        """ Base class virtual method for sizer use to get the best size
        """
        width = 100
        height = -1
        best = wx.Size(width, height)

        # Cache the best size so it doesn't need to be calculated again,
        # at least until some properties of the window change
        self.CacheBestSize(best)

        return best

    def on_char(self, evt):
        keycode = evt.GetKeyCode()
        log.debug("key down: %s" % keycode)
        if keycode == wx.WXK_RETURN:
            index = self.GetSelection()
            self.process_index(index)
        evt.Skip()

    def on_click(self, evt):
        index = self.HitTest(evt.GetPosition())
        if index >= 0:
            self.Select(index)
            self.process_index(index)

    def process_index(self, index):
        point_index = self.point_indexes[index]
        editor = self.task.active_editor
        layer = editor.layer_tree_control.get_edit_layer()
        print point_index, layer
        editor.layer_canvas.do_center_on_point_index(layer, point_index)

    def set_flagged(self, point_indexes):
        self.point_indexes = list(point_indexes)
        self.Set([str(i) for i in point_indexes])

    def recalc_view(self):
        editor = self.task.active_editor
        if editor is not None:
            self.editor = editor
            layer = editor.layer_tree_control.get_edit_layer()
            try:
                points = layer.get_flagged_point_indexes()
            except AttributeError:
                points = []
            self.set_flagged(points)

    def refresh_view(self):
        editor = self.task.active_editor
        if editor is not None:
            if self.editor != editor:
                self.recalc_view()
            else:
                self.Refresh()

    def activateSpringTab(self):
        self.recalc_view()

    def get_notification_count(self):
        self.recalc_view()
        return len(self.point_indexes)


class DownloadPanel(DownloadControl):
    def __init__(self, parent, task, **kwargs):
        self.task = task
        self.editor = None
        downloader = self.task.window.application.get_downloader()
        DownloadControl.__init__(self, parent, downloader, size=(400, -1), **kwargs)

    # turn the superclass attribute path into a property so we can override it
    # and pull out the paths from the preferences
    @property
    def path(self):
        prefs = self.task.preferences
        if prefs.download_directory:
            path = prefs.download_directory
        else:
            path = self.task.window.application.user_data_dir
        log.debug("download path: %s" % path)
        return path

    @path.setter
    def path(self, value):
        if value:
            prefs = self.task.preferences
            prefs.download_directory = value

    def refresh_view(self):
        self.Refresh()

    def activateSpringTab(self):
        self.refresh_view()

    def get_notification_count(self):
        self.refresh_view()
        return self.num_active


class SidebarPane(FrameworkFixedPane):
    # TaskPane interface ###################################################

    id = 'maproom.sidebar'
    name = 'Sidebar'

    movable = False
    caption_visible = False
    dock_layer = 9

    def flagged_cb(self, parent, task, **kwargs):
        self.flagged_control = FlaggedPointPanel(parent, task)

    def download_cb(self, parent, task, **kwargs):
        self.download_control = DownloadPanel(parent, task)

    def create_contents(self, parent):
        control = SpringTabs(parent, self.task, popup_direction="left")
        control.add_tab("Flagged Points", self.flagged_cb)
        control.add_tab("Downloads", self.download_cb)
        return control

    def refresh_active(self):
        self.control.update_notifications()
        active = self.control._radio
        if active is not None and active.is_shown:
            active.managed_window.refresh_view()


class HtmlHelpPane(FrameworkPane):
    # TaskPane interface ###################################################

    id = 'maproom.html_help_pane'
    name = 'HTML Help'
    code_header = "Markup"
    desc_header = "Appearance"
    help_text = """<h1>Title</h1>|<h1>Title</h1>
    <h2>Section</h2>|<h2>Section</h2>
    <h3>Subsection</h3>|<h3>Subsection</h3>
    <p>|start new paragraph
    <i>italics</i>|<i>italics</i>
    <b>bold</b>|<b>bold</b>
    <ul>|start list
    <li>list item|<ul><li>list item</li></ul>
    </ul>|end list
    """

    def create_contents(self, parent):
        control = PyClickableHtmlWindow(parent, -1, style=wx.NO_FULL_REPAINT_ON_RESIZE, size=(400, 300))
        control.SetPage(self.get_help_text())
        return control

    def get_help_text(self):
        lines = ["<table><tr><th>%s</th><th>%s</th>" % (self.code_header, self.desc_header)]
        for line in self.help_text.splitlines():
            if "|" in line:
                code, desc = line.split("|", 1)
                code = cgi.escape(code).replace("[RET]", "<br>")
                lines.append("<tr><td><tt>%s</tt></td><td>%s</td></tr>\n" % (code, desc))
            else:
                lines.append("<tr><td colspan=2>%s</td></tr>" % line)
        lines.append("</table>")
        return "\n".join(lines)


class RSTHelpPane(HtmlHelpPane):
    # TaskPane interface ###################################################

    id = 'maproom.rst_markup_help_pane'
    name = 'RST Help'
    help_text = """#*****[RET]Title[RET]*****|<h1>Title</h1>
    Section[RET]=======|<h2>Section</h2>
    Subsection[RET]----------|<h3>Subsection</h3>Separate paragraphs or new lists by blank lines.
    *italic*|<i>italic</i>
    **bold**|<b>bold</b>
    * list item[RET]* list item|<ul><li>list item</li><li>list item</li></ul>
    """


class MarkdownHelpPane(HtmlHelpPane):
    # TaskPane interface ###################################################

    id = 'maproom.markdown_help_pane'
    name = 'Markdown Help'
    help_text = """# Title|<h1>Title</h1>
    ## Section|<h2>Section</h2>
    ### Subsection|<h3>Subsection</h3>
    Separate paragraphs or new lists by blank lines.
    *italic*|<i>italic</i>
    **bold**|<b>bold</b>
    * list item[RET]* list item|<ul><li>list item</li><li>list item</li></ul>
    """
