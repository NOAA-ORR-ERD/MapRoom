import os
import cgi
import time
import datetime
import calendar
import bisect
import math

import numpy as np

import wx
from wx.lib.ClickableHtmlWindow import PyClickableHtmlWindow

# Enthought library imports.

from maproom.app_framework import preferences
from maproom.app_framework.ui.popuputil import SpringTabs
from maproom.app_framework.ui.download_manager import DownloadControl
from maproom.app_framework.ui.zoomruler import ZoomRuler
from maproom.app_framework.ui import compactgrid as cg
from maproom.app_framework.utils.textutil import pretty_seconds, parse_pretty_seconds

from .layer_tree_control import LayerTreeControl
from .ui.info_panels import LayerInfoPanel, SelectionInfoPanel
from .ui.triangle_panel import TrianglePanel
from .ui.merge_panel import MergePointsPanel
from .ui.undo_panel import UndoHistoryPanel

import logging
log = logging.getLogger(__name__)


class TimelinePanel(ZoomRuler):
    def __init__(self, parent, editor, **kwargs):
        ZoomRuler.__init__(self, parent, **kwargs)
        self.editor = editor
        self.current_time = None
 
    def init_playback(self):
        ZoomRuler.init_playback(self)
        self.step_rate = 0  # set as flag to indicate it needs to be updated

    def get_timeline_info(self):
        layers, start, end = self.editor.layer_manager.get_timestamped_layers(self.editor.layer_visibility, False)
        if start == 0.0:
            today = datetime.datetime.today()
            noon = today.replace(hour=12, minute=0, second=0, tzinfo=None)
            start = calendar.timegm(noon.timetuple())
        if end < start:
            end = start

        # Minimum of one hour
        padding = divmod((end - start) // 10, 60)[0] * 60
        if padding < 3600:
            padding = 3600

        t = datetime.datetime.utcfromtimestamp(start)
        if t.minute < 50:
            t = t.replace(minute=0, second=0, tzinfo=None)
        else:
            t = t.replace(hour=t.hour - 1, minute=0, second=0, tzinfo=None)
        t -= datetime.timedelta(seconds=padding)
        start = calendar.timegm(t.timetuple())

        t = datetime.datetime.utcfromtimestamp(end)
        if t.minute > 10:
            t = t.replace(hour=t.hour + 1, minute=0, second=0, tzinfo=None)
        else:
            t = t.replace(minute=0, second=0, tzinfo=None)
        t += datetime.timedelta(seconds=padding)
        end = calendar.timegm(t.timetuple())
        # print "DATES PROCESSED:", start, end
        info = {
            "format": "month",
            "earliest_time": start,
            "latest_time": end,
            "marks": [(l.start_time, l.end_time, l) for l in layers],
        }
        # print info
        return info

    def get_selected_range(self):
        r = self.ruler.selected_ranges
        return r

    def selection_started_callback(self, selected_ranges):
        if self.editor is None: return
        log.debug("selection started: %s" % str(selected_ranges))
        self.current_marks_in_selection = set()
        self.editor.refresh()

    def selection_extended_callback(self, selected_ranges, marks_in_selection):
        if self.editor is None: return
        log.debug("selection extended: %s" % str(selected_ranges))
        if marks_in_selection != self.current_marks_in_selection:
            self.editor.refresh()
            self.current_marks_in_selection = marks_in_selection
        else:
            log.debug("no marks changed; no refresh needed")

    def selection_finished_callback(self, selected_ranges):
        if self.editor is None: return
        log.debug("selection finished: %s" % str(selected_ranges))
        self.editor.refresh()

    def item_activation_callback(self, item):
        if self.editor is None: return
        layers = [item]
        self.editor.set_layer_visibility(layers)
        self.editor.layer_tree_control.Refresh()

    def over_item_callback(self, pos, item):
        if self.editor is None: return
        self.editor.frame.status_message(str(item))

    def not_over_item_callback(self, pos):
        if self.editor is None: return
        self.editor.frame.status_message("")

    def selected_item_callback(self, item):
        if self.editor is None: return
        self.editor.layer_tree_control.set_edit_layer(item)

    def selection_cleared_callback(self):
        if self.editor is None: return
        log.debug("selection cleared")
        self.current_time = None
        self.editor.refresh()

    def playback_start_callback(self):
        if self.editor is None: return
        self.GetParent().play.SetLabel("Pause")
        self.editor.start_movie_recording()

    def playback_pause_callback(self):
        if self.editor is None: return
        self.GetParent().play.SetLabel("Play")
        self.current_time = None
        self.editor.stop_movie_recording()

    def playback_callback(self, current_time):
        if self.editor is None: return
        log.debug("playback for time: %s" % time.strftime("%b %d %Y %H:%M", time.gmtime(current_time)))
        self.current_time = current_time
        self.editor.refresh()
        self.editor.add_frame_to_movie()


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


step_values = ['10m', '20m', '30m', '40m', '45m', '60m', '90m', '120m', '3hr', '4hr', '5hr', '6hr', '8hr', '10hr', '12hr', '16h', '24hr', '36hr', '48hr', '3d', '4d', '5d', '6d', '7d', '2wk', '3wk', '4wk']
step_values_as_seconds = [parse_pretty_seconds(a) for a in step_values]
rate_values = ['100ms', '200ms', '500ms', '1s', '2s', '3s', '4s', '5s', '10s', '15s', '20s']
rate_values_as_seconds = [parse_pretty_seconds(a) for a in rate_values]

class TimeStepPanelMixin(object):
    def setup_panel(self, step_value, step_rate):
        panel = wx.Panel(self)
        vsizer = wx.BoxSizer(wx.VERTICAL)

        self.add_header(panel, vsizer)

        sizer = wx.BoxSizer(wx.HORIZONTAL)

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

        cb = wx.ComboBox(panel, 500, choices=rate_values, style=wx.CB_DROPDOWN)
        try:
            i = rate_values_as_seconds.index(step_rate)
            cb.SetSelection(i)
        except ValueError:
            cb.ChangeValue(pretty_seconds(step_rate))
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
        log.debug("sending %s to %s" % (e, self.GetEventHandler()))
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
    def __init__(self, parent, editor, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        self.SetName("Timeline")

        sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.play = wx.Button(self, -1, "Play")
        self.play.Bind(wx.EVT_BUTTON, self.on_play)
        sizer.Add(self.play, 0, wx.EXPAND)

        self.steps = wx.Button(self, -1, "1m / 200ms", style=wx.BU_EXACTFIT)
        self.steps.Bind(wx.EVT_BUTTON, self.on_steps)
        sizer.Add(self.steps, 0, wx.EXPAND)

        self.timeline = TimelinePanel(self, editor)
        sizer.Add(self.timeline, 1, wx.EXPAND|wx.LEFT, 5)

        self.SetSizer(sizer)

        self.Bind(EVT_TIME_STEP_MODIFIED, self.on_set_steps)

    @property
    def current_time(self):
        return self.timeline.current_time

    @property
    def step_rate(self):
        return self.timeline.step_rate

    @property
    def selected_time_range(self):
        r = self.timeline.ruler.selected_ranges
        try:
            begin, end = r[0]
        except IndexError:
            begin = end = None
        return begin, end

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
        self.timeline.step_rate = 0  # Force recalc

    def recalc_view(self):
        log.debug("timeline recalc_view")
        self.timeline.rebuild(self.timeline)
        log.debug("step rate %d num %d" % (self.timeline.step_rate, self.timeline.num_marks))
        if self.timeline.step_rate == 0 and self.timeline.num_marks > 1:
            if self.timeline.num_marks > 1:
                interval = (self.timeline.highest_marker_value - self.timeline.lowest_marker_value) / (self.timeline.num_marks - 1)
                self.timeline.step_rate = .2
                log.debug(str((step_values_as_seconds, interval, interval/2, self.timeline._length)))
                self.timeline.step_value = step_values_as_seconds[bisect.bisect_left(step_values_as_seconds, interval)]
            if self.timeline.step_rate == 0:
                self.timeline.step_rate = 1
                self.timeline.step_value = 600
            log.debug(f"new timestep from interval {interval}: {self.timeline.step_value} / {self.timeline.step_rate}")
        self.update_ui()

    def refresh_view(self):
        log.debug("timeline refresh_view")
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

    def start_playback(self):
        self.timeline.start_playback()

    def pause_playback(self):
        self.timeline.pause_playback()

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
        log.debug("step=%s rate=%s" % (step, rate))
        if step is not None:
            self.timeline.step_value = step
        if rate is not None:
            self.timeline.step_rate = rate
        self.update_ui()


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
    def __init__(self, parent, editor, **kwargs):
        self.editor = editor
        self.point_indexes = []
        self.notification_count = 0
        wx.ListBox.__init__(self, parent, wx.ID_ANY, name="Flagged Points", **kwargs)
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
        editor = self.editor
        layer = editor.current_layer
        editor.layer_canvas.do_center_on_point_index(layer, point_index)

    def set_flagged(self, point_indexes):
        self.point_indexes = list(point_indexes)
        self.Set([str(i) for i in point_indexes])
        self.notification_count = len(point_indexes)

    def recalc_view(self):
        editor = self.editor
        layer = editor.current_layer
        try:
            points = layer.get_flagged_point_indexes()
        except AttributeError:
            points = []
        log.debug(f"flagged in {layer}: {str(points)}")
        self.set_flagged(points)

    def refresh_view(self):
        self.Refresh()


class DownloadPanel(DownloadControl):
    def __init__(self, parent, editor, **kwargs):
        self.editor = editor
        downloader = wx.GetApp().get_downloader()
        DownloadControl.__init__(self, parent, downloader, size=(400, -1), name="Downloads", **kwargs)

    # turn the superclass attribute path into a property so we can override it
    # and pull out the paths from the preferences
    @property
    def path(self):
        path = self.editor.preferences.download_directory
        log.debug("download path: %s" % path)
        return path

    @path.setter
    def path(self, value):
        if value:
            self.editor.preferences.download_directory = value

    def refresh_view(self):
        self.Refresh()

    def activateSpringTab(self):
        self.refresh_view()

    def get_notification_count(self):
        self.refresh_view()
        return self.num_active


class HtmlHelpPane(PyClickableHtmlWindow):
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

    def __init__(self, parent):
        PyClickableHtmlWindow.__init__(self, parent, -1, style=wx.NO_FULL_REPAINT_ON_RESIZE, size=(400, 300))
        self.SetPage(self.get_help_text())

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


# Layer debugging info panels

class LayerInfoTable(cg.VirtualTable):
    column_labels = []
    column_sizes = []

    def __init__(self, layer):
        self.layer = layer
        self.current_num_rows = 0
        cg.VirtualTable.__init__(self, len(self.column_labels), 0)

    def calc_last_valid_index(self):
        size = self.current_num_rows * self.items_per_row
        self.style = np.zeros(size, dtype=np.uint8)
        return size

    def get_label_at_index(self, index):
        return (index // self.items_per_row)

    def get_row_label_text(self, start_line, num_lines, step=1):
        last_line = min(start_line + num_lines, self.num_rows)
        for line in range(start_line, last_line, step):
            yield str(int(line))

    def calc_row_label_width(self, view_params):
        return view_params.calc_text_width("8" * len(str(int(self.num_rows))))

    def calc_cell_value(self, row, col, index):
        raise NotImplementedError

    def get_value_style(self, row, col):
        index, _ = self.get_index_range(row, col)
        try:
            text = str(self.calc_cell_value(row, col, index))
        except IndexError:
            text = f"{row}"
        try:
            s = self.style[index]
        except IndexError:
            s = 0
        return text, s

    def rebuild(self, layer):
        self.layer = layer
        self.init_boundaries()


class LayerInfoList(cg.CompactGrid):
    ui_name = "Layer Info"

    def __init__(self, parent, editor, **kwargs):
        self.editor = editor
        table = self.calc_table(None)
        cg.CompactGrid.__init__(self, table, editor.preferences, None, None, parent, size=(-1,800), name=self.ui_name)
        # The layer info list must change whenever a layer is selected, so the event on
        # the layer tree control is bound so this UI can be updated.
        editor.layer_tree_control.current_layer_changed_event += self.on_current_layer_changed

    def calc_table(self, layer):
        raise NotImplementedError

    def calc_line_renderer(self):
        return cg.VirtualTableLineRenderer(self, 1, widths=self.table.column_sizes, col_labels=self.table.column_labels)

    def on_current_layer_changed(self, evt):
        layer = evt[0]
        log.debug(f"on_current_layer_changed: {layer}")
        self.recalc_view()

    def send_caret_event(self, flags):
        self.process_item(*self.caret_handler.current.rc)

    def process_item(self, row, col):
        pass

    def recalc_view(self):
        log.debug(f"recalc_view: {self}")
        self.table.rebuild(self.editor.current_layer)
        cg.CompactGrid.recalc_view(self)

    def refresh_view(self):
        if self.IsShown():
            log.debug("refreshing %s" % self)
            if self.table.layer != self.editor.current_layer:
                self.recalc_view()
            else:
                cg.CompactGrid.refresh_view(self)
        else:
            log.debug("skipping refresh of hidden %s" % self)


class PointsTable(LayerInfoTable):
    column_labels = ["Longitude", "Latitude"]
    column_sizes = [16, 16]

    def calc_num_rows(self):
        try:
            num = len(self.layer.points)
        except AttributeError:
            num = 0
        return num

    def calc_cell_value(self, row, col, index):
        if col == 0:
            text = self.layer.points.x[row]
        else:
            text = self.layer.points.y[row]
        return text


class PointsList(LayerInfoList):
    ui_name = "Points"

    def calc_table(self, layer):
        return PointsTable(layer)

    def process_item(self, row, col):
        self.editor.layer_canvas.do_center_on_point_index(self.table.layer, row)


class PolygonTable(LayerInfoTable):
    column_labels = ["^Name", "^Start", "^Count", "^Code", "^Feature Name"]
    column_sizes = [12, 5, 5, 4, 12]

    def calc_num_rows(self):
        try:
            num = len(self.layer.geometry_list)
        except AttributeError:
            num = 0
        return num

    def calc_cell_value(self, row, col, index):
        geom_info = self.layer.geometry_list[row]
        if col == 0:
            val = geom_info.name
        elif col == 1:
            val = geom_info.start_index
        elif col == 2:
            val = geom_info.count
        elif col == 3:
            val = geom_info.feature_code
        else:
            val = geom_info.feature_name
        return val


class PolygonList(LayerInfoList):
    ui_name = "Polygons"

    def calc_table(self, layer):
        return PolygonTable(None)

    def process_item(self, row, col):
        try:
            geom_info = self.table.layer.geometry_list[row]
        except AttributeError:
            pass
        except IndexError:
            pass
        else:
            self.editor.layer_canvas.do_center_on_points(self.table.layer, geom_info.start_index, geom_info.count)
