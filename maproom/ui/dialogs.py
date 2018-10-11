import time
import types

import wx
import wx.lib.sized_controls as sc
import wx.lib.buttons as buttons

from omnivore_framework.utils.wx.dialogs import ObjectEditDialog

from ..library import coordinates
from ..library.textparse import parse_int_string
from ..library.marplot_icons import get_wx_bitmap, marplot_icons, marplot_icon_max_size, marplot_icon_id_to_category, marplot_icon_id_to_name
from ..mock import MockProject
from ..library.thread_utils import BackgroundWMSDownloader
from ..library.tile_utils import BackgroundTileDownloader
from ..library.host_utils import WMSHost, OpenTileHost
from ..library.simplify import VWSimplifier
from .. import menu_commands as mec

from .sliders import FloatSlider

import logging
log = logging.getLogger(__name__)


class FindPointDialog(sc.SizedDialog):

    def __init__(self, project):
        sc.SizedDialog.__init__(self, wx.GetTopLevelParent(project.control), wx.ID_ANY, "Find Points")

        panel = self.GetContentsPane()
        wx.StaticText(panel, -1, "Find point number, range of points\nor multiple ranges.\n\nSeparate ranges by commas, e.g: 1-4,8-10")

        self.text = wx.TextCtrl(panel, -1, "", name="FindPointDialog.text")
        self.text.SetSizerProps(expand=True)

        btn_sizer = self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL)
        self.Sizer.Add(btn_sizer, 0, 0, wx.EXPAND | wx.BOTTOM | wx.RIGHT, self.GetDialogBorder())

        self.layer = project.layer_tree_control.get_edit_layer()

        # Note: indexes are stored in zero-based array but need to be displayed
        # to the user as one-based
        indexes = self.layer.get_selected_point_indexes()
        if len(indexes):
            self.text.Value = str(indexes[-1] + 1)
        elif self.layer.points is not None:
            self.text.Value = str(len(self.layer.points) - 1)
        else:
            self.text.Value = ""
        self.text.SelectAll()

        self.Bind(wx.EVT_CLOSE, self.OnClose)

        self.Fit()

    def ShowModalWithFocus(self):
        self.text.SetFocus()
        return self.ShowModal()

    def OnClose(self, event):
        self.EndModal(wx.ID_CANCEL)

    def get_values(self):
        """Return indexes of points.

        Note that point indexes are stored internally numbered from zero,
        but the user expects indexes starting from 1.  Returned values are
        zero-based.
        """
        one_based_values, error = parse_int_string(self.text.Value)
        values = [x - 1 for x in one_based_values]
        return values, error


class JumpCoordsDialog(sc.SizedDialog):

    def __init__(self, layer_canvas, display_format):
        sc.SizedDialog.__init__(self, wx.GetTopLevelParent(layer_canvas), wx.ID_ANY, "Jump to Coordinates")

        panel = self.GetContentsPane()
        wx.StaticText(panel, -1, "Please enter the coordinates to jump to")

        self.coords_text = wx.TextCtrl(panel, -1, "", name="JumpCoordsDialog.coords_text")
        self.coords_text.SetSizerProps(expand=True)
        self.coords_text.Bind(wx.EVT_TEXT, self.OnText)

        btn_sizer = self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL)
        self.ok_btn = self.FindWindowById(wx.ID_OK)
        self.Sizer.Add(btn_sizer, 0, 0, wx.EXPAND | wx.BOTTOM | wx.RIGHT, self.GetDialogBorder())

        self.Bind(wx.EVT_CLOSE, self.OnClose)

        self.lat_lon = layer_canvas.get_world_point_from_projected_point(layer_canvas.projected_point_center)
        self.coords_text.Value = coordinates.format_coords_for_display(self.lat_lon[0], self.lat_lon[1], display_format)

        self.Fit()

    def ShowModalWithFocus(self):
        self.coords_text.SetFocus()
        return self.ShowModal()

    def OnClose(self, event):
        self.EndModal(wx.ID_CANCEL)

    def OnText(self, event):
        lat_lon_string = event.String

        try:
            self.lat_lon = coordinates.lat_lon_from_format_string(lat_lon_string)
            self.coords_text.SetBackgroundColour("#FFFFFF")
            valid = True
        except:
            self.lat_lon = None
            self.coords_text.SetBackgroundColour("#FF8080")
            valid = False
        self.ok_btn.Enable(valid)


class IconDialog(wx.Dialog):
    def __init__(self, parent, iid):
        wx.Dialog.__init__(self, parent, -1, "Choose MARPLOT Icon")
        self.num_cols = 5

        self.icon_list = wx.ScrolledWindow(self, wx.ID_ANY, style=wx.VSCROLL, name="IconDialog.icon_list")
        self.grid = wx.FlexGridSizer(cols=self.num_cols, hgap=2, vgap=2)

        self.Bind(wx.EVT_BUTTON, self.OnButton)
        self.Bind(wx.EVT_ENTER_WINDOW, self.on_enter)
        self.Bind(wx.EVT_LEAVE_WINDOW, self.on_leave)
        self.icon_list.SetScrollbars(0, marplot_icon_max_size[1], 0, (50 + self.num_cols - 1) // self.num_cols)

        icon_cats = [cat for cat, icons in marplot_icons]
        cat = wx.ListBox(self, -1, choices=icon_cats, name="IconDialog.cat")
        cat.Bind(wx.EVT_LISTBOX, self.on_category)
        cat_id = icon_cats.index(marplot_icon_id_to_category[iid])
        cat.SetSelection(cat_id)
        self.repopulate_grid(cat_id)

        self.icon_list.SetSizer(self.grid)
        self.icon_list.Layout()
        self.icon_list.Fit()

        self.name = wx.StaticText(self, -1, "", size=(marplot_icon_max_size[0] * (self.num_cols + 1), -1))

        vsiz = wx.BoxSizer(wx.VERTICAL)
        vsiz.Add(self.icon_list, 1, wx.EXPAND, 0)
        vsiz.Add(self.name, 0, wx.EXPAND, 0)

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(cat, 1, wx.EXPAND, 0)
        sizer.Add(vsiz, 0, wx.EXPAND, 0)
        self.SetSizer(sizer)
        self.Fit()

    def repopulate_grid(self, cat_id):
        self.grid.Clear(True)
        cat_name, icons = marplot_icons[cat_id]
        self.id_map = {}
        parent = self.icon_list
        for name, iid in icons:
            bmp = get_wx_bitmap(iid)
            wid = wx.NewId()
            self.id_map[wid] = iid
            b = buttons.GenBitmapButton(parent, wid, bmp, size=marplot_icon_max_size, style=wx.BORDER_NONE)
            b.SetBackgroundColour("black")
            self.grid.Add(b, flag=wx.ALIGN_CENTER_VERTICAL)
            b.Bind(wx.EVT_ENTER_WINDOW, self.on_enter)
            b.Bind(wx.EVT_LEAVE_WINDOW, self.on_leave)
        self.icon_list.SetScrollbars(0, marplot_icon_max_size[1], 0, (len(icons) + self.num_cols - 1) // self.num_cols)
        self.icon_list.Layout()

    def OnButton(self, event):
        iid = self.id_map[event.GetId()]
        self.EndModal(iid)

    def on_category(self, event):
        self.repopulate_grid(event.GetSelection())

    def on_enter(self, event):
        wid = event.GetId()
        if wid in self.id_map:
            iid = self.id_map[wid]
            name = marplot_icon_id_to_name[iid]
            self.name.SetLabel(name)

    def on_leave(self, event):
        self.name.SetLabel("")


class StyleDialog(wx.Dialog):
    displayed_style_types = ["Line style", "Line width", "Line color", "Start marker", "End marker", "Line transparency", "Fill style", "Fill color", "Fill transparency", "Text color", "Font", "Font size", "Text transparency", "Outline color", "Outline transparency", "Marplot icon"]

    def __init__(self, project, layers):
        wx.Dialog.__init__(self, project.control, -1, "Set Default Style", size=(300, -1))
        self.lm = project.layer_manager

        self.mock_project = MockProject(add_tree_control=True, default_styles=self.lm.default_styles)
        self.mock_project.control = None
        self.other = self.mock_project.layer_tree_control.get_edit_layer()
        self.other.type = "other"
        self.other.name = "other"
        self.other.layer_info_panel = self.displayed_style_types

        hbox = wx.BoxSizer(wx.HORIZONTAL)

        def set_style_override(self, style):
            self.style.copy_from(style)

        self.styleable_layers = list(layers)
        self.styleable_layers.append(self.other)
        for v in self.styleable_layers:
            self.mock_project.layer_manager.insert_layer([2], v)
            v.manager = self.lm
            v.style = self.lm.get_default_style_for(v)
            v.set_style = types.MethodType(set_style_override, v)

            # Restrict subset of styles displayed in dialog to the set shown
            # above in displayed_style_types
            valid_style_types = []
            for name in v.layer_info_panel:
                if name in self.displayed_style_types:
                    valid_style_types.append(name)
            v.layer_info_panel = valid_style_types
        self.obj_list = wx.ListBox(self, -1, choices=[v.name for v in self.styleable_layers])
        self.obj_list.Bind(wx.EVT_LISTBOX, self.on_category)
        hbox.Add(self.obj_list, 1, wx.EXPAND, 0)

        # Can't import from the top level because info_panels imports this
        # file, creating a circular import loop
        from .info_panels import LayerInfoPanel
        self.info = LayerInfoPanel(self, self.mock_project)
        self.info.display_panel_for_layer(self.mock_project, self.other)

        # Use the "other" layer so it has the most style items to force the
        # minimum client area to be big enough so there's no scrollbar
        vsiz = (400, self.info.GetBestVirtualSize()[1] + 50)
        self.info.SetMinSize(vsiz)
        self.info.Layout()
        hbox.Add(self.info, 4, wx.EXPAND, 0)

        # reset to first item in list
        self.obj_list.SetSelection(0)

        self.savebtn = wx.CheckBox(self, -1, "Save these styles as the default for future projects")

        self.applybtn = wx.CheckBox(self, -1, "Apply these styles to existing layers")

        btnsizer = wx.StdDialogButtonSizer()
        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()
        btnsizer.AddButton(btn)
        btn = wx.Button(self, wx.ID_CANCEL)
        btnsizer.AddButton(btn)
        btnsizer.Realize()

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(hbox, 1, wx.EXPAND, 0)
        sizer.Add(self.savebtn, 0, wx.EXPAND | wx.ALL, 10)
        sizer.Add(self.applybtn, 0, wx.EXPAND | wx.ALL, 10)
        sizer.Add(btnsizer, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.SetSizer(sizer)
        self.Fit()

    def use_layer(self, index):
        layer = self.styleable_layers[index]
        self.mock_project.layer_tree_control.layer = layer
        return layer

    def get_styles(self):
        d = {v.type: v.style.get_copy() for v in self.styleable_layers}
        return d

    def on_category(self, evt):
        index = evt.GetSelection()
        layer = self.use_layer(index)
        self.info.display_panel_for_layer(self.mock_project, layer)

    @property
    def save_for_future(self):
        return self.savebtn.IsChecked()

    @property
    def apply_to_current(self):
        return self.applybtn.IsChecked()


class WMSDialog(ObjectEditDialog):
    border = 5

    def __init__(self, parent, title, default=None):
        fields = [
            ('verify', 'url', 'URL: '),
            ('gauge', 'gauge', None),
            ('expando', 'status', ""),
            ('text', 'name', 'Server Name: '),
        ]
        self.verified_host = False
        ObjectEditDialog.__init__(self, parent, title, "Enter WMS Server Information:", fields, WMSHost, default)

        # Assume the host has been verified if there is a default URL
        control = self.controls['url']
        url = control.GetValue()
        self.verified_host = bool(url)
        self.check_enable()

    def on_verify(self, evt):
        control = self.controls['url']
        url = control.GetValue()
        self.check_server(url)

    def check_server(self, url):
        verify = self.buttons['url']
        verify.Enable(False)
        status = self.controls['status']
        status.SetValue("Checking server...\n")
        gauge = self.controls['gauge']
        for version in ['1.3.0', '1.1.1']:
            host = WMSHost("test", url, version)
            downloader = BackgroundWMSDownloader(host)
            server = downloader.server
            while True:
                if server.is_finished:
                    break
                time.sleep(.05)
                gauge.Pulse()
                wx.Yield()
            if server.is_valid():
                break

        gauge.SetValue(0)
        wx.Yield()
        name = self.controls['name']
        if server.is_valid():
            host.name = server.wms.identification.title
            status.AppendText("Found WMS server: %s\n" % host.name)
            name.SetValue(host.name)
            self.verified_host = host.version
        else:
            status.AppendText("Failed: %s\n" % server.error)
            name.SetValue("")
            self.verified_host = None
        verify.Enable(True)
        self.check_enable()

    def can_submit(self):
        return bool(self.verified_host)


def prompt_for_wms(parent, title, default=None):
    d = WMSDialog(parent, title, default)
    return d.show_and_get_value()


class TileServerDialog(ObjectEditDialog):
    border = 5

    def __init__(self, parent, title, default=None):
        fields = [
            ('verify list', 'urls', 'URLs: '),
            ('dropdown', 'url_format', 'URL Format', ['z/x/y', 'z/y/x']),
            ('text', 'suffix', 'Image file extension (usually automatically determined)'),
            ('gauge', 'gauge', None),
            ('expando', 'status', ""),
            ('text', 'name', 'Server Name: '),
        ]
        self.verified_urls = False
        ObjectEditDialog.__init__(self, parent, title, "Enter Tile Server Information:", fields, OpenTileHost, default)

        # Assume the host has been verified if there is a default URL
        control = self.controls['urls']
        urls = control.GetValue()
        self.verified_urls = urls
        self.check_enable()

    def on_verify(self, evt):
        control = self.controls['urls']
        urls = control.GetValue()
        self.check_server(urls)

    def check_server(self, urls):
        verify = self.buttons['urls']
        verify.Enable(False)
        status = self.controls['status']
        status.SetValue("Checking all round-robin URLs...\n")
        gauge = self.controls['gauge']
        success = False
        verified_urls = []

        def lookup(url, suffix, reverse):
            status.AppendText("Trying: %s..." % url)
            if suffix:
                status.AppendText(" (%s) " % suffix)
            host = OpenTileHost("test", [url], suffix=suffix, reverse_coords=reverse)
            downloader = BackgroundTileDownloader(host, None)
            req = downloader.request_tile(4, 3, 2)
            retval = False
            while True:
                found = False
                for r in downloader.get_finished():
                    found = found or r == req
                if found:
                    break
                time.sleep(.05)
                gauge.Pulse()
                wx.Yield()
            if req.is_finished and not req.error:
                status.AppendText("OK!\n")
                retval = True
                verified_urls.append(url)
            else:
                status.AppendText("Failed! (%s)\n" % req.error)
            return retval

        found_extension = False
        reverse_coords = False
        temp = self.get_new_object()
        self.get_edited_value_of(temp, "suffix")
        suffix = temp.suffix
        exts = list(OpenTileHost.known_suffixes)
        if suffix not in exts:
            exts.append(suffix)
        for url in urls.splitlines():
            if not url:
                continue
            if found_extension:
                success = lookup(url, suffix, reverse_coords)
                if not success:
                    break
            else:
                for ext in exts:
                    success = lookup(url, ext, reverse_coords)
                    if success:
                        found_extension = True
                        suffix = ext
                        break

        gauge.SetValue(0)
        wx.Yield()
        if success:
            self.verified_urls = verified_urls
            temp = self.get_new_object()
            temp.suffix = suffix
            self.set_initial_value_of(temp, "suffix")
        else:
            self.verified_urls = None
        verify.Enable(True)
        self.check_enable()

    def can_submit(self):
        name = self.controls['name']
        return bool(self.verified_urls) and bool(name.GetValue())


def prompt_for_tile(parent, title, default=None):
    d = TileServerDialog(parent, title, default)
    return d.show_and_get_value()


class SimplifyDialog(sc.SizedDialog):

    def __init__(self, project, layer, obj_type, obj_index, initial_ratio=1.0):
        sc.SizedDialog.__init__(self, project.control, wx.ID_ANY, "Simplify Polygon")

        self.project = project
        self.layer = layer
        self.obj_index = obj_index
        self.obj_type = obj_type

        panel = self.GetContentsPane()
        wx.StaticText(panel, -1, "Simplification Ratio\n\nFewest segments                                                                          Most segments")

        self.slider = FloatSlider(panel, -1, initial_ratio, 0.001, 1.0)
        self.slider.SetSizerProps(expand=True)
        self.slider.Bind(wx.EVT_TEXT, self.OnText)

        btn_sizer = self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL)
        self.ok_btn = self.FindWindowById(wx.ID_OK)
        self.Sizer.Add(btn_sizer, 0, 0, wx.EXPAND | wx.BOTTOM | wx.RIGHT, self.GetDialogBorder())

        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.Bind(wx.EVT_SLIDER, self.on_slider)

        self.Fit()

        points = layer.get_ring_points(self.obj_index).copy()
        self.simplifier = VWSimplifier(points)
        self.have_processed = False

    def on_slider(self, evt):
        cmd = mec.SimplifyPolygonCommand(self.layer, self.obj_type, self.obj_index, self.simplifier, self.slider.Value)
        self.project.process_command(cmd)
        self.have_processed = True

    def ShowModalWithFocus(self):
        self.coords_text.SetFocus()
        return self.ShowModal()

    def OnClose(self, event):
        self.EndModal(wx.ID_CANCEL)
        self.roll_back()

    def roll_back(self):
        if self.have_processed:
            self.project.undo()

    def OnText(self, event):
        lat_lon_string = event.String

        try:
            self.lat_lon = coordinates.lat_lon_from_format_string(lat_lon_string)
            self.coords_text.SetBackgroundColour("#FFFFFF")
            valid = True
        except:
            self.lat_lon = None
            self.coords_text.SetBackgroundColour("#FF8080")
            valid = False
        self.ok_btn.Enable(valid)
