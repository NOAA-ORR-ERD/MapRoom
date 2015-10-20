import time

import wx
import wx.lib.sized_controls as sc
import wx.lib.buttons as buttons
from wx.lib.expando import ExpandoTextCtrl, EVT_ETC_LAYOUT_NEEDED

from ..library import coordinates
from ..library.textparse import parse_int_string
from ..library.marplot_icons import *
from ..mock import MockProject
from ..library.thread_utils import BackgroundWMSDownloader, WMSHost


class FindPointDialog(sc.SizedDialog):

    def __init__(self, project):
        sc.SizedDialog.__init__(self, wx.GetTopLevelParent(project.control), wx.ID_ANY, "Find Points")

        panel = self.GetContentsPane()
        wx.StaticText(panel, -1, "Find point number, range of points\nor multiple ranges.\n\nSeparate ranges by commas, e.g: 1-4,8-10")

        self.text = wx.TextCtrl(panel, -1, "")
        self.text.SetSizerProps(expand=True)

        btn_sizer = self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL)
        self.Sizer.Add(btn_sizer, 0, 0, wx.EXPAND | wx.BOTTOM | wx.RIGHT, self.GetDialogBorder())

        self.layer = project.layer_tree_control.get_selected_layer()
        
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

        self.coords_text = wx.TextCtrl(panel, -1, "")
        self.coords_text.SetSizerProps(expand=True)
        self.coords_text.Bind(wx.EVT_TEXT, self.OnText)

        center_lat_lon = layer_canvas.get_world_point_from_projected_point(layer_canvas.projected_point_center)
        self.coords_text.Value = coordinates.format_coords_for_display(center_lat_lon[0], center_lat_lon[1], display_format)

        btn_sizer = self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL)
        self.Sizer.Add(btn_sizer, 0, 0, wx.EXPAND | wx.BOTTOM | wx.RIGHT, self.GetDialogBorder())

        self.Bind(wx.EVT_CLOSE, self.OnClose)

        self.Fit()
    
    def ShowModalWithFocus(self):
        self.coords_text.SetFocus()
        return self.ShowModal()

    def OnClose(self, event):
        self.EndModal(wx.ID_CANCEL)

    def OnText(self, event):
        lat_lon_string = event.String

        try:
            lat_lon = coordinates.lat_lon_from_format_string(lat_lon_string)
            if lat_lon == (-1, -1):
                self.coords_text.SetBackgroundColour("#FF8080")
            else:
                self.coords_text.SetBackgroundColour("#FFFFFF")
        except:
            self.coords_text.SetBackgroundColour("#FF8080")

class IconDialog(wx.Dialog):
    def __init__(self, parent, iid):
        wx.Dialog.__init__(self, parent, -1, "Choose MARPLOT Icon")
        self.num_cols = 5

        self.icon_list = wx.ScrolledWindow(self, wx.ID_ANY, style=wx.VSCROLL)
        self.grid = wx.FlexGridSizer(cols=self.num_cols, hgap=2, vgap=2)
        self.icon_id_to_name = [name for cat, icons in marplot_icons for name, index in icons]
        self.icon_id_to_cat = [cat for cat, icons in marplot_icons for name, index in icons]

        self.Bind(wx.EVT_BUTTON, self.OnButton)
        self.Bind(wx.EVT_ENTER_WINDOW, self.on_enter)
        self.Bind(wx.EVT_LEAVE_WINDOW, self.on_leave)
        self.icon_list.SetScrollbars(0, marplot_icon_max_size[1], 0, (50+self.num_cols-1)/self.num_cols)

        icon_cats = [cat for cat, icons in marplot_icons]
        cat = wx.ListBox(self, -1, choices=icon_cats)
        cat.Bind(wx.EVT_LISTBOX, self.on_category)
        cat_id = icon_cats.index(self.icon_id_to_cat[iid])
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
        self.icon_list.SetScrollbars(0, marplot_icon_max_size[1], 0, (len(icons)+self.num_cols-1)/self.num_cols)
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
            name = self.icon_id_to_name[iid]
            self.name.SetLabel(name)
    
    def on_leave(self, event):
        self.name.SetLabel("")


class StyleDialog(wx.Dialog):
    def __init__(self, project):
        wx.Dialog.__init__(self, project.control, -1, "Set Default Style", size=(300,-1))
        self.lm = project.layer_manager
        
        self.mock_project = MockProject()
        self.mock_project.control = None
        self.layer = self.mock_project.layer_tree_control.get_selected_layer()
        self.layer.style.copy_from(self.lm.default_style)
        self.layer.layer_info_panel = ["Line style", "Line width", "Line color", "Start marker", "End marker", "Line transparency", "Fill style", "Fill color", "Fill transparency", "Text color", "Font", "Font size", "Text transparency", "Marplot icon"]
        
        # Can't import from the top level because info_panels imports this
        # file, creating a circular import loop
        from info_panels import LayerInfoPanel
        self.info = LayerInfoPanel(self, self.mock_project)
        self.info.display_panel_for_layer(self.mock_project, self.layer)
        
        # Force the minimum client area to be big enough so there's no scrollbar
        vsiz = (400, self.info.GetBestVirtualSize()[1]+50)
        self.info.SetMinSize(vsiz)
        self.info.Layout()
        
        btnsizer = wx.StdDialogButtonSizer()
        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()
        btnsizer.AddButton(btn)
        btn = wx.Button(self, wx.ID_CANCEL)
        btnsizer.AddButton(btn)
        btnsizer.Realize()

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.info, 1, wx.EXPAND, 0)
        sizer.Add(btnsizer, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)
        self.SetSizer(sizer)
        self.Fit()

    def get_style(self):
        return self.layer.style


class AddWMSDialog(wx.Dialog):
    border = 5
    
    def __init__(self, parent):
        wx.Dialog.__init__(self, parent, -1, "Add New WMS Server", size=(500, -1), style=wx.DEFAULT_DIALOG_STYLE|wx.RESIZE_BORDER)
        
        self.verified_host = None
        
        sizer = wx.BoxSizer(wx.VERTICAL)

        text = wx.StaticText(self, -1, "Enter server URL:")
        sizer.Add(text, 0, wx.ALL|wx.EXPAND|wx.ALIGN_CENTER, self.border)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.url = wx.TextCtrl(self, -1, "", size=(500,-1))
        hbox.Add(self.url, 0, wx.ALL|wx.EXPAND, self.border)
        self.id_verify = wx.NewId()
        self.verify = wx.Button(self, self.id_verify, "Verify")
        hbox.Add(self.verify, 0, wx.ALL|wx.EXPAND|wx.ALIGN_CENTER, self.border)
        sizer.Add(hbox, 0, wx.ALL|wx.EXPAND, self.border)

        self.gauge = wx.Gauge(self, -1, 50, size=(500, 5))
        sizer.Add(self.gauge, 0, wx.ALL|wx.EXPAND, self.border)

        self.status = ExpandoTextCtrl(self, style=wx.ALIGN_LEFT|wx.TE_READONLY|wx.NO_BORDER)
        attr = self.GetDefaultAttributes()
        self.status.SetBackgroundColour(attr.colBg)
        sizer.Add(self.status, 1, wx.ALL|wx.EXPAND, self.border)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(self, -1, "Server Name")
        hbox.Add(label, 0, wx.ALL|wx.EXPAND|wx.ALIGN_CENTER, self.border)
        self.name = wx.TextCtrl(self, -1, "", size=(500,-1))
        hbox.Add(self.name, 0, wx.EXPAND, self.border)
        sizer.Add(hbox, 0, wx.ALL|wx.EXPAND, self.border)

        btnsizer = wx.StdDialogButtonSizer()
        self.ok_btn = wx.Button(self, wx.ID_OK)
        self.ok_btn.SetDefault()
        self.ok_btn.Enable(False)
        btnsizer.AddButton(self.ok_btn)
        btn = wx.Button(self, wx.ID_CANCEL)
        btnsizer.AddButton(btn)
        btnsizer.Realize()
        sizer.Add(btnsizer, 0, wx.ALL|wx.EXPAND, self.border)

        self.Bind(wx.EVT_BUTTON, self.on_button)
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.Bind(EVT_ETC_LAYOUT_NEEDED, self.on_resize)

        self.SetSizer(sizer)
        self.Fit()

    def on_button(self, event):
        if event.GetId() == self.id_verify:
            self.check_server(self.url.GetValue())
        elif event.GetId() == wx.ID_OK:
            self.EndModal(wx.ID_OK)
            event.Skip()
        else:
            self.EndModal(wx.ID_CANCEL)
            event.Skip()
    
    def check_server(self, url):
        self.verify.Enable(False)
        self.status.SetValue("Checking server...\n")
        for version in ['1.3.0', '1.1.1']:
            host = WMSHost("test", url, version)
            downloader = BackgroundWMSDownloader(host)
            wms = downloader.wms
            while True:
                if wms.is_finished:
                    break
                time.sleep(.05)
                self.gauge.Pulse()
                wx.Yield()
            if wms.is_valid():
                break
            
        self.gauge.SetValue(0)
        wx.Yield()
        if wms.is_valid():
            host.name = wms.wms.identification.title
            self.status.AppendText("Found WMS server: %s\n" % host.name)
            self.name.SetValue(host.name)
            self.ok_btn.Enable(True)
            self.verified_host = host
        else:
            self.status.AppendText("Failed: %s\n" % wms.error)
            self.name.SetValue("")
            self.ok_btn.Enable(False)
            self.verified_host = None
        self.verify.Enable(True)
    
    def get_host(self):
        new_name = self.name.GetValue()
        if new_name:
            self.verified_host.name = new_name
        return self.verified_host

    def on_resize(self, event):
        print "resized"
        self.Fit()

    def OnClose(self, event):
        self.EndModal(wx.ID_CANCEL)

