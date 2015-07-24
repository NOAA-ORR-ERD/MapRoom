import wx
import wx.lib.sized_controls as sc
import wx.lib.buttons as buttons

from ..library import coordinates
from ..library.textparse import parse_int_string
from ..library.marplot_icons import *
from ..mock import MockProject

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

    def __init__(self, layer_control, display_format):
        sc.SizedDialog.__init__(self, wx.GetTopLevelParent(layer_control), wx.ID_ANY, "Jump to Coordinates")

        panel = self.GetContentsPane()
        wx.StaticText(panel, -1, "Please enter the coordinates to jump to")

        self.coords_text = wx.TextCtrl(panel, -1, "")
        self.coords_text.SetSizerProps(expand=True)
        self.coords_text.Bind(wx.EVT_TEXT, self.OnText)

        center_lat_lon = layer_control.get_world_point_from_projected_point(layer_control.projected_point_center)
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
        self.layer.layer_info_panel = ["Line Style", "Line Width", "Line Color", "Start Marker", "End Marker", "Line Transparency", "Fill Style", "Fill Color", "Fill Transparency", "Text Color", "Font", "Font Size", "Text Transparency", "Marplot Icon"]
        
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
