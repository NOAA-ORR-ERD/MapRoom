import wx
import wx.lib.sized_controls as sc

from ..library import coordinates


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
