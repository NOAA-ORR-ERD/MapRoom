import wx
import wx.lib.sized_controls as sc

import library.coordinates as coordinates

class JumpCoordsDialog(sc.SizedDialog):
    def __init__(self, *a, **kw):
        sc.SizedDialog.__init__(self, *a, **kw)
        
        panel = self.GetContentsPane()
        wx.StaticText(panel, -1, "Please enter the coordinates to jump to")
        
        self.coords_text = wx.TextCtrl(panel, -1, "")
        self.coords_text.SetSizerProps(expand=True)
        self.coords_text.Bind(wx.EVT_TEXT, self.OnText)
        
        app = wx.GetApp()
        renderer = app.current_map.renderer
        center_lat_lon = renderer.get_world_point_from_projected_point(renderer.projected_point_center)
        self.coords_text.Value = coordinates.format_coords_for_display(center_lat_lon[0], center_lat_lon[1])
        
        btn_sizer = self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL)
        self.Sizer.Add(btn_sizer, 0, 0, wx.EXPAND | wx.BOTTOM | wx.RIGHT, self.GetDialogBorder())
        
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        
        self.Fit()
        
    def OnClose(self, event):
        self.EndModal(wx.ID_CANCEL)
        
    def OnText(self, event):
        lat_lon_string = event.String
        
        lat_lon = coordinates.lat_lon_from_format_string(lat_lon_string)
        if lat_lon == (-1, -1):
            self.coords_text.SetBackgroundColour( "#FF8080" )
        else:
            self.coords_text.SetBackgroundColour( "#FFFFFF" )