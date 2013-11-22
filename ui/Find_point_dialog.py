import wx
import wx.lib.sized_controls as sc

import library.coordinates as coordinates
from library.textparse import parse_int_string

class FindPointDialog(sc.SizedDialog):

    def __init__(self, *a, **kw):
        sc.SizedDialog.__init__(self, *a, **kw)

        panel = self.GetContentsPane()
        wx.StaticText(panel, -1, "Find point number, range of points\nor multiple ranges.\n\nSeparate ranges by commas, e.g: 1-4,8-10")

        self.text = wx.TextCtrl(panel, -1, "")
        self.text.SetSizerProps(expand=True)

        btn_sizer = self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL)
        self.Sizer.Add(btn_sizer, 0, 0, wx.EXPAND | wx.BOTTOM | wx.RIGHT, self.GetDialogBorder())

        app = wx.GetApp()
        self.layer = app.current_map.layer_tree_control.get_selected_layer()
        
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
        
        # hack -- couldn't find a way to set focus to the text widget otherwise
        wx.CallAfter(self.text.SetFocus)

    def OnClose(self, event):
        self.EndModal(wx.ID_CANCEL)
    
    def get_values(self):
        values, error = parse_int_string(self.text.Value)
        return values, error

