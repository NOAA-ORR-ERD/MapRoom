import wx
import wx.lib.sized_controls as sc

from ..library import coordinates
from ..library.textparse import parse_int_string

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

