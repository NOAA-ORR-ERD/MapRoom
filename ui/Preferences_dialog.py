import os
import sys
import wx
import wx.lib.sized_controls as sc

import app_globals

pref_choices = {
    "Coordinate Display Format": ( "degrees minutes seconds", "degrees decimal minutes", "decimal degrees" )
}

class PreferencesDialog(sc.SizedDialog):
    def __init__(self, *a, **kw):
        sc.SizedDialog.__init__(self, *a, **kw)
        
        panel = self.GetContentsPane()
        panel.SetSizerType("form")
        
        self.prefs = app_globals.preferences
        
        prefname = "Coordinate Display Format"
        wx.StaticText(panel, wx.ID_ANY, prefname)
        coord_format = wx.Choice(panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, choices = pref_choices[prefname])
        coord_format.SetStringSelection(self.prefs[prefname])
        coord_format.Bind(wx.EVT_CHOICE, self.OnCoordinateDisplayFormatChanged)
        
        self.Fit()
        self.MinSize = self.Size
        
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        
    def OnClose(self, event):
        self.Destroy()
        
    def OnCoordinateDisplayFormatChanged(self, event):
        self.prefs["Coordinate Display Format"] = event.String
        