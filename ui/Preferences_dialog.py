import os
import sys
import wx
import wx.lib.sized_controls as sc

import app_globals

pref_choices = {
    "Coordinate Display Format": ( "degrees minutes seconds", "degrees decimal minutes", "decimal degrees" ),
    "Scroll Zoom Speed": ( "Slow", "Medium", "Fast" ),
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
        
        prefname = "Scroll Zoom Speed"
        wx.StaticText(panel, wx.ID_ANY, prefname)
        zoom_speed = wx.Choice(panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, choices = pref_choices[prefname])
        zoom_speed.SetStringSelection(self.prefs[prefname])
        zoom_speed.Bind(wx.EVT_CHOICE, self.OnZoomSpeedChanged)
        
        self.Fit()
        self.MinSize = self.Size
        
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        
    def OnClose(self, event):
        self.Destroy()
        
    def OnCoordinateDisplayFormatChanged(self, event):
        self.prefs["Coordinate Display Format"] = event.String
        
    def OnZoomSpeedChanged(self, event):
        self.prefs["Scroll Zoom Speed"] = event.String
