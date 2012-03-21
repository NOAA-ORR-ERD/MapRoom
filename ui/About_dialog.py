import wx
import os
import app_globals

class About_dialog:
    LOGO_FILENAME = "ui/images/maproom_large.png"
    def show( self ):
        logo_filename = self.LOGO_FILENAME
        
        info = wx.AboutDialogInfo()

        info.SetIcon( wx.Icon( logo_filename, wx.BITMAP_TYPE_PNG ) )
        info.SetName( "Maproom" )
        info.SetCopyright( "Developed by NOAA" )

        if not app_globals.version.SOURCE_CONTROL_REVISION:
            info.SetVersion( app_globals.version.VERSION )
        else:
            info.SetVersion( "%s (r%s)" % (
                app_globals.version.VERSION, app_globals.version.SOURCE_CONTROL_REVISION,
            ) )

        info.SetDescription( "High-performance 2d mapping" )
        info.SetWebSite( "http://www.noaa.gov/" )

        dialog = wx.AboutBox( info )
