import wx
import os


class About_dialog:
    LOGO_FILENAME = "ui/images/maproom_large.png"
    def __init__( self, version ):
        self.version = version

    def run( self, scheduler ):
        logo_filename = self.LOGO_FILENAME
        if os.path.basename( os.getcwd() ) != "maproom":
            logo_filename = os.path.join( "maproom", logo_filename )

        info = wx.AboutDialogInfo()

        info.SetIcon( wx.Icon( logo_filename, wx.BITMAP_TYPE_PNG ) )
        info.SetName( "Maproom" )
        info.SetCopyright( "Developed by NOAA" )

        if not self.version.SOURCE_CONTROL_REVISION:
            info.SetVersion( self.version.VERSION )
        else:
            info.SetVersion( "%s (r%s)" % (
                self.version.VERSION, self.version.SOURCE_CONTROL_REVISION,
            ) )

        info.SetDescription( "High-performance 2d mapping" )
        info.SetWebSite( "http://www.noaa.gov/" )

        dialog = wx.AboutBox( info )
