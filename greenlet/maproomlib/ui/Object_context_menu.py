import os
import wx


class Object_context_menu( wx.Menu ):
    IMAGE_PATH = "ui/images"

    def __init__( self ):
        wx.Menu.__init__( self )

        image_path = self.IMAGE_PATH
        if os.path.basename( os.getcwd() ) != "maproom":
            image_path = os.path.join( "maproom", image_path )

        self.delete_selection = wx.MenuItem(
            self,
            wx.ID_DELETE,
            "&Delete Selection",
        )
        self.delete_selection.SetBitmap(
            wx.Bitmap( os.path.join( image_path, "delete_selection.png" ) ),
        )
        self.AppendItem( self.delete_selection )
