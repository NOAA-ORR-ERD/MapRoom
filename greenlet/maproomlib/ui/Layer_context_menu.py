import os
import wx


class Layer_context_menu( wx.Menu ):
    IMAGE_PATH = "ui/images"

    def __init__( self, layer, raisable, lowerable, layer_deletable ):
        wx.Menu.__init__( self )

        image_path = self.IMAGE_PATH
        if os.path.basename( os.getcwd() ) != "maproom":
            image_path = os.path.join( "maproom", image_path )

        self.raise_item = wx.MenuItem(
            self,
            wx.ID_UP,
            "Raise Layer",
        )
        self.raise_item.SetBitmap(
            wx.Bitmap( os.path.join( image_path, "raise.png" ) ),
        )
        self.AppendItem( self.raise_item )
        self.raise_item.Enable( raisable )

        self.lower_item = wx.MenuItem(
            self,
            wx.ID_DOWN,
            "Lower Layer",
        )
        self.lower_item.SetBitmap(
            wx.Bitmap( os.path.join( image_path, "lower.png" ) ),
        )
        self.AppendItem( self.lower_item )
        self.lower_item.Enable( lowerable )

        self.AppendSeparator()

        self.delete_layer_item = wx.MenuItem(
            self,
            wx.ID_REMOVE,
            "Delete Layer",
        )
        self.delete_layer_item.SetBitmap(
            wx.Bitmap( os.path.join( image_path, "delete_layer.png" ) ),
        )
        self.AppendItem( self.delete_layer_item )
        self.delete_layer_item.Enable( layer_deletable )
