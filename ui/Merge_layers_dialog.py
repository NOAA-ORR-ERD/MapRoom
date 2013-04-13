import wx
import app_globals

class Merge_layers_dialog:
    def __init__( self ):
        pass
    
    def show( self ):
        layers = [ layer for layer in app_globals.layer_manager.flatten() if layer.points != None ]
        
        if len( layers ) < 2:
            wx.MessageDialog(
                app_globals.application.frame,
                message = "Please open or create at least two vector layers to merge.",
                style = wx.OK | wx.ICON_ERROR,
            ).ShowModal()
            
            return
        
        layers.reverse()
        layer_names = [ str( layer.name ) for layer in layers ]
        
        dialog = wx.MultiChoiceDialog(
            app_globals.application.frame,
            "Please select two or more vector layers to merge together into one layer.\n\nOnly those layers that support merging are listed.",
            "Merge Layers",
            layer_names
        )
        
        # If there are exactly two layers, select them both as a convenience
        # to the user.
        if ( len( layers ) == 2 ):
            dialog.SetSelections( [ 0, 1 ] )
        
        result = dialog.ShowModal()
        if result == wx.ID_OK:
            selections = dialog.GetSelections()
        
            if len( selections ) != 2:
                wx.MessageDialog(
                    self.parent_frame,
                    message = "You must select exactly two layers to merge.",
                    style = wx.OK | wx.ICON_ERROR,
                ).ShowModal()
            else:
                app_globals.layer_manager.merge_layers( layers[ selections[ 0 ] ], layers[ selections[ 1 ] ] )
        dialog.Destroy()
