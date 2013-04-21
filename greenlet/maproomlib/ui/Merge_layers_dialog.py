import wx
from Wx_inbox import Wx_inbox


class Merge_layers_dialog:
    MERGEABLE_LAYER_TYPES = [
        "Line_point_layer",
    ]

    def __init__( self, parent_frame, root_layer ):
        self.parent_frame = parent_frame
        self.root_layer = root_layer
        self.inbox = Wx_inbox()

    def run( self, scheduler ):
        self.root_layer.inbox.send(
            request = "get_layers",
            response_box = self.inbox,
        )

        message = self.inbox.receive( request = "layers" )
        layers = [
            layer for layer in message.get( "layers" ) \
            if layer.__class__.__name__ in self.MERGEABLE_LAYER_TYPES
        ]

        if len( layers ) < 2:
            wx.MessageDialog(
                self.parent_frame,
                message = "Please open or create at least two vector layers to merge.",
                style = wx.OK | wx.ICON_ERROR,
            ).ShowModal()
            return

        layers.reverse()
        layer_names = \
            [ str( layer.name ) for layer in layers ]

        dialog = wx.MultiChoiceDialog(
            self.parent_frame,
            "Please select two or more vector layers to merge together " +
                "into one layer.\n\n" +
                "Only those layers that support merging are listed.",
            "Merge Layers",
            layer_names,
        )

        # If there are exactly two layers, select them both as a convenience
        # to the user.
        if len( layers ) == 2:
            dialog.SetSelections( [ 0, 1 ] )

        if dialog.ShowModal() != wx.ID_OK:
            return

        selections = dialog.GetSelections()
        if len( selections ) < 2:
            wx.MessageDialog(
                self.parent_frame,
                message = "At least two layers are required for merging.",
                style = wx.OK | wx.ICON_ERROR,
            ).ShowModal()
            return

        self.root_layer.inbox.send(
            request = "merge_layers",
            layers = [ layers[ index ] for index in selections ],
            response_box = self.inbox,
        )

        try:
            self.inbox.receive( request = "layer" )
        except Exception, error:
            wx.MessageDialog(
                self.parent_frame,
                message = str( error ),
                style = wx.OK | wx.ICON_ERROR,
            ).ShowModal()
