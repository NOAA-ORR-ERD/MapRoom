import wx
import maproomlib
import maproomlib.ui


class Triangulator:
    def __init__( self, frame, root_layer, transformer ):
        self.frame = frame
        self.root_layer = root_layer
        self.transformer = transformer
        self.inbox = maproomlib.ui.Wx_inbox()

    def run( self, scheduler ):
        import maproomlib.plugin

        self.root_layer.inbox.send( 
            request = "triangulate_selected",
            transformer = self.transformer,
            response_box = self.inbox,
        )

        try:
            self.inbox.receive()
        except ( maproomlib.plugin.Triangulation_error, NotImplementedError ), error:
            wx.MessageDialog(
                self.frame,
                message = str( error ),
                style = wx.OK | wx.ICON_ERROR,
            ).ShowModal()
