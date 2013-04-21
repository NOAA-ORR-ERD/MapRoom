import wx
import os.path
import maproomlib
import maproomlib.ui


class File_opener:
    """
    Prompts the user for a filename and then tries to load it as a layer,
    displaying any resulting error message to the user.
    """
    def __init__( self, frame, root_layer, command_stack, filename = None ):
        self.filename = filename
        self.frame = frame
        self.root_layer = root_layer
        self.command_stack = command_stack
        self.inbox = maproomlib.ui.Wx_inbox()
        self.outbox = maproomlib.utility.Outbox()

    def run( self, scheduler ):
        if self.filename is None:
            # TODO: These file types should be populated automatically based
            # on loaded plugins.
            file_types = (
                "All files (*.*)|*.*",
                "BNA files (*.bna)|*.bna",
                "GeoTIFF files (*.tif)|*.tif",
                "KAP files (*.kap)|*.kap",
                "Maproom Vector files (*.maproomv)|*.maproomv",
                "MOSS files (*.ms1)|*.ms1",
                "NGA DNC ZIP files (*.zip)|*.zip",
                "Shape files (*.shp)|*.shp",
                "Verdat files (*.verdat)|*.verdat",
            )

            dialog = wx.FileDialog(
                self.frame,
                style = wx.FD_FILE_MUST_EXIST,
                message = "Select a file to open",
                wildcard = "|".join( file_types )
            )

            if dialog.ShowModal() != wx.ID_OK:
                self.outbox.send(
                    request = "file_opener_done",
                )
                return

            self.filename = dialog.GetPath()

        self.root_layer.inbox.send(
            request = "get_layers",
            response_box = self.inbox,
        )

        message = self.inbox.receive( request = "layers" )
        layers = message.get( "layers" )
        filename = os.path.normcase( os.path.abspath( self.filename ) )
        replace_layer = None

        for layer in layers:
            if hasattr( layer, "filename" ) and layer.filename and \
               os.path.normcase( os.path.abspath( layer.filename ) ) == \
               filename:
                dialog = wx.MessageDialog(
                    self.frame,
                    message = 'That file is already loaded as layer "%s". Load the file again, replacing the existing layer?' % layer.name,
                    caption = "File already loaded",
                    style = wx.OK | wx.CANCEL | wx.ICON_QUESTION,
                )

                if dialog.ShowModal() != wx.ID_OK:
                    self.outbox.send(
                        request = "file_opener_done",
                    )
                    return

                replace_layer = layer
                break

        self.command_stack.inbox.send(
            request = "start_command"
        )
        self.root_layer.inbox.send(
            request = "load_layer",
            filename = self.filename,
            response_box = self.inbox,
        )

        try:
            self.inbox.receive( request = "layer", filename = self.filename )
        except Exception, error:
            wx.MessageDialog(
                self.frame,
                message = str( error ),
                style = wx.OK | wx.ICON_ERROR,
            ).ShowModal()
            self.outbox.send(
                request = "file_opener_done",
            )
            return

        if replace_layer:
            self.root_layer.inbox.send(
                request = "remove_layer",
                layer = replace_layer,
            )

        self.outbox.send(
            request = "file_opener_done",
        )
