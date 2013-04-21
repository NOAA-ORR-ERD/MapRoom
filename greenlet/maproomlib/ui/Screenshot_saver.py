import os
import wx
import maproomlib
import maproomlib.ui


class Screenshot_saver:
    """
    Prompts the user for a filename and then saves a viewport screenshot to
    it.
    """
    def __init__( self, frame, renderer ):
        self.frame = frame
        self.renderer = renderer
        self.inbox = maproomlib.ui.Wx_inbox()

    def run( self, scheduler ):
        file_types = (
            ( wx.BITMAP_TYPE_PNG, "PNG|*.png", ".png" ),
            ( wx.BITMAP_TYPE_JPEG, "JPEG|*.jpg", ".jpg" ),
        )

        # First, take a screenshot of the current viewport.
        self.renderer.inbox.send(
            request = "take_viewport_screenshot",
            response_box = self.inbox,
        )

        message = self.inbox.receive(
            request = "viewport_screenshot",
        )

        screenshot = message.get( "screenshot" )

        # Prompt for a filename and file type, and then save.
        dialog = wx.FileDialog(
            self.frame,
            style = wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
            message = "Select a file to save",
            wildcard = \
                "|".join( [ file_type[ 1 ] for file_type in file_types ] )
        )

        if dialog.ShowModal() != wx.ID_OK:
            return

        filename = dialog.GetPath()
        image_type = file_types[ dialog.GetFilterIndex() ][ 0 ]
        extension = file_types[ dialog.GetFilterIndex() ][ 2 ]

        # Tack an extension onto the filename if one isn't already present.
        if "." not in os.path.basename( filename ):
            new_filename = filename + extension
            if not os.path.exists( new_filename ):
                filename = new_filename

        try:
            screenshot.SaveFile( filename, image_type )
        except Exception, error:
            wx.MessageDialog(
                self.frame,
                message = str( error ),
                style = wx.OK | wx.ICON_ERROR,
            ).ShowModal()
