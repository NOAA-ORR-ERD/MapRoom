import wx
import maproomlib
import maproomlib.ui


class File_saver:
    """
    Prompts the user for a filename and then tries to save the currently
    selected layer to it, displaying any resulting error message to the user.
    """
    def __init__( self, frame, root_layer, command_stack,
                  always_prompt_for_filename ):
        self.frame = frame
        self.root_layer = root_layer
        self.command_stack = command_stack
        self.always_prompt_for_filename = always_prompt_for_filename
        self.inbox = maproomlib.ui.Wx_inbox()

    def run( self, scheduler ):
        # First, try just saving the file without prompting for a filename.
        if self.always_prompt_for_filename is False:
            self.root_layer.inbox.send(
                request = "save_selected",
                response_box = self.inbox,
            ),

            try:
                message = self.inbox.receive(
                    request = (
                        "saved",
                        "filename_needed",
                        "saver_needed",
                    )
                )

                # Success saving, so we're done.
                if message.get( "request" ) == "saved":
                    return
            except Exception, error:
                wx.MessageDialog(
                    self.frame,
                    message = str( error ),
                    style = wx.OK | wx.ICON_ERROR,
                ).ShowModal()
                return

        self.root_layer.inbox.send(
            request = "get_savers_for_selected",
            response_box = self.inbox,
        )

        try:
            message = self.inbox.receive( request = "savers" )
            savers = message.get( "savers" )
        except Exception, error:
            wx.MessageDialog(
                self.frame,
                message = str( error ),
                style = wx.OK | wx.ICON_ERROR,
            ).ShowModal()
            return

        # Prompt for a filename and file type, and then save.
        dialog = wx.FileDialog(
            self.frame,
            style = wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
            message = "Select a file to save",
            wildcard = \
                "|".join(
                    [ "%s|*.*" % saver.DESCRIPTION for saver in savers ]
                )
        )

        if dialog.ShowModal() != wx.ID_OK:
            return

        filename = dialog.GetPath()
        selected_saver = savers[ dialog.GetFilterIndex() ]

        self.root_layer.inbox.send(
            request = "save_selected",
            filename = filename,
            saver = selected_saver,
            response_box = self.inbox,
        )

        try:
            message = self.inbox.receive( request = "saved" )
        except Exception, error:
            wx.MessageDialog(
                self.frame,
                message = str( error ),
                style = wx.OK | wx.ICON_ERROR,
            ).ShowModal()
