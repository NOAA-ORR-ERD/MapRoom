import wx
import sys
import time
import logging
import traceback

def make_except_hook( logger ):
    MAX_ERROR_LINES = 15
    
    def except_hook( type, value, tb ):
        full_error = "".join( traceback.format_exception( type, value, tb ) )
        logger.critical( full_error )
        
        if sys.platform.startswith( "win" ):
            full_error = full_error.replace( "\n", "\r\n" )
        
        error_lines = full_error.split( "\n" )
        if len( error_lines ) > MAX_ERROR_LINES:
            error_lines = \
                [ error_lines[ 0 ], "..." ] + \
                error_lines[ -MAX_ERROR_LINES : ]
            error = "\n".join( error_lines )
        else:
            error = full_error
        
        try:
            """
            wx.TheClipboard.Open()
            wx.TheClipboard.SetData( wx.TextDataObject( full_error ) )
            wx.TheClipboard.Close()
            """
            time.sleep( 0.1 ) # Don't ask.
            wx.TheClipboard.Flush()
        except AttributeError:
            # If there's no clipboard, then the UI probably hasn't even
            # started yet, so bail.
            return
        
        wx.MessageDialog(
            wx.GetApp().GetTopWindow(),
            message = "An unexpected error has occurred:\n\n" + error +
                      "\n\nThe application will shut down now.",
            style = wx.OK | wx.ICON_ERROR,
        ).ShowModal()
        
        sys.exit( 1 )
    
    return except_hook

def notify_all_errors( logger ):
    sys.excepthook = make_except_hook( logger )
