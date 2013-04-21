import wx
import wx.richtext
import time
import logging
import logging.handlers
import maproomlib.utility as utility
from Wx_inbox import Wx_inbox


class Log_handler( logging.Handler ):
    def __init__( self, level = logging.NOTSET ):
        logging.Handler.__init__( self, level )
        self.outbox = utility.Outbox()

    def emit( self, record ):
        self.outbox.send( request = "log", record = record )


class Log_viewer:
    """
    A viewing window for messages sent to the Python logging system.
    """
    def __init__( self, parent_frame ):
        self.inbox = Wx_inbox( leak_warnings = False )
        self.window = None
        self.parent_frame = parent_frame
        self.records = []

        self.handler = Log_handler()
        self.handler.setLevel( logging.DEBUG )
        root_logger = logging.getLogger( "" )
        root_logger.addHandler( self.handler )

    def run( self, scheduler ):
        self.handler.outbox.subscribe(
            self.inbox, request = "log",
        )

        while True:
            message = self.inbox.receive(
                request = ( "show", "hide", "log" ),
            )

            request = message.pop( "request" )

            if request == "show":
                if self.window:
                    self.window.Show()
                    self.window.Raise()
                else:
                    self.window = Log_viewer_window( self.parent_frame )
                    for record in self.records:
                        self.window.add_record( record )
                    self.window.Show()
            elif request == "hide":
                self.window.Hide()
            elif request == "log":
                record = message.get( "record" )

                self.records.append( record )
                if self.window:
                    self.window.add_record( record )


class Log_viewer_window( wx.Frame ):
    """
    A viewing window for messages sent to the Python logging system.
    """
    DEFAULT_SIZE = ( 640, 480 )
    LEVEL_COLORS = {
        logging.DEBUG: "GRAY",
        logging.INFO: "BLACK",
        logging.WARNING: "BROWN",
        logging.ERROR: "ORANGE",
        logging.CRITICAL: "RED",
    }

    def __init__( self, parent_frame ):
        wx.Frame.__init__(
            self, None, wx.ID_ANY, "Log Viewer", size = self.DEFAULT_SIZE,
        )
        self.SetIcon( parent_frame.GetIcon() )

        self.text_area = wx.richtext.RichTextCtrl(
            self, wx.ID_ANY, style = wx.richtext.RE_READONLY | wx.VSCROLL,
        )
        self.text_area.GetCaret().Hide()
        self.sizer = wx.BoxSizer( wx.VERTICAL )
        self.sizer.Add( self.text_area, 1, wx.EXPAND )
        self.SetSizer( self.sizer )

    def add_record( self, record ):
        self.Freeze()
        self.text_area.AppendText(
            time.strftime(
                "%a, %b %d %Y %H:%M:%S ",
                time.localtime( record.created ),
            )
        )

        self.text_area.BeginTextColour(
            self.LEVEL_COLORS.get( record.levelno )
        )
        self.text_area.AppendText( "%s: " % record.levelname )
        self.text_area.BeginBold()
        if len( record.args ) > 0:
            self.text_area.AppendText( "%s " % ( record.msg % record.args ) )
        else:
            self.text_area.AppendText( "%s " % record.msg )
        self.text_area.EndBold()

        self.text_area.AppendText( "at %s:%s\n" % ( record.filename, record.lineno ) )
        self.text_area.EndTextColour()
        self.text_area.ShowPosition( self.text_area.GetLastPosition() )
        self.Thaw()
