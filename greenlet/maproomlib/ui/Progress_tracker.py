import wx
import maproomlib.ui
import maproomlib.utility as utility


class Progress_tracker( wx.Gauge ):
    """
    Tracks the progress of an object capable of reporting its progress
    (usually a layer as it's loading). Displays this progress as a message
    and progress bar in the status bar.
    """
    FIRST_STATUS_WIDTH = 200 # in pixels

    def __init__( self, status_bar, *watch_list ):
        wx.Gauge.__init__(
            self,
            status_bar,
            wx.ID_ANY,
        )

        self.inbox = maproomlib.ui.Wx_inbox()
        self.status_bar = status_bar
        self.status_bar.SetFieldsCount( 2 )
        self.status_bar.SetStatusWidths( [ self.FIRST_STATUS_WIDTH, -3 ] )

        self.Hide()

        for watch in watch_list:
            watch.outbox.subscribe(
                self.inbox,
                request = ( "start_progress", "end_progress" ),
            )

    def run( self, scheduler ):
        TICK_SECONDS = 0.05

        while True:
            message = self.inbox.receive( request = "start_progress" )

            tracking_id = message.get( "id" )

            self.status_bar.SetFieldsCount( 3 )
            self.status_bar.SetStatusWidths( [ self.FIRST_STATUS_WIDTH, -2, -1 ] )
            self.status_bar.SetStatusText( message.get( "message" ) + "...", 1 )

            # Position the progress bar directly over the third status bar
            # field.
            ( x, y, width, height ) = self.status_bar.GetFieldRect( 2 )
            self.SetPosition( ( x, y ) )
            self.SetSize( ( width, height ) )

            self.Show()
            self.Pulse()

            while True:
                try:
                    message = self.inbox.receive(
                        timeout = TICK_SECONDS,
                        request = "end_progress",
                        id = tracking_id,
                    )
                except utility.Timeout_error:
                    self.Pulse()
                    continue

                if message.get( "request" ) == "end_progress":
                    self.Hide()
                    self.status_bar.SetStatusText( message.get( "message" ) or "", 1 )
                    self.status_bar.SetFieldsCount( 2 )
                    self.status_bar.SetStatusWidths( [ self.FIRST_STATUS_WIDTH, -3 ] )
                    break
