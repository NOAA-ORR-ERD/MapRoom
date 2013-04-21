import wx
import pyproj
import maproomlib.ui
import maproomlib.utility as utility


class Mouse_tracker:
    """
    Tracks the mouse position and displays its current geographic position in
    the status bar.
    """
    def __init__( self, root_layer, status_bar, viewport ):
        self.inbox = maproomlib.ui.Wx_inbox()
        self.root_layer = root_layer
        self.status_bar = status_bar
        self.viewport = viewport
        self.latlong = pyproj.Proj( "+proj=latlong" )

        self.viewport.outbox.subscribe(
            self.inbox,
            request = (
                "mouse_left",
                "zoomed",
                "zoom_limit_reached",
            ),
        )

        self.viewport.window.Bind( wx.EVT_MOTION, self.position_moved )

    def position_moved( self, event ):
        event.Skip()
        self.inbox.send(
            request = "position_moved",
            pixel_position = event.GetPosition(),
        )

    def run( self, scheduler ):
        while True:
            message = self.inbox.receive(
                request = (
                    "position_moved",
                    "mouse_left",
                    "zoomed",
                    "zoom_limit_reached",
                ),
            )
            request = message.pop( "request" )

            if request == "position_moved":
                pixel_position = message.get( "pixel_position" )
                geo_point = \
                    self.viewport.pixel_to_geo( pixel_position, self.latlong )

                if geo_point is None:
                    continue
                
                self.status_bar.SetStatusText(
                    utility.format_lat_long_degrees_minutes(
                        *geo_point
                    ),
                    0,
                )
            elif request == "mouse_left":
                self.status_bar.SetStatusText( "", 0 )
            elif request == "zoomed":
                self.status_bar.SetStatusText( "", 1 )
            elif request == "zoom_limit_reached":
                self.status_bar.SetStatusText( message.get( "message" ), 1 )
