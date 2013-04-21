import math
import pyproj


class Gps_reader:
    RADIANS_TO_DEGREES = 180 / math.pi
    SERIAL_PORT = "COM1:" # TODO: Make this a preference.

    def __init__( self ):
        import garmin
        import maproomlib.utility as utility

        self.tick_interval_seconds = 5.0
        self.layer = None
        self.last_point_index = None
        self.projection = pyproj.Proj( "+proj=latlong" )
        self.inbox = utility.Inbox()
        self.gps = garmin.Garmin(
            garmin.SerialLink( self.SERIAL_PORT ),
        )
        self.gps.pvtOn()

    def run( self, scheduler ):
        import maproomlib.utility as utility
        # TODO: Turn off logging for garmin module.


        message = self.inbox.receive(
            request = "target_layer",
        )
        self.layer = message.get( "target_layer" )

        self.layer.points_layer.outbox.subscribe(
            self.inbox,
            request = ( "points_added", "line_points_added" ),
        )

        # TODO: When our target layer goes away, detect that and exit this
        # task as well.

        while True:
            try:
                message = self.inbox.receive(
                    request = (
                        "close",
                    ),
                    timeout = self.tick_interval_seconds,
                )
            except utility.Timeout_error:
                self.request_position()
                continue

            gps.pvtOff()
            self.layer.points_layer.outbox.unsubscribe( self.inbox )
            return
            

    def request_position( self ):
        # TODO: Handle any errors that may occur. Maybe just skip the point for
        # that tick if there is an error when fetching it.
        # TODO: If sync lost, somehow resync with GPS device.
        self.gps.getPvt( self.forward_position )

    def forward_position( self, pvt, record_number, total_points, tp ):
        point = self.layer.points_layer.make_points( 1, exact = True )
        point.x[ 0 ] = pvt.rlon * self.RADIANS_TO_DEGREES
        point.y[ 0 ] = pvt.rlat * self.RADIANS_TO_DEGREES

        #import random
        #point.x[ 0 ] += ( random.random() - 0.5 ) / 100.0
        #point.y[ 0 ] += ( random.random() - 0.5 ) / 100.0

        if self.last_point_index is None:
            # TODO: Somehow jump viewport to this first point.
            self.layer.points_layer.inbox.send(
                request = "add_points",
                points = point,
                projection = self.projection,

            )

            message = self.inbox.receive( request = "points_added" )
            self.last_point_index = message.get( "start_index" )
        else:
            self.layer.points_layer.inbox.send(
                request = "add_lines",
                points = point,
                projection = self.projection,
                from_index = self.last_point_index,

            )

            message = self.inbox.receive( request = "line_points_added" )
            self.last_point_index = message.get( "start_index" )
