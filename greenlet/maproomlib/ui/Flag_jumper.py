import maproomlib.utility as utility


class Flag_jumper:
    """
    Whenever the flagged points are updated, jumps the viewport to focus on
    them.
    """
    def __init__( self, root_layer, viewport ):
        self.root_layer = root_layer
        self.viewport = viewport
        self.inbox = utility.Inbox()

    def run( self, scheduler ):
        self.root_layer.outbox.subscribe(
            self.inbox,
            request = "flags_updated",
        )

        while True:
            message = self.inbox.receive( request = "flags_updated" )

            flag_count = message.get( "flag_count" )
            geo_origin = message.get( "origin" )
            geo_size = message.get( "size" )
            projection = message.get( "projection" )

            # If the points are directly on top of one another, then just center
            # the viewport on them without zooming.
            if geo_size == ( 0.0, 0.0 ):
                if geo_origin != ( 0.0, 0.0 ):
                    self.viewport.jump_geo_center(
                        geo_origin,
                        projection,
                    )
                continue

            # Add padding around the edge of the viewport, such that the flagged
            # points are centered within it. Add more padding if it's just a few
            # points so that the user can see some of the surrounding map as well.
            if flag_count <= 2:
                PADDING_FACTOR = 8.0
            elif flag_count <= 10:
                PADDING_FACTOR = 2.0
            else:
                PADDING_FACTOR = 0.1

            padding = max(
                geo_size[ 0 ] * PADDING_FACTOR,
                geo_size[ 1 ] * PADDING_FACTOR,
            )

            geo_origin = (
                geo_origin[ 0 ] - padding,
                geo_origin[ 1 ] - padding,
            )

            geo_size = (
                geo_size[ 0 ] + padding * 2,
                geo_size[ 1 ] + padding * 2,
            )

            self.viewport.jump_geo_boundary(
                geo_origin,
                geo_size,
                projection,
            )
