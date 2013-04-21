import logging
import numpy as np
import maproomlib.utility as utility
from Point_set_layer import Point_set_layer
from Selected_point_set_layer import Selected_point_set_layer


class Polygon_set_layer:
    """
    A set of polygons.

    :param name: textual name of the layer
    :type name: str
    :param points_layer: set of points that the polygons connect
    :type points_layer: plugin.Point_set_layer
    :param polygon_points: adjacency list where each index is a point index
                           and each row is:
                           [ index of the next point in the polygon,
                             index of the polygon containing these points ]
    :type polygon_points: scalar array of POLYGON_OPINTS_DTYPE
    :param polygons: array with one row per polygon:
                     [ index of start of polygon, point count, RGBA fill color ]
    :type polygon: Nx2 numpy array or rec.array of POLYGONS_DTYPE

    .. attribute:: name

        name of the layer

    .. attribute:: children

        list of child layers

    .. function:: get_polygons( origin, size, response_box )

        When a message with ``request = "get_polygons"`` is received within
        the :attr:`inbox`, a handler sends the polygon data to the given
        ``response_box``.

        Note that at this time, the ``origin`` and ``size`` are ignored, and
        all polygon data is sent without any culling.

        :param origin: lower-left corner of desired data in geographic
                       coordinates
        :type origin: ( float, float )
        :param size: dimensions of desired data in geographic coordinates
        :type size: ( float, float )
        :param response_box: response message sent here
        :type response_box: Inbox

        The response message is sent as follows::

            response_box.send( request = "polygons", points, polygons, projection, origin, size )

        ``points`` is an Nx4 numpy recarray with one row per point:
        [ point x, point y, point z, RGBA color of point ] and dtype
        [ ( "x", np.float32 ), ( "y", np.float32 ), ( "z", np.float32 ),
        ( "color", np.uint32 ) ].

        ``polygons_points`` is an scalar numpy array adjacency list where each
        index is a point index and each value is the index of the next point
        in the polygon of POLYGON_POINTS_DTYPE.

        ``polygons`` is an Nx2 numpy recarray with one row per polygon:
        [ index of start of polygon, RGBA fill color ] of POLYGONS_DTYPE.

        ``projection`` is the geographic projection of the provided points.

        ``origin`` and ``size`` are the original values that were passed in to
        the ``get_polygons`` request.
        """
    PLUGIN_TYPE = "layer"
    LAYER_TYPE = "polygon_set"
    EXTRA_CAPACITY_FACTOR = 2.0   # extra capacity to reserve
    DEFAULT_POLYGON_COLOR = utility.color_to_int( 0, 0.75, 0, 0.5 )
    POLYGONS_DTYPE = np.uint32
    POLYGON_POINTS_DTYPE = np.dtype( [ # One row per point.
        ( "next", np.uint32 ),    # Index of next adjacent point in polygon.
        ( "polygon", np.uint32 ), # Index of polygon this point is in.
    ] )
    POLYGONS_DTYPE = np.dtype( [  # One row per polygon.
        ( "start", np.uint32 ),   # Index of arbitrary point in this polygon.
        ( "count", np.uint32 ),   # Number of points in this polygon.
        ( "type", np.uint32 ),    # Type flag of this polygon.
        ( "color", np.uint32 ),   # Color of this polygon.
        ( "group", np.uint32 ),   # An outer polygon and all of its holes have
                                  # the same opaque group id.
    ] )

    def __init__( self, command_stack, name, points_layer, polygon_points,
                  polygons, polygon_count, selection_layer = None,
                  default_polygon_color = None ):
        self.command_stack = command_stack
        self.name = name
        self.points_layer = points_layer
        self.projection = points_layer.projection
        self.polygon_points = polygon_points.view( np.recarray )
        self.polygons = polygons.view( np.recarray )
        self.polygon_add_index = polygon_count
        self.invalid_polygons = None
        self.selection_layer = selection_layer
        self.default_polygon_color = default_polygon_color or \
            self.DEFAULT_POLYGON_COLOR
        self.children = []
        self.inbox = utility.Inbox()
        self.outbox = utility.Outbox()
        self.logger = logging.getLogger( __name__ )

    def run( self, scheduler ):
        self.points_layer.outbox.subscribe(
            self.inbox,
            request = (
                "points_added",
                "points_updated",
                "points_deleted",
                "points_undeleted",
                "size_changed",
            )
        )

        while True:
            message = self.inbox.receive(
                request = (
                    "get_polygons",
                    "set_polygon_starts",
                    "set_invalid_polygons",
                    "make_selection",
                    "points_added",
                    "points_updated",
                    "points_deleted",
                    "points_undeleted",
                    "size_changed",
                ),
            )
            request = message.pop( "request" )

            if request == "get_polygons":
                self.get_polygons( **message )
            elif request == "set_polygon_starts":
                self.set_polygon_starts( **message )
            elif request == "set_invalid_polygons":
                self.set_invalid_polygons( **message )
            elif request == "make_selection":
                self.make_selection( scheduler, **message )
            elif request == "points_added":
                # For now, only support adding one point at a time.
                if message.get( "count" ) != 1:
                    continue

                if message.get( "from_index" ) is None:
                    self.points_added_to_new_polygon( **message )
                else:
                    self.points_added_to_existing_polygon( **message )
            elif request in \
                ( "points_updated", "points_deleted", "points_undeleted" ):
                self.points_updated( **message )
            elif request == "size_changed":
                message[ "request" ] = request
                self.outbox.send( **message )

        self.points_layer.outbox.unsubscribe( self.inbox )

    def get_polygons( self, origin, size, response_box ):
        self.points_layer.inbox.send(
            request = "get_points",
            origin = origin,
            size = size,
            response_box = self.inbox,
        )
        message = self.inbox.receive( request = "points" )

        response_box.send(
            request = "polygons",
            points = message.get( "points" ),
            point_count = message.get( "count" ),
            polygon_points = self.polygon_points,
            polygons = self.polygons,
            polygon_count = self.polygon_add_index,
            projection = message.get( "projection" ),
            origin = origin,
            size = size,
        )

    def set_polygon_starts( self, start_indices ):
        self.points_layer.inbox.send(
            request = "get_points",
            origin = None,
            size = None,
            response_box = self.inbox,
        )
        message = self.inbox.receive( request = "points" )

        # For each given start index, set that point to be the start index for
        # the polygon that contains it.
        for start_index in start_indices:
            polygon_index = self.polygon_points.polygon[ start_index ]
            self.polygons.start[ polygon_index ] = start_index

        self.outbox.send(
            request = "polygons_updated",
            points = message.get( "points" ),
            polygon_points = self.polygon_points,
            polygons = self.polygons,
            polygon_count = self.polygon_add_index,
            updated_points = start_indices,
            projection = message.get( "projection" ),
        )

    def set_invalid_polygons( self, polygon_indices ):
        self.invalid_polygons = polygon_indices

    def make_selection( self, scheduler, object_indices, color, depth_unit,
                        response_box ):
        # If there are any points selected, bail. A polygon can only be
        # selected if nothing else is selected.
        if self.selection_layer and len( self.selection_layer.children ) > 0 \
           or not object_indices:
            response_box.send(
                request = "selection",
                layer = None,
            )
            return

        self.points_layer.inbox.send(
            request = "get_points",
            origin = None,
            size = None,
            response_box = self.inbox,
        )
        message = self.inbox.receive( request = "points" )
        points = message.get( "points" )

        # Ignore all polygons other than first one.
        polygon_index = object_indices[ 0 ]
        point_count = self.polygons.count[ polygon_index ]

        point_indices = np.empty(
            ( point_count * self.EXTRA_CAPACITY_FACTOR, ),
            np.uint32,
        )

        point_index = self.polygons.start[ polygon_index ]

        for selected_index in xrange( 0, point_count ):
            point_indices[ selected_index ] = point_index
            point_index = self.polygon_points.next[ point_index ]

        self.points_layer.inbox.send(
            request = "change_shown",
            shown_indices = point_indices,
            shown_count = point_count,
        )

        # This doesn't really make a selection. It just changes the points
        # shown in the points_layer.
        response_box.send(
            request = "selection",
            layer = None,
        )

    def points_added_to_new_polygon( self, layer, points, projection,
                                     start_index, count, from_index = None,
                                     to_layer = None, to_index = None,
                                     undo_recorded = None ):
        if start_index + count > len( self.polygon_points ):
            new_size = int( len( self.polygon_points ) * self.EXTRA_CAPACITY_FACTOR )
            self.logger.debug(
                "Growing polygon points array from %d points capacity to %d." % (
                    len( self.polygon_points ), new_size,
                ),
            )
            self.polygon_points = np.resize(
                self.polygon_points, ( new_size, ),
            ).view( np.recarray )

        if self.polygon_add_index >= len( self.polygons ):
            new_size = int( len( self.polygons ) * self.EXTRA_CAPACITY_FACTOR )
            self.logger.debug(
                "Growing polygons array from %d polygons capacity to %d." % (
                    len( self.polygons ), new_size,
                ),
            )
            self.polygons = np.resize(
                self.polygons, ( new_size, ),
            ).view( np.recarray )

        # This is essentially a one-point "polygon", so its next point is
        # itself.
        polygon_index = self.polygon_add_index
        self.polygon_points.next[ start_index ] = start_index
        self.polygon_points.polygon[ start_index ] = polygon_index
        self.polygons.start[ polygon_index ] = start_index
        self.polygons.count[ polygon_index ] = 1
        self.polygons.color[ polygon_index ] = self.default_polygon_color

        if polygon_index == 0:
            self.polygons.group[ polygon_index ] = 0
        else:
            # Just take the previous group id and increment it to make our
            # group id.
            self.polygons.group[ polygon_index ] = \
                self.polygons.group[ polygon_index - 1 ] + 1

        self.polygon_add_index += 1

        self.outbox.send(
            request = "polygons_updated",
            points = points,
            polygon_points = self.polygon_points,
            polygons = self.polygons,
            polygon_count = self.polygon_add_index,
            updated_points = [ start_index ],
            projection = projection,
        )

    def points_added_to_existing_polygon( self, layer, points, projection,
                                          start_index, count,
                                          from_index = None, to_layer = None,
                                          to_index = None, undo_recorded = None ):
        if start_index + count > len( self.polygon_points ):
            new_size = int( len( self.polygon_points ) * self.EXTRA_CAPACITY_FACTOR )
            self.logger.debug(
                "Growing polygon points array from %d points capacity to %d." % (
                    len( self.polygon_points ), new_size,
                ),
            )
            self.polygon_points = np.resize(
                self.polygon_points, ( new_size, ),
            ).view( np.recarray )

        # Insert the point at the correct location in the polygon.
        next_point_index = self.polygon_points.next[ from_index ]
        polygon_index = self.polygon_points.polygon[ from_index ]
        self.polygon_points.next[ from_index ] = start_index
        self.polygon_points.next[ start_index ] = next_point_index
        self.polygon_points.polygon[ start_index ] = polygon_index

        self.polygons.count[ polygon_index ] += count

        self.outbox.send(
            request = "polygons_updated",
            points = points,
            polygon_points = self.polygon_points,
            polygons = self.polygons,
            polygon_count = self.polygon_add_index,
            updated_points = [ from_index ],
            projection = projection,
        )

    def points_updated( self, layer, points, projection, indices,
                        undo_recorded = None ):
        start_indices = [
            self.polygons.start[
                self.polygon_points.polygon[ point_index ]
            ] for point_index in indices
        ]

        self.outbox.send(
            request = "polygons_updated",
            points = points,
            polygon_points = self.polygon_points,
            polygons = self.polygons,
            polygon_count = self.polygon_add_index,
            updated_points = start_indices,
            projection = projection,
        )

    @staticmethod
    def make_polygon_points( point_count ):
        """
        Make a default polygon points array with the given number of rows.
        """
        point_count = int( point_count * Polygon_set_layer.EXTRA_CAPACITY_FACTOR )

        return np.repeat(
            np.array( [
                ( 0, 0 ),
            ], dtype = Polygon_set_layer.POLYGON_POINTS_DTYPE ),
            point_count,
        ).view( np.recarray )

    @staticmethod
    def make_polygons( count ):
        """
        Make a default polygons array with the given number of rows.
        """
        count = int( count * Polygon_set_layer.EXTRA_CAPACITY_FACTOR )

        return np.repeat(
            np.array( [
                (
                    0, 0, 0,
                    Polygon_set_layer.DEFAULT_POLYGON_COLOR,
                    0,
                ),
            ], dtype = Polygon_set_layer.POLYGONS_DTYPE ),
            count,
        ).view( np.recarray )
