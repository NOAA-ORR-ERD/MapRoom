import numpy as np
import logging
import maproomlib.utility as utility
from Point_set_layer import Point_set_layer
from Selected_line_set_layer import Selected_line_set_layer


class Line_set_layer:
    """
    A set of lines.

    :param command_stack: undo/redo stack
    :type command_stack: maproomlib.utility.Command_stack
    :param name: textual name of the layer
    :type name: str
    :param points_layer: set of points that the lines connect
    :type points_layer: plugin.Point_set_layer
    :param lines: array with one row per line segment
                  [ index of first point, index of second point,
                  RGBA color of segment, segment type flag ]
    :type lines: Nx4 numpy array or rec.array with dtype:
                 [ ( "point1", np.uint32 ), ( "point2", np.uint32 ),
                 ( "type", np.uint32 ), ( "color", np.uint32 ].
    :param line_count: number of lines that are actually initialized
                       (starting from index 0)
    :type line_count: int
    :param line_width: width of the given lines in pixels
    :type line_width: float

    .. attribute:: name

        name of the layer

    .. attribute:: children

        list of child layers

    .. function:: get_lines( origin, size, response_box )

        When a message with ``request = "get_lines"`` is received within
        the :attr:`inbox`, a handler sends the line data to the given
        ``response_box``.

        Note that at this time, the ``origin`` and ``size`` are ignored, and
        all line data is sent without any culling.

        :param origin: lower-left corner of desired data in geographic
                       coordinates
        :type origin: ( float, float )
        :param size: dimensions of desired data in geographic coordinates
        :type size: ( float, float )
        :param response_box: response message sent here
        :type response_box: Inbox

        The response message is sent as follows::

            response_box.send( request = "lines", points, lines, projection, origin, size )

        ``points`` is an Nx4 numpy rec.array with one row per point:
        [ point x, point y, point z, RGBA color of point ] and dtype
        [ ( "x", np.float32 ), ( "y", np.float32 ), ( "z", np.float32 ),
        ( "color", np.uint32 ) ].

        ``projection`` is the geographic projection of the provided points.

        ``lines`` is an Nx4 numpy rec.array with one row per line segment:
        [ index of start of line, RGBA color of line ] and dtype
        [ ( "point1", np.uint32 ), ( "point2", np.uint32 ),
        ( "type", np.uint32 ), ( "color", np.uint32 ].

        ``origin`` and ``size`` are the original values that were passed in to
        the ``get_lines`` request.

    .. function:: make_selection( object_indices, response_box )

        When a message with ``request = "make_selection"`` is received within
        the :attr:`inbox`, a handler constructs a new
        :class:`Selected_line_set_layer` with the lines at the selected index.

        :param object_indices: indices of the selected lines within this layer
        :type object_indices: list of int
        :param response_box: where to send the response message
        :type response_box: utility.Inbox

        The created layer is sent in a message to the ``response_box`` as
        follows::

            response_box.send( request = "selection", layer )

    .. function:: points_updated( layer, points, projection, indices ):

        Any received ``points_updated`` messages are simply forwarded on to
        the :attr:`outbox` with an additional ``point_map`` parameter added on.

    .. function:: points_deleted( layer, points, projection, indices ):

        Any received ``points_deleted`` messages are simply forwarded on to
        the :attr:`outbox` with an additional ``point_map`` parameter added on.

        Additionally, if possible, a line segment is added to fill in the hole
        left by the deleted point.

    .. function:: points_undeleted( layer, points, projection, indices ):

        Any received ``points_undeleted`` messages are simply forwarded on to
        the :attr:`outbox` with an additional ``point_map`` parameter added on.

        Additionally, if possible, the line segment that was added to fill in
        the hole left by the deleted point is removed.

    .. function:: points_added( layer, points, projection, start_index, count,
                                to_layer, to_index ):

        When a message with ``request = "points_added"`` is received within
        the :attr:`inbox`, a handler checks whether a line in this layer
        needs to be split by the last point of the added points, and if so,
        splits it.

        :param layer: layer to which points were added
        :type layer: Point_set_layer
        :param points: array with one row per geographic point added:
                       [ point x, point y, point z, RGBA color of point ]
        :type points: Nx4 numpy array or rec.array with dtype:
                      [ ( "x", np.float32 ), ( "y", np.float32 ),
                      ( "z", np.float32 ), ( "color", np.uint32 ) ]
        :param projection: geographic projection of the given points
        :type projection: pyproj.Proj
        :param start_index: index of first point added
        :type start_index: int
        :param count: number of points added, starting from ``start_index``
        :type count: int
        :param to_layer: layer containing the last point in ``points`` (if
                         any)
        :type to_layer: layer or NoneType
        :param to_index: index of this object within ``to_layer`` (if any)
        :type to_index: int or NoneType

        Note that if :attr:`to_layer` is not this layer, then no split occurs.

        The deleted line is sent in a messages to the :attr:`outbox` as
        follows::

            outbox.send( request = "lines_deleted", layer, indices )

        The two added lines are sent in a messages to the :attr:`outbox` as
        follows::

            outbox.send( request = "lines_added", layer, points, lines, projection, indices )

        The ``points_added`` message is also forwarded to the :attr:`outbox`
        with an additional ``point_map`` parameter added on.

    .. function:: line_points_added( layer, points, projection, selected_start_index, start_index, count, selected_end_index ):

        When a message with ``request = "line_points_added"`` is received
        within the :attr:`inbox`, a handler creates a series of new line
        segments connecting the given points.

        :param layer: layer to which points were added
        :type layer: Point_set_layer
        :param points: array with one row per geographic point added:
                       [ point x, point y, point z, RGBA color of point ]
        :type points: Nx4 numpy array or rec.array with dtype:
                      [ ( "x", np.float32 ), ( "y", np.float32 ),
                      ( "z", np.float32 ), ( "color", np.uint32 ) ]
        :param projection: geographic projection of the given points
        :type projection: pyproj.Proj
        :param selected_start_index: index of selected start point (existing
                                     point that starts the line)
        :type selected_start_index: int
        :param start_index: index of first point added
        :type start_index: int
        :param count: number of points added, starting from ``start_index``
        :type count: int
        :type selected_end_index: index of end point (existing point that
                                  ends the line, if any)
        :type selected_end_index: int or NoneType

        The added lines are sent in a messages to the :attr:`outbox` as
        follows::

            outbox.send( request = "lines_added", layer, points, lines, projection, indices )

    .. function:: get_properties( response_box, indices ):

        When a message with ``request = "get_properties"`` is received within
        the :attr:`inbox`, a handler sends the property data for this layer
        to the given ``response_box``.

        :param response_box: response message sent here
        :type response_box: Inbox
        :param indices: ignore
        :type indices: object

        The response message is sent as follows::

            response_box.send( request = "properties", properties )

        ``properties`` is a tuple of the properties for this layer.
    """

    PLUGIN_TYPE = "layer"
    LAYER_TYPE = "line_set"
    EXTRA_CAPACITY_FACTOR = 2.0   # extra capacity to reserve
    DEFAULT_LINE_COLOR = utility.color_to_int( 0, 0, 0, 1 )
    DEFAULT_LINE_TYPE = 0
    DEFAULT_LINE_WIDTH = 2.0
    SELECTED_LINE_COLOR = utility.color_to_int( 1, 0.6, 0, 0.75 )
    SELECTED_LINE_WIDTH = 10.0
    LINES_DTYPE = np.dtype( [    # One row per line segment.
        ( "point1", np.uint32 ), # Index of first point in line segment.
        ( "point2", np.uint32 ), # Index of second point in line segment.
        ( "type", np.uint32 ),   # Type flag of this line segment.
        ( "color", np.uint32 ),  # Color of this line segment.
    ] )
    LINES_POINTS_DTYPE = np.dtype( [
        ( "points", "2u4" ),
        ( "type", np.uint32 ),
        ( "color", np.uint32 ),
    ] )

    def __init__( self, command_stack, name, points_layer, lines, line_count,
                  line_width, default_line_color = None ):
        self.command_stack = command_stack
        self.name = name
        self.points_layer = points_layer
        self.projection = points_layer.projection
        self.lines = lines.view( np.recarray )
        self.default_line_color = \
            default_line_color or self.DEFAULT_LINE_COLOR

        # point index ->
        # [ index of containing line * 2 (+1 for 2nd point in segment), ... ]
        # This indexing scheme is a relatively simple way to map a point not
        # only to its containing line segments, but also to its respective
        # positions within those line segments (point1 or point2).
        self.point_map = {}

        self.add_index = line_count
        self.line_width = line_width
        self.children = []
        self.inbox = utility.Inbox()
        self.outbox = utility.Outbox()
        self.logger = logging.getLogger( __name__ )
        self.generate_point_map()

    def generate_point_map( self ):
        # Given an array of info about line segments, each of which contains
        # two point indices, generate a point to line map as described above.
        from operator import itemgetter
        from itertools import groupby, imap, chain

        lines_point1 = self.lines.point1
        lines_point2 = self.lines.point2
        add_index = self.add_index

        # 1. Create an iterator of tuples of the form:
        # ( point_index, doubled line_index )
        # 2. Use groupby() to convert that to tuples of the form:
        # ( point_index, [ ( point_index, doubled line_index ), ... ]
        # 3. Transform that into a list of tuples of the form:
        # ( point_index, [ doubled line_index, ... ]
        # 4. Convert the whole thing into a dict of the form:
        # { point_index: [ doubled line_index, ... ], ... }

        self.point_map = dict( [
            ( point_index, map( itemgetter( 1 ), doubled_line_indices ) )
            for ( point_index, doubled_line_indices )
            in groupby(
                sorted( chain(
                    imap(
                        lambda ( line_index, point_index ): \
                            ( point_index, line_index * 2 ),
                        enumerate( lines_point1[ : add_index ] )
                    ),
                    imap(
                        lambda ( line_index, point_index ): \
                            ( point_index, line_index * 2 + 1 ),
                        enumerate( lines_point2[ : add_index ] )
                    ),
                ) ),
                itemgetter( 0 ),
            )
        ] )

    def run( self, scheduler ):
        self.points_layer.outbox.subscribe(
            self.inbox,
            request = (
                "points_updated",
                "points_deleted",
                "points_undeleted",
                "points_added",
                "line_points_added",
                "size_changed",
            ),
        )

        while True:
            message = self.inbox.receive(
                request = (
                    "get_lines",
                    "get_point_map",
                    "get_point_range",
                    "make_selection",
                    "delete",
                    "undelete",
                    "points_updated",
                    "points_deleted",
                    "points_undeleted",
                    "points_added",
                    "line_points_added",
                    "line_points_deleted",
                    "size_changed",
                    "get_properties",
                    "move",
                ),
            )
            request = message.pop( "request" )

            if request == "get_lines":
                self.get_lines( **message )
            elif request == "get_point_map":
                self.get_point_map( **message )
            elif request == "get_point_range":
                self.get_point_range( **message )
            elif request == "make_selection":
                self.make_selection( scheduler, **message )
            elif request == "delete":
                self.delete( **message )
            elif request == "undelete":
                self.undelete( **message )
            elif request == "points_updated":
                message[ "request" ] = request
                message[ "point_map" ] = self.point_map
                self.outbox.send( **message )
            elif request == "points_deleted":
                self.points_deleted( **message )
                message[ "request" ] = request
                message[ "point_map" ] = self.point_map
                self.outbox.send( **message )
            elif request == "points_undeleted":
                self.points_undeleted( **message )
                message[ "request" ] = request
                message[ "point_map" ] = self.point_map
                self.outbox.send( **message )
            elif request == "points_added":
                self.points_added( **message )
                message[ "request" ] = request
                message[ "point_map" ] = self.point_map
                self.outbox.send( **message )
            elif request == "line_points_added":
                self.line_points_added( **message )
            elif request == "line_points_deleted":
                self.line_points_deleted( **message )
            elif request == "size_changed":
                message[ "request" ] = request
                self.outbox.send( **message )
            elif request == "get_properties":
                response_box = message.get( "response_box" )
                response_box.send(
                    request = "properties",
                    properties = (),
                )
            elif request == "move":
                pass

        self.points_layer.outbox.unsubscribe( self.inbox )

    def get_lines( self, origin, size, response_box ):
        self.points_layer.inbox.send(
            request = "get_points",
            origin = origin,
            size = size,
            response_box = self.inbox,
        )
        message = self.inbox.receive( request = "points" )

        response_box.send(
            request = "lines",
            points = message.get( "points" ),
            point_count = message.get( "count" ),
            lines = self.lines,
            line_count = self.add_index,
            projection = message.get( "projection" ),
            line_width = self.line_width,
            origin = origin,
            size = size,
            point_map = self.point_map,
        )

    def get_point_map( self, response_box ):
        response_box.send(
            request = "point_map",
            point_map = self.point_map,
        )

    def get_point_range( self, start_point_index, end_point_index,
                         response_box ):
        self.points_layer.inbox.send(
            request = "get_points",
            origin = None,
            size = None,
            response_box = self.inbox,
        )
        message = self.inbox.receive( request = "points" )
        points = message.get( "points" )

        indices1 = self.get_point_range_in_one_direction(
            start_point_index, end_point_index, points,
            default_direction = True,
        )
        indices2 = self.get_point_range_in_one_direction(
            start_point_index, end_point_index, points,
            default_direction = False,
        )

        # Default to the shortest non-null set of indices.
        if len( indices1 ) == 0:
            point_indices = tuple( indices2 )
        elif len( indices2 ) == 0:
            point_indices = tuple( indices1 )
        elif len( indices1 ) <= len( indices2 ):
            point_indices = tuple( indices1 )
        else:
            point_indices = tuple( indices2 )

        response_box.send(
            request = "point_range",
            point_indices = point_indices,
        )

    def get_point_range_in_one_direction( self, start_point_index,
                                          end_point_index, points,
                                          default_direction ):
        point_index = start_point_index
        point_indices = set( [ point_index ] )

        if default_direction:
            default_index = 0
            other_index = 1
        else:
            default_index = 1
            other_index = 0

        while True:
            adjacent = \
                self.points_in_common_lines(
                    points,
                    start_index = point_index,
                    count = 1,
                    allow_single_point = True,
                )

            # If there aren't any adjacent points on lines, bail.
            if adjacent[ 0 ] is None and adjacent[ 1 ] is None:
                return set()

            if adjacent[ default_index ] is not None and \
               adjacent[ default_index ] not in point_indices:
                point_index = adjacent[ default_index ]
            elif adjacent[ other_index ] is not None and \
               adjacent[ other_index ] not in point_indices:
                point_index = adjacent[ other_index ]
            # If we've already seen both of the adjacent points, bail.
            else:
                return set()

            # If we've reached the end point, success! Send the indices of the
            # points we've accumulated.
            if point_index == end_point_index:
                point_indices.remove( start_point_index )
                return point_indices
            
            point_indices.add( point_index )

    def make_selection( self, scheduler, object_indices, color, depth_unit,
                        response_box ):
        line_count = len( object_indices )
        lines = Selected_line_set_layer.make_lines(
            line_count,
        )

        for ( selected_index, line_index ) in enumerate( object_indices ):
            lines[ selected_index ] = self.lines[ line_index ]
            lines.color[ selected_index ] = color or self.SELECTED_LINE_COLOR

        s = "s" if line_count > 1 else ""

        selection = Selected_line_set_layer(
            self.command_stack,
            name = "Selected Line%s" % s,
            lines_layer = self,
            line_width = self.SELECTED_LINE_WIDTH,
            lines = lines,
            indices = object_indices,
            line_count = line_count,
            color = color or self.SELECTED_LINE_COLOR,
        )
        scheduler.add( selection.run )

        response_box.send(
            request = "selection",
            layer = selection,
        )

    def delete( self, indices = None, record_undo = True ):
        if indices is None or len( indices ) == 0:
            indices = range( 0, self.add_index )

        count = len( indices )
        deleted_lines = self.make_lines(
            count, exact = True, color = self.default_line_color,
        )

        # Delete lines by setting both of their point indices to the same value.
        for ( deleted_index, line_index ) in enumerate( indices ):
            deleted_lines[ deleted_index ] = self.lines[ line_index ]

            self.lines.point1[ line_index ] = 0
            self.lines.point2[ line_index ] = 0

            self.point_map[ deleted_lines.point1[ deleted_index ] ].remove(
                line_index * 2,
            )
            self.point_map[ deleted_lines.point2[ deleted_index ] ].remove(
                line_index * 2 + 1,
            )

        self.outbox.send(
            request = "lines_deleted",
            layer = self,
            indices = indices,
            undo_recorded = record_undo,
        )

        if record_undo is True:
            s = "s" if count > 1 else ""

            self.command_stack.inbox.send(
                request = "add",
                description = "Delete Line%s" % s,
                redo = lambda: self.inbox.send(
                    request = "delete",
                    indices = indices,
                    record_undo = False,
                ),
                undo = lambda: self.inbox.send(
                    request = "undelete",
                    lines = deleted_lines,
                    indices = indices,
                    record_undo = False,
                ),
            )

    def undelete( self, lines, indices, record_undo = True ):
        for ( deleted_index, line_index ) in enumerate( indices ):
            self.lines.point1[ line_index ] = lines.point1[ deleted_index ]
            self.lines.point2[ line_index ] = lines.point2[ deleted_index ]

            self.point_map.setdefault(
                lines.point1[ deleted_index ], list(),
            ).append( line_index * 2 )
            self.point_map.setdefault(
                lines.point2[ deleted_index ], list(),
            ).append( line_index * 2 + 1 )

        self.outbox.send(
            request = "lines_undeleted",
            layer = self,
            points = self.points_layer.points,
            lines = self.lines,
            projection = self.points_layer.projection,
            indices = indices,
        )

        if record_undo is True:
            s = "s" if count > 1 else ""

            self.command_stack.inbox.send(
                request = "add",
                description = "Undelete Lines%s" % s,
                redo = lambda: self.inbox.send(
                    request = "undelete",
                    lines = lines,
                    indices = indices,
                    record_undo = False,
                ),
                undo = lambda: self.inbox.send(
                    request = "delete",
                    indices = indices,
                    record_undo = False,
                ),
            )

    def add_lines( self, lines, description = None, record_undo = True ):
        count = len( lines )
        start_index = self.add_index

        if start_index + count > len( self.lines ):
            new_size = int( len( self.lines ) * self.EXTRA_CAPACITY_FACTOR )
            self.logger.debug(
                "Growing lines array from %d lines capacity to %d." % (
                    len( self.lines ), new_size,
                ),
            )
            self.lines = np.resize(
                self.lines, ( new_size, ),
            ).view( np.recarray )

        self.lines[ start_index: start_index + count ] = lines
        self.add_index += count

        for line_index in range( count ):
            doubled_index = ( start_index + line_index ) * 2

            self.point_map.setdefault(
                lines.point1[ line_index ], list(),
            ).append( doubled_index )

            self.point_map.setdefault(
                lines.point2[ line_index ], list(),
            ).append( doubled_index + 1 )

        self.outbox.send(
            request = "lines_added",
            layer = self,
            points = self.points_layer.points,
            lines = self.lines,
            projection = self.projection,
            indices = range( start_index, start_index + count ),
            undo_recorded = record_undo,
        )

        if record_undo is True:
            s = "s" if count > 1 else ""

            self.command_stack.inbox.send(
                request = "add",
                description = description or "Add Line%s" % s,
                redo = lambda: self.inbox.send(
                    request = "undelete",
                    lines = lines,
                    indices = range( start_index, start_index + count ),
                    record_undo = False,
                ),
                undo = lambda: self.inbox.send(
                    request = "delete",
                    indices = range( start_index, start_index + count ),
                    record_undo = False,
                ),
            )
    
    def points_in_common_lines( self, points, start_index, count,
                                allow_single_point = False ):
        """
        Given a point, find the line segments that contain it. Then, return a
        list of other point indices within those line segments.
        """
        # We only handle the common case of connecting two points after a
        # single point is deleted. If there's any other situation, or the
        # deleted point is in more than two lines, bail.
        if count != 1:
            return ( None, None )

        doubled_line_indices = self.point_map.get( start_index )
        if doubled_line_indices is None or len( doubled_line_indices ) == 0:
            return ( None, None )
        if len( doubled_line_indices ) == 1 and allow_single_point is False:
            return ( None, None )
        doubled_line_indices = list( doubled_line_indices )

        # Weed out lines that contain deleted points other than the one
        # that's at start_index. This is only necessary because we leave
        # mappings for deleted points within self.point_map (to make
        # undeletion easier).
        for doubled_index in list( doubled_line_indices ):
            point1_index = self.lines.point1[ doubled_index // 2 ]
            point2_index = self.lines.point2[ doubled_index // 2 ]

            if point1_index != start_index and \
               np.isnan( points.x[ point1_index ] ):
                doubled_line_indices.remove( doubled_index )

            if point2_index != start_index and \
               np.isnan( points.x[ point2_index ] ):
                doubled_line_indices.remove( doubled_index )

        # If we're not left with exactly two lines, bail.
        if len( doubled_line_indices ) != 2:
            if allow_single_point is False or \
               len( doubled_line_indices ) != 1:
                return ( None, None )

        # Determine the indices of the other points.
        start_line_index = doubled_line_indices[ 0 ] // 2
        if doubled_line_indices[ 0 ] % 2 == 0:
            point1 = self.lines.point2[ start_line_index ]
        else:
            point1 = self.lines.point1[ start_line_index ]

        if len( doubled_line_indices ) == 1:
            point2 = None
        else:
            end_line_index = doubled_line_indices[ 1 ] // 2
            if doubled_line_indices[ 1 ] % 2 == 0:
                point2 = self.lines.point2[ end_line_index ]
            else:
                point2 = self.lines.point1[ end_line_index ]

        return ( point1, point2 )

    def points_deleted( self, layer, points, projection, indices,
                        undo_recorded = None ):
        """
        When a single point is deleted, try to keep its containing line
        contiguous by stitching up the "hole" left in the line with a new line
        segment.
        """
        if len( indices ) != 1:
            return

        line = self.make_lines(
            1, exact = True, color = self.default_line_color,
        )
        ( point1_index, point2_index ) = \
            self.points_in_common_lines( points, indices[ 0 ], 1 )
        if point1_index is None or point2_index is None:
            return

        ( line.point1[ 0 ], line.point2[ 0 ] ) = \
            ( point1_index, point2_index )

        self.lines[ self.add_index: self.add_index + 1 ] = line

        self.point_map.setdefault(
            line.point1[ 0 ], list(),
        ).append( self.add_index * 2 )
        self.point_map.setdefault(
            line.point2[ 0 ], list(),
        ).append( self.add_index * 2 + 1 )

        self.add_index += 1

        self.outbox.send(
            request = "lines_added",
            layer = self,
            points = points,
            lines = self.lines,
            projection = projection,
            indices = ( self.add_index - 1, ),
            undo_recorded = undo_recorded,
        )

    def points_undeleted( self, layer, points, projection, indices ):
        """
        When a single point is undeleted, look for a line segment that was
        added during the point's original deletion. If found, remove it. This
        is essentially undoing what :meth:`points_deleted()` does above.
        """
        if len( indices ) != 1:
            return

        # Get the two other points within the line segments that the newly
        # undeleted point is in.
        ( point1, point2 ) = \
            self.points_in_common_lines( points, indices[ 0 ], 1 )

        # Determine whether those two points themselves share a common line
        # segment.
        line_indices1 = set(
            np.array( self.point_map.get( point1, list() ) ) // 2,
        )
        line_indices2 = set(
            np.array( self.point_map.get( point2, list() ) ) // 2,
        )
        common_indices = list(
            line_indices1.intersection( line_indices2 ),
        )

        # We're only interested if there's exactly one common line.
        if len( common_indices ) != 1:
            return

        self.delete( common_indices, record_undo = False )

    def points_added( self, layer, points, projection, start_index, count,
                      from_index = None, to_layer = None, to_index = None,
                      record_undo = True, undo_recorded = None ):
        if to_layer != self:
            return

        # Perform a split. Start by removing the line on which the last point
        # in points was added.
        deleted_line = self.lines[ to_index: to_index + 1 ].copy()
        self.lines.point1[ to_index ] = 0
        self.lines.point2[ to_index ] = 0
        self.point_map[ deleted_line.point1[ 0 ] ].remove( to_index * 2 )
        self.point_map[ deleted_line.point2[ 0 ] ].remove( to_index * 2 + 1 )

        self.outbox.send(
            request = "lines_deleted",
            layer = self,
            indices = ( to_index, ),
            undo_recorded = record_undo,
        )

        if self.add_index + 2 > len( self.lines ):
            new_size = int( len( self.lines ) * self.EXTRA_CAPACITY_FACTOR )
            self.logger.debug(
                "Growing lines array from %d lines capacity to %d." % (
                    len( self.lines ), new_size,
                ),
            )
            self.lines = np.resize(
                self.lines, ( new_size, ),
            ).view( np.recarray )

        # Now connect the point to the surrounding points with new line
        # segments, thereby completing the split.
        added_point_index = start_index + count - 1
        our_start_index = self.add_index

        lines = self.make_lines(
            2, exact = True, color = self.default_line_color,
        )

        lines.point1[ 0 ] = deleted_line.point1[ 0 ]
        lines.point2[ 0 ] = added_point_index
        lines.point1[ 1 ] = added_point_index
        lines.point2[ 1 ] = deleted_line.point2[ 0 ]
        self.lines[ self.add_index : self.add_index + 2 ] = lines

        self.point_map.setdefault(
            deleted_line.point1[ 0 ], list(),
        ).append( self.add_index * 2 )

        self.point_map.setdefault(
            added_point_index, list()
        ).append( self.add_index * 2 + 1 )

        self.add_index += 1

        self.point_map.setdefault(
            added_point_index, list()
        ).append( self.add_index * 2 )

        self.point_map.setdefault(
            deleted_line.point2[ 0 ], list(),
        ).append( self.add_index * 2 + 1 )

        self.add_index += 1
        
        self.outbox.send(
            request = "lines_added",
            layer = self,
            points = points,
            lines = self.lines,
            projection = projection,
            indices = ( our_start_index, our_start_index + 1 ),
            undo_recorded = record_undo,
        )

        if record_undo is True:
            def redo():
                self.inbox.send(
                    request = "delete",
                    indices = ( to_index, ),
                    record_undo = False,
                )
                self.inbox.send(
                    request = "undelete",
                    lines = lines,
                    indices = ( our_start_index, our_start_index + 1 ),
                    record_undo = False,
                )

            def undo():
                self.inbox.send(
                    request = "undelete",
                    lines = deleted_line,
                    indices = ( to_index, ),
                    record_undo = False,
                ),
                self.inbox.send(
                    request = "delete",
                    indices = ( our_start_index, our_start_index + 1 ),
                    record_undo = False,
                )

            self.command_stack.inbox.send(
                request = "add",
                description = "Split Line",
                redo = redo,
                undo = undo,
            )

    def line_points_added( self, layer, points, projection,
                           selected_start_index, start_index, count,
                           selected_end_index, record_undo = True,
                           undo_recorded = None ):
        our_start_index = self.add_index
        orig_count = count

        if our_start_index + count + \
           ( 1 if selected_end_index is not None else 0 ) > len( self.lines ):
            new_size = int( len( self.lines ) * self.EXTRA_CAPACITY_FACTOR )
            self.logger.debug(
                "Growing lines array from %d lines capacity to %d." % (
                    len( self.lines ), new_size,
                ),
            )
            self.lines = np.resize(
                self.lines, ( new_size, ),
            ).view( np.recarray )

        # Make a series of lines connecting the selected point to each of the
        # added points.
        if count > 0:
            lines = self.make_lines(
                count, exact = True, color = self.default_line_color,
            )

            lines.point1[ 0 ] = selected_start_index
            lines.point1[ 1: ] = np.arange( start_index, start_index + count - 1 )
            lines.point2 = np.arange( start_index, start_index + count )

            self.lines[ our_start_index : our_start_index + count ] = lines

            for line_index in \
                range( our_start_index, our_start_index + count ):

                index = line_index - our_start_index
                self.point_map.setdefault(
                    lines.point1[ index ], list(),
                ).append( line_index * 2 )
                self.point_map.setdefault(
                    lines.point2[ index ], list(),
                ).append( line_index * 2 + 1 )

            self.add_index += count

        # If an optional end index was given, then add one last line.
        # But only if there isn't already a line segment there.
        if selected_end_index is not None:
            start_lines = set( [
                index // 2 for index in
                self.point_map.get( selected_start_index, [] )
            ] )
            end_lines = set( [
                index // 2 for index in
                self.point_map.get( selected_end_index, [] )
            ] )

            # Only add a line segment if there isn't already one between the
            # start and end points.
            if not start_lines.intersection( end_lines ):
                line = self.make_lines(
                    1, exact = True, color = self.default_line_color,
                )

                line.point1[ 0 ] = selected_start_index
                line.point2[ 0 ] = selected_end_index
                self.lines[ self.add_index : self.add_index + 1 ] = line

                self.point_map.setdefault(
                    selected_start_index, list(),
                ).append( self.add_index * 2 )

                self.point_map.setdefault(
                    selected_end_index, list(),
                ).append( self.add_index * 2 + 1 )

                self.add_index += 1
                count += 1

        if count == 0:
            return

        self.outbox.send(
            request = "lines_added",
            layer = self,
            points = points,
            lines = self.lines,
            projection = projection,
            indices = range( our_start_index, our_start_index + count ),
            undo_recorded = record_undo,
        )

        if record_undo is True:
            s = "s" if count > 1 else ""

            self.command_stack.inbox.send(
                request = "add",
                description = "Add Line%s" % s,
                redo = lambda: self.inbox.send(
                    request = "line_points_added",
                    layer = layer,
                    points = points,
                    projection = projection,
                    selected_start_index = selected_start_index,
                    start_index = start_index,
                    count = orig_count,
                    selected_end_index = selected_end_index,
                    record_undo = False,
                ),
                undo = lambda: self.inbox.send(
                    request = "line_points_deleted",
                    layer = layer,
                    selected_start_index = selected_start_index,
                    start_index = start_index,
                    count = orig_count,
                    selected_end_index = selected_end_index,
                    record_undo = False,
                )
            )

    def line_points_deleted( self, layer, selected_start_index, start_index,
                             count, selected_end_index, record_undo = True,
                             undo_recorded = None ):
        if selected_start_index is None or selected_end_index is None or \
           count > 0:
            return

        start_line_indices = set(
            np.array(
                self.point_map.get( selected_start_index, list() )
            ) // 2,
        )
        end_line_indices = set(
            np.array(
                self.point_map.get( selected_end_index, list() )
            ) // 2,
        )
        common_indices = list(
            start_line_indices.intersection( end_line_indices ),
        )

        self.delete( common_indices, record_undo = False )

    @staticmethod
    def make_lines( count, exact = False, color = None ):
        """
        Make a default line segments array with the given number of rows, plus
        some extra capacity for future additions.
        """
        if not exact:
            count = int( count * Line_set_layer.EXTRA_CAPACITY_FACTOR )

        return np.repeat(
            np.array( [
                (
                    0,
                    0,
                    Line_set_layer.DEFAULT_LINE_TYPE,
                    color or Line_set_layer.DEFAULT_LINE_COLOR,
                ),
            ], dtype = Line_set_layer.LINES_DTYPE ),
            count,
        ).view( np.recarray )
