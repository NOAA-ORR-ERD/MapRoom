import bisect
import pyproj
import logging
import numpy as np
import maproomlib.utility as utility
from Selected_point_set_layer import Selected_point_set_layer


class Point_set_layer:
    """
    A set of points.

    :param command_stack: undo/redo stack
    :type command_stack: maproomlib.utility.Command_stack
    :param name: textual name of the layer
    :type name: str
    :param points: array with one row per geographic point:
                   [ point x, point y, point z, RGBA color of point ]
    :type points: Nx4 numpy array or rec.array with dtype:
                  [ ( "x", np.float32 ), ( "y", np.float32 ),
                  ( "z", np.float32 ), ( "color", np.uint32 ) ]
    :param point_count: number of points that are actually initialized
                        (starting from index 0)
    :type point_count: int
    :param point_size: size of the given points in pixels
    :type point_size: float
    :param projection: geographic projection of the given points
    :type projection: pyproj.Proj
    :param seams: list of point indices indicating where a new layer started
                  (only if the points were merged together from multiple
                  different layers, otherwise None)
    :type seams: list of ints or NoneType

    .. attribute:: name

        name of the layer

    .. attribute:: projection

        geographic projection of the points (pyproj.Proj)

    .. attribute:: children

        list of child layers

    .. function:: get_points( origin, size, response_box )

        When a message with ``request = "get_points"`` is received within the
        :attr:`inbox`, a handler sends the point data to the given
        ``response_box``.

        Note that at this time, the ``origin`` and ``size`` are ignored, and
        all point data is sent without any culling.

        :param origin: lower-left corner of desired data in geographic
                       coordinates
        :type origin: ( float, float )
        :param size: dimensions of desired data in geographic coordinates
        :type size: ( float, float )
        :param response_box: response message sent here
        :type response_box: Inbox

        The response message is sent as follows::

            response_box.send( request = "points", points, projection, origin, size )

        ``points`` is an Nx4 numpy rec.array with one row per point:
        [ point x, point y, point z, RGBA color of point ] and dtype
        [ ( "x", np.float32 ), ( "y", np.float32 ), ( "z", np.float32 ),
        ( "color", np.uint32 ) ].

        ``projection`` is the geographic projection of the provided points.

        ``origin`` and ``size`` are the original values that were passed
        in to the ``get_points`` request.

    .. function:: make_selection( object_indices, color, response_box )

        When a message with ``request = "make_selection"`` is received within
        the :attr:`inbox`, a handler constructs a new
        :class:`Selected_point_set_layer` with the points at the given
        indices.

        :param object_indices: indices of the selected points within this
                               layer
        :type object_indices: list of int
        :param color: color that the selected points should be (optional)
        :type color: int or NoneType
        :param response_box: where to send the response message
        :type response_box: utility.Inbox

        The created layer is sent in a message to the ``response_box`` as
        follows::

            response_box.send( request = "selection", layer )

    .. function:: make_flag( object_indices, color, response_box )

        When a message with ``request = "make_flag"`` is received within the
        :attr:`inbox`, a handler constructs a new
        :class:`Selected_point_set_layer` with the points at the given
        indices.

        :param object_indices: indices of the selected points within this
                               layer
        :type object_indices: list of int
        :param color: color that the flagged points should be (optional)
        :type color: int or NoneType
        :param response_box: where to send the response message
        :type response_box: utility.Inbox

        The created layer is sent in a message to the ``response_box`` as
        follows::

            response_box.send( request = "flag", layer )

    .. function:: move( movement_vector, start_index, count )

        When a message with ``request = "move"`` is received within the
        :attr:`inbox`, a handler shifts the position of some or all points
        in this layer.

        :param movement_vector: geographic vector used to shift the points,
                                in the same projection as the points
                                themselves
        :type movement_vector: ( float, float )
        :param start_index: index of first point to move (defaults to 0)
        :type start_index: int
        :param count: number of points to move, starting from ``start_index``
                      (defaults to rest of points)
        :type count: int

        The moved points are sent in a messages to the :attr:`outbox` as
        follows::

            outbox.send( request = "points_updated", layer, points, projection, indices )

        ``layer`` is this layer.

        ``points`` is a points array of just the updated points, of the same
        type as passed to :class:`Point_set_layer()`. ``projection`` is the
        geographic projection of the provided points. ``start_index`` is the
        index of the first moved point, and ``count`` is the number of points
        moved.

    .. function:: delete( start_index, count, record_undo = True )

        When a message with ``request = "delete"`` is received within the
        :attr:`inbox`, a handler deletes some or all points in this layer.

        :param start_index: index of first point to delete (defaults to 0)
        :type start_index: int
        :param count: number of points to delete, starting from
                      ``start_index`` (defaults to 1)
        :type count: int
        :param record_undo: whether to make this operation undoable (defaults
                            to True)
        :type record_undo: bool

        Note that deleting points is implemented simply by setting their x and
        y coordinates to NaN. This has the advantages of requiring no copies
        and keeping indices stable.

        The deleted points are sent in a messages to the :attr:`outbox` as
        follows::

            outbox.send( request = "points_deleted", layer, points, projection, indices )

        The parameters are just like those of the ``points_updated`` message
        below, except the points are those from this layer's own set of
        points, and ``projection`` is the geographic projection of these
        points.

    .. function:: undelete( points, start_index, count, record_undo = True )

        When a message with ``request = "undelete"`` is received within the
        :attr:`inbox`, a handler undeletes some or all points in this layer.

        :param points: array with one row per undeleted point, the original
                       data to restore:
                       [ point x, point y, point z, RGBA color of point ]
        :type points: Nx4 numpy array or rec.array with dtype:
                      [ ( "x", np.float32 ), ( "y", np.float32 ),
                      ( "z", np.float32 ), ( "color", np.uint32 ) ]
        :param start_index: index of first point to undelete
        :type start_index: int
        :param count: number of points to undelete, starting from
                      ``start_index``
        :type count: int
        :param record_undo: whether to make this operation undoable (defaults
                            to True)
        :type record_undo: bool

        The undeleted points are sent in a messages to the :attr:`outbox` as
        follows::

            outbox.send( request = "points_undeleted", layer, points, projection, indices )

        The parameters are just like those of the ``points_updated`` message
        below, except the points are those from this layer's own set of
        points, and ``projection`` is the geographic projection of these
        points.

    .. function:: add_points( points, projection, to_layer = None, to_index = None, record_undo = True )

        When a message with ``request = "add_points"`` is received within the
        :attr:`inbox`, a handler adds the given points to this layer.

        :param points: array with one row per geographic point to add:
                       [ point x, point y, point z, RGBA color of point ]
        :type points: Nx4 numpy array or rec.array with dtype:
                      [ ( "x", np.float32 ), ( "y", np.float32 ),
                      ( "z", np.float32 ), ( "color", np.uint32 ) ]
        :param projection: geographic projection of the given points
        :type projection: pyproj.Proj
        :param to_layer: layer containing the last point in ``points`` (if
                         any)
        :type to_layer: layer or NoneType
        :param to_index: index of last point in ``points`` within ``to_layer``
                         (if any)
        :type to_index: int or NoneType
        :param record_undo: whether to make this operation undoable (defaults
                            to True)
        :type record_undo: bool

        The added points are sent in a messages to the :attr:`outbox` as
        follows::

            outbox.send( request = "points_added", layer, points, projection, start_index, count )

        The parameters are just like those of the ``points_updated`` message
        below, except the points are those from this layer's own set of
        points, and ``projection`` is the geographic projection of these
        points.

    .. function:: add_lines( points, projection, to_layer = None, to_index = None, record_undo = True )

        When a message with ``request = "add_lines"`` is received within the
        :attr:`inbox`, a handler adds the given points to this layer.

        :param points: array with one row per geographic point to add:
                       [ point x, point y, point z, RGBA color of point ]
        :type points: Nx4 numpy array or rec.array with dtype:
                      [ ( "x", np.float32 ), ( "y", np.float32 ),
                      ( "z", np.float32 ), ( "color", np.uint32 ) ]
        :param projection: geographic projection of the given points
        :type projection: pyproj.Proj
        :param to_layer: layer containing the last point in ``points`` (if
                         any)
        :type to_layer: layer or NoneType
        :param to_index: index of last point in ``points`` within ``to_layer``
                         (if any)
        :type to_index: int or NoneType
        :param record_undo: whether to make this operation undoable (defaults
                            to True)
        :type record_undo: bool

        The added points are sent in a messages to the :attr:`outbox` as
        follows::

            outbox.send( request = "line_points_added", layer, points, projection, selected_start_index, start_index, count, selected_end_index )

        The parameters are just like those of the ``points_updated`` message
        below, except the points are those from this layer's own set of
        points, and ``projection`` is the geographic projection of these
        points. Also, ``selected_start_index`` is the index of the selected
        start point from which the line should start. And
        ``selected_end_index`` is the index of the existing point where the
        line should end (if any).

    .. function:: get_properties( response_box, indices ):

        When a message with ``request = "get_properties"`` is received within
        the :attr:`inbox`, a handler sends the property data for the given
        point `indices`` to the given ``response_box``.

        :param response_box: response message sent here
        :type response_box: Inbox
        :param indices: point indices within this layer
        :type indices: sequence

        The response message is sent as follows::

            response_box.send( request = "properties", properties )

        ``properties`` is a tuple of the properties for the requested point.
        If the point has a depth, then it will be the only property. If the
        point does not have a depth, then ``properties`` will be an empty
        tuple.
    """

    PLUGIN_TYPE = "layer"
    LAYER_TYPE = "point_set"
    EXTRA_CAPACITY_FACTOR = 2.0   # extra capacity to reserve
    DEFAULT_POINT_COLOR = utility.color_to_int( 0, 0, 0, 1 )
    DEFAULT_POINT_SIZE = 4.0
    SELECTED_POINT_COLOR = utility.color_to_int( 1, 0.6, 0, 0.75 )
    SELECTED_POINT_SIZE = 15.0
    POINTS_DTYPE = np.dtype( [
        ( "x", np.float32 ),
        ( "y", np.float32 ),
        ( "z", np.float32 ),
        ( "color", np.uint32 ),
    ] )
    POINTS_XY_DTYPE = np.dtype( [
        ( "xy", "2f4" ),
        ( "z", np.float32 ),
        ( "color", np.uint32 ),
    ] )
    POINTS_XYZ_DTYPE = np.dtype( [
        ( "xyz", "3f4" ),
        ( "color", np.uint32 ),
    ] )

    def __init__( self, command_stack, name, points, point_count, point_size,
                  projection, seams = None, origin = None, size = None,
                  default_point_color = None, shown_indices = None ):
        self.command_stack = command_stack
        self.name = name
        self.points = points.view( np.recarray )
        self.add_index = point_count
        self.point_size = point_size
        self.projection = projection
        self.seams = seams or []
        self.origin = origin
        self.size = size
        self.default_point_color = \
            default_point_color or self.DEFAULT_POINT_COLOR
        self.shown_indices = shown_indices
        self.shown_count = 0 if shown_indices is None else len( shown_indices )
        self.lines_layer = None

        self.children = []
        self.inbox = utility.Inbox()
        self.outbox = utility.Outbox()
        self.logger = logging.getLogger( __name__ )

    def run( self, scheduler ):
        while True:
            message = self.inbox.receive(
                request = (
                    "get_points",
                    "make_selection",
                    "make_flag",
                    "move",
                    "scale_depth",
                    "delete",
                    "undelete",
                    "add_points",
                    "add_lines",
                    "delete_lines",
                    "undelete_lines",
                    "get_properties",
                    "set_property",
                    "find_duplicates",
                    "merge_duplicates",
                    "change_shown",
                    "add_shown",
                    "clear_shown",
                    "delete_shown",
                    "close",
                ),
            )
            request = message.pop( "request" )

            if request == "get_points":
                self.get_points( **message )
            elif request == "make_selection":
                self.make_selection( scheduler, **message )
            elif request == "make_flag":
                self.make_flag( scheduler, **message )
            elif request == "move":
                self.move( **message )
            elif request == "scale_depth":
                self.scale_depth( **message )
            elif request == "delete":
                self.delete( **message )
            elif request == "undelete":
                self.undelete( **message )
            elif request == "add_points":
                self.add_points( **message )
            elif request == "add_lines":
                self.add_lines( **message )
            elif request == "delete_lines":
                self.delete_lines( **message )
            elif request == "undelete_lines":
                self.undelete_lines( **message )
            elif request == "get_properties":
                self.get_properties( **message )
            elif request == "set_property":
                self.set_property( **message )
            elif request == "find_duplicates":
                self.find_duplicates( scheduler, **message )
            elif request == "merge_duplicates":
                self.merge_duplicates( scheduler, **message )
            elif request == "change_shown":
                self.change_shown( **message )
            elif request == "add_shown":
                self.add_shown( **message )
            elif request == "clear_shown":
                self.change_shown( shown_indices = () )
            elif request == "delete_shown":
                if self.shown_count > 0:
                    self.delete( indices = self.shown_indices[ : self.shown_count ] )
                    self.change_shown( shown_indices = () )
            elif request == "close":
                self.outbox.send(
                    request = "points_nuked",
                    points = self.points,
                    projection = self.projection,
                )
                return

    def get_points( self, response_box, origin = None, size = None,
                    request = None ):
        response_box.send(
            request = request or "points",
            points = self.points,
            projection = self.projection,
            count = self.add_index,
            point_size = self.point_size,
            origin = self.origin,
            size = self.size,
            shown_indices = self.shown_indices,
            shown_count = self.shown_count,
        )

    def make_selection( self, scheduler, object_indices, color, depth_unit,
                        response_box ):
        point_count = len( object_indices )
        points = Selected_point_set_layer.make_points(
            point_count,
        )

        for ( selected_index, point_index ) in enumerate( object_indices ):
            points[ selected_index ] = self.points[ point_index ]
            points.color[ selected_index ] = color or self.SELECTED_POINT_COLOR

        s = "s" if point_count > 1 else ""

        selection = Selected_point_set_layer(
            self.command_stack,
            name = "Selected Point%s" % s,
            points_layer = self,
            point_size = self.SELECTED_POINT_SIZE,
            points = points,
            indices = object_indices,
            point_count = point_count,
            color = color or self.SELECTED_POINT_COLOR,
            depth_unit = depth_unit,
            lines_layer = self.lines_layer,
        )
        scheduler.add( selection.run )

        response_box.send(
            request = "selection",
            layer = selection,
        )

    def make_flag( self, scheduler, object_indices, color, response_box ):
        point_count = len( object_indices )
        points = Selected_point_set_layer.make_points(
            point_count, color = color or self.SELECTED_POINT_COLOR,
        )

        for ( selected_index, point_index ) in enumerate( object_indices ):
            points[ selected_index ] = self.points[ point_index ]
            points.color[ selected_index ] = color or self.SELECTED_POINT_COLOR

        s = "s" if point_count > 1 else ""

        flag = Selected_point_set_layer(
            self.command_stack,
            name = "Flagged Point%s" % s,
            points_layer = self,
            point_size = self.SELECTED_POINT_SIZE,
            points = points,
            indices = object_indices,
            point_count = point_count,
            color = color or self.SELECTED_POINT_COLOR,
        )
        scheduler.add( flag.run )

        response_box.send(
            request = "flag",
            layer = flag,
        )

    def move( self, movement_vector, indices, cumulative = False,
              record_undo = True ):
        points = self.points.view( self.POINTS_XY_DTYPE ).xy

        if cumulative is False:
            points[ indices ] += movement_vector
        else:
            self.update_dimensions()

        self.outbox.send(
            request = "points_updated",
            layer = self,
            points = self.points,
            projection = self.projection,
            indices = indices,
            undo_recorded = record_undo,
        )

        if record_undo is True and cumulative is True:
            reverse_movement_vector = (
                -movement_vector[ 0 ],
                -movement_vector[ 1 ],
            )
            s = "s" if len( indices ) > 1 else ""

            self.command_stack.inbox.send(
                request = "add",
                description = "Move Point%s" % s,
                redo = lambda: self.inbox.send(
                    request = "move",
                    movement_vector = movement_vector,
                    cumulative = False,
                    indices = indices,
                    record_undo = False,
                ),
                undo = lambda: self.inbox.send(
                    request = "move",
                    movement_vector = reverse_movement_vector,
                    cumulative = False,
                    indices = indices,
                    record_undo = False,
                ),
            )

    def update_dimensions( self ):
        self.origin = (
            np.nanmin( self.points.x[ : self.add_index ] ),
            np.nanmin( self.points.y[ : self.add_index ] ),
        )
        self.size = (
            np.nanmax( self.points.x[ : self.add_index ] ) - self.origin[ 0 ],
            np.nanmax( self.points.y[ : self.add_index ] ) - self.origin[ 1 ],
        )

        self.outbox.send(
            request = "size_changed",
            origin = self.origin,
            size = self.size,
        )

    def scale_depth( self, scale_factor, start_index = 0, count = None ):
        if count is None:
            count = len( self.points ) - start_index

        self.points.z[ start_index : start_index + count ] *= scale_factor

        self.outbox.send(
            request = "depths_updated",
            layer = self,
            points = self.points,
            projection = self.projection,
            start_index = start_index,
            count = count,
        )

    def delete( self, indices = None, record_undo = True ):
        if indices is None or len( indices ) == 0:
            indices = range( 0, self.add_index )

        count = len( indices )
        deleted_points = self.make_points(
            count, exact = True, color = self.default_point_color,
        )
        points_xy = self.points.view( self.POINTS_XY_DTYPE ).xy

        # Delete points by setting their coordinate values to NaN.
        for ( deleted_index, point_index ) in enumerate( indices ):
            deleted_points[ deleted_index ] = self.points[ point_index ]
            points_xy[ point_index : point_index + 1 ].fill( np.nan )

        self.update_dimensions()

        self.outbox.send(
            request = "points_deleted",
            layer = self,
            points = self.points,
            projection = self.projection,
            indices = indices,
            undo_recorded = record_undo,
        )

        if record_undo is True:
            s = "s" if count > 1 else ""

            self.command_stack.inbox.send(
                request = "add",
                description = "Delete Point%s" % s,
                redo = lambda: self.inbox.send(
                    request = "delete",
                    indices = indices,
                    record_undo = False,
                ),
                undo = lambda: self.inbox.send(
                    request = "undelete",
                    points = deleted_points,
                    indices = indices,
                    record_undo = False,
                ),
            )

    def undelete( self, points, indices, record_undo = True ):
        for ( deleted_index, point_index ) in enumerate( indices ):
            self.points[ point_index ] = points[ deleted_index ]

        self.update_dimensions()

        self.outbox.send(
            request = "points_undeleted",
            layer = self,
            points = self.points,
            projection = self.projection,
            indices = indices,
        )

        if record_undo is True:
            s = "s" if count > 1 else ""

            self.command_stack.inbox.send(
                request = "add",
                description = "Undelete Point%s" % s,
                redo = lambda: self.inbox.send(
                    request = "undelete",
                    points = points,
                    indices = indices,
                    record_undo = False,
                ),
                undo = lambda: self.inbox.send(
                    request = "delete",
                    indices = indices,
                    record_undo = False,
                ),
            )

    def add_points( self, points, projection, from_index = None,
                    to_layer = None, to_index = None, record_undo = True,
                    start_index = None, description = None ):
        # If the user is trying to add a point on top of an existing point,
        # refuse to do so.
        if to_layer and isinstance( to_layer, self.__class__ ):
            return

        count = len( points )
        start_index = self.add_index

        if start_index + count > len( self.points ):
            new_size = int( len( self.points ) * self.EXTRA_CAPACITY_FACTOR )
            self.logger.debug(
                "Growing points array from %d points capacity to %d." % (
                    len( self.points ), new_size,
                ),
            )
            self.points = np.resize(
                self.points, ( new_size, ),
            ).view( np.recarray )

        if projection.srs != self.projection.srs:
            ( points.x, points.y ) = pyproj.transform(
                projection, self.projection,
                points.x, points.y,
            )

        self.points[ start_index: start_index + count ] = points
        self.points.color[ start_index: start_index + count ] = \
            self.default_point_color
        self.add_index += count

        if from_index is not None:
            self.points.z[ start_index: start_index + count ] = \
                self.points.z[ from_index ]

        self.update_dimensions()

        if self.shown_indices != None:
            self.add_shown(
                list( range(
                    start_index,
                    start_index + count,
                ) ),
            )

        self.outbox.send(
            request = "points_added",
            layer = self,
            points = self.points,
            projection = self.projection,
            start_index = start_index,
            count = count,
            from_index = from_index,
            to_layer = to_layer,
            to_index = to_index,
            undo_recorded = record_undo,
        )

        if record_undo is True:
            s = "s" if count > 1 else ""

            self.command_stack.inbox.send(
                request = "add",
                description = description or "Add Point%s" % s,
                redo = lambda: self.inbox.send(
                    request = "undelete",
                    points = points,
                    indices = range( start_index, start_index + count ),
                    record_undo = False,
                ),
                undo = lambda: self.inbox.send(
                    request = "delete",
                    indices = range( start_index, start_index + count ),
                    record_undo = False,
                ),
            )

    def add_lines( self, points, projection, from_index, to_layer = None,
                   to_index = None, record_undo = True ):
        # Can't connect two points in different layers.
        if to_layer and to_layer != self:
            return

        # Only add line points if there's a lines layer.
        if self.lines_layer is None:
            return

        count = len( points )
        selected_start_index = from_index
        start_index = self.add_index

        if start_index + count > len( self.points ):
            new_size = int( len( self.points ) * self.EXTRA_CAPACITY_FACTOR )
            self.logger.debug(
                "Growing points array from %d points capacity to %d." % (
                    len( self.points ), new_size,
                ),
            )
            self.points = np.resize(
                self.points, ( new_size, ),
            ).view( np.recarray )

        if projection.srs != self.projection.srs:
            ( points.x, points.y ) = pyproj.transform(
                projection, self.projection,
                points.x, points.y,
            )

        if to_layer is None or to_index is None or to_layer != self:
            self.points[ start_index: start_index + count ] = points
            self.add_index += count
        else:
            # Since to_layer and to_index are set, that means the last point
            # is already in the points array. (The line is being connected to
            # an existing point.) So copy all points but the last one.
            count -= 1
            if count > 0:
                self.points[ start_index: start_index + count ] = points[ : -1 ]
                self.add_index += count

        self.points.color[ start_index: start_index + count ] = \
            self.default_point_color

        if selected_start_index is not None:
            self.points.z[ start_index: start_index + count ] = \
                self.points.z[ selected_start_index ]

        self.update_dimensions()

        self.outbox.send(
            request = "line_points_added",
            layer = self,
            points = self.points,
            projection = self.projection,
            selected_start_index = selected_start_index,
            start_index = start_index,
            count = count,
            selected_end_index = to_index,
            undo_recorded = record_undo,
        )

        if count == 0:
            return

        if record_undo is True:
            s = "s" if count > 1 else ""

            self.command_stack.inbox.send(
                request = "add",
                description = "Add Line%s" % s,
                redo = lambda: self.inbox.send(
                    request = "undelete_lines",
                    layer = self,
                    points = points,
                    selected_start_index = selected_start_index,
                    start_index = start_index,
                    count = count,
                    selected_end_index = to_index,
                    record_undo = False,
                ),
                undo = lambda: self.inbox.send(
                    request = "delete_lines",
                    layer = self,
                    selected_start_index = selected_start_index,
                    start_index = start_index,
                    count = count,
                    selected_end_index = to_index,
                    record_undo = False,
                )
            )

    def delete_lines( self, layer, selected_start_index, start_index, count,
                      selected_end_index, record_undo = True ):
        if count > 0:
            self.delete(
                range( start_index, start_index + count ),
                record_undo,
            )
        else:
            self.outbox.send(
                request = "line_points_deleted",
                layer = self,
                selected_start_index = selected_start_index,
                start_index = start_index,
                count = count,
                selected_end_index = selected_end_index,
                undo_recorded = record_undo,
            )

    def undelete_lines( self, layer, points, selected_start_index,
                        start_index, count, selected_end_index,
                        record_undo = True ):
        if count > 0:
            self.undelete(
                points,
                range( start_index, start_index + count ),
                record_undo,
            )
        else:
            self.outbox.send(
                request = "line_points_added",
                layer = self,
                points = self.points,
                projection = self.projection,
                selected_start_index = selected_start_index,
                start_index = start_index,
                count = count,
                selected_end_index = selected_end_index,
                undo_recorded = record_undo,
            )

    def get_properties( self, response_box, indices = None ):
        if not indices:
            response_box.send(
                request = "properties",
                properties = (),
            )
            return

        depth_value = self.points.z[ indices[ 0 ] ]

        if len( indices ) == 1:
            point_number = utility.Property(
                "Point number",
                indices[ 0 ] + 1,
                type = int,
                mutable = False,
            )
            if np.isnan( depth_value ):
                response_box.send(
                    request = "properties",
                    properties = (
                        point_number,
                    ),
                )
                return
        else:
            point_number = utility.Property(
                "Point numbers",
                [ index + 1 for index in indices ],
                type = list,
                mutable = False,
            )
            if np.isnan( depth_value ):
                response_box.send(
                    request = "properties",
                    properties = (
                        point_number,
                    ),
                )
                return

            # Don't report a single depth value for multiple points.
            depth_value = np.nan

        depth = utility.Property(
            "Depth",
            depth_value,
            type = np.float32,
            min = -999999,
            max = 999999,
            indices = indices,
        )

        response_box.send(
            request = "properties",
            properties = (
                depth,
                point_number,
            ),
        )

    def set_property( self, property, value, response_box = None,
                      record_undo = True ):
        if record_undo is True:
            orig_values = [ self.points.z[ index ] for index in property.indices ]

        try:
            if hasattr( value, "__iter__" ):
                if len( value ) == 1:
                    property.update( value[ 0 ] )
                    for index in property.indices:
                        self.points.z[ index ] = property.value
                else:
                    property.update( np.nan )
                    for ( one_value, index ) in zip( value, property.indices ):
                        self.points.z[ index ] = one_value
            else:
                property.update( value )
                for index in property.indices:
                    self.points.z[ index ] = property.value
        except ValueError, error:
            if response_box:
                response_box.send( exception = error )
            return

        if response_box:
            response_box.send(
                request = "property_updated",
                layer = self,
                property = property,
            )

        self.outbox.send(
            request = "property_updated",
            layer = self,
            property = property,
        )

        if record_undo is True:
            self.command_stack.inbox.send(
                request = "add",
                description = "Set Depth",
                redo = lambda: self.inbox.send(
                    request = "set_property",
                    property = property,
                    value = value,
                    record_undo = False,
                ),
                undo = lambda: self.inbox.send(
                    request = "set_property",
                    property = property,
                    value = orig_values,
                    record_undo = False,
                ),
            )

    def find_duplicates( self, scheduler, distance_tolerance, depth_tolerance,
                         response_box ):
        if self.add_index == 0:
            response_box.send(
                request = "duplicates",
                duplicates = [],
                layer = self,
            )
            return

        try:
            from scipy.spatial.ckdtree import cKDTree
        except ImportError, error:
            response_box.send( exception = error )
            return
        except ValueError, error:
            response_box.send( exception = ImportError( str( error ) ) )
            return

        points = self.points.view(
            self.POINTS_XY_DTYPE
        ).xy[ : self.add_index ].copy()

        latlong = pyproj.Proj( "+proj=latlong" )

        # If necessary, convert points to lat-long before find duplicates.
        # This makes the distance tolerance work properly.
        if self.projection.srs != latlong.srs:
            points = points.view(
                [ ( "x", np.float32 ), ( "y", np.float32 ) ]
            ).view( np.recarray )
            latlong_transformer = utility.Transformer( latlong )
            latlong_transformer.transform_many(
                points, points, self.projection, set_cache = False
            )

        # cKDTree doesn't handle NaNs gracefully, but it does handle infinity
        # values. So replace all the NaNs with infinity.
        points = points.view( np.float32 )
        points[ np.isnan( points ) ] = np.inf
        tree = cKDTree( points )

        ( _, indices_list ) = tree.query(
            points,
            2, # Number of points to return per point given.
            distance_upper_bound = distance_tolerance,
        )

        duplicates = set()

        for ( n, indices ) in enumerate( indices_list ):
            # cKDTree uses the point count to indicate a missing neighbor, so
            # filter out those values from the results.
            indices = [
                index for index in sorted( indices )
                if index != self.add_index
            ]

            if len( indices ) < 2:
                continue                

            # If this layer was merged from multiple different layers, and if
            # all point indices in the current list are from the same source
            # layer, then skip this list of duplicates.
            if self.seams:
                seams = set()
                for index in indices:
                    seams.add( bisect.bisect( self.seams, index ) )

                if len( seams ) < 2:
                    continue

            # Filter out points not within the depth tolerance from one
            # another. The depth_tolerance is a percentage.
            depth_0 = self.points.z[ indices[ 0 ] ]
            depth_1 = self.points.z[ indices[ 1 ] ]
            smaller_depth = min( abs( depth_0 ), abs( depth_1 ) ) or 1.0

            depth_difference = abs(
                ( depth_0 - depth_1 ) / smaller_depth
            ) * 100.0

            if depth_difference > depth_tolerance:
                continue

            duplicates.add( tuple( indices ) )

            if n % 100 == 0:
                scheduler.switch()

        response_box.send(
            request = "duplicates",
            duplicates = list( duplicates ),
            points = self.points,
            projection = self.projection,
            layer = self,
        )

    def merge_duplicates( self, scheduler, indices, points_in_lines ):
        points_to_delete = set()

        for ( point_0, point_1 ) in indices:
            point_0_in_line = point_0 in points_in_lines
            point_1_in_line = point_1 in points_in_lines

            # If each point in the pair is within a line, then skip it since
            # we don't know how to merge such points.
            if point_0_in_line and point_1_in_line:
                continue

            if np.isnan( self.points.x[ point_0 ] ) or \
               np.isnan( self.points.x[ point_1 ] ):
                continue

            # If only one of the points is within a line, then delete the
            # other point in the pair.
            if point_0_in_line:
                points_to_delete.add( point_1 )
            elif point_1_in_line:
                points_to_delete.add( point_0 )
            # Otherwise, arbitrarily delete one of the points
            else:
                points_to_delete.add( point_1 )

        if len( points_to_delete ) > 0:
            self.delete( list( points_to_delete ) )

    def change_shown( self, shown_indices, shown_count = 0 ):
        self.shown_indices = shown_indices[ :shown_count ]
        self.shown_count = shown_count

        self.outbox.send(
            request = "shown_updated",
            shown_indices = self.shown_indices,
            shown_count = shown_count,
        )

    def add_shown( self, add_shown_indices ):
        old_count = self.shown_count
        old_capacity = len( self.shown_indices )
        self.shown_count = old_count + len( add_shown_indices )

        if old_count == 0:
            self.shown_indices = np.empty(
                ( self.shown_count * self.EXTRA_CAPACITY_FACTOR, ),
                np.uint32,
            )
            self.shown_indices[ 0: self.shown_count ] = add_shown_indices
        elif self.shown_count > old_capacity:
            new_capacity = old_capacity * self.EXTRA_CAPACITY_FACTOR
            self.logger.debug(
                "Growing shown indices array from %d points capacity to %d." % (
                    old_capacity, new_capacity,
                ),
            )
            self.shown_indices = np.resize(
                self.shown_indices, ( new_capacity, ),
            )

        self.shown_indices[ old_count: ] = add_shown_indices

        self.outbox.send(
            request = "shown_updated",
            shown_indices = self.shown_indices,
            shown_count = self.shown_count,
        )

    @staticmethod
    def make_points( count, exact = False, color = None ):
        """
        Make a default points array with the given number of rows, plus some
        extra capacity for future additions.
        """
        if not exact:
            count = int( count * Point_set_layer.EXTRA_CAPACITY_FACTOR )

        return np.repeat(
            np.array( [
                (
                    np.nan, np.nan, np.nan,
                    color or Point_set_layer.DEFAULT_POINT_COLOR,
                ),
            ], dtype = Point_set_layer.POINTS_DTYPE ),
            count,
        ).view( np.recarray )
