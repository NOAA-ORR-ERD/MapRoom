import numpy as np
import logging
import maproomlib.utility as utility
from Point_set_layer import Point_set_layer


class Triangle_set_layer:
    """
    A set of triangles.

    :param command_stack: undo/redo stack
    :type command_stack: maproomlib.utility.Command_stack
    :param name: textual name of the layer
    :type name: str
    :param points_layer: set of points that the triangles connect
    :type points_layer: plugin.Point_set_layer
    :param triangles: array with one row per triangle
                  [ index of first point, index of second point,
                    index of third point ]
    :type triangles: Nx3 numpy array or rec.array with dtype:
                 [ ( "point1", np.uint32 ), ( "point2", np.uint32 ),
                   ( "point3", np.uint32 ) ].
    :param triangle_count: number of triangles that are actually initialized
                       (starting from index 0)
    :type triangle_count: int

    .. attribute:: name

        name of the layer

    .. attribute:: children

        list of child layers

    .. function:: get_triangles( response_box )

        When a message with ``request = "get_triangles"`` is received within
        the :attr:`inbox`, a handler sends the triangle data to the given
        ``response_box``.

        :param response_box: response message sent here
        :type response_box: Inbox

        The response message is sent as follows::

            response_box.send( request = "triangles", points, triangles, projection )

        ``points`` is an Nx4 numpy rec.array with one row per point:
        [ point x, point y, point z, RGBA color of point ] and dtype
        [ ( "x", np.float32 ), ( "y", np.float32 ), ( "z", np.float32 ),
        ( "color", np.uint32 ) ].

        ``projection`` is the geographic projection of the provided points.

        ``triangles`` is an Nx3 numpy rec.array with one row per triangle
        segment: [ index of first point, index of second point, index of third
        point ] and dtype [ ( "point1", np.uint32 ), ( "point2", np.uint32 ),
        ( "point3", np.uint32 ) ].
    """

    PLUGIN_TYPE = "layer"
    LAYER_TYPE = "triangle_set"
    EXTRA_CAPACITY_FACTOR = 2.0   # extra capacity to reserve
    TRIANGLES_DTYPE = np.dtype( [ # One row per triangle.
        ( "point1", np.uint32 ),  # Index of first point in triangle.
        ( "point2", np.uint32 ),  # Index of second point in triangle.
        ( "point3", np.uint32 ),  # Index of third point in triangle.
    ] )
    TRIANGLES_POINTS_DTYPE = np.dtype( [
        ( "points", "3u4" ),
    ] )

    def __init__( self, command_stack, name, points_layer, triangles,
                  triangle_count ):
        self.command_stack = command_stack
        self.name = name
        self.points_layer = points_layer
        self.projection = points_layer.projection
        self.triangles = triangles.view( np.recarray )
        self.add_index = triangle_count
        self.children = []
        self.inbox = utility.Inbox()
        self.outbox = utility.Outbox()
        self.logger = logging.getLogger( __name__ )

    def run( self, scheduler ):
        while True:
            message = self.inbox.receive(
                request = (
                    "get_triangles",
                ),
            )
            request = message.pop( "request" )

            if request == "get_triangles":
                self.get_triangles( **message )

    def get_triangles( self, response_box, request = None ):
        self.points_layer.inbox.send(
            request = "get_points",
            origin = None,
            size = None,
            response_box = self.inbox,
        )
        message = self.inbox.receive( request = "points" )

        response_box.send(
            request = request or "triangles",
            points = message.get( "points" ),
            point_count = message.get( "count" ),
            triangles = self.triangles,
            triangle_count = self.add_index,
            projection = message.get( "projection" ),
        )

    @staticmethod
    def make_triangles( count, exact = False ):
        """
        Make a default triangles array with the given number of rows, plus
        some extra capacity for future additions.
        """
        if not exact:
            count = int( count * Triangle_set_layer.EXTRA_CAPACITY_FACTOR )

        return np.repeat(
            np.array( [
                ( 0, 0, 0 ),
            ], dtype = Triangle_set_layer.TRIANGLES_DTYPE ),
            count,
        ).view( np.recarray )
