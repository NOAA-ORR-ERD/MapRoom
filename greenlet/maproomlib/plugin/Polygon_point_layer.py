import os.path
import logging
import functools
import pyproj
import numpy as np
from maproomlib.plugin.Composite_layer import Composite_layer
from maproomlib.plugin.Selection_layer import Selection_layer
from maproomlib.plugin.Point_set_layer import Point_set_layer
from maproomlib.plugin.Polygon_set_layer import Polygon_set_layer
from maproomlib.plugin.Bna_saver import Bna_saver


class Load_polygon_error( Exception ):
    """ 
    An error occurring when attempting to load a polygon file.
            
    :param message: the textual message explaining the error
    """
    def __init__( self, message = None ):
        Exception.__init__( self, message )


class Polygon_point_layer( Composite_layer ):
    """
    A set of data consisting of points and polygons that connect them.

    :param filename: full path of polygon file to open
    :type filename: str
    :param command_stack: undo/redo stack
    :type command_stack: maproomlib.utility.Command_stack
    :param plugin_loader: used to load the appropriate plugin for a file
    :type plugin_loader: maproomlib.utility.Plugin_loader
    :param parent: parent layer containing this layer (if any)
    :type parent: object or NoneType
    :param points_layer: points contained in this layer (optional)
    :type points_layer maproomlib.plugin.Point_set_layer
    :param polygons_layer: polygons contained in this layer (optional)
    :type polygons_layer: maproomlib.plugin.Polygon_set_layer
    :param saver: default saver to use when this layer is saved (optional)
    :type saver: callable
    :param name: name of this layer, if different than basename of filename
    :type saver: str

    .. attribute:: name

        name of the layer (derived from the filename)

    .. attribute:: projection

        geographic lat-long projection of the data (pyproj.Proj)

    .. attribute:: children

        list of child layers

    .. attribute:: origin

        lower-left corner of the points' bounding box in geographic
        coordinates

    .. attribute:: size

        dimensions of the points' bounding box in geographic coordinates
    """
    PLUGIN_TYPE = "vector_layer"
    DEFAULT_POINT_COUNT = 2000
    DEFAULT_POLYGON_COUNT = 1000
    POINTS_LAYER_NAME = None
    POLYGONS_LAYER_NAME = "Boundary polygons"

    def __init__( self, filename, command_stack, plugin_loader, parent,
                  points_layer = None, polygons_layer = None, saver = None,
                  name = None ):
        Composite_layer.__init__(
            self, command_stack, plugin_loader, parent,
            name if name else os.path.basename( filename ) if filename
            else "New BNA",
            saver = saver, supported_savers = ( Bna_saver, ),
        )
        self.filename = filename
        self.logger = logging.getLogger( __name__ )
        self.points_layer = points_layer
        self.polygons_layer = polygons_layer
        self.projection = pyproj.Proj( "+proj=latlong" )

        if points_layer is None:
            point_count = 0
            points = Point_set_layer.make_points(
                self.DEFAULT_POINT_COUNT,
            )

            self.points_layer = Point_set_layer(
                command_stack,
                self.POINTS_LAYER_NAME, points, point_count,
                Point_set_layer.DEFAULT_POINT_SIZE, self.projection,
            )

            polygon_count = 0
            polygons = Polygon_set_layer.make_polygons(
                self.DEFAULT_POLYGON_COUNT,
            )
            polygon_points = Polygon_set_layer.make_polygon_points(
                self.DEFAULT_POINT_COUNT,
            )

            self.polygons_layer = Polygon_set_layer(
                command_stack, self.POLYGONS_LAYER_NAME, self.points_layer,
                polygon_points, polygons, polygon_count,
            )

        # Don't show any points until a polygon is selected.
        self.points_layer.shown_indices = ()
        self.selection_layer = Selection_layer(
            self.command_stack, self.plugin_loader, self,
        )
        self.polygons_layer.selection_layer = self.selection_layer

        self.children = [
            self.polygons_layer,
            self.selection_layer,
            self.points_layer,
        ]

        for child in self.children:
            child.outbox.subscribe(
                self.inbox,
                request = (
                    "start_progress",
                    "end_progress",
                    "selection_updated",
                ),
            )

    def run( self, scheduler ):
        Composite_layer.run(
            self,
            scheduler,
            get_properties = self.get_properties,
            add_points = self.add_points,
            replace_selection = self.replace_selection,
            clear_selection = self.clear_selection,
            delete_selection = self.delete_selection,
            find_duplicates = self.find_duplicates,
            triangulate = self.triangulate,
        )

    def get_properties( self, response_box, indices = None ):
        response_box.send(
            request = "properties",
            properties = (
                self.name,
            )
        )

    def add_points( self, points, projection, to_layer = None,
                    to_index = None, layer = None ):
        # If there are any points selected or the user has just selected a
        # polygon, bail without adding a point.
        if len( self.selection_layer.children ) > 0 or \
           to_layer == self.polygons_layer:
            return

        self.points_layer.inbox.send(
            request = "add_points",
            points = points,
            projection = projection,
            to_layer = to_layer,
            to_index = to_index,
        )

    def replace_selection( self, **message ):
        if message.get( "layer" ) not in self.children:
            self.points_layer.inbox.send(
                request = "clear_shown",
            )

        self.outbox.send(
            request = "replace_selection",
            **message
        )

    def clear_selection( self, record_undo = True ):
        self.points_layer.inbox.send(
            request = "clear_shown",
        )

        self.outbox.send(
            request = "clear_selection",
            record_undo = record_undo,
        )

    def delete_selection( self ):
        # If nothing else is selected, then consider this a request to delete
        # the "selected" polygon. The polygon isn't really selected but simply
        # indicated as shown points.
        if len( self.selection_layer.children ) == 0:
            self.points_layer.inbox.send(
                request = "delete_shown",
            )

        self.outbox.send(
            request = "delete_selection",
        )

    def find_duplicates( self, distance_tolerance, depth_tolerance,
                         response_box, layer = None ):
        response_box.send(
            exception = NotImplementedError(
                "Polygon layers do not support finding duplicate points.",
            ),
        )

    def triangulate( self, transformer, response_box ):
        response_box.send(
            exception = NotImplementedError(
                "Polygon layers do not support triangulation.",
            ),
        )

    origin = property( lambda self: self.points_layer.origin )
    size = property( lambda self: self.points_layer.size )
