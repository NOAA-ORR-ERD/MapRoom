import os.path
import bisect
import pyproj
import random
import logging
import functools
import collections
import numpy as np
import maproomlib.utility.hazpy.unit_conversion as unit_conversion
import maproomlib.utility as utility
from maproomlib.plugin.Composite_layer import Composite_layer
from maproomlib.plugin.Selection_layer import Selection_layer
from maproomlib.plugin.Flag_layer import Flag_layer
from maproomlib.plugin.Point_set_layer import Point_set_layer
from maproomlib.plugin.Line_set_layer import Line_set_layer
from maproomlib.plugin.Triangle_set_layer import Triangle_set_layer
from maproomlib.plugin.Polygon_point_layer import Polygon_point_layer
from maproomlib.plugin.Polygon_set_layer import Polygon_set_layer
from maproomlib.plugin.Depth_label_set_layer import Depth_label_set_layer
from maproomlib.plugin.Verdat_saver import Verdat_saver
from maproomlib.plugin.Maproom_saver import Maproom_saver


class Triangulation_error( Exception ):
    """ 
    An error occuring when attempting to triangulate a line point layer.
            
    :param message: the textual message explaining the error
    """
    def __init__( self, message = None ):
        Exception.__init__( self, message )


Contour_info = collections.namedtuple(
    "Contour_info",
    "name color metadata_tag",
)


class Line_point_layer( Composite_layer ):
    """
    A set of data consisting of points and line segments that connect (some
    of) those points.

    :param filename: full path of verdat file to open (optional)
    :type filename: str or NoneType
    :param command_stack: undo/redo stack
    :type command_stack: maproomlib.utility.Command_stack
    :param plugin_loader: used to load the appropriate plugin for a file
    :type plugin_loader: maproomlib.utility.Plugin_loader
    :param parent: parent layer containing this layer (if any)
    :type parent: object or NoneType
    :param points_layer: layer to use for points (optional, ignored if
                         filename is given)
    :type points_layer: Point_set_layer or NoneType
    :param lines_layer: layer to use for lines (optional, ignored if
                        filename is given)
    :type lines_layer: Line_set_layer or NoneType
    :param depth_unit: unit to use for point depth (optional, defaults to
                       "unknown", ignored if filename is given )
    :type depth_unit: str or NoneType

    .. attribute:: name

        name of the layer (derived from the filename)

    .. attribute:: projection

        geographic projection of the data (pyproj.Proj)

    .. attribute:: children

        list of child layers

    .. function:: get_properties( response_box, indices ):

        When a message with ``request = "get_properties"`` is received within
        the :attr:`inbox`, a handler sends the property data for this layer
        to the given ``response_box``.

        :param response_box: response message sent here
        :type response_box: Inbox
        :param indices: ignored
        :type indices: object

        The response message is sent as follows::

            response_box.send( request = "properties", properties )

        ``properties`` is a tuple of the properties for this layer.

    .. attribute:: origin

        lower-left corner of the points' bounding box in geographic
        coordinates

    .. attribute:: size

        dimensions of the points' bounding box in geographic coordinates
    """
    PLUGIN_TYPE = "vector_layer"
    DEPTH_UNITS = [ "meters", "feet", "fathoms", "unknown" ]
    DEFAULT_POINT_COUNT = 2000
    DEFAULT_LINE_COUNT = 1000
    DEFAULT_COLORS = [
        utility.color_to_int( 0, 0, 1.0, 1 ),
        utility.color_to_int( 0, 0.75, 0, 1 ),
        utility.color_to_int( 0.5, 0, 1.0, 1 ),
        utility.color_to_int( 1.0, 0.5, 0, 1 ),
    ]
    next_default_color_index = 0
    POINTS_LAYER_NAME = None
    LINES_LAYER_NAME = "Boundary lines"
    TRIANGLES_LAYER_NAME = "Bathymetry triangles"
    TRIANGULATION_PREFIX = "Triangulation of"
    DEFAULT_LABELS = object()

    def __init__( self, filename, command_stack, plugin_loader, parent,
                  points_layer = None, lines_layer = None,
                  depth_unit = None, origin = None, size = None,
                  default_depth = 1, saver = None, triangles_layer = None,
                  name = None, labels_layer = DEFAULT_LABELS,
                  hide_labels = True ):
        Composite_layer.__init__(
            self, command_stack, plugin_loader, parent,
            name if name else os.path.basename( filename ) if filename else
                "New Verdat",
            child_subscribe_requests = [ "property_updated" ],
            saver = saver,
            supported_savers = ( Verdat_saver, Maproom_saver ),
        )

        self.filename = filename
        self.logger = logging.getLogger( __name__ )

        if points_layer:
            self.projection = points_layer.projection
        else:
            self.projection = pyproj.Proj( "+proj=latlong" )

        self.default_depth = utility.Property(
            "Default depth",
            default_depth,
            type = np.float32,
            min = -999999,
            max = 999999,
        )

        color = self.DEFAULT_COLORS[
            Line_point_layer.next_default_color_index
        ]

        Line_point_layer.next_default_color_index = (
            Line_point_layer.next_default_color_index + 1
        ) % len( self.DEFAULT_COLORS )

        if points_layer is None:
            point_count = 0
            points = Point_set_layer.make_points( self.DEFAULT_POINT_COUNT )
            points.color = color
            self.points_layer = Point_set_layer(
                self.command_stack,
                self.POINTS_LAYER_NAME, points, point_count,
                Point_set_layer.DEFAULT_POINT_SIZE, self.projection,
                origin = origin, size = size,
                default_point_color = color,
            )
        else:
            point_count = points_layer.add_index
            self.points_layer = points_layer

        if lines_layer is None:
            line_count = 0
            lines = Line_set_layer.make_lines( self.DEFAULT_LINE_COUNT )
            lines.color = color
            self.lines_layer = Line_set_layer(
                self.command_stack,
                self.LINES_LAYER_NAME, self.points_layer, lines, line_count,
                Line_set_layer.DEFAULT_LINE_WIDTH,
                default_line_color = color,
            )
            self.points_layer.lines_layer = self.lines_layer
        else:
            line_count = lines_layer.add_index
            self.lines_layer = lines_layer

        self.triangles_layer = triangles_layer

        if depth_unit is None:
            depth_unit = "unknown"

        try:
            self.depth_unit = utility.Property(
                "Depth unit",
                depth_unit,
                choices = self.DEPTH_UNITS,
            )
        except ValueError:
            raise utility.Load_plugin_error(
                'Invalid depth unit "%s".' % depth_unit
            )

        self.point_count = utility.Property(
            "Point count",
            point_count,
            type = int,
            mutable = False,
        )

        # Flag layer: Works independently of the user's selection.
        self.flag_layer = Flag_layer(
            self.command_stack, self.plugin_loader,
        )

        # Standard selection layer: Changes what's selected in response to
        # selection messages.
        self.selection_layer = Selection_layer(
            self.command_stack, self.plugin_loader, parent = self,
            depth_unit = depth_unit,
        )
        if labels_layer == self.DEFAULT_LABELS:
            self.labels_layer = Depth_label_set_layer(
                "Depth labels", self.points_layer, self.command_stack,
            )
        else:
            self.labels_layer = labels_layer

        self.children = [ child for child in [
            self.flag_layer,
            self.selection_layer,
            self.lines_layer,
            self.labels_layer,
            self.points_layer,
        ] if child is not None ]

        if self.triangles_layer is not None:
            self.children.insert( 2, self.triangles_layer )

        if self.labels_layer and hide_labels and point_count > 0:
            self.hidden_children.add( 
                self.labels_layer,
            )

        requests = (
            "start_progress",
            "end_progress",
            "selection_updated",
            "flags_updated",
            "property_updated",
        )

        self.points_layer.outbox.subscribe(
            self.inbox,
            request = (
                "points_added",
                "points_updated",
                "points_deleted",
                "points_nuked",
                "line_points_added",
                "line_points_deleted",
            ) + requests,
        )

        self.lines_layer.outbox.subscribe(
            self.inbox,
            request = (
                "lines_added",
                "lines_deleted",
            ) + requests,
        )

        for child in self.children:
            if child in ( self.points_layer, self.lines_layer ):
                continue
            child.outbox.subscribe(
                self.inbox,
                request = requests,
            )

    def run( self, scheduler ):
        Composite_layer.run(
            self,
            scheduler,
            get_properties = self.get_properties,
            add_points = self.add_points,
            replace_selection = self.replace_selection,
            property_updated = self.property_updated,
            find_duplicates = self.find_duplicates,
            triangulate = functools.partial( self.triangulate, scheduler ),
            contour = functools.partial( self.contour, scheduler ),
            merge = self.merge,
            points_added = self.delete_triangles,
            points_updated = self.delete_triangles,
            points_deleted = self.delete_triangles,
            line_points_added = self.delete_triangles,
            line_points_deleted = self.delete_triangles,
            lines_added = self.delete_triangles,
            lines_deleted = self.delete_triangles,
        )

    def get_properties( self, response_box, indices = None ):
        # Only update the point count on demand: Whenever the properties are
        # requested.
        self.points_layer.inbox.send(
            request = "get_points",
            response_box = self.inbox,
        )
        message = self.inbox.receive(
            request = "points",
        )

        points = message.get( "points" )
        count = message.get( "count" )

        for index in range( 0, count ):
            if np.isnan( points.x[ index ] ):
                count -= 1

        self.point_count.value = count

        response_box.send(
            request = "properties",
            properties = (
                self.name,
                self.depth_unit,
                self.default_depth,
                self.point_count,
            )
        )

    def add_points( self, points, projection, to_layer = None,
                    to_index = None, layer = None ):
        # If something's selected, then let the selection layer handle adding
        # the points. This prevents duplicate points from being added.
        if len( self.selection_layer.children ) > 0:
            return

        points = points.copy()
        points.z = self.default_depth.value

        self.points_layer.inbox.send(
            request = "add_points",
            points = points,
            projection = projection,
            to_layer = to_layer,
            to_index = to_index,
        )

    def replace_selection( self, layer, object_indices, record_undo = True ):
        if layer == self.points_layer and object_indices:
            self.points_layer.inbox.send(
                request = "get_points",
                response_box = self.inbox,
            )
            message = self.inbox.receive(
                request = "points",
            )
            self.default_depth.value = message.get( "points" ).z[
                object_indices[ -1 ]
            ]

        self.outbox.send(
            request = "replace_selection",
            layer = layer,
            object_indices = object_indices,
            record_undo = record_undo,
        )

    def property_updated( self, layer, property ):
        if layer == self.points_layer and property.name == "Depth":
            self.default_depth.value = property.value

    def find_duplicates( self, distance_tolerance, depth_tolerance,
                         response_box, layer = None ):
        unique_id = "find_duplicates %s" % self.name
        self.outbox.send(
            request = "start_progress",
            id = unique_id,
            message = "Finding duplicate points in %s" % self.name,
        )

        self.points_layer.inbox.send(
            request = "find_duplicates",
            distance_tolerance = distance_tolerance,
            depth_tolerance = depth_tolerance,
            response_box = self.inbox,
        )

        try:
            message = self.inbox.receive( request = "duplicates" )
        except ImportError, error:
            self.outbox.send(
                request = "end_progress",
                id = unique_id,
            )
            response_box.send( exception = error )         
            return

        duplicates = message.get( "duplicates" )

        self.lines_layer.inbox.send(
            request = "get_point_map",
            response_box = self.inbox,
        )

        message = self.inbox.receive( request = "point_map" )
        point_map = message.get( "point_map" )

        response_box.send(
            request = "duplicates",
            duplicates = duplicates,
            layer = self.points_layer,
            flag_layer = self.flag_layer,
            points_in_lines = set( point_map.keys() ),
        )

        self.outbox.send(
            request = "end_progress",
            id = unique_id,
        )

    def triangulate( self, scheduler, q, a, transformer, response_box ):
        from pytriangle import triangulate_simple
        from maproomlib.utility.Shape import point_in_polygon

        unique_id = "triangulate %s" % self.name
        self.outbox.send(
            request = "start_progress",
            id = unique_id,
            message = "Triangulating"
        )

        # Get the points and lines from the one merged layer.
        self.lines_layer.inbox.send(
            request = "get_lines",
            origin = None,
            size = None,
            response_box = transformer( self.inbox ),
        )
        message = self.inbox.receive(
            request = "lines",
        )

        orig_points = message.get( "points" )
        orig_point_count = message.get( "point_count" )
        orig_lines = message.get( "lines" )
        orig_line_count = message.get( "line_count" )

        # Determine the boundaries in this layer.
        try:
            ( boundaries, non_boundary_points ) = \
                utility.find_boundaries(
                    points = orig_points,
                    point_count = orig_point_count,
                    lines = orig_lines,
                    line_count = orig_line_count,
                )
        except utility.Find_boundaries_error, error:
            self.outbox.send(
                request = "end_progress",
                id = unique_id,
            )
            response_box.send(
                exception = Triangulation_error( str( error ) ),
            )
            return

        # Calculate a hole point for each boundary.
        hole_points_xy = np.empty(
            ( len( boundaries ), 2 ), np.float32,
        )

        MAX_SEARCH_COUNT = 10000

        def generate_inside_hole_point( boundary ):
            boundary_size = len( boundary )
            inside = False
            search_count = 0

            while inside is False:
                # Pick three random boundary points and take the average of
                # their coordinates.
                ( point1, point2, point3 ) = random.sample( boundary, 3 )

                candidate_x = \
                    ( orig_points.x[ point1 ] + orig_points.x[ point2 ] + \
                      orig_points.x[ point3 ] ) / 3.0
                candidate_y = \
                    ( orig_points.y[ point1 ] + orig_points.y[ point2 ] + \
                      orig_points.y[ point3 ] ) / 3.0

                inside = point_in_polygon(
                    points_x = orig_points.x,
                    points_y = orig_points.y,
                    point_count = orig_point_count,
                    polygon = np.array( boundary, np.uint32 ),
                    x = candidate_x,
                    y = candidate_y,
                )

                search_count += 1
                if search_count > MAX_SEARCH_COUNT:
                    raise Triangulation_error(
                        "Cannot find a boundary hole for triangulation."
                )

            return ( candidate_x, candidate_y )

        def generate_outside_hole_point( boundary ):
            boundary_size = len( boundary )
            inside = True
            search_count = 0

            while inside is True:
                # Pick two consecutive boundary points, take the average of
                # their coordinates, then perturb randomly.
                point1 = random.randint( 0, boundary_size - 1 )
                point2 = ( point1 + 1 ) % boundary_size 

                candidate_x = \
                    ( orig_points.x[ point1 ] + orig_points.x[ point2 ] ) \
                    / 2.0
                candidate_y = \
                    ( orig_points.y[ point1 ] + orig_points.y[ point2 ] ) \
                    / 2.0

                candidate_x *= random.random() + 0.5
                candidate_y *= random.random() + 0.5

                inside = point_in_polygon(
                    points_x = orig_points.x,
                    points_y = orig_points.y,
                    point_count = orig_point_count,
                    polygon = np.array( boundary, np.uint32 ),
                    x = candidate_x,
                    y = candidate_y,
                )

                search_count += 1
                if search_count > MAX_SEARCH_COUNT:
                    raise Triangulation_error(
                        "Cannot find an outer boundary hole for triangulation."
                )

            return ( candidate_x, candidate_y )

        try:
            for ( boundary_index, ( boundary, area ) ) in enumerate( boundaries ):
                if len( boundary ) < 3: continue

                # The "hole" point for the outer boundary (first in the list)
                # should be outside of it.
                if boundary_index == 0:
                    hole_points_xy[ boundary_index ] = \
                        generate_outside_hole_point( boundary )
                else:
                    hole_points_xy[ boundary_index ] = \
                        generate_inside_hole_point( boundary )
        except Triangulation_error, error:
            self.outbox.send(
                request = "end_progress",
                id = unique_id,
            )
            response_box.send( exception = error )
            return

        # Triangle will crash or hang if given NaN data, so filter out points
        # with NaN coordinates and the line segments that contain them.
        points_xy = orig_points.view(
            Point_set_layer.POINTS_XY_DTYPE
        ).xy[ : orig_point_count ].view( np.float32 ).copy()
        points_z = orig_points.z[ : orig_point_count ].copy()
        lines_points = orig_lines.view(
            Line_set_layer.LINES_POINTS_DTYPE,
        ).points[ : orig_line_count ].view( np.uint32 ).copy()

        # First filter NaNs from the point data.
        nans = np.any( np.isnan( points_xy ), axis = 1 )
        nanless = np.logical_not( nans )
        nanless_points_xy = points_xy[ nanless ]
        nanless_points_z = points_z[ nanless ]
        nan_indices = np.nonzero( nans )[ 0 ]
        nan_index_count = len( nan_indices )

        # Then filter points containing NaNs from the line segments, adjusting
        # point indices accordingly.
        nanless_lines_points = np.empty(
            lines_points.shape,
            lines_points.dtype,
        )
        nanless_line_index = 0

        for line_index in xrange( 0, orig_line_count ):
            ( point1_index, point2_index ) = \
                lines_points[ line_index ]

            previous_nan_counts = (
                bisect.bisect_left( nan_indices, point1_index ),
                bisect.bisect_left( nan_indices, point2_index ),
            )

            # If this line segment contains a NaN point, skip the line
            # segment.
            if previous_nan_counts[ 0 ] < nan_index_count and \
               nan_indices[ previous_nan_counts[ 0 ] ] == point1_index:
                continue
            if previous_nan_counts[ 1 ] < nan_index_count and \
               nan_indices[ previous_nan_counts[ 1 ] ] == point2_index:
                continue

            nanless_lines_points[ nanless_line_index ] = \
                lines_points[ line_index ] - previous_nan_counts

            nanless_line_index += 1

        nanless_lines_points = nanless_lines_points[ : nanless_line_index ]

        # Triangulate!
        try:
            params = "V"
            if ( q is not None ):
                params = params + "q" + str( q )
            if ( a is not None ):
                params = params + "a" + str( a )
            print params
            ( triangle_points_xy, triangle_points_z, line_segments, triangles ) = \
                triangulate_simple(
                    params,
                    nanless_points_xy,
                    nanless_points_z,
                    nanless_lines_points,
                    hole_points_xy,
                )
        except ( RuntimeError, EOFError ):
            self.outbox.send(
                request = "end_progress",
                id = unique_id,
            )

            error = Triangulation_error(
                "An error occurred during triangulation.",
            )
            response_box.send( exception = error )

            return

        # Store the newly triangulated triangle data as a new layer.
        point_count = len( triangle_points_xy )
        points = Point_set_layer.make_points( point_count )
        points.view( Point_set_layer.POINTS_XY_DTYPE ).xy[
            0: point_count
        ] = triangle_points_xy
        points.z[ 0: point_count ] = triangle_points_z

        line_count = len( line_segments )
        lines = Line_set_layer.make_lines( line_count )
        lines.view( Line_set_layer.LINES_POINTS_DTYPE ).points[
            0: line_count
        ] = line_segments

        projection = transformer.projection
        color = self.points_layer.default_point_color

        points.color = color
        lines.color = color

        points_layer = Point_set_layer(
            self.command_stack,
            Line_point_layer.POINTS_LAYER_NAME, points, point_count,
            Point_set_layer.DEFAULT_POINT_SIZE, transformer.projection,
            default_point_color = color,
        )

        lines_layer = Line_set_layer(
            self.command_stack,
            Line_point_layer.LINES_LAYER_NAME, points_layer, lines,
            line_count, Line_set_layer.DEFAULT_LINE_WIDTH,
            default_line_color = color,
        )

        points_layer.lines_layer = lines_layer

        triangles_layer = Triangle_set_layer(
            self.command_stack, self.TRIANGLES_LAYER_NAME, points_layer,
            triangles, triangle_count = len( triangles ),
        )

        if str( self.name ).startswith( self.TRIANGULATION_PREFIX ):
            name = self.name
        else:
            name = "%s %s" % ( self.TRIANGULATION_PREFIX, self.name )

        triangulation = Line_point_layer(
            None, self.command_stack, self.plugin_loader, self.parent,
            points_layer, lines_layer, self.depth_unit.value, self.origin,
            self.size, saver = Verdat_saver,
            triangles_layer = triangles_layer,
            name = name,
        )

        self.parent.inbox.send(
            request = "replace_layers",
            layer = triangulation,
            layers_to_replace = ( self, ),
        )

        scheduler.add( triangulation.run )

        self.outbox.send(
            request = "end_progress",
            id = unique_id,
        )
        response_box.send( success = True )

    CONTOUR_INFO = (
        Contour_info(
            "Light",
            utility.color_to_int( 0.0, 1.0, 1.0, 0.5 ),
            "FORECASTLIGHT",
        ),
        Contour_info(
            "Medium",
            utility.color_to_int( 0.33, 0.66, 1.0, 0.75 ),
            "FORECASTMEDIUM",
        ),
        Contour_info(
            "Heavy",
            utility.color_to_int( 0.33, 0.33, 0.62, 0.75 ),
            "FORECASTHEAVY",
        ),
    )

    def contour( self, scheduler, levels, neighbor_count, grid_steps,
                 buffer_factor, response_box ):
        from maproomlib.utility.Contour import contour_points

        unique_id = "contour %s" % self.name
        self.outbox.send(
            request = "start_progress",
            id = unique_id,
            message = "Contouring"
        )

        # Hack to allow the start_progress message to be sent out.
        try:
            self.inbox.receive( request = "timeout", timeout = 0.01 )
        except utility.Timeout_error:
            pass

        contours = contour_points( 
            points = self.points_layer.points.view(
                Point_set_layer.POINTS_XY_DTYPE,
            ).xy[ : self.points_layer.add_index ].copy(),
            levels = np.array( levels, np.float32 ),
            neighbor_count = neighbor_count,
            grid_steps = grid_steps,
            buffer_factor = buffer_factor,
        )
        contours_info = iter( self.CONTOUR_INFO )
        children = []
        metadata = []

        for contour in contours:
            point_data = []
            polygon_point_data = []
            polygon_data = []
            point_index = 0

            try:
                contour_info = contours_info.next()
            except StopIteration:
                contour_info = self.CONTOUR_INFO[ -1 ]

            if not contour:
                continue

            for ( polygon_index, ( group, polygon ) ) in \
                enumerate( contour ):
                start_point_index = point_index

                # Skip the last point, which is a repeat.
                for point in polygon[ : -1 ]:
                    point_data.append( point )
                    polygon_point_data.append(
                        ( point_index + 1, polygon_index ),
                    )

                    point_index += 1

                # Alter the last point in the polygon so its next points
                # wraps back to the start.
                polygon_point_data.pop()
                polygon_point_data.append(
                    ( start_point_index, polygon_index ),
                )

                polygon_data.append( (
                    start_point_index,
                    point_index - start_point_index,
                    0,
                    contour_info.color,
                    group,
                ) )

            point_count = len( point_data )
            points = Point_set_layer.make_points( point_count )
            points.view( Point_set_layer.POINTS_XY_DTYPE ).xy[
                0: point_count
            ] = point_data

            origin = (
                np.nanmin( points.x[ : point_count ] ),
                np.nanmin( points.y[ : point_count ] ),
            )
            size = (
                np.nanmax( points.x[ : point_count ] ) - origin[ 0 ],
                np.nanmax( points.y[ : point_count ] ) - origin[ 1 ],
            )

            polygon_point_count = len( polygon_point_data )
            polygon_points = Polygon_set_layer.make_polygon_points(
                polygon_point_count,
            )
            polygon_points[ : polygon_point_count ] = polygon_point_data

            polygon_count = len( polygon_data )
            polygons = Polygon_set_layer.make_polygons(
                polygon_count,
            )
            polygons[ : polygon_count ] = polygon_data

            points_layer = Point_set_layer(
                self.command_stack, None, points, point_count,
                Point_set_layer.DEFAULT_POINT_SIZE, self.projection,
                origin = origin, size = size,
            )

            polygons_layer = Polygon_set_layer(
                self.command_stack, None, points_layer,
                polygon_points, polygons, polygon_count,
                default_polygon_color = contour_info.color,
            )

            child = Polygon_point_layer(
                None, self.command_stack, self.plugin_loader, self.parent,
                points_layer,
                polygons_layer,
                saver = None,
                name = contour_info.name,
            )
            scheduler.add( child.run )

            children.append( child )
            metadata.append( ( contour_info.metadata_tag, child ) )

        existing_metadata = dict( self.parent.children_metadata ) \
            if self.parent.children_metadata else {}

        for ( insert_index, ( metadata_tag, child ) ) in \
            enumerate( metadata ):
            child_to_replace = existing_metadata.get( metadata_tag )

            if child_to_replace:
                self.parent.inbox.send(
                    request = "replace_layers",
                    layer = child,
                    layers_to_replace = ( child_to_replace, ),
                    insert_index = insert_index,
                )
            else:
                self.parent.inbox.send(
                    request = "add_layer",
                    layer = child,
                    insert_index = insert_index,
                )

        non_contour_metadata = filter(
            lambda item: not item[ 0 ].startswith( "FORECAST" ),
            self.parent.children_metadata
        )

        self.parent.children_metadata = metadata + non_contour_metadata

        self.outbox.send(
            request = "end_progress",
            id = unique_id,
        )
        response_box.send( success = True )

    def merge( self, layers, response_box ):
        # Prepare to merge all the layers together (including this one).
        layers.append( self )
        point_count = 0
        line_count = 0
        depth_unit = None
        seams = []
        origin = None if self.origin is None else list( self.origin )
        size = None if self.size is None else list( self.size )

        for layer in layers:
            if layer.__class__ != self.__class__:
                response_box.send(
                    exception = ValueError(
                        "Can only merge layers of the same type.",
                    ),
                )
                return

            if depth_unit is None:
                depth_unit = str( layer.depth_unit )
            elif str( layer.depth_unit ) == "unknown":
                response_box.send(
                    exception = ValueError(
                        "Cannot merge layers with unknown depth units.",
                    ),
                )
                return
            elif str( layer.depth_unit ) != depth_unit:
                response_box.send(
                    exception = ValueError(
                        "Cannot merge layers with different depth units.",
                    ),
                )
                return

            if layer.origin is not None:
                if origin is None:
                    origin = list( layer.origin )
                else:
                    origin[ 0 ] = min( origin[ 0 ], layer.origin[ 0 ] )
                    origin[ 1 ] = min( origin[ 1 ], layer.origin[ 1 ] )

            if layer.size is not None:
                if size is None:
                    size = list( layer.size )
                else:
                    size[ 0 ] = max( size[ 0 ], layer.size[ 0 ] )
                    size[ 1 ] = max( size[ 1 ], layer.size[ 1 ] )

            seams.append( point_count )
            point_count += layer.points_layer.add_index
            line_count += layer.lines_layer.add_index

        color = self.DEFAULT_COLORS[
            Line_point_layer.next_default_color_index
        ]

        points_layer = Point_set_layer(
            self.command_stack,
            self.POINTS_LAYER_NAME,
            Point_set_layer.make_points( point_count ),
            point_count,
            Point_set_layer.DEFAULT_POINT_SIZE,
            self.projection,
            seams = seams,
            origin = None if origin is None else tuple( origin ),
            size = None if size is None else tuple( size ),
            default_point_color = color,
        )

        lines_layer = Line_set_layer(
            self.command_stack,
            self.LINES_LAYER_NAME,
            points_layer,
            Line_set_layer.make_lines( line_count ),
            line_count,
            Line_set_layer.DEFAULT_LINE_WIDTH,
            default_line_color = color,
        )
        points_layer.lines_layer = lines_layer

        # Merge all the points/lines arrays by copying them into the new
        # arrays.
        point_index = 0
        line_index = 0

        for layer in layers:
            layer_point_count = layer.points_layer.add_index

            points_layer.points[
                point_index : point_index + layer_point_count
            ] = layer.points_layer.points[ : layer_point_count ]

            layer_line_count = layer.lines_layer.add_index

            # Copy the line data and also adjust the contained point indices
            # so that they index into the new points layer.
            lines_layer.lines[
                line_index : line_index + layer_line_count
            ] = layer.lines_layer.lines[ : layer_line_count ]

            if point_index > 0:
                lines_layer.lines.point1[
                    line_index : line_index + layer_line_count
                ] += point_index
                lines_layer.lines.point2[
                    line_index : line_index + layer_line_count
                ] += point_index

            point_index += layer_point_count
            line_index += layer_line_count

        points_layer.add_index = point_index
        lines_layer.add_index = line_index
        lines_layer.generate_point_map()

        points_layer.points.color = color
        lines_layer.lines.color = color
        Line_point_layer.next_default_color_index = (
            Line_point_layer.next_default_color_index + 1
        ) % len( self.DEFAULT_COLORS )

        # Create a merged layer from the new arrays.
        merged = Line_point_layer(
            None,
            self.command_stack,
            self.plugin_loader,
            self.parent,
            points_layer,
            lines_layer,
            str( self.depth_unit ),
        )

        response_box.send(
            request = "merged",
            layer = merged,
        )

    def delete_triangles( self, **message ):
        # Called if anything changes in the layer. When that happens, the
        # child triangle set layer is deleted so that the triangles don't
        # remain in an invalid (non-Delaunay) state. The user can then
        # re-triangulate if desired.

        # Skip the removal if this is just the result of an operation being
        # undone. Or if there are no triangles. Or if there are triangles but
        # they're already hidden.
        if message.get( "undo_recorded" ) is False:
            return
        if self.triangles_layer is None:
            return
        if self.triangles_layer in self.hidden_children:
            return

        self.hide_layer( self.triangles_layer, description = "" )

    def set_property( self, property, value, response_box = None,
                      record_undo = True ):
        old_value = property.value

        Composite_layer.set_property(
            self, property, value, response_box, record_undo,
        )

        if property != self.depth_unit or old_value == "unknown" or \
           property.value == "unknown":
            return

        self.convert_units( old_value, property.value )

    def convert_units( self, old_units, new_units ):
        # When the depth units change, convert all the point depths
        # accordingly. Also convert the default depth value for this layer.
        conversion_factor = unit_conversion.convert(
            "length", old_units, new_units, 1,
        )

        self.set_property(
            property = self.default_depth,
            value = self.default_depth.value * conversion_factor,
            record_undo = False,
        )

        self.points_layer.inbox.send(
            request = "scale_depth",
            scale_factor = conversion_factor,
        )

    origin = property( lambda self: self.points_layer.origin )
    size = property( lambda self: self.points_layer.size )
