import re
import pyproj
import numpy as np
import maproomlib.utility as utility
from maproomlib.plugin.Point_set_layer import Point_set_layer
from maproomlib.plugin.Line_set_layer import Line_set_layer
from maproomlib.plugin.Line_point_layer import Line_point_layer
from maproomlib.plugin.Verdat_saver import Verdat_saver


WHITESPACE_PATTERN = re.compile( "\s+" )


def Verdat_loader( filename, command_stack, plugin_loader, parent ):
    """
    A plugin for loading vertex data from a DOGS-style verdat file.
    """

    in_file = file( filename, "rU" )

    header_line = in_file.readline().strip()
    header = WHITESPACE_PATTERN.split( header_line )
    if header[ 0 ] == "DOGS":
        not_actually_header = None
    else:
        not_actually_header = header_line

    if len( header ) == 2:
        depth_unit = header[ 1 ].lower()
    else:
        depth_unit = "unknown"

    lower_left = ( None, None )
    upper_right = ( None, None )

    point_data = []
    while True:
        line_str = not_actually_header or in_file.readline().strip()
        not_actually_header = None
        line = line_str.split( "," )

        data = tuple( map( float, line ) )
        if data == ( 0, 0, 0, 0 ):
            break
        if len( data ) != 4:
            raise utility.Load_plugin_error(
                "The vector file %s is invalid." % filename,
            )
        ( index, longitude, latitude, depth ) = data
        lower_left = (
            longitude if lower_left[ 0 ] is None \
                      else min( longitude, lower_left[ 0 ] ),
            latitude if lower_left[ 1 ] is None \
                     else min( latitude, lower_left[ 1 ] ),
        )
        upper_right = (
            max( longitude, upper_right[ 0 ] ),
            max( latitude, upper_right[ 1 ] ),
        )

        point_data.append( ( longitude, latitude, depth ) )

    origin = lower_left
    size = (
        upper_right[ 0 ] - lower_left[ 0 ],
        upper_right[ 1 ] - lower_left[ 1 ],
    )

    boundary_count = int( in_file.readline() )
    line_segments = []
    point_index = 0
    start_point_index = 0

    for boundary_index in range( boundary_count ):
        # -1 to make zero-indexed
        end_point_index = int( in_file.readline() ) - 1
        point_index = start_point_index

        # Skip "boundaries" that are only one or two points.
        if end_point_index - start_point_index + 1 < 3:
            start_point_index = end_point_index + 1
            continue

        while point_index < end_point_index:
            line_segments.append( ( point_index, point_index + 1 ) )
            point_index += 1

        # Close the boundary by connecting the first point to the last.
        line_segments.append( ( point_index, start_point_index ) )

        start_point_index = end_point_index + 1

    in_file.close()

    point_count = len( point_data )
    points = Point_set_layer.make_points( point_count )
    points.view( Point_set_layer.POINTS_XYZ_DTYPE ).xyz[
        0: point_count
    ] = np.array( point_data, dtype = np.float32 )

    line_count = len( line_segments )
    lines = Line_set_layer.make_lines( line_count )
    lines.view( Line_set_layer.LINES_POINTS_DTYPE ).points[
        0: line_count
    ] = line_segments

    projection = pyproj.Proj( "+proj=latlong" )
    color = Line_point_layer.DEFAULT_COLORS[
        Line_point_layer.next_default_color_index
    ]

    points.color = color
    lines.color = color

    points_layer = Point_set_layer(
        command_stack,
        Line_point_layer.POINTS_LAYER_NAME, points, point_count,
        Point_set_layer.DEFAULT_POINT_SIZE, projection,
        origin = origin, size = size, default_point_color = color,
    )

    lines_layer = Line_set_layer(
        command_stack,
        Line_point_layer.LINES_LAYER_NAME, points_layer, lines, line_count,
        Line_set_layer.DEFAULT_LINE_WIDTH,
        default_line_color = color,
    )

    points_layer.lines_layer = lines_layer

    return Line_point_layer(
        filename, command_stack, plugin_loader, parent,
        points_layer, lines_layer, depth_unit, origin, size,
        saver = Verdat_saver,
    )


Verdat_loader.PLUGIN_TYPE = "file_loader"
