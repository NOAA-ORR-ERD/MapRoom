import pyproj
import zipfile
import numpy as np
from maproomlib.plugin.Point_set_layer import Point_set_layer
from maproomlib.plugin.Line_set_layer import Line_set_layer
from maproomlib.plugin.Line_point_layer import Line_point_layer
from maproomlib.plugin.Maproom_saver import Maproom_saver


class Load_maproom_error( Exception ):
    pass


def Maproom_loader( filename, command_stack, plugin_loader, parent ):
    """
    A plugin for loading data from a Maproom file.

    This function currently assumes that the file contains Verdat array data.
    """
    try:
        zip = zipfile.ZipFile( file = filename, mode = "r" )
    except ( IOError, OSError, zipfile.BadZipfile ):
        raise Load_maproom_error(
            "The Maproom vector file %s is invalid." % filename,
        )

    zip.close()
    in_file = open( filename, "rb" )
    file_data = np.load( in_file )
    data = {}

    # If necessary, convert all arrays to native byte order.
    for key in file_data.files:
        if file_data[ key ].dtype.isnative:
            data[ key ] = file_data[ key ]
        else:
            data[ key ] = file_data[ key ].byteswap().newbyteorder()

    points = data[ "points" ].view( np.recarray )
    lines = data[ "lines" ].view( np.recarray )
    point_count = int( data[ "point_count" ] )
    line_count = int( data[ "line_count" ] )
    projection = pyproj.Proj( str( data[ "projection" ] ) )
    depth_unit = str( data[ "depth_unit" ] )
    default_depth = data[ "default_depth" ]
    origin = tuple( data[ "origin" ] )
    size = tuple( data[ "size" ] )

    in_file.close()

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
        default_depth, saver = Maproom_saver,
    )


Maproom_loader.PLUGIN_TYPE = "file_loader"
