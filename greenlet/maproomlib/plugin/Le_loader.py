import os
import pyproj
import numpy as np
import maproomlib.utility as utility
from maproomlib.plugin.Line_set_layer import Line_set_layer
from maproomlib.plugin.Point_set_layer import Point_set_layer
from maproomlib.plugin.Line_point_layer import Line_point_layer
from maproomlib.plugin.Composite_layer import Composite_layer
from maproomlib.plugin.Flag_layer import Flag_layer
from maproomlib.plugin.Moss_saver import Moss_saver


class Load_le_error( Exception ):
    """ 
    An error occuring when attempting to load a binary LE file.
            
    :param message: the textual message explaining the error
    """
    def __init__( self, message = None ):
        Exception.__init__( self, message )


# Assume that the file data is big-endian.
HEADER_DTYPE = np.dtype( [
    ( "version1", ">S10" ),
    ( "day", ">h" ),
    ( "month", ">h" ),
    ( "year", ">h" ),
    ( "hour", ">h" ),
    ( "minute", ">h" ),
    ( "current_time", ">f" ),
    ( "version2", ">f" ),
    ( "point_count", ">l" ),
] )


POINTS_DTYPE = np.dtype( [
    ( "y", ">f" ),
    ( "x", ">f" ),
    ( "release_time", ">f" ),
    ( "age_when_released", ">f" ),
    ( "beach_height", ">f" ),
    ( "nmap", ">l" ),
    ( "pollutant", ">l" ),
    ( "wind_key", ">l" ),
] )


LE_FLOATING_METADATA = "INWATER"
LE_BEACHED_METADATA = "ONBEACH"
LE_FLOATING_LAYER_NAME = "Floating oil"
LE_BEACHED_LAYER_NAME = "Beached oil"
LE_NORMAL_COLOR = utility.color_to_int( 0, 0, 0, 0.75 )
LE_UNCERTAINTY_COLOR = utility.color_to_int( 0.75, 0, 0, 0.75 )
LE_FLOATING_POINT_SIZE = Point_set_layer.DEFAULT_POINT_SIZE
LE_BEACHED_POINT_SIZE = LE_FLOATING_POINT_SIZE * 2
LE_BEACHED_FLAG = -50


def load_LE_file( filename, command_stack, plugin_loader, color,
                  parent, layer_name_suffix = "" ):
    if not os.path.exists( filename ):
        return ( [], [] )

    in_file = open( filename, "rb" )

    header = np.fromfile(
        in_file, HEADER_DTYPE, count = 1,
    ).view( np.recarray )
    point_count = header.point_count[ 0 ]

    point_data = np.fromfile(
        in_file, POINTS_DTYPE, count = point_count,
    ).view( np.recarray )

    in_file.close()

    # Assume Western hemisphere.
    point_data.x[ : ] = -point_data.x[ : ]

    beached = ( point_data.beach_height == LE_BEACHED_FLAG )
    beached_point_data = point_data[ beached ]
    floating_point_data = point_data[ np.logical_not( beached ) ]
    children = []
    metadata = []
    projection = pyproj.Proj( "+proj=latlong" )

    if len( floating_point_data ) > 0:
        floating_point_count = len( floating_point_data )
        floating_points = Point_set_layer.make_points( floating_point_count )
        floating_points.x[ : floating_point_count ] = floating_point_data.x
        floating_points.y[ : floating_point_count ] = floating_point_data.y
        floating_points.color = color
        floating_origin = (
            np.nanmin( floating_points.x[ : floating_point_count ] ),
            np.nanmin( floating_points.y[ : floating_point_count ] ),
        )
        floating_size = (
            np.nanmax( floating_points.x[ : floating_point_count ] ) - \
                floating_origin[ 0 ],
            np.nanmax( floating_points.y[ : floating_point_count ] ) - \
                floating_origin[ 1 ],
        )

        points_layer = Point_set_layer(
            command_stack, None,
            floating_points, floating_point_count, LE_FLOATING_POINT_SIZE,
            projection,
            origin = floating_origin,
            size = floating_size,
            default_point_color = color,
        )
        lines_layer = Line_set_layer(
            command_stack, None, points_layer,
            Line_set_layer.make_lines( 10 ), 0,
            Line_set_layer.DEFAULT_LINE_WIDTH,
        )
        points_layer.lines_layer = lines_layer

        children.append(
            Line_point_layer(
                filename, command_stack, plugin_loader,
                parent, points_layer, lines_layer, 
                name = LE_FLOATING_LAYER_NAME + layer_name_suffix,
                labels_layer = None,
            ),
        )
        metadata.append( (
            LE_FLOATING_METADATA + layer_name_suffix.upper(),
            children[ -1 ],
        ) )

    if len( beached_point_data ) > 0:
        beached_point_count = len( beached_point_data )
        beached_points = Point_set_layer.make_points( beached_point_count )
        beached_points.x[ : beached_point_count ] = beached_point_data.x
        beached_points.y[ : beached_point_count ] = beached_point_data.y
        beached_points.color = color
        beached_origin = (
            np.nanmin( beached_points.x[ : beached_point_count ] ),
            np.nanmin( beached_points.y[ : beached_point_count ] ),
        )
        beached_size = (
            np.nanmax( beached_points.x[ : beached_point_count ] ) - \
                beached_origin[ 0 ],
            np.nanmax( beached_points.y[ : beached_point_count ] ) - \
                beached_origin[ 1 ],
        )

        points_layer = Point_set_layer(
            command_stack, None,
            beached_points, beached_point_count, LE_BEACHED_POINT_SIZE,
            projection,
            origin = beached_origin,
            size = beached_size,
            default_point_color = color,
        )
        lines_layer = Line_set_layer(
            command_stack, None, points_layer,
            Line_set_layer.make_lines( 10 ), 0,
            Line_set_layer.DEFAULT_LINE_WIDTH,
        )
        points_layer.lines_layer = lines_layer

        children.append(
            Line_point_layer(
                filename, command_stack, plugin_loader,
                parent, points_layer, lines_layer, 
                name = LE_BEACHED_LAYER_NAME + layer_name_suffix,
                labels_layer = None,
            ),
        )
        metadata.append( (
            LE_BEACHED_METADATA + layer_name_suffix.upper(),
            children[ -1 ],
        ) )

    return ( children, metadata )


NORMAL_FILENAME_TAG = "FORCST"
UNCERTAINTY_FILENAME_TAG = "UNCRTN"
UNCERTAINTY_SUFFIX = " uncertainty"


def Le_loader( filename, command_stack, plugin_loader, parent ):
    ( dir_name, filename ) = os.path.split( filename )
    normal_pieces = filename.split( NORMAL_FILENAME_TAG )
    uncertainty_pieces = filename.split( UNCERTAINTY_FILENAME_TAG )

    group = Composite_layer(
        command_stack, plugin_loader, parent,
        name = os.path.basename( filename ),
        supported_savers = [ Moss_saver ],
        filename = os.path.join( dir_name, filename ),
    )

    # If the filename contains NORMAL_FILENAME_TAG, then try to load the
    # corresponding filename containing the UNCERTAINTY_FILENAME_TAG as well.
    if len( normal_pieces ) == 2:
        uncertainty_filename = \
            normal_pieces[ 0 ] + UNCERTAINTY_FILENAME_TAG + normal_pieces[ 1 ]
        ( children, metadata ) = \
            load_LE_file(
                os.path.join( dir_name, filename ),
                command_stack, plugin_loader, LE_NORMAL_COLOR,
                parent = group,
            )
        ( uncertainty_children, uncertainty_metadata ) = \
            load_LE_file(
                os.path.join( dir_name, uncertainty_filename ),
                command_stack, plugin_loader, LE_UNCERTAINTY_COLOR,
                parent = group,
                layer_name_suffix = UNCERTAINTY_SUFFIX,
            )
        children += uncertainty_children
        metadata += uncertainty_metadata

    # If the filename contains the UNCERTAINTY_FILENAME_TAG, then try to load
    # the corresponding filename containing the NORMAL_FILENAME_TAG as well.
    elif len( uncertainty_pieces ) == 2:
        normal_filename = \
            uncertainty_pieces[ 0 ] + NORMAL_FILENAME_TAG + \
            uncertainty_pieces[ 1 ]
        ( children, metadata ) = \
            load_LE_file(
                os.path.join( dir_name, normal_filename ),
                command_stack, plugin_loader, LE_NORMAL_COLOR,
                parent = group,
            )
        ( uncertainty_children, uncertainty_metadata ) = \
            load_LE_file(
                os.path.join( dir_name, filename ),
                command_stack, plugin_loader, LE_UNCERTAINTY_COLOR,
                parent = group,
                layer_name_suffix = UNCERTAINTY_SUFFIX,
            )
        children += uncertainty_children
        metadata += uncertainty_metadata

    # Otherwise, just load the file by itself as a normal layer.
    else:
        ( children, metadata ) = load_LE_file(
            os.path.join( dir_name, normal_filename ),
            command_stack, plugin_loader, LE_NORMAL_COLOR,
        )

    if len( children ) == 0:
        raise Load_le_error( "The LE file is invalid." )

    group.flag_layer = Flag_layer(
        command_stack, plugin_loader,
    )

    group.children = children
    group.children_metadata = metadata

    return group


Le_loader.PLUGIN_TYPE = "file_loader"
