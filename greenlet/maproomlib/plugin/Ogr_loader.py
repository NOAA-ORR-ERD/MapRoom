import os
import pyproj
import numpy as np
from osgeo import ogr
import maproomlib.utility as utility
from maproomlib.plugin.Polygon_point_layer import Polygon_point_layer
from maproomlib.plugin.Line_point_layer import Line_point_layer
from maproomlib.plugin.Bna_saver import Bna_saver
from maproomlib.plugin.Verdat_saver import Verdat_saver
from maproomlib.plugin.Point_set_layer import Point_set_layer
from maproomlib.plugin.Line_set_layer import Line_set_layer
from maproomlib.plugin.Polygon_set_layer import Polygon_set_layer


class Load_ogr_error( Exception ):
    """ 
    An error occuring when attempting to load an OGR file.
            
    :param message: the textual message explaining the error
    """
    def __init__( self, message = None ):
        Exception.__init__( self, message )


def layer_geometries( layer ):
    """
    Generator that yields all the geometry objects contained in the given
    OGR layer.
    """
    # If we don't call GetFeatureCount() first, then GetNextFeature()
    # results in an error!
    layer.GetFeatureCount()

    feature = layer.GetNextFeature()
    result = []

    while feature:
        geom = feature.GetGeometryRef()
        if geom:
            yield geom

        for child in child_geometries( geom ):
            yield child

        feature = layer.GetNextFeature()


def child_geometries( geom ):
    """
    Generator that yields all the geometry objects that are children of the
    given OGR geometry object.
    """
    child_count = geom.GetGeometryCount()

    if child_count == 0:
        return

    for child_index in xrange( 0, child_count ):
        child_geom = geom.GetGeometryRef( child_index )
        if not child_geom: continue
        yield child_geom

        for grandchild in child_geometries( child_geom ):
            yield grandchild


def Ogr_loader( filename, command_stack, plugin_loader, parent ):
    """
    Load an OGR vector data file from the given filename and return either a
    Polygon_point_layer or a Line_point_layer as appropriate. 
    """
    ogr.UseExceptions()
    dataset = ogr.Open( str( filename ) )

    if dataset is None:
        raise Load_ogr_error(
            "The OGR file %s is invalid." % filename,
        )

    lower_left = ( None, None )
    upper_right = ( None, None )
    point_data = []
    line_segments = []
    point_colors = []
    line_colors = []
    polygon_point_data = []
    polygon_data = []
    point_index = 0
    polygon_count = 0
    scheduler = utility.Scheduler.current()
    first_geom_type = None
    DEFAULT_COLOR = utility.color_to_int( 0.66, 0, 0, 1 )
    SECONDARY_COLOR = utility.color_to_int( 0.45, 0.7, 1.0, 1 )
    LINE_WIDTH = 3

    layers = [
        dataset.GetLayer( layer_index )
        for layer_index in xrange( 0, dataset.GetLayerCount() )
    ]

    if not layers:
        raise Load_ogr_error(
            "The OGR file %s does not contain any layers." % filename,
        )

    for layer in layers:
        projection = pyproj.Proj( layer.GetSpatialRef().ExportToProj4() )

        for geom in layer_geometries( layer ):
            geom_type = geom.GetGeometryType()
            if first_geom_type is None:
                first_geom_type = geom_type

            # Skip unsupported geometry types.
            if geom_type not in \
               ( ogr.wkbLineString, ogr.wkbLineString25D, ogr.wkbPoint ):
                continue

            geom_point_count = geom.GetPointCount()
            start_point_index = point_index
            longitude = None
            latitude = None
            area = 0.0
            geom_line_count = 0

            for geom_point_index in range( geom_point_count ):
                point = geom.GetPoint( geom_point_index )

                # See http://alienryderflex.com/polygon_area/
                if geom_point_index > 0:
                    area += \
                        ( longitude + point[ 0 ] ) * ( latitude - point[ 1 ] )

                longitude = point[ 0 ]
                latitude = point[ 1 ]

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

                point_data.append( ( longitude, latitude ) )

                if first_geom_type in ( ogr.wkbPolygon, ogr.wkbMultiPolygon ):
                    if geom_point_index < geom_point_count - 1:
                        # See maproomlib.plugin.Polygon_set_layer.POLYGON_POINTS_DTYPE.
                        polygon_point_data.append(
                            ( point_index + 1, polygon_count )
                        )
                    else:
                        polygon_point_data.append(
                            ( start_point_index, polygon_count )
                        )
                elif first_geom_type in \
                     ( ogr.wkbLineString, ogr.wkbLineString25D ):
                    if geom_point_index < geom_point_count - 1:
                        # See maproomlib.plugin.Line_set_layer.LINES_DTYPE
                        # (point1 and point2).
                        line_segments.append(
                            ( point_index, point_index + 1 ),
                        )
                        geom_line_count += 1

                point_index += 1

                if point_index % 100 == 0:
                    scheduler.switch()

            # If this is a loop, calculate its area. If it's
            # counter-clockwise, then change the color. This is a special-case
            # to support loop current and eddy coloring.
            if point_data[ start_point_index ] == point_data[ -1 ] and \
               area > 0.0:
                color = SECONDARY_COLOR
            else:
                color = DEFAULT_COLOR
            point_colors.extend( ( color, ) * geom_point_count )
            line_colors.extend( ( color, ) * geom_line_count )

            if first_geom_type in ( ogr.wkbPolygon, ogr.wkbMultiPolygon ):
                # See maproomlib.plugin.Polygon_set_layer.POLYGONS_DTYPE.
                polygon_data.append( (
                    start_point_index,
                    geom_point_count,
                    0,
                    Polygon_set_layer.DEFAULT_POLYGON_COLOR,
                ) )

            polygon_count += 1

    origin = lower_left
    size = (
        upper_right[ 0 ] - lower_left[ 0 ],
        upper_right[ 1 ] - lower_left[ 1 ],
    )

    # Create a points layer with the loaded point data. Unfortunately there's
    # a fair amount of boilerplate necessary.
    point_count = len( point_data )
    points = Point_set_layer.make_points( point_count )
    points[
        : point_count
    ].view( Point_set_layer.POINTS_XY_DTYPE ).xy = point_data
    points.color[ : point_count ] = point_colors

    points_layer = Point_set_layer(
        command_stack,
        layer.GetName(), points, point_count,
        Point_set_layer.DEFAULT_POINT_SIZE, projection,
        origin = origin, size = size, default_point_color = point_colors[ 0 ],
    )

    # If there's any polygon data, create a polygons layer with it and return
    # a composite polygon/point layer.
    if polygon_data:
        polygon_points = Polygon_set_layer.make_polygon_points( point_count )
        polygon_points[ : point_count ] = polygon_point_data

        polygon_count = len( polygon_data )
        polygons = Polygon_set_layer.make_polygons( polygon_count )
        polygons[ : polygon_count ] = polygon_data

        polygons_layer = Polygon_set_layer(
            command_stack,
            Polygon_point_layer.POLYGONS_LAYER_NAME, points_layer,
            polygon_points, polygons, polygon_count,
        )

        return Polygon_point_layer(
            filename, command_stack, plugin_loader, parent,
            points_layer, polygons_layer,
            saver = Bna_saver,
        )

    # Otherwise, create a lines layer with the line data.
    line_count = len( line_segments )
    lines = Line_set_layer.make_lines( line_count )
    if line_count > 0:
        lines[
            : line_count
        ].view( Line_set_layer.LINES_POINTS_DTYPE ).points = line_segments
        lines.color[ : line_count ] = line_colors

    lines_layer = Line_set_layer(
        command_stack,
        Line_point_layer.LINES_LAYER_NAME, points_layer, lines, line_count,
        LINE_WIDTH,
        default_line_color = line_colors[ 0 ],
    )

    points_layer.lines_layer = lines_layer

    # Return a composite line/point layer.
    line_point_layer = Line_point_layer(
        filename, command_stack, plugin_loader, parent,
        points_layer, lines_layer, None, origin, size,
        saver = Verdat_saver,
    )

    line_point_layer.hidden_children.add( points_layer )

    return line_point_layer


Ogr_loader.PLUGIN_TYPE = "file_loader"
