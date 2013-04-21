import os
import hashlib
import pyproj
import numpy as np
from osgeo import ogr
import maproomlib.utility as utility
from maproomlib.plugin.Line_point_layer import Line_point_layer
from maproomlib.plugin.Point_set_layer import Point_set_layer
from maproomlib.plugin.Line_set_layer import Line_set_layer


class Load_dnc_error( Exception ):
    """ 
    An error occuring when attempting to load a DNC file.
            
    :param message: the textual message explaining the error
    """
    def __init__( self, message = None ):
        Exception.__init__( self, message )


def Dnc_loader( filename, command_stack, plugin_loader, parent ):
    MAX_ZIP_COUNT = 10

    if not utility.Extracted_zip_cache.zip_file( filename ):
        raise Load_dnc_error(
            "The file %s is not a zip file." % filename,
        )

    data_file = open( filename, "rb" )
    header = data_file.read( 512 )
    data_file.close()

    name = os.path.splitext( os.path.basename( filename ) )[ 0 ]
    header_hash = hashlib.sha1( header ).hexdigest()

    zip_cache = utility.Extracted_zip_cache( MAX_ZIP_COUNT )
    extracted_path = zip_cache.get( key = header_hash )

    if extracted_path is None:
        zip_cache.set( key = header_hash, value = filename )
        extracted_path = zip_cache.get( key = header_hash )

        if extracted_path is None:
            raise Load_dnc_error(
                "The zip file %s could not be extracted." % filename,
            )

    extracted_path = os.path.abspath( extracted_path )
    dnc_paths = [
        path for path in os.listdir( extracted_path )
        if path.startswith( "DNC" ) and
           os.path.isdir( os.path.join( extracted_path, path ) )
    ]

    if len( dnc_paths ) < 1:
        raise Load_dnc_error(
            "The zip file %s does not contain a DNC directory." % filename,
        )

    dnc_path = os.path.join( extracted_path, dnc_paths[ 0 ] )
    inner_dnc_paths = [
        path for path in os.listdir( dnc_path )
        if not path.startswith( "." ) and
           os.path.isdir( os.path.join( dnc_path, path ) )
    ]

    if len( inner_dnc_paths ) < 1:
        raise Load_dnc_error(
            "The zip file %s does not contain a DNC directory." % filename,
        )

    dnc_path = os.path.join(
        dnc_path, inner_dnc_paths[ 0 ]
    ).replace( "\\", "/" )

    if dnc_path[ 0 ] != "/":
        dnc_path = "/" + dnc_path

    full_path = "gltp:/vrf%s" % str( dnc_path )

    ogr.UseExceptions()
    dataset = ogr.Open( full_path )

    if dataset is None:
        raise Load_dnc_error(
            "The DNC file %s is invalid." % filename,
        )

    lower_left = ( None, None )
    upper_right = ( None, None )
    children = []
    point_data = []
    line_segments = []
    point_index = 0
    scheduler = utility.Scheduler.current()

    layers = [
        dataset.GetLayerByName( "coastl@ecr(*)_line" ),  # coastline
        dataset.GetLayerByName( "soundp@hyd(*)_point" ), # bathymetry points
    ]

    if None in layers:
        raise Load_dnc_error(
            "The DNC file %s does not contain the expected layers." % filename,
        )

    for layer in layers:
        projection = pyproj.Proj( layer.GetSpatialRef().ExportToProj4() )

        # If we don't call GetFeatureCount() first, then GetNextFeature()
        # results in an error!
        layer.GetFeatureCount()

        feature = layer.GetNextFeature()

        while feature is not None:
            geom = feature.GetGeometryRef()

            if geom is None:
                feature = layer.GetNextFeature()
                continue

            geom_type = geom.GetGeometryType()

            if geom_type == ogr.wkbPoint:
                # "hdp" = Hydrographic Depth
                depth_field_index = feature.GetFieldIndex( "hdp" )
            elif geom_type == ogr.wkbLineString:
                # "crv" = Depth Curve or Contour Value
                depth_field_index = feature.GetFieldIndex( "crv" )
            else: # Skip unsupported geometry types.
                feature = layer.GetNextFeature()
                continue

            geom_point_count = geom.GetPointCount()
            depth = feature.GetField( depth_field_index ) \
                    if depth_field_index != -1 else 1

            for geom_point_index in range( geom_point_count ):
                point = geom.GetPoint( geom_point_index )
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

                point_data.append( ( longitude, latitude, depth ) )

                if geom_type == ogr.wkbLineString and \
                   geom_point_index < geom_point_count - 1:
                    line_segments.append( ( point_index, point_index + 1 ) )

                point_index += 1

                if point_index % 100 == 0:
                    scheduler.switch()
            
            feature = layer.GetNextFeature()

    origin = lower_left
    size = (
        upper_right[ 0 ] - lower_left[ 0 ],
        upper_right[ 1 ] - lower_left[ 1 ],
    )

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

    color = Line_point_layer.DEFAULT_COLORS[
        Line_point_layer.next_default_color_index
    ]

    points.color = color
    lines.color = color

    points_layer = Point_set_layer(
        command_stack,
        layer.GetName(), points, point_count,
        Point_set_layer.DEFAULT_POINT_SIZE, projection,
        origin = origin, size = size,
        default_point_color = color,
    )
    children.append( points_layer )

    lines_layer = Line_set_layer(
        command_stack,
        layer.GetName(), points_layer, lines, line_count,
        Line_set_layer.DEFAULT_LINE_WIDTH,
        default_line_color = color,
    )
    children.append( lines_layer )

    points_layer.lines_layer = lines_layer

    return Line_point_layer(
        filename, command_stack, plugin_loader, parent,
        points_layer = points_layer,
        lines_layer = lines_layer,
        depth_unit = "meters",
        origin = origin,
        size = size,
    )


Dnc_loader.PLUGIN_TYPE = "file_loader"
