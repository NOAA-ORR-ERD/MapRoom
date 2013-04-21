import os
import re
import pyproj
import numpy as np
import collections
import maproomlib.utility as utility
from maproomlib.plugin.Polygon_point_layer import Polygon_point_layer
from maproomlib.plugin.Polygon_set_layer import Polygon_set_layer
from maproomlib.plugin.Line_set_layer import Line_set_layer
from maproomlib.plugin.Point_set_layer import Point_set_layer
from maproomlib.plugin.Line_point_layer import Line_point_layer
from maproomlib.plugin.Composite_layer import Composite_layer
from maproomlib.plugin.Flag_layer import Flag_layer
from maproomlib.plugin.Moss_saver import Moss_saver


class Load_moss_error( Exception ):
    """ 
    An error occuring when attempting to load a MOSS file.
            
    :param message: the textual message explaining the error
    """
    def __init__( self, message = None ):
        Exception.__init__( self, message )


ATTRIBUTE_NAMES = (
    "MAPLAND",
    "FORECASTUNCERTAINTY",
    "FORECASTLIGHT",
    "FORECASTMEDIUM",
    "FORECASTHEAVY",
)


Layer_info = collections.namedtuple(
    "Layer_info",
    "name color",
)


ATTRIBUTE_LAYER_INFO = {
    "MAPLAND": Layer_info(
        "Land",
        utility.color_to_int( 0, 0.75, 0, 0.5 ),
    ),
    "FORECASTUNCERTAINTY": Layer_info(
        "Uncertainty",
        utility.color_to_int( 0.75, 0, 0, 0.1 ),
    ), 
    "FORECASTLIGHT": Layer_info(
        "Light",
        utility.color_to_int( 0.0, 1.0, 1.0, 0.5 ),
    ),
    "FORECASTMEDIUM": Layer_info(
        "Medium",
        utility.color_to_int( 0.33, 0.66, 1.0, 0.75 ),
    ),
    "FORECASTHEAVY": Layer_info(
        "Heavy",
        utility.color_to_int( 0.33, 0.33, 0.62, 0.75 ),
    ),
}


"""
The moss .ms1 file is a fixed-width format. Each file is composed of a series
of section. Each section starts with a section header line followed several
point lines. Each section can contain multiple polygons.

Section header line: 56 characters wide
  char   0-4: Unique section ID number (negative, meaning points are lon/lat)
  char 15-44: Attribute name of this section (used in multiple sections)
  char 50-54: Number of points in this section

Point line: 22 characters wide
  char   0-9: Longitude
  char 10-19: Latitude
  char 20-21: Flag (1 indicates first point of an inner polygon, 0 otherwise)
"""
POLYGON_SECTION_CHAR_COUNT = 56
POLYGON_SECTION_ID_SLICE = slice( 0, 5 )
POLYGON_SECTION_ATTRIBUTE_NAME_SLICE = slice( 15, 45 )
POLYGON_SECTION_POINT_COUNT_SLICE = slice( 50, 55 )
POLYGON_POINT_CHAR_COUNT = 22
POLYGON_POINT_LONGITUDE_SLICE = slice( 0, 10 )
POLYGON_POINT_LATITUDE_SLICE = slice( 10, 20 )
POLYGON_POINT_FLAG_SLICE = slice( 20, 22 )
POLYGON_POINT_FLAG_NORMAL = 0
POLYGON_POINT_FLAG_START_INNER_POLYGON = 1
SKIP_HOLES = True


def parse_section_header( line ):
    return (
        int( line[ POLYGON_SECTION_ID_SLICE ] ),
        line[ POLYGON_SECTION_ATTRIBUTE_NAME_SLICE ].strip(),
        int( line[ POLYGON_SECTION_POINT_COUNT_SLICE ] ),
    )


def parse_point( line ):
    return (
        float( line[ POLYGON_POINT_LONGITUDE_SLICE ] ),
        float( line[ POLYGON_POINT_LATITUDE_SLICE ] ),
        int( line[ POLYGON_POINT_FLAG_SLICE ] ),
    )


class Layer_data:
    def __init__( self, point_data = None, polygon_point_data = None,
                  polygon_data = None, point_index = 0, polygon_index = 0,
                  lower_left = None, upper_right = None ):
        self.point_data = point_data or []
        self.polygon_point_data = polygon_point_data or []
        self.polygon_data = polygon_data or []
        self.point_index = point_index
        self.polygon_index = polygon_index
        self.lower_left = lower_left or ( None, None )
        self.upper_right = upper_right or ( None, None )


def load_polygons( filename, command_stack, plugin_loader, parent ):
    layers = {} # attribute name -> Layer_data
    current_layer = None
    section_id = None
    section_attribute = None
    previous_section_attribute = None
    section_point_count = None
    polygon_index = 0
    start_point_index = 0
    hole = False

    in_file = open( filename, "rU" )

    for line in in_file:
        if not line.strip(): continue
        line = line.rstrip( "\n" )

        # If this is a header line, parse it as such. Also end the current
        # polygon in the previous section.
        if len( line ) > POLYGON_POINT_CHAR_COUNT:
            ( section_id, new_section_attribute, section_point_count ) = \
                parse_section_header( line )
            if new_section_attribute not in ATTRIBUTE_NAMES:
                section_id = None
                continue

            previous_section_attribute = section_attribute
            section_attribute = new_section_attribute

            if current_layer:
                # Alter the last point in the polygon so its next point wraps
                # back around to the start.
                current_layer.polygon_point_data.pop()
                current_layer.polygon_point_data.append( (
                    start_point_index, current_layer.polygon_index,
                ) )
                current_layer.polygon_data.append( (
                    start_point_index,
                    current_layer.point_index - start_point_index,
                    0,
                    ATTRIBUTE_LAYER_INFO[ previous_section_attribute ].color,
                    start_point_index,
                ) )
                current_layer.polygon_index += 1

            current_layer = layers.setdefault(
                section_attribute,
                Layer_data(),
            )
            start_point_index = current_layer.point_index
            hole = False
            continue

        # Otherwise, this is a point. If we're in a section other than one of
        # few we're actually interested in, ignore the point.
        if section_id is None:
            continue

        if hole and SKIP_HOLES:
            continue

        ( longitude, latitude, flag ) = parse_point( line )

        # Skip any point that repeats the start point.
        if current_layer.point_index != start_point_index and \
           ( longitude, latitude ) == \
           current_layer.point_data[ start_point_index ]:
            continue

        current_layer.lower_left = (
            longitude if current_layer.lower_left[ 0 ] is None \
                      else min( longitude, current_layer.lower_left[ 0 ] ),
            latitude if current_layer.lower_left[ 1 ] is None \
                     else min( latitude, current_layer.lower_left[ 1 ] ),
        )
        current_layer.upper_right = (
            max( longitude, current_layer.upper_right[ 0 ] ),
            max( latitude, current_layer.upper_right[ 1 ] ),
        )

        # If indicated, end the current polygon and start a new polygon.
        if flag == POLYGON_POINT_FLAG_START_INNER_POLYGON:
            hole = True
            if SKIP_HOLES: continue
            current_layer.polygon_point_data.pop()
            current_layer.polygon_point_data.append( (
                start_point_index, current_layer.polygon_index,
            ) )
            current_layer.polygon_data.append( (
                start_point_index,
                current_layer.point_index - start_point_index,
                0,
                ATTRIBUTE_LAYER_INFO[ section_attribute ].color,
                start_point_index,
            ) )

            start_point_index = current_layer.point_index
            current_layer.polygon_index += 1

        current_layer.point_data.append( ( longitude, latitude ) )
        current_layer.polygon_point_data.append( (
            current_layer.point_index + 1, current_layer.polygon_index,
        ) )
        current_layer.point_index += 1

    in_file.close()

    # End the last polygon.
    current_layer.polygon_point_data.pop()
    current_layer.polygon_point_data.append( (
        start_point_index, current_layer.polygon_index,
    ) )
    current_layer.polygon_data.append( (
        start_point_index,
        current_layer.point_index - start_point_index,
        0,
        ATTRIBUTE_LAYER_INFO[ section_attribute ].color,
        start_point_index,
    ) )

    projection = pyproj.Proj( "+proj=latlong" )

    children = []
    metadata = [] # [ ( attribute name, child layer object ), ... ]

    for attribute_name in ATTRIBUTE_NAMES:
        layer = layers.get( attribute_name )
        if not layer: continue
        layer_name = ATTRIBUTE_LAYER_INFO[ attribute_name ].name
        point_count = len( layer.point_data )
        polygon_point_count = len( layer.polygon_point_data )
        polygon_count = len( layer.polygon_data )

        origin = layer.lower_left
        size = (
            layer.upper_right[ 0 ] - layer.lower_left[ 0 ],
            layer.upper_right[ 1 ] - layer.lower_left[ 1 ],
        )

        points = Point_set_layer.make_points( point_count )
        points.view( Point_set_layer.POINTS_XY_DTYPE ).xy[
            0: point_count
        ] = layer.point_data

        polygon_points = Polygon_set_layer.make_polygon_points(
            polygon_point_count,
        )
        polygon_points[ : polygon_point_count ] = layer.polygon_point_data

        polygons = Polygon_set_layer.make_polygons(
            polygon_count
        )
        polygons[ : polygon_count ] = layer.polygon_data

        points_layer = Point_set_layer(
            command_stack, None, points, point_count,
            Point_set_layer.DEFAULT_POINT_SIZE, projection,
            origin = origin, size = size,
        )

        polygons_layer = Polygon_set_layer(
            command_stack, None, points_layer, polygon_points, polygons,
            polygon_count, default_polygon_color = \
                ATTRIBUTE_LAYER_INFO[ attribute_name ].color,
        )

        child = Polygon_point_layer(
            None, command_stack, plugin_loader, parent,
            points_layer,
            polygons_layer,
            saver = None,
            name = layer_name,
        )

        children.append( child )
        metadata.append( ( attribute_name, child ) )

    return ( children, metadata )


LE_HEADER_CHAR_COUNT = 55
LE_HEADER_LABEL_SLICE = slice( 15, 23 )
LE_HEADER_LABEL = "LE POINT"
LE_PROPERTY_SEPARATOR_PATTERN = re.compile( ",\s*" )
LE_PROPERTY_FLOATING_LABEL = "INWATER"
LE_PROPERTY_BEACHED_LABEL = "ONBEACH"
LE_FLOATING_LAYER_NAME = "Floating oil"
LE_BEACHED_LAYER_NAME = "Beached oil"
LE_NORMAL_COLOR = utility.color_to_int( 0, 0, 0, 0.75 )
LE_UNCERTAINTY_COLOR = utility.color_to_int( 0.75, 0, 0, 0.75 )
LE_FLOATING_POINT_SIZE = Point_set_layer.DEFAULT_POINT_SIZE
LE_BEACHED_POINT_SIZE = LE_FLOATING_POINT_SIZE * 2


def load_LEs( filename, property_filename, command_stack, plugin_loader,
              color, parent, layer_name_suffix = "" ):
    if not os.path.exists( filename ) or \
       not os.path.exists( property_filename ):
        return ( [], [] )

    beached_point_data = []
    floating_point_data = []

    in_file = open( filename, "rU" )
    in_property_file = open( property_filename, "rU" )

    for line in in_file:
        if not line.strip(): continue
        line = line.rstrip( "\n" )

        # Skip header lines.
        if len( line ) >= LE_HEADER_CHAR_COUNT and \
           line[ LE_HEADER_LABEL_SLICE ] == LE_HEADER_LABEL:
            continue

        ( longitude, latitude ) = line.split()[ :2 ]
        longitude = float( longitude )
        latitude = float( latitude )

        # Depth is in meters, mass is in kg, density is in g/cm^3, age is
        # hours since release.
        property_line = in_property_file.readline().rstrip( "\n" )
        if not property_line.strip(): break

        ( le_id, le_type, pollutant, depth, mass, density, age, status ) = \
            LE_PROPERTY_SEPARATOR_PATTERN.split( property_line )
        depth = float( depth )

        if status == LE_PROPERTY_FLOATING_LABEL:
            floating_point_data.append( ( longitude, latitude, depth ) )
        elif status == LE_PROPERTY_BEACHED_LABEL:
            beached_point_data.append( ( longitude, latitude, depth ) )

    in_file.close()
    in_property_file.close()

    children = []
    metadata = []
    projection = pyproj.Proj( "+proj=latlong" )

    if floating_point_data:
        floating_point_count = len( floating_point_data )
        floating_points = Point_set_layer.make_points( floating_point_count )
        floating_points.view( Point_set_layer.POINTS_XYZ_DTYPE ).xyz[
            0: floating_point_count
        ] = floating_point_data
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
            LE_PROPERTY_FLOATING_LABEL + layer_name_suffix.upper(),
            children[ -1 ],
        ) )

    if beached_point_data:
        beached_point_count = len( beached_point_data )
        beached_points = Point_set_layer.make_points( beached_point_count )
        beached_points.view( Point_set_layer.POINTS_XYZ_DTYPE ).xyz[
            0: beached_point_count
        ] = beached_point_data
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
            LE_PROPERTY_BEACHED_LABEL + layer_name_suffix.upper(),
            children[ -1 ],
        ) )

    return ( children, metadata )


POLYGON_FILE_EXTENSION = "ms1"
NORMAL_LE_FILE_EXTENSION = "ms4"
NORMAL_LE_PROPERTY_FILE_EXTENSION = "ms5"
UNCERTAINTY_LE_FILE_EXTENSION = "ms6"
UNCERTAINTY_LE_PROPERTY_FILE_EXTENSION = "ms7"


def Moss_loader( filename, command_stack, plugin_loader, parent ):
    base_filename = os.path.splitext( filename )[ 0 ]

    group = Composite_layer(
        command_stack, plugin_loader, parent,
        name = os.path.basename( base_filename ),
        supported_savers = [ Moss_saver ],
        filename = filename,
    )
    group.flag_layer = Flag_layer(
        command_stack, plugin_loader,
    )

    # Load floating and beached oil LEs.
    # Load floating and beached oil uncertainty LEs.
    ( LEs, metadata ) = load_LEs(
        ".".join( ( base_filename, NORMAL_LE_FILE_EXTENSION ) ),
        ".".join( ( base_filename, NORMAL_LE_PROPERTY_FILE_EXTENSION ) ),
        command_stack,
        plugin_loader,
        LE_NORMAL_COLOR,
        parent = group,
    )
    ( uncertainty_LEs, uncertainty_metadata ) = load_LEs(
        ".".join( ( base_filename, UNCERTAINTY_LE_FILE_EXTENSION ) ),
        ".".join( ( base_filename, UNCERTAINTY_LE_PROPERTY_FILE_EXTENSION ) ),
        command_stack,
        plugin_loader,
        LE_UNCERTAINTY_COLOR,
        parent = group,
        layer_name_suffix = " uncertainty",
    )
    LEs += uncertainty_LEs
    metadata += uncertainty_metadata

    # Load oil contours.
    ( children, group.children_metadata ) = \
        load_polygons(
            ".".join( ( base_filename, POLYGON_FILE_EXTENSION ) ),
            command_stack,
            plugin_loader,
            parent = group,
        )

    group.children = [ group.flag_layer ] + children + LEs
    group.hidden_children = set( LEs )
    group.children_metadata += metadata

    return group


Moss_loader.PLUGIN_TYPE = "file_loader"
