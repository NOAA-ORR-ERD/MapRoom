import os
import shutil
import numpy as np
from maproomlib.plugin.Polygon_point_layer import Polygon_point_layer
from maproomlib.plugin.Point_set_layer import Point_set_layer


class Moss_save_error( Exception ):
    def __init__( self, message, points_layer = None, points = None ):
        Exception.__init__( self, message )
        self.points_layer = points_layer
        self.points = points


SECTION_ID_CHAR_COUNT = 5
SECTION_ATTRIBUTE_NAME_LEFT_PAD = " " * 10
SECTION_ATTRIBUTE_NAME_CHAR_COUNT = 35
SECTION_POINT_COUNT_CHAR_COUNT = 5
POINT_CHAR_COUNT = 22
POINT_LONGITUDE_CHAR_COUNT = 10
POINT_LATITUDE_CHAR_COUNT = 10
POINT_FLAG_CHAR_COUNT = 2
POINT_FLAG_NORMAL = 0
POINT_FLAG_START_INNER_POLYGON = 1


def format_section_header( section_id, attribute_name, point_count ):
    return "".join( (
        str( section_id ).rjust( SECTION_ID_CHAR_COUNT ),
        SECTION_ATTRIBUTE_NAME_LEFT_PAD,
        attribute_name.ljust( SECTION_ATTRIBUTE_NAME_CHAR_COUNT ),
        str( point_count ).rjust( SECTION_POINT_COUNT_CHAR_COUNT ),
        "\n",
    ) )


def format_point( point, flag = POINT_FLAG_NORMAL ):
    return "".join( (
        str( point[ 0 ] ).rjust( POINT_LONGITUDE_CHAR_COUNT ),
        str( point[ 1 ] ).rjust( POINT_LATITUDE_CHAR_COUNT ),
        str( flag ).rjust( POINT_FLAG_CHAR_COUNT ),
        "\n",
    ) )


def save_polygons( save_file, section_id, attribute_name, layer ):
    points_layer = layer.points_layer
    polygons_layer = layer.polygons_layer
    group = None

    for polygon_index in xrange( polygons_layer.polygon_add_index ):
        # + 1 for the repeated start point
        point_count = polygons_layer.polygons.count[ polygon_index ] + 1
        start_point_index = polygons_layer.polygons.start[ polygon_index ]
        point_index = start_point_index
        section_id -= 1

        # Reduce the point count by the number of NaN points.
        while True:
            if np.isnan( points_layer.points[ point_index ][ 0 ] ):
                point_count -= 1

            point_index = polygons_layer.polygon_points.next[ point_index ]
            if point_index == start_point_index:
                break

        if point_count <= 1:
            continue

        # If this is a new group, then this is a transition to a new top-level
        # polygon.
        if polygons_layer.polygons.group[ polygon_index ] != group:
            # Write out the section header for this polygon.
            save_file.write(
                format_section_header( section_id, attribute_name, point_count ),
            )
            group = polygons_layer.polygons.group[ polygon_index ]
            flag = POINT_FLAG_NORMAL
        else:
            flag = POINT_FLAG_START_INNER_POLYGON

        # Write out the boundary point data for this polygon.
        start_point_index = polygons_layer.polygons.start[ polygon_index ]
        first_valid_point_index = None
        point_index = start_point_index

        while True:
            if not np.isnan( points_layer.points[ point_index ][ 0 ] ):
                save_file.write(
                    format_point( points_layer.points[ point_index ], flag ),
                )
                flag = POINT_FLAG_NORMAL
                if first_valid_point_index is None:
                    first_valid_point_index = point_index

            point_index = polygons_layer.polygon_points.next[ point_index ]

            if point_index == start_point_index:
                if first_valid_point_index is None:
                    break

                # Repeat the start point.
                save_file.write(
                    format_point(
                        points_layer.points[ first_valid_point_index ],
                    ),
                )
                break

    return section_id


def copy_polygons( in_file, save_file, known_attribute_names ):
    from Moss_loader import parse_section_header

    copy_points = False

    for line in in_file:
        orig_line = line
        if not line.strip(): continue
        line = line.rstrip( "\n" )

        # If this is a header line, parse it as such.
        if len( line ) > POINT_CHAR_COUNT:
            ( section_id, section_attribute, section_point_count ) = \
                parse_section_header( line )

            # If this is an unknown section, then copy it to the save file.
            if section_attribute in known_attribute_names:
                copy_points = False
            else:
                copy_points = True
                save_file.write( orig_line )

            continue

        # Otherwise, this is a point. Copy it to the save file if in an
        # appropriate section.
        if copy_points:
            save_file.write( orig_line )


LE_HEADER_ID_CHAR_COUNT = 5
LE_HEADER_LABEL_CHAR_COUNT = 18
LE_HEADER_LABEL = "LE POINT".rjust( LE_HEADER_LABEL_CHAR_COUNT )
LE_HEADER_FLAG_CHAR_COUNT = 32
LE_HEADER_FLAG = "1".rjust( LE_HEADER_FLAG_CHAR_COUNT )
LE_HEADER_LABEL_AND_FLAG = LE_HEADER_LABEL + LE_HEADER_FLAG
LE_LONGITUDE_CHAR_COUNT = 10
LE_LATITUDE_CHAR_COUNT = 10
LE_DEPTH_CHAR_COUNT = 2


def save_LEs( save_file, properties_save_file, points_layer ):
    points = points_layer.points

    for le_index in xrange( points_layer.add_index ):
        if np.isnan( points.z[ le_index ] ):
            z = 0
        else:
            z = points.z[ le_index ]

        save_file.write( "".join( (
            str( -( le_index + 1 ) ).rjust( LE_HEADER_ID_CHAR_COUNT ),
            LE_HEADER_LABEL_AND_FLAG,
            "\n",
            str( points.x[ le_index ] ).rjust( LE_LONGITUDE_CHAR_COUNT ),
            str( points.y[ le_index ] ).rjust( LE_LATITUDE_CHAR_COUNT ),
            str( z ).rjust( LE_DEPTH_CHAR_COUNT ),
            "\n",
        ) ) )

        # TODO: Also save property data for each LE to properties_save_file.


def points_of_invalid_polygons( layer ):
    points_layer = layer.points_layer
    polygons_layer = layer.polygons_layer
    invalid_polygons = polygons_layer.invalid_polygons
    points = []

    if invalid_polygons is None or len( invalid_polygons ) == 0:
        return points

    # Make a list of all the points in invalid polygons.
    for polygon_index in invalid_polygons:
        # + 1 for the repeated start point
        start_point_index = polygons_layer.polygons.start[ polygon_index ]
        point_index = start_point_index

        while True:
            if not np.isnan( points_layer.points[ point_index ][ 0 ] ):
                points.append( point_index )

            point_index = polygons_layer.polygon_points.next[ point_index ]
            if point_index == start_point_index:
                break

    return points


POLYGON_FILE_EXTENSION = "ms1"
SPILL_INFO_FILE_EXTENSION = "ms3"
NORMAL_LE_FILE_EXTENSION = "ms4"
NORMAL_LE_PROPERTY_FILE_EXTENSION = "ms5"
UNCERTAINTY_LE_FILE_EXTENSION = "ms6"
UNCERTAINTY_LE_PROPERTY_FILE_EXTENSION = "ms7"

EXTENSIONS_OF_FILES_TO_COPY = [
    SPILL_INFO_FILE_EXTENSION,
    NORMAL_LE_FILE_EXTENSION,
    NORMAL_LE_PROPERTY_FILE_EXTENSION,
    UNCERTAINTY_LE_FILE_EXTENSION,
    UNCERTAINTY_LE_PROPERTY_FILE_EXTENSION,
]

LE_FLOATING_METADATA = "INWATER"
LE_BEACHED_METADATA = "ONBEACH"


def Moss_saver( layer, filename ):
    """
    Save the contents of the layer as a set of MOSS files based on the path in
    the given :arg:`filename`.
    """
    for ( attribute_name, child ) in layer.children_metadata:
        if not isinstance( child, Polygon_point_layer ):
            continue

        invalid_points = points_of_invalid_polygons( child )
        if invalid_points:
            raise Moss_save_error(
                "MOSS files only support valid polygons.",
                points_layer = child.points_layer,
                points = invalid_points,
            )

    if layer.filename:
        source_base_filename = os.path.splitext( layer.filename )[ 0 ]
    else:
        source_base_filename = None
    dest_base_filename = os.path.splitext( filename )[ 0 ]

    # If the layer being saved was originally loaded from a set of MOSS files,
    # then use them as the basis of newly saved MOSS files.
    in_filename = ".".join( ( source_base_filename, POLYGON_FILE_EXTENSION ) )
    if source_base_filename and os.path.isfile( in_filename ):
        in_file = open( ".".join( in_filename ), "rU" )
    else:
        in_file = None

    save_file = open(
        ".".join( ( dest_base_filename, POLYGON_FILE_EXTENSION ) ), "wb",
    )
    section_id = 0
    known_attribute_names = []

    # Save in-memory polygons to file.
    for ( attribute_name, child ) in layer.children_metadata:
        if not isinstance( child, Polygon_point_layer ):
            continue

        known_attribute_names.append( attribute_name )
        section_id = save_polygons(
            save_file,
            section_id,
            attribute_name,
            child,
        )

    # Also copy polygons that we ignored when we initially read in the source
    # MOSS file.
    if in_file:
        copy_polygons(
            in_file,
            save_file,
            known_attribute_names,
        )
        in_file.close()

    save_file.close()

    # If there are source MOSS files, copy several files verbatim from them.
    if in_file:
        for extension in EXTENSIONS_OF_FILES_TO_COPY:
            try:
                shutil.copyfile(
                    ".".join( ( source_base_filename, extension ) ),
                    ".".join( ( dest_base_filename, extension ) ),
                )
            # If source and dest are actually the same file, skip the copy.
            except shutil.Error:
                pass
    # Otherwise, we need to generate LE files ourself.
    else:
        le_save_file = open(
            ".".join( ( dest_base_filename, NORMAL_LE_FILE_EXTENSION ) ), "wb",
        )
        le_property_save_file = open(
            ".".join( ( dest_base_filename, NORMAL_LE_PROPERTY_FILE_EXTENSION ) ), "wb",
        )
        uncertainty_le_save_file = None
        uncertainty_le_property_save_file = None

        for ( attribute_name, child ) in layer.children_metadata:
            if isinstance( child, Polygon_point_layer ):
                continue

            if "UNCERTAINTY" in attribute_name:
                if uncertainty_le_save_file is None:
                    uncertainty_le_save_file = open(
                        ".".join( ( dest_base_filename, UNCERTAINTY_LE_FILE_EXTENSION ) ), "wb",
                    )
                    uncertainty_le_property_save_file = open(
                        ".".join( ( dest_base_filename, UNCERTAINTY_LE_PROPERTY_FILE_EXTENSION ) ), "wb",
                    )
                save_LEs( uncertainty_le_save_file, uncertainty_le_property_save_file, child.points_layer )
            else:
                save_LEs( le_save_file, le_property_save_file, child.points_layer )

    return ".".join( ( dest_base_filename, POLYGON_FILE_EXTENSION ) )


Moss_saver.PLUGIN_TYPE = "file_saver"
Moss_saver.DESCRIPTION = "MOSS"
