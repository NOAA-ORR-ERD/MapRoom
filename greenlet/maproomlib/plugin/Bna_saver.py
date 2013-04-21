import os.path
import numpy as np


class Bna_save_error( Exception ):
    def __init__( self, message ):
        Exception.__init__( self, message )


def Bna_saver( layer, filename ):
    """
    Save the contents of the layer to the path in the given :arg:`filename`.
    """
    if "." not in os.path.basename( filename ):
        new_filename = filename + ".bna"
        if not os.path.exists( new_filename ):
            filename = new_filename

    save_file = None
    points_layer = layer.points_layer
    polygons_layer = layer.polygons_layer
    POLYGON_FORMAT = '"%s","%s",%d\n'
    POINT_FORMAT = "%.12f,%.12f\n"

    for polygon_index in xrange( polygons_layer.polygon_add_index ):
        # + 1 for the repeated start point
        point_count = polygons_layer.polygons.count[ polygon_index ] + 1
        start_point_index = polygons_layer.polygons.start[ polygon_index ]
        point_index = start_point_index

        # Reduce the point count by the number of NaN points.
        while True:
            if np.isnan( points_layer.points[ point_index ][ 0 ] ):
                point_count -= 1

            point_index = polygons_layer.polygon_points.next[ point_index ]
            if point_index == start_point_index:
                break

        if point_count <= 1:
            continue

        if save_file is None:
            save_file = open( filename, "w" )

        # Write out the description line for this polygon.
        save_file.write(
            POLYGON_FORMAT % (
                polygon_index,           # TODO: Support custom polygon id.
                polygons_layer.polygons.type[ polygon_index ],
                point_count,
            )
        )

        # Write out the boundary point data for this polygon.
        start_point_index = polygons_layer.polygons.start[ polygon_index ]
        first_valid_point_index = None
        point_index = start_point_index

        while True:
            if not np.isnan( points_layer.points[ point_index ][ 0 ] ):
                save_file.write(
                    POINT_FORMAT % (
                        points_layer.points[ point_index ][ 0 ],
                        points_layer.points[ point_index ][ 1 ],
                    )
                )
                if first_valid_point_index is None:
                    first_valid_point_index = point_index

            point_index = polygons_layer.polygon_points.next[ point_index ]

            if point_index == start_point_index:
                if first_valid_point_index is None:
                    break

                # Repeat the start point.
                # TODO: Support polylines, not just polygons.
                save_file.write(
                    POINT_FORMAT % (
                        points_layer.points[ first_valid_point_index ][ 0 ],
                        points_layer.points[ first_valid_point_index ][ 1 ],
                    )
                )
                break

    if save_file is None:
        raise Bna_save_error( "BNA files require at least one polygon." )

    save_file.close()

    return filename


Bna_saver.PLUGIN_TYPE = "file_saver"
Bna_saver.DESCRIPTION = "BNA"
