import os.path
import pyproj
import numpy as np
import maproomlib.utility as utility


class Verdat_save_error( Exception ):
    def __init__( self, message, points = None ):
        Exception.__init__( self, message )
        self.points = points


def Verdat_saver( layer, filename ):
    """
    Save the contents of the layer to the path in the given :arg:`filename`.
    """
    from maproomlib.utility.Shape import points_outside_polygon

    if "." not in os.path.basename( filename ):
        new_filename = filename + ".verdat"
        if not os.path.exists( new_filename ):
            filename = new_filename

    points = layer.points_layer.points
    lines = layer.lines_layer.lines

    # If necessary, convert point data to a lat-long projection.
    target_projection = pyproj.Proj( "+proj=latlong" )
    if layer.projection.srs != target_projection.srs:
        transformer = utility.Transformer( target_projection )
        points = points.copy()

        transformer.transform_many(
            points, points, layer.projection, set_cache = False,
        )

    ( boundaries, non_boundary_points ) = \
        utility.find_boundaries(
            points = points,
            point_count = layer.points_layer.add_index,
            lines = lines,
            line_count = layer.lines_layer.add_index,
        )

    # Ensure that all points are within (or on) the outer boundary.
    outside_point_indices = points_outside_polygon(
        points.x,
        points.y,
        point_count = layer.points_layer.add_index,
        polygon = np.array( boundaries[ 0 ][ 0 ], np.uint32 ),
    )

    if len( outside_point_indices ) > 0:
        raise Verdat_save_error(
            "Points occur outside of the Verdat boundary.",
            points = tuple( outside_point_indices ),
        )

    # Open the output file and write its header.
    save_file = open( filename, "w" )
    save_file.write( "DOGS" )
    if layer.depth_unit.value and layer.depth_unit.value != "unknown":
        save_file.write( "\t%s\n" % layer.depth_unit.value.upper() )
    else:
        save_file.write( "\n" )

    boundary_endpoints = []
    POINT_FORMAT = "%3d, %4.6f, %4.6f, %3.3f\n"
    file_point_index = 1 # one-based instead of zero-based

    # Write all boundary points to file.
    #print "writing boundaries"
    for ( boundary_index, ( boundary, area ) ) in enumerate( boundaries ):
        # If the outer boundary's area is positive, then reverse its
        # points so that they're wound counter-clockwise.
        #print "index:", boundary_index, "area:", area, "len( boundary ):", len( boundary )
        if boundary_index == 0:
            if area > 0.0:
                boundary = reversed( boundary )
        # If any other boundary has a negative area, then reverse its
        # points so that they're wound clockwise.
        elif area < 0.0:
            boundary = reversed( boundary )

        for point_index in boundary:
            save_file.write( POINT_FORMAT % (
                file_point_index,
                points.x[ point_index ],
                points.y[ point_index ],
                points.z[ point_index ],
            ) )
            file_point_index += 1

        boundary_endpoints.append( file_point_index - 1 )

    # Write non-boundary points to file.
    for point_index in non_boundary_points:
        x = points.x[ point_index ]
        if np.isnan( x ):
            continue

        y = points.y[ point_index ]
        z = points.z[ point_index ]

        save_file.write( POINT_FORMAT % (
            file_point_index,
            x, y, z,
        ) )
        file_point_index += 1

    # Dummy zero points to signal end of points section.
    save_file.write( POINT_FORMAT % ( 0, 0.0, 0.0, 0.0 ) )

    # Write the number of boundaries, followed by each boundary endpoint
    # index.
    save_file.write( "%d\n" % len( boundary_endpoints ) )

    for endpoint in boundary_endpoints:
        save_file.write( "%d\n" % endpoint )

    return filename


Verdat_saver.PLUGIN_TYPE = "file_saver"
Verdat_saver.DESCRIPTION = "Verdat"
