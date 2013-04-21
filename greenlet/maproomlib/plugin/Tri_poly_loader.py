import re
import pyproj
import numpy as np
import maproomlib.utility as utility
from maproomlib.plugin.Point_set_layer import Point_set_layer
from maproomlib.plugin.Line_set_layer import Line_set_layer
from maproomlib.plugin.Line_point_layer import Line_point_layer
from maproomlib.utility.hazpy.file_tools import triangle_files


def Tri_poly_loader( filename, command_stack, plugin_loader, parent ):
    """
    A plugin for loading vertex data from a "poly" file, as used by the
    "triangle" program.

    the .poly file gets pointed to, but it gets it vertices from the .node file

    Limitations:
      Boundary markers and holes are ignored
      It only handles a single attribute per node.
      It is assumed that the coordinates are lat-long
      The attribute is assumed to be depth, in unknown units
    """
    nodefilename = filename.rsplit(".",1)[0] + ".node"

    depth_unit = "unknown"

    ## fixme -- there should be some better exception handling here:
    # read the node file
    (PointData, Attributes, BoundaryMarkers) = triangle_files.ReadNodeFile(nodefilename)
    Attributes = Attributes[:,0:1]
    # read the poly file
    (line_segments, Holes, BoundaryMarkers) = triangle_files.ReadPolyFile(filename)
 
    lower_left =  tuple( PointData.min(0) )
    upper_right = tuple( PointData.max(0) )
    origin = lower_left
    size = (
        upper_right[ 0 ] - lower_left[ 0 ],
        upper_right[ 1 ] - lower_left[ 1 ],
    )

    point_count = len( PointData )
    points = Point_set_layer.make_points( point_count )
    points.view( Point_set_layer.POINTS_XYZ_DTYPE ).xyz[
        0: point_count
        ] = np.c_[PointData.astype(np.float32), Attributes.astype(np.float32)]

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
        Line_point_layer.POINTS_LAYER_NAME,
        points, point_count,
        Point_set_layer.DEFAULT_POINT_SIZE,
        projection,
        origin = origin, size = size,
        default_point_color = color,
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
        saver = None,
    )


Tri_poly_loader.PLUGIN_TYPE = "file_loader"
