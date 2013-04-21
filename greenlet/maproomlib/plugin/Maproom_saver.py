import os.path
import numpy as np


def Maproom_saver( layer, filename ):
    """
    Save the contents of the arg:`layer` to the path in the given
    :arg:`filename`.

    This function currently assumes that the arg:`layer` is a
    :class:`maproomlib.plugin.Line_point_layer`.
    """
    if "." not in os.path.basename( filename ):
        new_filename = filename + ".maproomv"
        if not os.path.exists( new_filename ):
            filename = new_filename

    points = layer.points_layer.points
    lines = layer.lines_layer.lines

    out_file = open( filename, "wb" )

    np.savez(
        out_file,
        points = points,
        lines = lines,
        point_count = layer.points_layer.add_index,
        line_count = layer.lines_layer.add_index,
        projection = layer.projection.srs,
        depth_unit = layer.depth_unit.value,
        default_depth = layer.default_depth.value,
        origin = layer.origin,
        size = layer.size,
    )

    out_file.close()
    return filename


Maproom_saver.PLUGIN_TYPE = "file_saver"
Maproom_saver.DESCRIPTION = "Maproom vector"
