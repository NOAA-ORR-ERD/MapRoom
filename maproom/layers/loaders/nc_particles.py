"""
loader for nc_particles files

i.e. GNOME output
"""

import os
import numpy as np
#import re

from common import BaseLoader
from maproom.layers import LineLayer

import logging
progress_log = logging.getLogger("progress")

from post_gnome import nc_particles

class ParticleLoader(BaseLoader):
    mime = "application/x-nc_particles"
    
    layer_types = ["line"]
    
    extensions = [".nc"]
    
    name = "nc_particles"
    
    def load(self, metadata, manager):
        """
        load the nc_particles file

        :param metadata: the metadata object from the file opener guess object.

        :param manager: The layer manager

        """

        layer = LineLayer(manager=manager)

        # load the data
        progress_log.info("Loading from %s" % metadata.uri)
        (layer.load_error_string,
         f_points,
         f_depths,
         f_line_segment_indexes,
         layer.depth_unit) = load_nc_particles_file(metadata.uri)

        if (layer.load_error_string == ""):
            progress_log.info("Finished loading %s" % metadata.uri)
            layer.set_data(f_points, f_depths, f_line_segment_indexes)
            layer.file_path = metadata.uri
            layer.name = os.path.split(layer.file_path)[1]
            layer.mime = "application/x-maproom-verdat" #fixme: wrong mime type!
        return [layer]

def load_nc_particles_file(file_path):

        reader = nc_particles.Reader(file_path)
        data = reader.get_timestep( len(reader.times)-1 ) #fixme: only the last timestep for now!
        f_points = np.c_[data['longitude'], data['latitude']]
        num_points = f_points.shape[0]
        load_error_string = ""
        depth_unit = "unknown"
        f_depths = np.zeros((num_points,), dtype = np.float32)
        f_line_segment_indexes = np.zeros( (num_points, 2), dtype=np.uint32 )
        return ( load_error_string,
                 f_points,
                 f_depths,
                 f_line_segment_indexes,
                 depth_unit)


# def write_layer_as_verdat(f, layer):
#     boundaries = Boundaries(layer, allow_branches=False)
#     errors, error_points = boundaries.check_errors()
#     if errors:
#         raise PointsError("Layer can't be saved as Verdat:\n\n%s" % "\n\n".join(errors), error_points)
    
#     points = layer.points
#     lines = layer.line_segment_indexes

#     f.write("DOGS")
#     if layer.depth_unit != None and layer.depth_unit != "unknown":
#         f.write("\t{0}\n".format(layer.depth_unit.upper()))
#     else:
#         f.write("\n")

#     boundary_endpoints = []
#     POINT_FORMAT = "%3d, %4.6f, %4.6f, %3.3f\n"
#     file_point_index = 1  # one-based instead of zero-based

#     ticks = (boundaries.num_points() / VerdatLoader.points_per_tick) + 1
#     progress_log.info("TICKS=%d" % ticks)

#     # write all boundary points to the file
#     # print "writing boundaries"
#     for (boundary_index, boundary) in enumerate(boundaries):
#         # if the outer boundary's area is positive, then reverse its
#         # points so that they're wound counter-clockwise
#         # print "index:", boundary_index, "area:", area, "len( boundary ):", len( boundary )
#         if boundary_index == 0:
#             if boundary.area > 0.0:
#                 boundary = reversed(boundary)
#         # if any other boundary has a negative area, then reverse its
#         # points so that they're wound clockwise
#         elif boundary.area < 0.0:
#             boundary = reversed(boundary)

#         for point_index in boundary:
#             f.write(POINT_FORMAT % (
#                 file_point_index,
#                 points.x[point_index],
#                 points.y[point_index],
#                 points.z[point_index],
#             ))
#             file_point_index += 1
            
#             if file_point_index % VerdatLoader.points_per_tick == 0:
#                 progress_log.info("Saved %d points" % file_point_index)

#         boundary_endpoints.append(file_point_index - 1)

#     # Write non-boundary points to file.
#     for point_index in boundaries.non_boundary_points:
#         x = points.x[point_index]
#         if np.isnan(x):
#             continue

#         y = points.y[point_index]
#         z = points.z[point_index]

#         f.write(POINT_FORMAT % (
#             file_point_index,
#             x, y, z,
#         ))
#         file_point_index += 1
        
#         if file_point_index % VerdatLoader.points_per_tick == 0:
#             progress_log.info("Saved %d points" % file_point_index)

#     # zero record signals the end of the points section
#     f.write(POINT_FORMAT % (0, 0.0, 0.0, 0.0))

#     # write the number of boundaries, followed by each boundary endpoint index
#     f.write("%d\n" % len(boundary_endpoints))

#     for endpoint in boundary_endpoints:
#         f.write("{0}\n".format(endpoint))
    
#     progress_log.info("Saved verdat")
