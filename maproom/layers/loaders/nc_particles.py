import os
import numpy as np
#import re


from common import BaseLoader
from maproom.layers import PointLayer
from maproom.renderer import data_types

from post_gnome import nc_particles

#WHITESPACE_PATTERN = re.compile("\s+")

class UGridLoader(BaseLoader):
    mime = "application/x-nc_particles"
    
    layer_types = ["point"]
    
    extensions = [".nc"]
    
    name = "nc_particles"

    def find_depths(self, grid):
        found = grid.find_data_sets('sea_floor_depth_below_geoid', 'node')
        if found:
            return found.pop()
        return None
    
    def load(self, metadata, manager):
        reader = from post_gnome import nc_particles.Reader(metadata.uri)
        data = reader.get_timestep(0) # only one timestep now!
        nodes = np.c_[data['longitude'], data['latitude']]


        layers = []

        depths = 0.0

        layer = PointLayer(manager=manager)
        layer.file_path = metadata.uri
        #layer.set_data(nodes, depths, edges)
        layer.set_data(nodes, depths, [])

        layer.depth_unit = dataset.attributes.get('units', 'unknown')
        layer.name = os.path.split(layer.file_path)[1]
        layer.mime = self.mime
        layers.append(layer)

        return layers
    
    # def save_to_local_file(self, filename, layer):
    #     n = np.alen(layer.points)
    #     points = layer.points.view(data_types.POINT_XY_VIEW_DTYPE).xy[0:n]
    #     depths = layer.points.view(data_types.POINT_XY_VIEW_DTYPE).z[0:n]
    #     if layer.type == "triangle":
    #         lines = None
    #         n = np.alen(layer.triangles)
    #         faces = layer.triangles.view(data_types.TRIANGLE_POINTS_VIEW_DTYPE).point_indexes
    #     elif layer.type == "line":
    #         faces = None
    #         lines = layer.line_segment_indexes.view(data_types.LINE_SEGMENT_POINTS_VIEW_DTYPE).points

    #     grid = UGrid(points, faces, lines)
    #     dataset = DataSet('depth', location='node', data=depths)
    #     dataset.attributes['units'] = layer.depth_unit
    #     dataset.attributes['standard_name'] = "sea_floor_depth_below_geoid"
    #     dataset.attributes['positive'] = "down"
    #     grid.add_data(dataset)
    #     grid.save_as_netcdf(filename)
