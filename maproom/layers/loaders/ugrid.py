import os
import numpy as np
import re

from pyugrid.ugrid import UGrid, DataSet

from common import BaseLoader
from maproom.layers import LineLayer, TriangleLayer
from maproom.renderer import data_types

WHITESPACE_PATTERN = re.compile("\s+")

class UGridLoader(BaseLoader):
    mime = "application/x-hdf"
    
    layer_types = ["point", "line", "triangle"]
    
    extensions = [".nc"]
    
    name = "UGrid"

    def find_depths(self, grid):
        found = grid.find_data_sets('sea_floor_depth_below_geoid', 'node')
        if found:
            return found.pop()
        return None
    
    def load(self, metadata, manager):
        layers = []
        
        ug = UGrid.from_ncfile(metadata.uri, load_data=True)
        dataset = self.find_depths(ug)
        if dataset:
            depths = dataset.data
        else:
            depths = 0.0
        if ug.edges is not None and len(ug.edges) > 0:
            layer = LineLayer(manager=manager)
            layer.set_data(ug.nodes, depths, ug.edges)
            layer.depth_unit = dataset.attributes.get('units', 'unknown')
            layer.file_path = metadata.uri
            layer.name = os.path.split(layer.file_path)[1]
            layer.mime = self.mime
            layers.append(layer)

        if ug.faces is not None and len(ug.faces) > 0:
            layer = TriangleLayer(manager=manager)
            layer.set_data(ug.nodes, depths, ug.faces)
            layer.depth_unit = dataset.attributes.get('units', 'unknown')
            layer.file_path = metadata.uri
            layer.name = os.path.split(layer.file_path)[1]
            layer.mime = self.mime
            layers.append(layer)
        return layers
    
    def check(self, layer):
        return True
    
    def save_to_local_file(self, filename, layer):
        n = np.alen(layer.points)
        points = layer.points.view(data_types.POINT_XY_VIEW_DTYPE).xy[0:n]
        depths = layer.points.view(data_types.POINT_XY_VIEW_DTYPE).z[0:n]
        if layer.type == "triangle":
            lines = None
            n = np.alen(layer.triangles)
            faces = layer.triangles.view(data_types.TRIANGLE_POINTS_VIEW_DTYPE).point_indexes
        elif layer.type == "line":
            faces = None
            lines = layer.line_segment_indexes.view(data_types.LINE_SEGMENT_POINTS_VIEW_DTYPE).points

        grid = UGrid(points, faces, lines)
        dataset = DataSet('depth', location='node', data=depths)
        dataset.attributes['units'] = layer.depth_unit
        dataset.attributes['standard_name'] = "sea_floor_depth_below_geoid"
        dataset.attributes['positive'] = "down"
        grid.add_data(dataset)
        grid.save_as_netcdf(filename)
