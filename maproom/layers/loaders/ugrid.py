import os
import numpy as np
import re

from pyugrid.ugrid import UGrid

from common import PointsError, BaseLoader
from maproom.layers import TriangleLayer
from maproom.renderer import data_types

WHITESPACE_PATTERN = re.compile("\s+")

class UGridLoader(BaseLoader):
    mime = "application/x-hdf"
    
    layer_type = "triangle"
    
    name = "UGrid"
    
    def load(self, metadata, manager):
        layer = TriangleLayer(manager=manager)
        
        ug = UGrid.from_ncfile(metadata.uri)

        layer.set_data(ug.nodes, ug.depths, ug.faces)
        layer.file_path = metadata.uri
        layer.name = os.path.split(layer.file_path)[1]
        layer.mime = self.mime
        return layer
    
    def check(self, layer):
        return True
    
    def save_to_local_file(self, filename, layer):
        n = np.alen(layer.points)
        points = layer.points.view(data_types.POINT_XY_VIEW_DTYPE).xy[0:n]
        print points
        depths = layer.points.view(data_types.POINT_XY_VIEW_DTYPE).z[0:n]
        print depths
        n = np.alen(layer.triangles)
        faces = layer.triangles.view(data_types.TRIANGLE_POINTS_VIEW_DTYPE).point_indexes
        print faces

        grid = UGrid(points, faces, [], depths, layer.depth_unit)
        grid.save_as_netcdf(filename)
