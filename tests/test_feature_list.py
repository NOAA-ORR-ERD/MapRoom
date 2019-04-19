import os

import numpy as np

from mock import *

from maproom import loaders
from maproom.loaders import shapefile
from maproom.layers import PolygonParentLayer
from maproom.library.Boundary import Boundaries, PointsError

from maproom.renderer.gl.data_types import make_points_from_xy



class TestFeatureList(object):
    def setup(self):
        self.project = MockProject()
        self.proj = self.project.projection

    def test_line_feature_list(self):
        self.project.load_file("../TestData/Verdat/000011pts.verdat")
        layer = self.project.layer_manager.get_nth_oldest_layer_of_type("line", 1)
        out = layer.calc_output_feature_list()

        uri = os.path.join(os.getcwd(), "tmp.test_line_feature_list.shp")
        shapefile.write_feature_list_as_shapefile(uri, layer.points, out, self.proj)

        loader = shapefile.ShapefileLoader()
        layers = loader.load_layers(uri, self.project.layer_manager)
        test_layer = layers[0]
        test_out = test_layer.calc_output_feature_list()

        uri2 = os.path.join(os.getcwd(), "tmp.test_line_feature_list2.shp")
        shapefile.write_feature_list_as_shapefile(uri2, test_layer.points, test_out, self.proj)

        layers = loader.load_layers(uri2, self.project.layer_manager)
        test2_layer = layers[0]
        test2_out = test2_layer.calc_output_feature_list()
        print(test_out)
        print(test2_out)
        assert test2_out == test_out

    def test_shapefile_feature_list(self):
        self.project.load_file("../TestData/Shapefiles/square.shp", "application/x-maproom-shapefile")
        layer = self.project.layer_manager.get_nth_oldest_layer_of_type("shapefile", 1)
        layer.create_rings()
        out = layer.calc_output_feature_list()
        print(out)

        uri = os.path.join(os.getcwd(), "tmp.test_shapefile_feature_list.rings.shp")
        shapefile.write_rings_as_shapefile(uri, layer, layer.points, layer.rings, layer.ring_adjacency, self.proj)

        uri = os.path.join(os.getcwd(), "tmp.test_shapefile_feature_list.feature_list.shp")
        shapefile.write_feature_list_as_shapefile(uri, layer.points, out, self.proj)

        loader = shapefile.ShapefileLoader()
        layers = loader.load_layers(uri, self.project.layer_manager)
        test_layer = layers[0]
        test_out = test_layer.calc_output_feature_list()
        print(test_out)
        assert out == test_out



if __name__ == "__main__":
    import sys

    t = TestFeatureList()
    t.setup()
    # t.test_delete_polygon()
    # t.setup()
    # t.test_delete_hole()
    t.test_line_feature_list()
