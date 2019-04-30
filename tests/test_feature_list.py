import os

import numpy as np

from mock import *

from maproom import loaders
from maproom.loaders import shapefile
from maproom import layers as ly
from maproom.library.Boundary import Boundaries, PointsError

from maproom.renderer.gl.data_types import make_points_from_xy


def check_layer_points(layer, other, other_point_offset):
    # other_point_offset: when a polyline layer gets loaded as a shapefile,
    # there are no bounding box or center points so the points start at index =
    # 0. The original layer, though, doesn't start the polyline points until
    # after the center point index, so need this offset to compare points.
    count = len(layer.points)
    print(layer.points.x)
    print(other.points.x[other_point_offset:])
    assert np.allclose(layer.points.x[:count], other.points.x[other_point_offset:other_point_offset + count])
    assert np.allclose(layer.points.y[:count], other.points.y[other_point_offset:other_point_offset + count])

def save_and_check_layer(layer, layer_point_offset, label, project):
    out = layer.calc_output_feature_list()
    print(out)
    boundary = out[0][2].start_index
    print(boundary)
    print(boundary.points)
    print(boundary.point_indexes)
    uri = os.path.join(os.getcwd(), f"tmp.test_{label}.shp")
    shapefile.write_feature_list_as_shapefile(uri, out, project.projection)
    uri2 = os.path.join(os.getcwd(), f"tmp.test_{label}.bna")
    shapefile.write_feature_list_as_bna(uri2, out, project.projection)

    loader = shapefile.ShapefileLoader()
    layers = loader.load_layers(uri, project.layer_manager)
    test_layer = layers[0]
    test_out = test_layer.calc_output_feature_list()
    layers = loader.load_layers(uri2, project.layer_manager)
    test2_layer = layers[0]
    test2_out = test_layer.calc_output_feature_list()
    print(test_out)
    print(test2_out)
    assert test2_out == test_out
    check_layer_points(test2_layer, layer, layer_point_offset)


class TestFeatureList(object):
    def setup(self):
        self.project = MockProject()
        self.proj = self.project.projection

    def test_line_feature_list(self):
        self.project.load_file("../TestData/Verdat/000011pts.verdat")
        layer = self.project.layer_manager.get_nth_oldest_layer_of_type("line", 1)
        out = layer.calc_output_feature_list()

        uri = os.path.join(os.getcwd(), "tmp.test_line_feature_list.shp")
        shapefile.write_feature_list_as_shapefile(uri, out, self.proj)

        loader = shapefile.ShapefileLoader()
        layers = loader.load_layers(uri, self.project.layer_manager)
        test_layer = layers[0]
        test_out = test_layer.calc_output_feature_list()

        uri2 = os.path.join(os.getcwd(), "tmp.test_line_feature_list2.shp")
        shapefile.write_feature_list_as_shapefile(uri2, test_out, self.proj)

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

        uri = os.path.join(os.getcwd(), "tmp.test_shapefile_feature_list.feature_list.shp")
        shapefile.write_feature_list_as_shapefile(uri, out, self.proj)

        loader = shapefile.ShapefileLoader()
        layers = loader.load_layers(uri, self.project.layer_manager)
        test_layer = layers[0]
        test_out = test_layer.calc_output_feature_list()
        print(test_out)
        assert out == test_out


    def test_shapefile_formats(self):
        self.project.load_file("../TestData/BNA/00011polys_001486pts.bna")
        layer = self.project.layer_manager.get_nth_oldest_layer_of_type("shapefile", 1)
        layer.create_rings()
        out = layer.calc_output_feature_list()
        print(out)

        uri = os.path.join(os.getcwd(), "tmp.test_shapefile_formats.shp")
        shapefile.write_feature_list_as_shapefile(uri, out, self.proj)
        uri2 = os.path.join(os.getcwd(), "tmp.test_shapefile_formats.bna")
        shapefile.write_feature_list_as_bna(uri2, out, self.proj)

        loader = shapefile.ShapefileLoader()
        layers = loader.load_layers(uri, self.project.layer_manager)
        test_layer = layers[0]
        test_out = test_layer.calc_output_feature_list()
        layers = loader.load_layers(uri2, self.project.layer_manager)
        test2_layer = layers[0]
        test2_out = test_layer.calc_output_feature_list()
        print(test_out)
        print(test2_out)
        assert test2_out == test_out

    def test_annotation_rect(self):
        lm = self.project.layer_manager
        parent = ly.AnnotationLayer(manager=lm)
        lm.insert_layer([3], parent)
        
        layer = ly.RectangleVectorObject(manager=lm)
        layer.set_opposite_corners(
            (-16.6637485204,-1.40163099748),
            (9.65688930428,-19.545688433))
        lm.insert_layer([3, 2], layer)
        
        # Calculate bounds of the annotation layers to set up their
        # points/lines arrays
        parent.update_bounds()
        
        save_and_check_layer(layer, 0, "annotation_rect", self.project)

    def test_annotation_multi_rect(self):
        lm = self.project.layer_manager
        parent = ly.AnnotationLayer(manager=lm)
        lm.insert_layer([3], parent)
        
        layer1 = ly.RectangleVectorObject(manager=lm)
        layer1.set_opposite_corners(
            (-20, 20),
            (-10, 30))
        lm.insert_layer([3, 2], layer1)
        
        layer2 = ly.RectangleVectorObject(manager=lm)
        layer2.set_opposite_corners(
            (0, 10),
            (10, 0))
        lm.insert_layer([3, 2], layer2)
        
        # Calculate bounds of the annotation layers to set up their
        # points/lines arrays
        parent.update_bounds()
        
        out = parent.calc_output_feature_list()
        print(out)
        boundary = out[0][2].start_index
        print(boundary)
        print(boundary.points)
        print(boundary.point_indexes)
        label = "multi_rect"
        uri = os.path.join(os.getcwd(), f"tmp.test_{label}.shp")
        shapefile.write_feature_list_as_shapefile(uri, out, self.project.projection)
        uri2 = os.path.join(os.getcwd(), f"tmp.test_{label}.bna")
        shapefile.write_feature_list_as_bna(uri2, out, self.project.projection)

    def test_annotation_polyline(self):
        lm = self.project.layer_manager
        parent = ly.AnnotationLayer(manager=lm)
        lm.insert_layer([3], parent)
        
        layer = ly.PolylineObject(manager=lm)
        layer.set_points([
            (-15,-2),
            (5, -8),
            (10, -20),
            (8, -5),
            (-17, -10),
            ])
        lm.insert_layer([3, 2], layer)
        
        # Calculate bounds of the annotation layers to set up their
        # points/lines arrays
        parent.update_bounds()
        
        save_and_check_layer(layer, layer.center_point_index + 1, "annotation_polyline", self.project)

    def test_annotation_polygon(self):
        lm = self.project.layer_manager
        parent = ly.AnnotationLayer(manager=lm)
        lm.insert_layer([3], parent)
        
        layer = ly.PolygonObject(manager=lm)
        layer.set_points([
            (-15,-2),
            (5, -8),
            (10, -20),
            (8, -5),
            (-17, -10),
            ])
        lm.insert_layer([3, 2], layer)
        
        # Calculate bounds of the annotation layers to set up their
        # points/lines arrays
        parent.update_bounds()
        
        save_and_check_layer(layer, layer.center_point_index + 1, "annotation_polygon", self.project)

    def test_annotation_multi_polyline(self):
        lm = self.project.layer_manager
        parent = ly.AnnotationLayer(manager=lm)
        lm.insert_layer([3], parent)
        
        layer = ly.PolylineObject(manager=lm)
        layer.set_points([
            (-20, 0),
            (-10, 0),
            (-10, 5),
            (-5, 10),
            (-10, 15),
            (-20, 10),
            ])
        lm.insert_layer([3, 2], layer)
        
        layer = ly.PolylineObject(manager=lm)
        layer.set_points([
            (0, 15),
            (10, 17),
            (10, 20),
            (8, 22),
            (10, 30),
            (5, 27),
            (-5, 20),
            ])
        lm.insert_layer([3, 2], layer)
        
        # Calculate bounds of the annotation layers to set up their
        # points/lines arrays
        parent.update_bounds()
        
        out = parent.calc_output_feature_list()
        print(out)
        boundary = out[0][2].start_index
        print(boundary)
        print(boundary.points)
        print(boundary.point_indexes)
        label = "multi_polyline"
        uri = os.path.join(os.getcwd(), f"tmp.test_{label}.shp")
        shapefile.write_feature_list_as_shapefile(uri, out, self.project.projection)
        uri2 = os.path.join(os.getcwd(), f"tmp.test_{label}.bna")
        shapefile.write_feature_list_as_bna(uri2, out, self.project.projection)

    def test_annotation_multi_many(self):
        lm = self.project.layer_manager
        parent = ly.AnnotationLayer(manager=lm)
        lm.insert_layer([3], parent)
        
        layer1 = ly.RectangleVectorObject(manager=lm)
        layer1.set_opposite_corners(
            (-20, 20),
            (-10, 30))
        lm.insert_layer([3, 2], layer1)
        
        layer2 = ly.RectangleVectorObject(manager=lm)
        layer2.set_opposite_corners(
            (0, 10),
            (10, 0))
        lm.insert_layer([3, 2], layer2)
        
        layer = ly.PolylineObject(manager=lm)
        layer.set_points([
            (-20, 0),
            (-10, 0),
            (-10, 5),
            (-5, 10),
            (-10, 15),
            (-20, 10),
            ])
        lm.insert_layer([3, 2], layer)
        
        layer = ly.PolygonObject(manager=lm)
        layer.set_points([
            (0, 15),
            (10, 17),
            (10, 20),
            (8, 22),
            (10, 30),
            (5, 27),
            (-5, 20),
            ])
        lm.insert_layer([3, 2], layer)
        
        # Calculate bounds of the annotation layers to set up their
        # points/lines arrays
        parent.update_bounds()
        
        out = parent.calc_output_feature_list()
        print(out)
        boundary = out[0][2].start_index
        print(boundary)
        print(boundary.points)
        print(boundary.point_indexes)
        label = "multi_many"
        uri = os.path.join(os.getcwd(), f"tmp.test_{label}.shp")
        with pytest.raises(RuntimeError):
            # Shapefiles can't contain both polygons and polylines
            shapefile.write_feature_list_as_shapefile(uri, out, self.project.projection)
        uri = os.path.join(os.getcwd(), f"tmp.test_{label}.json")
        shapefile.write_feature_list_as_shapefile(uri, out, self.project.projection)
        uri2 = os.path.join(os.getcwd(), f"tmp.test_{label}.bna")
        shapefile.write_feature_list_as_bna(uri2, out, self.project.projection)


if __name__ == "__main__":
    import sys

    t = TestFeatureList()
    t.setup()
    # t.test_delete_polygon()
    # t.setup()
    # t.test_delete_hole()
    # t.test_shapefile_formats()
    # t.test_annotation_rect()
    # t.test_annotation_polyline()
    # t.test_annotation_polygon()
    # t.test_annotation_multi_rect()
    t.test_annotation_multi_many()
    # t.test_annotation_multi_polyline()
