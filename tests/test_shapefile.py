import os

import numpy as np

from mock import *

from maproom.layers import loaders, PolygonParentLayer
from maproom.library.Boundary import Boundaries, PointsError

from maproom.renderer.gl.data_types import make_points_from_xy


class TestEmptyShapefile(object):
    def setup(self):
        self.project = MockProject()

    def test_empty_layer(self):
        lm = self.project.layer_manager
        layer = PolygonParentLayer(manager=lm)
        layer.new()
        lm.insert_loaded_layer(layer)

        points = make_points_from_xy([(1.,2.), (1., 5.), (0., 3.)])
        layer.replace_ring(0, points, True)
        assert 1 == len(layer.rings)
        assert 1 == len(layer.geometry_list)
        assert 3 == len(layer.ring_adjacency)

        points = make_points_from_xy([(8.,2.), (9., 5.), (8., 3.), (9.,3.)])
        layer.replace_ring(0, points)
        assert 1 == len(layer.rings)
        assert 1 == len(layer.geometry_list)
        assert 4 == len(layer.ring_adjacency)

        points = make_points_from_xy([(8.,5.), (9., 7.), (8., 4.), (9.,1.), (10., 4.)])
        layer.replace_ring(0, points, new_boundary=True)
        assert 2 == len(layer.rings)
        assert 2 == len(layer.geometry_list)
        assert 9 == len(layer.ring_adjacency)

class TestBNAShapefile(object):
    def setup(self):
        self.project = MockProject()
        self.project.load_file("../TestData/BNA/00003polys_000035pts.bna", "application/x-maproom-bna")
        self.bna = self.project.layer_manager.get_nth_oldest_layer_of_type("shapefile", 1)
        self.bna.create_rings()

    def test_simple(self):
        print(self.bna)
        print(self.bna.points)
        print(len(self.bna.points))
        assert 33 == np.alen(self.bna.points)
        
        uri = os.path.join(os.getcwd(), "tmp.3polys.shp")
        loaders.save_layer(self.bna, uri)
        
        layer = self.project.raw_load_first_layer(uri, "application/x-maproom-shapefile")
        assert 33 == np.alen(layer.points)

    def test_add_polygon(self):
        uri = os.path.normpath(os.getcwd() + "/../TestData/Verdat/000011pts.verdat")
        layer = self.project.raw_load_first_layer(uri, "application/x-maproom-verdat")
        boundary = layer.select_outer_boundary()
        self.bna.create_rings()
        self.bna.replace_ring(0, boundary.get_points(), 0, True)
        assert len(self.bna.rings) == 4

    def test_add_commit_polygon(self):
        uri = os.path.normpath(os.getcwd() + "/../TestData/Verdat/000011pts.verdat")
        layer = self.project.raw_load_first_layer(uri, "application/x-maproom-verdat")
        layer.feature_code = 0
        layer.ring_indexes = []
        self.bna.commit_editing_layer(layer)
        assert len(self.bna.rings) == 5
        uri = os.path.join(os.getcwd(), "tmp.add_commit.shp")
        loaders.save_layer(self.bna, uri)

    def test_replace_polygon(self):
        uri = os.path.normpath(os.getcwd() + "/../TestData/Verdat/000011pts.verdat")
        layer = self.project.raw_load_first_layer(uri, "application/x-maproom-verdat")
        layer.feature_code = 0
        layer.ring_indexes = [1, 2]
        layer.new_boundary = False
        layer.num_boundaries = 2
        layer.new_hole = False
        self.bna.commit_editing_layer(layer)
        assert len(self.bna.rings) == 3

    def test_replace_and_add_polygon(self):
        uri = os.path.normpath(os.getcwd() + "/../TestData/Verdat/000011pts.verdat")
        layer = self.project.raw_load_first_layer(uri, "application/x-maproom-verdat")
        layer.feature_code = 0
        layer.ring_indexes = [1]
        layer.new_boundary = False
        layer.num_boundaries = 2
        layer.new_hole = False
        self.bna.commit_editing_layer(layer)
        assert len(self.bna.rings) == 4

    def test_delete_polygon(self):
        self.bna.delete_ring(0)
        assert len(self.bna.rings) == 2

class TestESRIShapefile(object):
    def setup(self):
        self.project = MockProject()
        self.project.load_file("../TestData/Shapefiles/square.shp", "application/x-maproom-shapefile")
        self.layer = self.project.layer_manager.get_nth_oldest_layer_of_type("shapefile", 1)
        self.layer.create_rings()

    def test_delete_polygon(self):
        self.layer.delete_ring(0)
        assert len(self.layer.rings) == 0

    def test_delete_hole(self):
        self.layer.delete_ring(1)
        assert len(self.layer.rings) == 1

    def test_save_compare(self):
        uri = os.path.normpath(os.getcwd() + "/tmp.square.shp")
        loaders.save_layer(self.layer, uri)
        layer = self.project.raw_load_first_layer(uri, "application/x-maproom-shapefile")
        print(layer.points)
        print(self.layer.points)
        assert layer.points[0]['x'] == self.layer.points[0]['x']



if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        project = MockProject()
        uri = sys.argv[1]
        project.load_file(uri, "application/x-maproom-shapefile")
        for layer in project.layer_manager.flatten():
            try:
                layer.create_rings()
            except AttributeError:
                pass
            else:
                for r in layer.rings:
                    print(f"ring: {r}")
                    p = layer.points[r["start"]:r["start"] + r["count"]]
                    print(f"points: {p}")
        out_uri = os.path.join(os.getcwd(), "out.shp")
        loaders.save_layer(layer, out_uri)
        reloaded_layer = project.raw_load_first_layer(out_uri, "application/x-maproom-shapefile")
        assert np.alen(reloaded_layer.points) == np.alen(layer.points)
    else:
        t = TestBNAShapefile()
        t.setup()
        t.test_simple()
        t.setup()
        t.test_add_polygon()
        t.setup()
        t.test_delete_polygon()
        t = TestESRIShapefile()
        t.setup()
        t.test_delete_polygon()
        t.setup()
        t.test_delete_hole()
