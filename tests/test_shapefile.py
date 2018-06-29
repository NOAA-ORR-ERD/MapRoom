import os

import numpy as np

from pyugrid.ugrid import UGrid

from mock import *

from maproom.layers import loaders, TriangleLayer
from maproom.library.Boundary import Boundaries, PointsError

class TestShapefile(object):
    def setup(self):
        self.project = MockProject()
        self.project.load_file("../TestData/BNA/00003polys_000035pts.bna", "application/x-maproom-bna")
        self.bna = self.project.layer_manager.get_layer_by_invariant(1)
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
        self.bna.replace_ring_with_resizing(0, boundary, True, False)


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
        t = TestShapefile()
        t.setup()
        t.test_simple()
        t.test_add_polygon()
