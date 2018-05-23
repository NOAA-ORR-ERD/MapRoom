import os

import numpy as np

from shapely.geometry import box, Polygon, MultiPolygon, LineString, MultiLineString

from mock import *

from maproom.layers import loaders, TriangleLayer
from maproom.library.Boundary import Boundaries


class ClipBase(object):
    def test_clip_quarters(self):
        """from http://pcjericks.github.io/py-gdalogr-cookbook/geometry.html#quarter-polygon-and-create-centroids
        """
        geom_poly = self.get_clip_object()
        print("geom_poly")
        print(geom_poly)
        
        # Create 4 square polygons
        geom_poly_envelope = geom_poly.bounds
        minX = geom_poly_envelope[0]
        minY = geom_poly_envelope[1]
        maxX = geom_poly_envelope[2]
        maxY = geom_poly_envelope[3]

        '''
        coord0----coord1----coord2
        |           |           |
        coord3----coord4----coord5
        |           |           |
        coord6----coord7----coord8
        '''
        coord0 = minX, maxY
        coord1 = minX+(maxX-minX)/2, maxY
        coord2 = maxX, maxY
        coord3 = minX, minY+(maxY-minY)/2
        coord4 = minX+(maxX-minX)/2, minY+(maxY-minY)/2
        coord5 = maxX, minY+(maxY-minY)/2
        coord6 = minX, minY
        coord7 = minX+(maxX-minX)/2, minY
        coord8 = maxX, minY

        clipTopLeft = box(coord3[0], coord3[1], coord1[0], coord1[1])
        print("TopLeft")
        print(clipTopLeft.bounds)
        clipBottomRight = box(coord7[0], coord7[1], coord5[0], coord5[1])
        print("BottomRight")
        print(clipBottomRight.bounds)
        
        # Note: polygons must be in a list
        clip = MultiPolygon([clipTopLeft, clipBottomRight])
        
        print("clip")
        print(clip.is_valid)
        print(clip.bounds)

        # Intersect 4 squares polygons with test polygon
        clip_result = clip.intersection(geom_poly)
        print("clip_result")
        print(clip_result)
        for i, x in enumerate(clip_result.geoms):
            print("clip result %d:" % (i, ))
            print(x)


class TestShapelyVerdat(ClipBase):
    def setup(self):
        self.manager = MockManager()
        self.verdat = self.manager.load_first_layer("../TestData/Verdat/negative-depth-triangles.verdat", "application/x-maproom-verdat")

    def get_clip_points(self):
        boundary = self.manager.get_outer_boundary(self.verdat)
        print(boundary)
        points = boundary.get_xy_point_float64()
        print("points tuples:")
        print(points)
        print(points.__array_interface__)
        print(points.shape)
        print(id(points))
        print(points.flags)
#        points = np.require(points, np.float64, ["C", "OWNDATA"])
#        print id(points)
#        print points.flags
        return points

    def get_clip_object(self):
        points = self.get_clip_points()
        poly = Polygon(points)
        print(poly.bounds)
        return poly


class TestShapelyUGridLines(ClipBase):
    def setup(self):
        self.manager = MockManager()
        self.verdat = self.manager.load_first_layer("../TestData/UGrid/jetty.nc", "application/x-nc_ugrid")

    def get_clip_points(self):
        boundary = self.manager.get_outer_boundary(self.verdat)
        print(boundary)
        points = boundary.get_xy_point_float64()
        print("points tuples:")
        print(points)
        print(points.__array_interface__)
        print(points.shape)
        print(id(points))
        print(points.flags)
#        points = np.require(points, np.float64, ["C", "OWNDATA"])
#        print id(points)
#        print points.flags
        return points

    def get_clip_object(self):
        points = self.get_clip_points()
        poly = LineString(points)
        print(poly.bounds)
        return poly


class TestShapelyBNA(ClipBase):
    def setup(self):
        self.manager = MockManager()
        self.layer = self.manager.load_first_layer("../TestData/BNA/00011polys_001486pts.bna", "application/x-maproom-bna")

    def get_clip_points(self):
        poly = self.layer.rings
        start = poly.start[3]
        count = poly.count[3]
        boundary = self.layer.points
        print(boundary)
        points = np.c_[boundary.x[start:start + count], boundary.y[start:start + count]]
        print("points tuples:")
        print(points)
        print(points.__array_interface__)
        print(points.shape)
        print(id(points))
        print(points.flags)
        points = np.require(points, np.float64, ["C", "OWNDATA"])
        print(id(points))
        print(points.flags)
        return points

    def get_clip_object(self):
        points = self.get_clip_points()
        poly = Polygon(points)
        print(poly.bounds)
        return poly

    def test_clip_rect(self):
        w_r = ((-117.21561639930427, 32.56583226472319), (-117.03282298967024, 32.67074950148222))
        self.layer.crop_rectangle(w_r)

    def test_clip_multipolygon(self):
        w_r = ((-117.2658988858227, 32.63249047752771), (-117.07014217231291, 32.69989978984453))
        self.layer.crop_rectangle(w_r)


if __name__ == "__main__":
#    t = TestShapelyVerdat()
#    t.setup()
#    t.test_clip_quarters()
#    t = TestShapelyUGridLines()
#    t.setup()
#    t.test_clip_quarters()
    t = TestShapelyBNA()
    t.setup()
    t.test_clip_rect()
    t.test_clip_multipolygon()
