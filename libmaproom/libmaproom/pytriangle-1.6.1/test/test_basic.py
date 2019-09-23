#!/usr/bin/env python

"""
test the very basics of pytriangle
"""

# these tests assume that pytriangle is installed
import pytriangle
import numpy as np
from numpy.testing import assert_array_equal

##fixme: need tests for mis-matched data --i.e number of xy points not matching the number of depths.

def test_import():
    import pytriangle

#def test_dummy():
#    """ a dummy test to print stuff, etc """
#    print "I'm here"
#    print dir(pytriangle)
#    raise Exception

def test_triangulate():
    # about the simplest case there is

    points = np.array( (( 0,  0),
                        (10,  0),
                        ( 5,  5),
                        ( 5,  2),
                        ),
                       dtype=np.float64)
    depths = np.array( (1, 2, 3, 4),
                              dtype=np.float32)
    lines = np.array( ((0,1),
                       (1,2),
                       (2,0),
                       ),
                      dtype=np.uint32)
    holes = np.array( ((10, 10),),
                      dtype = np.float64)

    out_points_xy, out_points_z, out_lines, out_triangles  =  \
          pytriangle.triangulate_simple("", points, depths, lines, holes)

    print("out_points_xy:", out_points_xy)
    print("out_points_z:",out_points_z)
    print("out_lines:",out_lines)
    print("out_triangles:",out_triangles)
    print(out_points_xy.dtype)
    print(out_triangles.dtype)

    assert_array_equal(points, out_points_xy)
    assert_array_equal(depths, out_points_z)


def test_convex_hull():
    """
    Do just a convex hull, rather than a full triangulation

    more here to show how it's done
    """
    points_xy = np.random.rand(20, 2)
    points_xy = points_xy.astype(np.float64)  # jsut to make sure
    points_z = np.zeros((points_xy.shape[0]), dtype=np.float32)
    lines = np.zeros((0, 2), dtype=np.uint32)
    holes = np.zeros((0, 2), dtype=np.float64)

    (out_points_xy,
     out_points_z,
     out_lines,
     out_triangles) = pytriangle.triangulate_simple("-c",
                                                    points_xy,
                                                    points_z,
                                                    lines,
                                                    holes)
    hull_coords = out_points_xy[out_lines[:, 0]]

    # not sure what to test here:
    assert hull_coords.shape[1] == 2
    assert hull_coords.dtype == np.float64


if __name__ == "__main__":
    test_triangulate()
