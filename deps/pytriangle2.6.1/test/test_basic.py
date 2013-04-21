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
                       dtype=np.float32)
    depths = np.array( (1, 2, 3, 4),
                              dtype=np.float32)
    lines = np.array( ((0,1),
                       (1,2),
                       (2,0),
                       ),
                      dtype=np.uint32)
    holes = np.array( ((10, 10),),
                      dtype = np.float32)
    
    out_points_xy, out_points_z, out_lines, out_triangles  =  \
          pytriangle.triangulate_simple(points,
                                        depths,
                                        lines,
                                        holes,
                                        )
     
    print "out_points_xy:", out_points_xy
    print "out_points_z:",out_points_z
    print "out_lines:",out_lines
    print "out_triangles:",out_triangles
    print out_points_xy.dtype
    print out_triangles.dtype
    
    assert_array_equal(points, out_points_xy)
    assert_array_equal(depths, out_points_z)
    
    
    #raise Exception
