#!/usr/bin/env python

# simple unit tests for contour code.

import numpy as np
import pytest
import py_contour
import random

def test_basic():
    """ does it run at all """
    x = range(3)
    y = range(4)
    z = np.arange(12).reshape((3,4))
    segs = py_contour.contour(z,x,y,[2,3])

def test_mismatched_dims():
    """ x and y need to be the right size """
    x = range(3)
    y = range(4)
    z = np.arange(12).reshape((3,4))
    with pytest.raises(ValueError):
        segs = py_contour.contour(z,x,x,[2,3])

def test_sort_segments_triangle():
    """on triangle -- easiers option"""
    # about as easy as it gets
    segs = [((0,0),(1,1)),
            ((1,1),(0,2)),
            ((0,2),(0,0)),
            ]
    polygons = py_contour.sort_segments(segs)
    assert len(polygons) == 1
    assert polygons[0] == [(0, 2), (0, 0), (1, 1), (0, 2)]

def segments_from_poly(poly):
    """
    build a segment list from a polygon
    
    utility for making tests
    """
    segs = []
    for i in range(len(poly)-1):
        segs.append( tuple(poly[i:i+2]) )
    return segs

def merge_polys(*args):
    """
    build a randomized segment list from a bunch of polygons
    
    utility for making tests
    """
    segs = []
    for poly in args:
        segs += segments_from_poly(poly)
    # mix them up:
    random.seed(10) # so it will always be the same
    random.shuffle(segs)

    return segs


def test_polyline():
    """ a non-closed polyline -- should return as a single line """
    polyline = [(1, 0), (0, 1), (1, 3), (0, 4)]
    segs = merge_polys(polyline)

    polygons = py_contour.sort_segments(segs)
    for poly in polygons:
        print poly
    assert len(polygons) == 1
    # Note: order could be different if algorithm changes
    assert polygons[0] == [(1, 0), (0, 1), (1, 3), (0, 4)]


def test_sort_segments_two_polys():
    poly1 = [(0, 2), (0, 0), (1, 1), (0, 2)]
    poly2 = [(2, 0), (2, 2), (4, 2), (5, -1), (3, -2), (2, 0) ]

    segs = merge_polys(poly1, poly2)

    polygons = py_contour.sort_segments(segs)
    for poly in polygons:
        print poly
    assert len(polygons) == 2
    # Note: order could be different if algorithm changes
    assert polygons[0] == [(2, 2), (4, 2), (5, -1), (3, -2), (2, 0), (2, 2)]
    assert polygons[1] == [(0, 2), (0, 0), (1, 1), (0, 2)]

def test_sort_segments_polygon_polyline():
    """ this one has a polgon, and non-closed poly line """
    poly1 = [(0, 2), (0, 0), (1, 1), (0, 2)]
    poly2 = [(2, 0), (2, 2), (4, 2), (5, -1), (3, -2), (2, 0) ]
    polyline = [(1, 0), (0, 1), (1, 3), (0, 4)]

    segs = merge_polys(poly1, poly2, polyline)

    polygons = py_contour.sort_segments(segs)
    print "Polygons:"
    for poly in polygons:
        print poly
    assert len(polygons) == 3
    # Note: order could be different if algorithm changes
    assert polygons[0] == [(4, 2), (5, -1), (3, -2), (2, 0), (2, 2), (4, 2)]
    assert polygons[1] == [(0, 0), (1, 1), (0, 2), (0, 0)]
    assert polygons[2] == [(1, 0), (0, 1), (1, 3), (0, 4)]

def test_touching_polys():
    """ two polygons that touch -- should come back as one
    
        NOTE: It would be better to have it come back as two, but
              Very hard to get a unique solution to this! And this could
              return it either crossing or not...
    """
    poly1 = [(0, 0), (1, 1), (0, 1), (-1, 1), (0, 0)]
    poly2 = [(-1, -1), (0, 0), (1, -1), (0, -1), (-1, -1)]

    segs = merge_polys(poly1, poly2)
    
    polygons = py_contour.sort_segments(segs)
    print "Polygons:"
    for poly in polygons:
        print poly

    assert len(polygons) == 1
    # Note: order could be different if algorithm changes
    assert polygons[0] == [(0, 0), (1, -1), (0, -1), (-1, -1), (0, 0), (1, 1), (0, 1), (-1, 1), (0, 0)]
    



if __name__ == "__main__":
    # test_polyline()
    # test_sort_segments_triangle()
    # test_sort_segments_two_polys()
    # test_sort_segments_polygon_polyline()
    # test_touching_polys()
    pass

