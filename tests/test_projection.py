#!/usr/bin/env python

"""
tests for the Projection classes
"""
import pytest

from maproom.library.projection import Projection, NullProjection

def test_init():
	"""can it even be initialized"""
	p = Projection("+proj=merc +units=m +over")

# def test_init_epsg():
#     """ can it be initialized with an epsg code"""

#     p = Projection("epsg:3857") # spherical mercater

def test_identity():
    """
    we don't want folks to use a proj latlong projection -- issues with conversion to radians
    """
    with pytest.raises(ValueError):
        p = Projection("+proj=latlong")

    with pytest.raises(ValueError):
        p = Projection("+proj=longlat")


def test_Null():
    proj = NullProjection()
    orig_point = (-62.242000,  12.775000)
    new_point = proj.projection(orig_point[0], orig_point[1])

    assert orig_point == new_point

