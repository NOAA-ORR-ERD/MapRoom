"""
test for various nc_particles functionality
"""

import os

from nose.tools import *

import numpy as np

#from mock import *

from maproom.layers.loaders.nc_particles import nc_particles_file_loader

this_dir = os.path.split(os.path.abspath(__file__))[0]
test_data_dir = os.path.normpath(os.path.join(this_dir,"../TestData/")) 

sample_nc_file = os.path.join(test_data_dir, "NC_particles/Mobile_test.nc")

def test_iterator():
    for i, data in enumerate( nc_particles_file_loader(sample_nc_file) ):
        print "next time step:"
        assert data['f_points'].shape == (1000, 2)
    assert i == 24

