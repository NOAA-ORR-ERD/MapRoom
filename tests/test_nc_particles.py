"""
test for various nc_particles functionality
"""

import os

import numpy as np

from mock import maproom_dir

from maproom.layers.loaders.nc_particles import nc_particles_file_loader

sample_nc_file = os.path.join(maproom_dir, "TestData/NC_particles/Mobile_test.nc")

def test_iterator():
    for i, data in enumerate( nc_particles_file_loader(sample_nc_file) ):
        print "next time step:"
        points, status_codes, time = data
        assert points.shape == (1000, 2)
    assert i == 24
