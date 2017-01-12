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
        points, status_codes, status_code_map, time, warnings = data
        print "next time step:", points.shape
        assert points.shape == (100, 2)
    assert i == 12
