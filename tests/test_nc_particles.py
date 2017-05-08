"""
test for various nc_particles functionality
"""

import os

import numpy as np

from mock import maproom_dir

from maproom.layers.loaders.nc_particles import nc_particles_file_loader

def test_iterator():
    sample_nc_file = os.path.join(maproom_dir, "TestData/NC_particles/Mobile_test.nc")
    for i, data in enumerate( nc_particles_file_loader(sample_nc_file) ):
        points, status_codes, status_code_map, time, warnings = data
        print "next time step:", points.shape
        assert points.shape == (100, 2)
        assert warnings is None
    assert i == 12

def test_old_status():
    sample_nc_file = os.path.join(maproom_dir, "TestData/NC_particles/gnome_1.3.9_particles.nc")
    for i, data in enumerate( nc_particles_file_loader(sample_nc_file) ):
        points, status_codes, status_code_map, time, warnings = data
        print "next time step:", points.shape
        assert points.shape == (1000, 2)
        assert warnings is None
    assert i == 0

def test_spurious_values():
    sample_nc_file = os.path.join(maproom_dir, "TestData/NC_particles/new_york_harbor_save.nc")
    for i, data in enumerate( nc_particles_file_loader(sample_nc_file) ):
        points, status_codes, status_code_map, time, warnings = data
        print "next time step:", i, points.shape
        if i < 14 or i == 18 or i > 37:
            assert warnings is None
        else:
            assert "spurious" in str(warnings)
    assert i == 36

