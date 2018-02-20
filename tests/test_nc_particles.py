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

if __name__ == "__main__":
    import sys
    from post_gnome import nc_particles

    for testfile in sys.argv[1:]:
        reader = nc_particles.Reader(testfile)
        print("\n\n\n%s *****************" % testfile)
        status_id = "status_codes"
        try:
            attributes = reader.get_attributes(status_id)
        except KeyError:
            # try "status" instead
            status_id = "status"
            attributes = reader.get_attributes(status_id)
        print("Using '%s' for status code identifier" % status_id)
        meanings = attributes['flag_meanings']
        print("Meanings: %s" % str(meanings))
        print("Variables: %s" % str(reader.variables))

        for timestep in range(len(reader.times)):
            data = reader.get_timestep(timestep, variables=reader.variables)
            for variable in reader.variables:
                if variable in data:
                    d = data[variable]
                    print "timestep %d:" % timestep, variable, d.dtype, d.shape
                    #print d
                else:
                    print variable, "(not in timestep %d)" % timestep
