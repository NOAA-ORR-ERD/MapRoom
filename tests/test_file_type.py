#!/usr/bin/env python

"""
Tests for the binary file types

Designed to be run with nose

"""
import os

from nose.tools import *

from maproom.file_type import binary

from peppy2.utils.file_guess import FileGuess

this_dir = os.path.split(os.path.abspath(__file__))[0]
test_data_dir = os.path.normpath(os.path.join(this_dir,"../../TestData/")) 

def test_identify_nc_particles():
    """ make sure it works with a netcdf3 (CDF) format file """
    recognizer = binary.NC_ParticleRecognizer()
    guess = FileGuess( os.path.join(test_data_dir, "NC_particles", "Mobile_test.nc") )

    assert recognizer.identify(guess) == binary.NC_ParticleRecognizer.id

def test_identify_nc_particles_ugrid():
    """ make sure it rejects a ugrid netcdf file """
    recognizer = binary.NC_ParticleRecognizer()
    guess = FileGuess( os.path.join(test_data_dir, "UGrid", "21_triangles.nc") )


    assert recognizer.identify(guess) is None

def test_identify_hdf():
    """ make sure it works with a netcdf4 (HDF) ugrid file """
    recognizer = binary.HDF5Recognizer()
    guess = FileGuess( os.path.join(test_data_dir, "UGrid", "21_triangles.nc") )

    assert recognizer.identify(guess) == binary.HDF5Recognizer.id
