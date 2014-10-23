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
test_data_dir = os.path.normpath(os.path.join(this_dir,"../TestData/")) 


## fixme -- need test to make sure that only one identifier returns for each file.

## Auto-generated tests from various file types, etc.
FILES = [ ( (os.path.join(test_data_dir, "NC_particles", "Mobile_test.nc") ), binary.NC_ParticleRecognizer ),
          ( (os.path.join(test_data_dir, "UGrid", "21_triangles.nc") ), binary.HDF5Recognizer ),
        ]

def test_identify():
    for filename, identifier in  FILES:
        yield check_filetype, filename, identifier

def check_filetype(filename, identifier):
    recognizer = identifier()
    guess = FileGuess( filename)
    assert recognizer.identify(guess) == identifier.id

def test_identify_nc_particles_ugrid():
    """ make sure it rejects a ugrid netcdf file """
    recognizer = binary.NC_ParticleRecognizer()
    guess = FileGuess( os.path.join(test_data_dir, "UGrid", "21_triangles.nc") )
    assert recognizer.identify(guess) is None
