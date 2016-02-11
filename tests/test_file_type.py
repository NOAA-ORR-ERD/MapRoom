"""
Tests for the binary file types
"""
import os

from mock import *

from maproom.file_type import binary, text, image

from omnivore.utils.file_guess import FileGuess

this_dir = os.path.split(os.path.abspath(__file__))[0]
test_data_dir = os.path.normpath(os.path.join(this_dir,"../TestData/")) 


## List for auto-generated tests from various file types, etc.
## fixme -- maybe this should scan the TestData dir instead of hard coding all of them...
FILES = [ 
          # nc_particles
          ( (os.path.join(test_data_dir, "NC_particles", "Mobile_test.nc") ), binary.NC_ParticleRecognizer ),
          ( (os.path.join(test_data_dir, "NC_particles", "script_guam.nc") ), binary.NC_ParticleRecognizer ),
          # ugrid
          ( (os.path.join(test_data_dir, "UGrid", "21_triangles.nc") ), binary.UGRID_Recognizer ),
          ( (os.path.join(test_data_dir, "UGrid", "2_triangles.nc") ), binary.UGRID_Recognizer ),
          ( (os.path.join(test_data_dir, "UGrid", "full_example.nc") ), binary.UGRID_Recognizer ),
          # verdat
          ( (os.path.join(test_data_dir, "Verdat", "000011pts.verdat") ), text.VerdatRecognizer ),
          ( (os.path.join(test_data_dir, "Verdat", "MobileBay.dat") ), text.VerdatRecognizer ),
          # BNA
          ( (os.path.join(test_data_dir, "BNA", "00011polys_001486pts.bna") ), text.BNARecognizer ),
          ( (os.path.join(test_data_dir, "BNA", "Haiti.bna") ), text.BNARecognizer ),
          # GDAL images
          ( (os.path.join(test_data_dir, "ChartsAndImages", "11361_1.KAP") ), image.GDALRecognizer),
          ( (os.path.join(test_data_dir, "ChartsAndImages", "NOAA18649_small.png") ), image.GDALRecognizer),
          ( (os.path.join(test_data_dir, "ChartsAndImages", "Admiralty-0463-2.tif") ), image.GDALRecognizer),
         ]

## set of all recognizers from the above list
RECOGNIZERS = { item[1] for item in FILES }


def test_identify():
    """test generator for positive check"""
    for filename, identifier in  FILES:
        yield check_filetype, filename, identifier

def check_filetype(filename, identifier):
    recognizer = identifier()
    guess = FileGuess(filename)
    assert recognizer.identify(guess) == identifier.id


def test_not_identity():
    """test generator for negative check"""
    for recognizer in RECOGNIZERS:
        for filename, identifier in FILES:
            if recognizer is not identifier:
                yield check_not_filetype, filename, recognizer

def check_not_filetype(filename, Recognizer):
    recognizer = Recognizer()
    guess = FileGuess(filename)
    print filename, guess, Recognizer.id, recognizer.identify(guess)
    assert not recognizer.identify(guess) == Recognizer.id

def test_identify_nc_particles_ugrid():
    """ make sure it rejects a ugrid netcdf file """
    recognizer = binary.NC_ParticleRecognizer()
    guess = FileGuess(os.path.join(test_data_dir, "UGrid", "21_triangles.nc"))
    assert recognizer.identify(guess) is None
