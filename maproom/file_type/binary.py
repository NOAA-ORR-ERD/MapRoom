from traits.api import HasTraits, provides

from peppy2.file_type.i_file_recognizer import IFileRecognizer

@provides(IFileRecognizer)
class HDF5Recognizer(HasTraits):
    """Recognizer for HDF5
    
    """
    id = "application/x-hdf"
    
    # GDAL recognizes HDF files, so this needs to be before the GDAL recognizer
    before = "image/x-gdal"
    
    def identify(self, guess):
        byte_stream = guess.get_utf8()
        if byte_stream[0:8] == "\211HDF\r\n\032\n":
            return self.id
