from osgeo import gdal
gdal.UseExceptions()

from traits.api import HasTraits, provides

from peppy2.file_type.i_file_recognizer import IFileRecognizer

import logging
log = logging.getLogger(__name__)

@provides(IFileRecognizer)
class GDALRecognizer(HasTraits):
    """Default plain text identifier based on percentage of non-ASCII bytes.
    
    """
    id = "image/x-gdal"
    
    before = "image/common"
    
    def identify(self, guess):
        try:
            dataset = gdal.Open(guess.metadata.uri)
        except RuntimeError:
            log.debug("GDAL can't open %s; not an image")
            return None
        return "image/x-gdal"
