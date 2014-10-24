from osgeo import gdal
gdal.UseExceptions()

from traits.api import HasTraits, provides

from peppy2.file_type.i_file_recognizer import IFileRecognizer

import logging
log = logging.getLogger(__name__)

@provides(IFileRecognizer)
class GDALRecognizer(HasTraits):
    """Check to see if GDAL can open this -- if so, it returns OK

       fixme -- ideally, this would check not ony GDAL, but whether
       it is a dataset we know how to deal with.    
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
