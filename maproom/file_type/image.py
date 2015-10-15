from osgeo import gdal
gdal.UseExceptions()

from traits.api import HasTraits, provides

from omnimon.file_type.i_file_recognizer import IFileRecognizer

import logging
log = logging.getLogger(__name__)

@provides(IFileRecognizer)
class GDALRecognizer(HasTraits):
    """Check to see if GDAL can open this as a raster file.

    Some vector files can be opened by GDAL but these are not recognized by
    this class.
    """
    id = "image/x-gdal"
    
    before = "image/common"
    
    def identify(self, guess):
        try:
            dataset = gdal.Open(guess.metadata.uri)
        except RuntimeError:
            log.debug("GDAL can't open %s; not an image")
            return None
        if dataset.RasterCount > 0:
            return "image/x-gdal"
        return None
