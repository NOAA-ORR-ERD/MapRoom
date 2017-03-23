from fs.opener import opener
from osgeo import gdal
gdal.UseExceptions()

from traits.api import HasTraits, provides

from omnivore.file_type.i_file_recognizer import IFileRecognizer

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
            fs, relpath = opener.parse(guess.metadata.uri)
            if not fs.hassyspath(relpath):
                return None
            file_path = fs.getsyspath(relpath)
            if file_path.startswith("\\\\?\\"):  # GDAL doesn't support extended filenames
                file_path = file_path[4:]
            dataset = gdal.Open(file_path)
        except RuntimeError:
            log.debug("GDAL can't open %s; not an image")
            return None
        if dataset is not None and dataset.RasterCount > 0:
            return "image/x-gdal"
        return None
