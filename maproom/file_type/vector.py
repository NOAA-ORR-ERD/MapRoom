from fs.opener import opener
from osgeo import ogr
ogr.UseExceptions()

from traits.api import HasTraits, provides

from omnivore.file_type.i_file_recognizer import IFileRecognizer

import logging
log = logging.getLogger(__name__)

@provides(IFileRecognizer)
class OGRRecognizer(HasTraits):
    """Check to see if OGR can open this as a vector shapefile.

    """
    id = "application/x-maproom-shapefile"
    
    before = "text/*"
    
    def identify(self, guess):
        try:
            fs, relpath = opener.parse(guess.metadata.uri)
            if not fs.hassyspath(relpath):
                return None
            file_path = fs.getsyspath(relpath)
            if file_path.startswith("\\\\?\\"):  # GDAL doesn't support extended filenames
                file_path = file_path[4:]
            dataset = ogr.Open(file_path)
        except RuntimeError:
            log.debug("OGR can't open %s; not an image")
            return None
        if dataset is not None and dataset.GetLayerCount() > 0:
            return self.id
        return None
