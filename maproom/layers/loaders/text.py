import os
import numpy as np
import re

from fs.opener import fsopen

from maproom.library.accumulator import accumulator
from maproom.library.lat_lon_parser import parse_coordinate_text

from common import BaseLayerLoader
from maproom.layers import LineLayer

import logging
progress_log = logging.getLogger("progress")

WHITESPACE_PATTERN = re.compile("\s+")


class TextMixin(object):
    def load_layers(self, metadata, manager, **kwargs):
        layer = LineLayer(manager=manager)

        progress_log.info("Loading from %s" % metadata.uri)
        layer.load_error_string, f_points, f_depths, f_line_segment_indexes = self.load_text(metadata.uri)
        if (layer.load_error_string == ""):
            progress_log.info("Finished loading %s" % metadata.uri)
            layer.set_data(f_points, f_depths, f_line_segment_indexes)
            layer.file_path = metadata.uri
            layer.name = "%s (%s)" % (os.path.split(layer.file_path)[1], self.mime)
            layer.mime = self.mime
        return [layer]

    def load_text(self, uri):
        in_file = fsopen(uri, "r")
        text = in_file.read()
        in_file.close()

        mime, points, num_unmatched = parse_coordinate_text(text)
        #log.debug("%s: (%d unmatched): %s" % (mime, num_unmatched, points))
        return "",  np.asarray(points, dtype=np.float64), 0.0, []


class LatLonTextLoader(TextMixin, BaseLayerLoader):
    mime = "text/latlon"

    layer_types = ["line"]

    extensions = [".txt"]

    name = "Text Lat/Lon"


class LonLatTextLoader(TextMixin, BaseLayerLoader):
    mime = "text/lonlat"

    layer_types = ["line"]

    extensions = [".txt"]

    name = "Text Lon/Lat"