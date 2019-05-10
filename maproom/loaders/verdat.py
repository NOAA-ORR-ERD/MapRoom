import os
import numpy as np
import re

from sawx.filesystem import fsopen as open

from maproom.library.accumulator import accumulator
from maproom.library.Boundary import Boundaries, PointsError
from maproom.library.verdat_utils import load_verdat_file, write_layer_as_verdat

from .common import BaseLayerLoader
from maproom.layers import LineLayer

import logging
progress_log = logging.getLogger("progress")

WHITESPACE_PATTERN = re.compile("\s+")


def identify_loader(file_guess):
    if file_guess.is_text:
        if file_guess.sample_data.startswith(b"DOGS"):
            mime = "application/x-maproom-verdat"
            return dict(mime=mime, loader=VerdatLoader())


class VerdatLoader(BaseLayerLoader):
    mime = "application/x-maproom-verdat"

    layer_types = ["line"]

    extensions = [".verdat", ".dat"]

    name = "Verdat"

    points_per_tick = 5000

    def load_layers(self, uri, manager, **kwargs):
        layer = LineLayer(manager=manager)

        progress_log.info("Loading from %s" % uri)
        (layer.load_error_string, layer.load_warning_string,
         f_points,
         f_depths,
         f_line_segment_indexes,
         layer.depth_unit) = load_verdat_file(uri, self.points_per_tick)
        if (layer.load_error_string == ""):
            progress_log.info("Finished loading %s" % uri)
            layer.set_data(f_points, f_depths, f_line_segment_indexes)
            layer.file_path = uri
            layer.name = os.path.split(layer.file_path)[1]
            layer.mime = self.mime
        return [layer]

    def save_to_fh(self, fh, layer):
        return write_layer_as_verdat(fh, layer, self.points_per_tick)
