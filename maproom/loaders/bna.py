import os

from sawx.filesystem import fsopen as open

import numpy as np
from shapely.geometry import Polygon, LineString

from maproom.library.shapely_utils import add_maproom_attributes_to_shapely_geom
from maproom.library.bna_utils import load_bna_file, save_bna_file
from maproom.layers import RNCLoaderLayer

from .common import BaseLayerLoader

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


def identify_loader(file_guess):
    if file_guess.is_text and file_guess.uri.lower().endswith(".bna"):
        lines = file_guess.sample_lines
        if b".KAP" in lines[0]:
            return dict(mime="application/x-maproom-rncloader", loader=RNCLoader())


class RNCLoader(BaseLayerLoader):
    mime = "application/x-maproom-rncloader"

    layer_types = ["rncloader"]

    extensions = [".bna"]

    name = "RNCLoader"

    layer_class = RNCLoaderLayer

    def can_save_layer(self, layer):
        return False

    def load_layers(self, uri, manager, **kwargs):
        layer = self.layer_class(manager=manager)

        (layer.load_error_string,
         f_ring_points,
         f_ring_starts,
         f_ring_counts,
         f_ring_identifiers) = load_bna_file(uri, regimes=[0, 360])
        progress_log.info("Creating layer...")
        if (layer.load_error_string == ""):
            layer.set_data(f_ring_points, f_ring_starts, f_ring_counts,
                           f_ring_identifiers)
            layer.file_path = uri
            layer.name = os.path.split(layer.file_path)[1]
            layer.mime = self.mime
        return [layer]

    def save_to_fh(self, fh, layer):
        save_bna_file(fh, layer)
