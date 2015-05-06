"""Loader for annotation layers

"""

import os
import numpy as np
#import re

from common import BaseLayerLoader
from maproom.layers.float_canvas import FloatCanvasLayer

import logging
progress_log = logging.getLogger("progress")


class FloatCanvasJSONLoader(BaseLayerLoader):
    """Loader for FloatCanvas save files

    """
    mime = "application/x-float_canvas_json"
    layer_types = ["annotation"]
    extensions = [".fc"]
    name = "FloatCanvas JSON Layer"
    
    def load_layers(self, metadata, manager):
        """Load the FloatCanvas save file

        :param metadata: the metadata object from the file opener guess object.

        :param manager: The layer manager

        """
        layers = []
        with open(metadata.uri, "r") as fh:
            text = fh.read()
            layer = FloatCanvasLayer(manager=manager)
            layer.load_fc_json(text)
            layers.append(layer)
        return layers
