"""
loader for nc_particles files

i.e. GNOME output
"""

import os
import numpy as np
#import re

from maproom.app_framework.filesystem import fsopen as open

from .common import BaseLayerLoader
from ..library.gps_utils import GarminGPSDataset
from maproom.layers.vector_object import AnnotationLayer, PolylineObject, OverlayIconObject

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


def identify_loader(file_guess):
    if file_guess.is_text:
        if b'xmlns="http://www.topografix.com/GPX/1/1"' in file_guess.sample_data:
                return dict(mime="text/garmin-gpx", loader=GarminGPSLoader())


class GarminGPSLoader(BaseLayerLoader):
    """
    loader for nc_particles file

    creates a bunch of particle layers, one for each tiem step in the file
    """

    mime = "text/garmin-gpx"
    layer_types = ["annotation"]
    extensions = [".gpx"]
    name = "gpx"

    def can_save_layer(self, layer):
        return False

    def load_layers(self, uri, manager, **kwargs):
        """
        load the nc_particles file

        :param uri: the uri object from the file opener guess object.

        :param manager: The layer manager

        """
        layers = []
        parent = AnnotationLayer(manager=manager)
        parent.file_path = uri
        parent.mime = self.mime  # fixme: tricky here, as one file has multiple layers
        parent.name = os.path.split(parent.file_path)[1]

        xml = open(uri).read()
        gps = GarminGPSDataset(xml)
        for waypoint in gps.waypoints:
            layer = OverlayIconObject(manager=manager)
            layer.set_location((waypoint.lon, waypoint.lat))
            layer.set_style(None)
            layer.name = f"{waypoint.name}: {waypoint.time}"
            layers.append(layer)

        points = []
        for point in gps.track:
            points.append((point.lon, point.lat))
        layer = PolylineObject(manager=manager)
        layer.set_points(points)
        layer.set_style(None)
        layer.name = gps.name
        layers.append(layer)

        progress_log.info("Finished loading %s" % uri)
        layers.reverse()
        layers[0:0] = [parent]
        log.debug("Adding layers: %s" % ("\n".join([str(lr) for lr in layers])))
        return layers
