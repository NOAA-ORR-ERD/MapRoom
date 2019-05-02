# coding=utf8

from .svg import SVGLayer

import logging
log = logging.getLogger(__name__)


class NOAALogo(SVGLayer):
    """NOAA logo overlay layer
    """
    name = "NOAA Logo"

    type = "noaa_logo"

    default_svg_source = "template://noaa_logo.svg"

    def __init__(self, manager):
       super().__init__(manager, None, x_percentage=1.0, y_percentage=1.0, magnification=0.1)
