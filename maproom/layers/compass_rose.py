# coding=utf8

from .svg import SVGLayer

import logging
log = logging.getLogger(__name__)


class CompassRose(SVGLayer):
    """Compass Rose layer

    Shows a compass rose or north-up arrow as a graphic overlay
    """
    name = "Compass Rose"

    type = "compass_rose"

    default_svg_source = "template://compass_rose.svg"
