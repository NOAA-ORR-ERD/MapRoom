# coding=utf8

from ..library import rect
from ..library.svg_utils import SVGOverlay

from .base import StickyResizableLayer

import logging
log = logging.getLogger(__name__)


class SVGLayer(StickyResizableLayer):
    """SVG screen layer

    Shows an svg as an overlay layer
    """
    name = "SVG"

    type = "svg"

    # SVG source text goes here
    default_svg_source = ""

    skip_on_insert = True

    bounded = False

    def __init__(self, manager, svg_source=None, x_percentage=1.0, y_percentage=0.0, magnification=0.2):
        super().__init__(manager, x_percentage, y_percentage, magnification)
        self._svg = None
        self._svg_source = None
        self.svg_source = self.default_svg_source if svg_source is None else svg_source

    @property
    def svg(self):
        if self._svg is None:
            self._svg = SVGOverlay(self.svg_source)
        return self._svg

    @property
    def svg_source(self):
        return self._svg_source

    @svg_source.setter
    def svg_source(self, value):
        self._svg_source = value
        self._svg = None  # force recreation of svg overlay

    def render_screen(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        if picker.is_active:
            return
        log.log(5, f"Rendering svg {self.name} rose!!! pick={picker}")

        if self.svg.height > 0:
            object_ar = self.svg.width / self.svg.height
        else:
            object_ar = 1.0

        usable_w = s_r[1][0] - s_r[0][0] - 2 * self.x_offset
        usable_h = s_r[1][1] - s_r[0][1] - 2 * self.y_offset
        if usable_h > 0:
            usable_ar = usable_w / usable_h
        else:
            usable_ar = 1.0

        if object_ar > usable_ar:
            # wider object than screen, so max based on width
            max_w = usable_w
            max_h = usable_h / object_ar
        else:
            # taller object than screen; max based on height
            max_h = usable_h
            max_w = usable_w / usable_ar

        w = max_w * self.magnification
        h = max_h * self.magnification

        x = s_r[0][0] + ((usable_w - w) * self.x_percentage) + self.x_offset
        y = s_r[1][1] - ((usable_h - h) * self.y_percentage) - self.y_offset

        r = rect.get_rect_of_points([(x, y), (x + w, y - h)])

        renderer.draw_screen_svg(r, self.svg)
