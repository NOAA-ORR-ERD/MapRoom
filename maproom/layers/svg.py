# coding=utf8

# Enthought library imports.
from traits.api import on_trait_change, Unicode, Str, Any, Float

from ..library import rect
from ..library.svg_utils import SVGOverlay

from .base import ScreenLayer

import logging
log = logging.getLogger(__name__)


class SVGLayer(ScreenLayer):
    """SVG screen layer

    Shows an svg as an overlay layer
    """
    name = "SVG"

    type = "svg"

    x_percentage = Float(1.0)

    y_percentage = Float(0.0)

    magnification = Float(0.2)

    # SVG source text goes here
    svg_source = Str("")

    svg = Any

    skip_on_insert = True

    # class attributes

    bounded = False

    layer_info_panel = ["X location", "Y location", "Magnification"]

    x_offset = 10
    y_offset = 10

    def _svg_default(self):
        return SVGOverlay(self.svg_source)

    @on_trait_change('svg_source')
    def svg_changed(self):
        self.svg = self._svg_default()

    ##### serialization

    def x_percentage_to_json(self):
        return self.x_percentage

    def x_percentage_from_json(self, json_data):
        self.x_percentage = json_data['x_percentage']

    def y_percentage_to_json(self):
        return self.y_percentage

    def y_percentage_from_json(self, json_data):
        self.y_percentage = json_data['y_percentage']

    def magnification_to_json(self):
        return self.magnification

    def magnification_from_json(self, json_data):
        self.magnification = json_data['magnification']

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
