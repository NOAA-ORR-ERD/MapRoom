# coding=utf8

from sawx.filesystem import fsopen as open

from ..library import rect
from ..library.svg_utils import SVGOverlay
from ..renderer import int_to_color_floats

from .base import StickyResizableLayer

import logging
log = logging.getLogger(__name__)


svg_error_box = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   width="50mm"
   height="50mm"
   viewBox="0 0 50 50"
   version="1.1"
   id="svg8"
   inkscape:version="0.92.2 5c3e80d, 2017-08-06"
   sodipodi:docname="missing.svg">
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1">
    <rect
       style="color:#000000;clip-rule:nonzero;display:inline;overflow:visible;visibility:visible;opacity:1;isolation:auto;mix-blend-mode:normal;color-interpolation:sRGB;color-interpolation-filters:linearRGB;solid-color:#000000;solid-opacity:1;fill:none;fill-opacity:1;fill-rule:nonzero;stroke:#e45c2a;stroke-width:5;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1;paint-order:markers stroke fill;color-rendering:auto;image-rendering:auto;shape-rendering:auto;text-rendering:auto;enable-background:accumulate"
       id="rect815"
       width="50"
       height="50"
       x="0"
       y="0" />
  </g>
</svg>"""


class SVGLayer(StickyResizableLayer):
    """SVG screen layer

    Shows an svg as an overlay layer
    """
    name = "SVG"

    type = "svg"

    layer_info_panel = ["X location", "Y location", "Magnification", "SVG status", "Load SVG"]

    default_svg_source = svg_error_box

    def __init__(self, manager, svg_source=None, x_percentage=1.0, y_percentage=0.0, magnification=0.2):
        super().__init__(manager, x_percentage, y_percentage, magnification)
        self._svg = None
        self._svg_source = None
        self.svg_parse_error = None
        self.svg_source = self.default_svg_source if svg_source is None else svg_source

    @property
    def svg(self):
        if self._svg is None:
            try:
                self._svg = SVGOverlay(self.svg_source)
            except Exception as e:
                self.svg_parse_error = str(e)
                self._svg = SVGOverlay(svg_error_box)
            else:
                self.svg_parse_error = None
        return self._svg

    @property
    def svg_source(self):
        return self._svg_source

    @svg_source.setter
    def svg_source(self, value):
        if value.startswith("template://") or value.endswith(".svg"):
            try:
                value = open(value).read()
            except IOError as e:
                log.error(f"Failed reading svg source file {value}: {e}")
        self._svg_source = value
        self._svg = None  # force recreation of svg overlay

    def svg_source_to_json(self):
        return self._svg_source

    def svg_source_from_json(self, json_data):
        self.svg_source = json_data['svg_source']

    def get_undo_info(self):
        return (self.svg_source,)

    def restore_undo_info(self, undo_info):
        s = undo_info[0]
        self.svg_source = s

    def render_screen(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        log.log(5, f"Rendering svg {self.name} rose!!! pick={picker}")

        svg = self.svg
        if self.svg_parse_error is not None:
            log.error(f"Can't render SVG: {self.svg_parse_error}")
        if svg is None:
            # even the backup SVG is bad; don't render
            return

        if svg.height > 0:
            object_ar = svg.width / svg.height
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

        # SVG coordinate origin is lower left
        x = s_r[0][0] + ((usable_w - w) * self.x_percentage) + self.x_offset
        y = s_r[0][1] + ((usable_h - h) * self.y_percentage) + self.y_offset

        bounding_box = rect.get_rect_of_points([(x, y), (x + w, y + h)])
        if picker.is_active:
            c = picker.get_polygon_picker_colors(self, 1)[0]
            r, g, b, a = int_to_color_floats(c)
            w = s_r[1][0] - s_r[0][0] - 2 * self.x_offset - w
            h = s_r[1][1] - s_r[0][1] - 2 * self.y_offset - h
            self.usable_screen_size = (w, h)
            renderer.draw_screen_rect(bounding_box, r, g, b, a, flip=False)
        else:
            renderer.draw_screen_svg(bounding_box, svg)
