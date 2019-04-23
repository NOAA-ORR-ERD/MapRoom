# coding=utf8
import math
import bisect

from ..library import rect
from ..library.coordinates import haversine_at_const_lat, km_to_string, ft_to_string
from ..renderer import int_to_color_floats, int_to_color_uint8, int_to_html_color_string, alpha_from_int, ImageData

from .base import StickyLayer

import logging
log = logging.getLogger(__name__)


class Title(StickyLayer):
    """Text overlay for titles or other annotations that are fixed relative
    to pixel coordinates.
    """
    name = "Title"

    type = "title"

    mouse_mode_toolbar = "StickyLayerToolBar"

    # layer_info_panel = ["X location", "Y location", "Text color", "Font", "Font size", "Border width", "Line style", "Line width", "Line color", "Fill style", "Fill color"]
    layer_info_panel = ["X location", "Y location", "Text color", "Font", "Font size"]

    selection_info_panel = ["Text", "Text format"]

    skip_on_insert = True

    bounded = False

    background = True

    pickable = True

    x_offset = 20
    y_offset = 20

    def __init__(self, manager, text="Title", x_percentage=0.5, y_percentage=1.0):
        super().__init__(manager, x_percentage, y_percentage)
        self._user_text = text
        self.image_data = None

        self.text_width = 50
        self.text_height = 50

        self.rebuild_needed = True

    @property
    def user_text(self):
        return self._user_text

    @user_text.setter
    def user_text(self, value):
        self._user_text = value
        self.rebuild_needed = True

    def user_text_to_json(self):
        return self._user_text

    def user_text_from_json(self, json_data):
        self._user_text = json_data['user_text']

    def get_text_box(self):
        return self

    def set_style(self, style):
        super().set_style(style)
        self.rebuild_needed = True  # Force rebuild to change font style

    def get_image_array(self):
        from maproom.library.numpy_images import OffScreenHTML
        bg = int_to_color_uint8(self.style.fill_color)
        h = OffScreenHTML(self.screen_width, min(self.style.font_size * 10, self.screen_height), bg)
        c = int_to_html_color_string(self.style.text_color)
        arr = h.get_numpy(self.user_text, c, self.style.font, self.style.font_size, self.style.text_format)
        return arr

    def rebuild_image(self, renderer):
        """Update renderer

        """
        if self.rebuild_needed:
            renderer.release_textures()
            self.image_data = None
        if self.image_data is None:
            s_r = renderer.canvas.screen_rect
            self.screen_width = s_r[1][0]
            self.screen_height = s_r[1][1]
            raw = self.get_image_array()
            self.image_data = ImageData(raw.shape[1], raw.shape[0])
            self.text_width = raw.shape[1]
            self.text_height = raw.shape[0]
            self.image_data.load_numpy_array(None, raw, None)
            log.debug(f"rebuild_image: new image created: {self.text_width},{self.text_height}")
        renderer.set_image_screen(self.image_data)

    def rebuild_renderer(self, renderer, in_place=False):
        """Update renderer

        """
        self.rebuild_image(renderer)
        self.rebuild_needed = False

    def render_screen(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        log.log(5, "Rendering scale!!! pick=%s" % (picker))

        if self.screen_width != s_r[1][0] or self.screen_height != s_r[1][1]:
            self.rebuild_needed = True
        if self.rebuild_needed:
            self.rebuild_renderer(renderer)
        bounding_box = self.calc_bounding_box(s_r, self.text_width, self.text_height)
        renderer.set_image_to_screen_rect(self.image_data, bounding_box)
        if picker.is_active:
            c = picker.get_polygon_picker_colors(self, 1)[0]
            r, g, b, a = int_to_color_floats(c)
            renderer.draw_screen_rect(bounding_box, r, g, b, a, flip=False)
        else:
            # Only render text when we're not drawing the picker framebuffer
            renderer.prepare_to_render_screen_objects()
            alpha = alpha_from_int(self.style.text_color)
            renderer.draw_image(self, picker, alpha)
