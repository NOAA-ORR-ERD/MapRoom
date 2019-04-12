# coding=utf8
import math
import bisect

from ..library import rect
from ..library.coordinates import haversine_at_const_lat, km_to_string, ft_to_string

from .base import StickyLayer

import logging
log = logging.getLogger(__name__)


class Scale(StickyLayer):
    """Scale layer

    Shows a scale in miles/meters
    """
    name = "Scale"

    type = "scale"

    layer_info_panel = ["X location", "Y location"]

    skip_on_insert = True

    bounded = False

    background = True

    km_steps = [item for sublist in [[i * math.pow(10, scale) for i in [1, 2, 5]] for scale in range(-3, 5)] for item in sublist]
    km_step_count = len(km_steps)

    # in feet, the steps 1 through 2000 are taken from a subset of the above
    # list (of course, referring to feet here in this list), then next steps
    # are in units of miles (5280 feet)
    ft_steps = list(km_steps[9:20])
    ft_steps.extend([item for sublist in [[5280.0 * i * math.pow(10, scale) for i in [1, 2, 5]] for scale in range(0, 5)] for item in sublist])
    ft_step_count = len(ft_steps)

    # length of the scale bar in pixels
    reference_pixel_length = 50

    line_width = 2.0
    tick_length = 8
    tick_spacing = 5

    x_offset = 10
    y_offset = 20

    def get_visibility_dict(self, project):
        prefs = project.preferences
        d = dict()
        d["layer"] = prefs.show_scale
        return d

    def resize(self, renderer, world_rect, screen_rect):
        center = rect.center(world_rect)
        degrees_lon_per_pixel = float(rect.width(world_rect)) / float(rect.width(screen_rect))
        self.km_per_pixel = haversine_at_const_lat(degrees_lon_per_pixel, center[1])
        self.km_length = self.get_step_size(self.reference_pixel_length * self.km_per_pixel, self.km_steps, self.km_step_count)

        self.ft_per_pixel = self.km_per_pixel * 3.28084 * 1000.0
        self.ft_length = self.get_step_size(self.reference_pixel_length * self.ft_per_pixel, self.ft_steps, self.ft_step_count)

    def get_step_size(self, reference_size, steps, count):
        return steps[min(bisect.bisect(steps, abs(reference_size)), count - 1)]

    def render_screen(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        if picker.is_active:
            return
        log.log(5, "Rendering scale!!! pick=%s" % (picker))
        self.resize(renderer, w_r, s_r)

        km_label = km_to_string(self.km_length)
        km_length = self.km_length / self.km_per_pixel
        # print "km_length", self.km_length, "length", length

        ft_label = ft_to_string(self.ft_length)
        ft_length = self.ft_length / self.ft_per_pixel
        # print "ft_length", self.ft_length, "length", length

        w = s_r[1][0] - s_r[0][0] - 2 * self.x_offset - max(km_length, ft_length)
        h = s_r[1][1] - s_r[0][1] - 2 * self.y_offset

        x = s_r[0][0] + (w * self.x_percentage) + self.x_offset
        y = s_r[1][1] - (h * self.y_percentage) - self.y_offset

        size = renderer.get_drawn_string_dimensions(km_label)
        renderer.draw_screen_lines([(x, y - self.tick_length), (x, y), (x + self.tick_spacing + km_length, y), (x + self.tick_spacing + km_length, y - self.tick_length)], width=self.line_width)
        renderer.draw_screen_string((x + self.tick_spacing, y - size[1] - 1), km_label)
        size = renderer.get_drawn_string_dimensions(ft_label)
        renderer.draw_screen_lines([(x, y + self.tick_length), (x, y), (x + self.tick_spacing + ft_length, y), (x + self.tick_spacing + ft_length, y + self.tick_length)], width=self.line_width)
        renderer.draw_screen_string((x + self.tick_spacing, y + 1), ft_label)
