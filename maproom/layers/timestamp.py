# coding=utf8
import math
import bisect
import time

# Enthought library imports.
from traits.api import Unicode, Str, Float

from ..library import rect
from ..library.coordinates import haversine_at_const_lat, km_to_string, ft_to_string

from base import ScreenLayer

import logging
log = logging.getLogger(__name__)


class Timestamp(ScreenLayer):
    """Displays the current date & time in the image, useful for timeline
    playback
    """
    name = "Timestamp"

    type = "timestamp"

    x_percentage = Float(1.0)

    y_percentage = Float(0.0)

    # class attributes

    layer_info_panel = ["X location", "Y location"]

    skip_on_insert = True

    bounded = False

    background = True

    x_offset = 10
    y_offset = 20

    def x_percentage_to_json(self):
        return self.x_percentage

    def x_percentage_from_json(self, json_data):
        self.x_percentage = json_data['x_percentage']

    def y_percentage_to_json(self):
        return self.y_percentage

    def y_percentage_from_json(self, json_data):
        self.y_percentage = json_data['y_percentage']

    def render_screen(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        if picker.is_active:
            return
        current_time = renderer.canvas.project.timeline.current_time
        if current_time is not None:
            timestamp = time.strftime("%b %d %Y %H:%M", time.gmtime(current_time))
        else:
            begin, end = renderer.canvas.project.timeline.selected_time_range
            if begin is not None:
                t1 = time.strftime("%b %d %Y %H:%M", time.gmtime(begin))
                t2 = time.strftime("%b %d %Y %H:%M", time.gmtime(end))
                timestamp = "%s - %s" % (t1, t2)
            else:
                return
        log.log(5, "Rendering timeline at %s!!! pick=%s" % (timestamp, picker))

        size = renderer.get_drawn_string_dimensions(timestamp)

        w = s_r[1][0] - s_r[0][0] - 2 * self.x_offset - size[0]
        h = s_r[1][1] - s_r[0][1] - 2 * self.y_offset

        x = s_r[0][0] + (w * self.x_percentage) + self.x_offset
        y = s_r[1][1] - (h * self.y_percentage) - self.y_offset

        renderer.draw_screen_string((x, y), timestamp)
