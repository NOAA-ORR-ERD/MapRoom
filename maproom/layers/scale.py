# coding=utf8
import math
import bisect
import numpy as np

# Enthought library imports.
from traits.api import Unicode, Str

from ..library import rect

from base import ScreenLayer

import logging
log = logging.getLogger(__name__)

class Scale(ScreenLayer):
    """Scale layer
    
    Shows a scale in miles/meters
    """
    name = Unicode("Scale")
    
    type = Str("scale")
    
    skip_on_insert = True
    
    
    km_steps = [item for sublist in [[i*math.pow(10,scale) for i in [1,2,5]] for scale in range(-3,5)] for item in sublist]
    km_step_count = len(km_steps)
    
    # in feet, the steps 1 through 2000 are taken from a subset of the above
    # list (of course, referring to feet here in this list), then next steps
    # are in units of miles (5280 feet)
    ft_steps = list(km_steps[9:20])
    ft_steps.extend([item for sublist in [[5280.0*i*math.pow(10,scale) for i in [1,2,5]] for scale in range(0,5)] for item in sublist])
    ft_step_count = len(ft_steps)

    # length of the scale bar in pixels
    reference_pixel_length = 50
    
    line_width = 2.0
    tick_length = 8
    tick_spacing = 5
    
    x_offset = 10
    y_offset = 20
    
    def get_visibility_dict(self):
        prefs = self.manager.project.task.get_preferences()
        d = dict()
        d["layer"] = prefs.show_scale
        return d

    def resize(self, renderer, world_rect, screen_rect):
        c = renderer.canvas
        z = c.zoom_level
        center = rect.center(world_rect)
        degrees_lon_per_pixel = float(rect.width(world_rect)) / float(rect.width(screen_rect))
        self.km_per_pixel = self.haversine(degrees_lon_per_pixel, center[1])
        self.km_length = self.get_step_size(self.reference_pixel_length * self.km_per_pixel, self.km_steps, self.km_step_count)
        
        self.ft_per_pixel = self.km_per_pixel * 3.28084 * 1000.0
        self.ft_length = self.get_step_size(self.reference_pixel_length * self.ft_per_pixel, self.ft_steps, self.ft_step_count)
    
    def get_step_size(self, reference_size, steps, count):
        return steps[min(bisect.bisect(steps, abs(reference_size)), count - 1)]
    
    def haversine(self, deg_lon, deg_lat):
        lon = math.radians(deg_lon)
        lat = math.radians(deg_lat)
        clat = math.cos(lat)
        slon = math.sin(lon/2)
        a = clat * clat * slon * slon
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        d = 6371.0 * c
        return d

    def render_screen(self, renderer, w_r, p_r, s_r, layer_visibility, layer_index_base, picker):
        if picker.is_active:
            return
        log.log(5, "Rendering scale!!! pick=%s" % (picker))
        render_window = renderer.canvas
        self.resize(renderer, w_r, s_r)

        x = s_r[0][0] + self.x_offset
        y = s_r[1][1] - self.y_offset

        if self.km_length < 1.0:
            s = "%d m" % (self.km_length * 1000)
        else:
            s = "%d km" % self.km_length
        length = self.km_length / self.km_per_pixel
        #print "km_length", self.km_length, "length", length
        size = renderer.get_drawn_string_dimensions(s)
        renderer.draw_screen_lines([(x, y - self.tick_length), (x, y), (x + self.tick_spacing + length, y), (x + self.tick_spacing + length, y - self.tick_length)], width=self.line_width)
        renderer.draw_screen_string((x + self.tick_spacing, y - size[1] - 1), s)
    
        if self.ft_length < 5000:
            s = "%d ft" % self.ft_length
        else:
            s = "%d mi" % (self.ft_length / 5280)
        length = self.ft_length / self.ft_per_pixel
        #print "ft_length", self.ft_length, "length", length
        size = renderer.get_drawn_string_dimensions(s)
        renderer.draw_screen_lines([(x, y + self.tick_length), (x, y), (x + self.tick_spacing + length, y), (x + self.tick_spacing + length, y + self.tick_length)], width=self.line_width)
        renderer.draw_screen_string((x + self.tick_spacing, y + 1), s)
