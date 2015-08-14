# coding=utf8
import bisect
import numpy as np

# Enthought library imports.
from traits.api import Unicode, Str

from ..library import rect, coordinates

from base import ScreenLayer

import logging
log = logging.getLogger(__name__)

class Grid(ScreenLayer):
    """Root layer
    
    Only one root layer per project.
    """
    name = Unicode("Lon/Lat Grid")
    
    type = Str("grid")
    
    skip_on_insert = True
    
    LINE_WIDTH = 1.0
    LINE_COLOR = (0, 0, 0, 0.75)
    # a nice amount of spacing between lines
    REFERENCE_PIXEL_SIZE = (100, 100)
    
    def resize(self, renderer, world_rect, screen_rect):
        prefs = renderer.canvas.project.task.get_preferences()
        if prefs.coordinate_display_format == "decimal degrees":
            self.grid = DecimalDegreeGridLines()
        else:
            self.grid = DegreeMinuteGridLines()
        self.lat_step = self.grid.get_step_size(0)
        self.lon_step = self.grid.get_step_size(0)

        degrees_lon_per_pixel = float(rect.width(world_rect)) / float(rect.width(screen_rect))
        degrees_lat_per_pixel = float(rect.height(world_rect)) / float(rect.height(screen_rect))

        self.lon_step = self.grid.get_step_size(self.REFERENCE_PIXEL_SIZE[0] * degrees_lon_per_pixel)
        self.lon_steps = np.arange(
            world_rect[0][0] + self.lon_step - world_rect[0][0] % self.lon_step,
            world_rect[1][0],
            self.lon_step,
            dtype=np.float64)

        self.lat_step = self.grid.get_step_size(self.REFERENCE_PIXEL_SIZE[1] * degrees_lat_per_pixel)
        self.lat_steps = np.arange(
            world_rect[0][1] + self.lat_step - world_rect[0][1] % self.lat_step,
            world_rect[1][1],
            self.lat_step,
            dtype=np.float64)

    # fixme == this should be able to get the various rects from the render_window object...
    def render_screen(self, renderer, world_rect, projected_rect, screen_rect, layer_visibility, layer_index_base, picker):
        if (not layer_visibility["layer"] or picker.is_active):
            return
        log.log(5, "Rendering grid!!! visible=%s, pick=%s" % (layer_visibility["layer"], picker))
        render_window = renderer.canvas
#        print "projected_rect = %r" % (projected_rect,)
#        print "screen_rect = %r" % (screen_rect,)
        self.resize(renderer, world_rect, screen_rect)
#        print "lon_step = " + str(self.lon_step)
#        print "lat_step = " + str(self.lat_step)
#        print "world_rect = " + str(world_rect)

        for longitude in self.lon_steps:

#            print "  longitude = " + str(longitude)
            if (longitude < -180 or longitude > 180):
                continue
            w_p = (longitude, world_rect[0][1])
            s_p = render_window.get_screen_point_from_world_point(w_p)
            s = self.grid.format_lon_line_label(longitude)
            size = renderer.get_drawn_string_dimensions(s)
            renderer.draw_screen_line((s_p[0], screen_rect[0][1] + size[1] + 5),
                                             (s_p[0], screen_rect[1][1]))
            """
            for offset in xrange( 200 ):
                renderer.draw_screen_string( ( s_p[ 0 ] - size[ 0 ] / 2, screen_rect[ 0 ][ 1 ] + offset * 2 ), s )
            """
            renderer.draw_screen_string((s_p[0] - size[0] / 2, screen_rect[0][1]), s)

        for latitude in self.lat_steps:

#            print "  latitude = " + str(latitude)
            if (latitude < -89 or latitude > 89):
                continue
            w_p = (world_rect[0][0], latitude)
            s_p = render_window.get_screen_point_from_world_point(w_p)
            s = self.grid.format_lat_line_label(latitude)
            size = renderer.get_drawn_string_dimensions(s)
            renderer.draw_screen_line((screen_rect[0][0], s_p[1]),
                                             (screen_rect[1][0] - size[0] - 5, s_p[1]))
            renderer.draw_screen_string(
                (screen_rect[1][0] - size[0] - 3, s_p[1] - size[1] / 2 - 1), s)


class GridLines(object):
    def get_step_size(self, reference_size):
        return self.STEPS[min(
            bisect.bisect(self.STEPS, abs(reference_size)),
            self.STEP_COUNT - 1,
        )]
    
class DegreeMinuteGridLines(GridLines):
    DEGREE = np.float64(1.0)
    MINUTE = DEGREE / 60.0
    SECOND = MINUTE / 60.0

    STEPS = (
        MINUTE,
        MINUTE * 2,
        MINUTE * 3,
        MINUTE * 4,
        MINUTE * 5,
        MINUTE * 10,
        MINUTE * 15,
        MINUTE * 20,
        MINUTE * 30,
        DEGREE,
        DEGREE * 2,
        DEGREE * 3,
        DEGREE * 4,
        DEGREE * 5,
        DEGREE * 10,
        DEGREE * 15,
        DEGREE * 20,
        DEGREE * 30,
        DEGREE * 40,
    )
    STEP_COUNT = len(STEPS)
    
    def format_lat_line_label(self, latitude):
        return coordinates.format_lat_line_label(latitude)

    def format_lon_line_label(self, longitude):
        return coordinates.format_lon_line_label(longitude)

class DecimalDegreeGridLines(GridLines):
    DEGREE = np.float64(1.0)
    TENTH = DEGREE / 10.0
    HUNDREDTH = DEGREE / 100.0

    STEPS = (
        HUNDREDTH,
        HUNDREDTH * 2,
        HUNDREDTH * 5,
        TENTH,
        TENTH * 2,
        TENTH * 5,
        DEGREE,
        DEGREE * 2,
        DEGREE * 3,
        DEGREE * 4,
        DEGREE * 5,
        DEGREE * 10,
        DEGREE * 15,
        DEGREE * 20,
        DEGREE * 30,
        DEGREE * 40,
    )
    STEP_COUNT = len(STEPS)
    
    def format_lat_line_label(self, latitude):
        ( degrees, direction ) = \
            coordinates.float_to_degrees(latitude, directions=("N", "S"))

        return u" %.2f° %s " % (degrees, direction)

    def format_lon_line_label(self, longitude):
        ( degrees, direction ) = \
            coordinates.float_to_degrees(longitude, directions=("E", "W"))

        return u" %.2f° %s " % (degrees, direction)
