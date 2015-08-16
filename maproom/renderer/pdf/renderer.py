import os
import sys
import wx
import numpy as np

from peppy2 import get_image_path

import maproom.library.rect as rect
from .. import data_types, int_to_color_floats
from .. import BaseRenderer


class ReportLabRenderer(BaseRenderer):
    def __init__(self, canvas, layer):
        self.canvas = canvas
        self.layer = layer
        self.image_textures = None
        self.image_projected_rects = []

    def prepare_to_render_projected_objects(self):
        pass

    def prepare_to_render_screen_objects(self):
        pass
    
    def set_points(self, xy, depths, color=None, num_points=-1):
        pass
    
    def set_lines(self, xy, indexes, color):
        pass
    
    def draw_lines(self,
                   layer_index_base,
                   picker,
                   style,
                   selected_line_segment_indexes=[],
                   flagged_line_segment_indexes=[]):  # flagged_line_segment_indexes not yet used
        pass

    def draw_selected_lines(self, style, selected_line_segment_indexes=[]):
        pass

    def draw_points(self,
                    layer_index_base,
                    picker,
                    point_size,
                    selected_point_indexes=[],
                    flagged_point_indexes=[]):  # flagged_line_segment_indexes not yet used
        pass

    def draw_selected_points(self, point_size, selected_point_indexes=[]):
        pass

    def draw_labels_at_points(self, values, screen_rect, projected_rect):
        pass

    def set_triangles(self, triangle_point_indexes, triangle_point_colors):
        pass

    def draw_triangles(self, line_width):
        pass

    def set_image_projection(self, image_data, projection):
        pass

    def set_image_screen(self, image_data):
        pass
    
    def set_image_center_at_screen_point(self, image_data, center, screen_rect):
        pass
    
    def release_textures(self):
        pass

    def draw_image(self, layer_index_base, picker, alpha=1.0):
        pass

    def set_invalid_polygons(self, polygons, polygon_count):
        pass

    def set_polygons(self, polygons, point_adjacency_array):
        pass

    def draw_polygons(self, layer_index_base, picker,
                      polygon_colors, line_color, line_width,
                      broken_polygon_index=None):
        pass

    def draw_screen_line(self, point_a, point_b, width=1.0, red=0.0, green=0.0, blue=0.0, alpha=1.0, stipple_factor=1, stipple_pattern=0xFFFF, xor=False):
        print "line", point_a[0], point_a[1], point_b[0], point_b[1]
        self.canvas.pdf.line(point_a[0], point_a[1], point_b[0], point_b[1])

    def draw_screen_lines(self, points, width=1.0, red=0.0, green=0.0, blue=0.0, alpha=1.0, stipple_factor=1, stipple_pattern=0xFFFF, xor=False):
        pass

    def draw_screen_markers(self, markers, style):
        pass

    def draw_screen_box(self, r, red=0.0, green=0.0, blue=0.0, alpha=1.0, width=1.0, stipple_factor=1, stipple_pattern=0xFFFF):
        pass

    def draw_screen_rect(self, r, red=0.0, green=0.0, blue=0.0, alpha=1.0):
        pass

    def get_drawn_string_dimensions(self, text):
        return (100, 20)  # FIXME: need to figure out ReportLab font metrics

    def draw_screen_string(self, point, text):
        pass

    # Vector object drawing routines

    def fill_object(self, layer_index_base, picker, style):
        pass

    def outline_object(self, layer_index_base, picker, style):
        pass
